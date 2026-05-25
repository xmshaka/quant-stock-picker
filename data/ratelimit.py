"""限流 / 熔断 / 退避重试 - 统一的请求治理层

设计要点:
1. TokenBucket: 令牌桶限流, 平滑控制每个数据源的 QPS
2. CircuitBreaker: 三态熔断 (CLOSED/OPEN/HALF_OPEN), 防止把已经挂掉的源继续打爆
3. retry_with_backoff: 指数退避 + 抖动, 避免雪崩
4. RateLimitedSession: 把上面三件套打包给 requests.Session 用

所有状态都是进程内的, 多进程场景下需要换成 Redis 实现 (后续 P5 可做)
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Optional

from loguru import logger


# ════════════════════════════════════════════════════════════
# 1. 令牌桶限流
# ════════════════════════════════════════════════════════════
class TokenBucket:
    """线程安全的令牌桶
    
    qps: 每秒生成的令牌数 (= 稳态 QPS)
    burst: 桶容量, 允许的瞬时突发数 (默认 = qps, 即不突发)
    """

    def __init__(self, qps: float, burst: Optional[float] = None, name: str = ""):
        if qps <= 0:
            raise ValueError("qps must be > 0")
        self.qps = qps
        self.burst = burst if burst is not None else qps
        self.name = name
        self._tokens = self.burst
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
        # 统计
        self.total_acquired = 0
        self.total_waited_seconds = 0.0

    def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """获取 tokens 个令牌, 不够则阻塞等待
        
        timeout: 最长等待秒数, None=无限等待
        return: True=成功, False=超时
        """
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        wait_start = time.monotonic()

        while True:
            with self._lock:
                now = time.monotonic()
                # 按时间补充令牌
                elapsed = now - self._last_refill
                self._tokens = min(self.burst, self._tokens + elapsed * self.qps)
                self._last_refill = now

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self.total_acquired += tokens
                    self.total_waited_seconds += (now - wait_start)
                    return True

                # 计算需要等多久
                need = tokens - self._tokens
                wait = need / self.qps

            if deadline is not None and (time.monotonic() + wait) > deadline:
                logger.warning(f"[RateLimit:{self.name}] 等待超时 (need {wait:.2f}s)")
                return False

            # 轻微抖动避免线程同步唤醒
            time.sleep(wait + random.uniform(0, 0.05))

    def stats(self) -> dict:
        return {
            "name": self.name,
            "qps": self.qps,
            "burst": self.burst,
            "tokens_now": round(self._tokens, 2),
            "total_acquired": self.total_acquired,
            "avg_wait_ms": round(
                self.total_waited_seconds * 1000 / max(self.total_acquired, 1), 2
            ),
        }


# ════════════════════════════════════════════════════════════
# 2. 熔断器
# ════════════════════════════════════════════════════════════
class CircuitState(str, Enum):
    CLOSED = "closed"        # 正常放行
    OPEN = "open"            # 熔断, 直接拒绝
    HALF_OPEN = "half_open"  # 半开, 试探性放行


@dataclass
class CircuitBreaker:
    """简化的三态熔断器
    
    failure_threshold: 连续失败多少次进入 OPEN
    recovery_seconds:  OPEN 多久后转 HALF_OPEN 试探
    half_open_max:     HALF_OPEN 状态最多放行多少试探请求
    """
    failure_threshold: int = 10
    recovery_seconds: int = 300
    half_open_max: int = 3
    name: str = ""

    state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    consecutive_failures: int = field(default=0, init=False)
    opened_at: float = field(default=0.0, init=False)
    half_open_inflight: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    # 统计
    total_rejected: int = field(default=0, init=False)
    total_opened_times: int = field(default=0, init=False)

    def allow_request(self) -> bool:
        """是否允许这次请求通过"""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                # 检查是否到了恢复时间
                if time.monotonic() - self.opened_at >= self.recovery_seconds:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_inflight = 1
                    logger.info(f"[Circuit:{self.name}] OPEN -> HALF_OPEN, 开始试探")
                    return True
                self.total_rejected += 1
                return False

            # HALF_OPEN
            if self.half_open_inflight < self.half_open_max:
                self.half_open_inflight += 1
                return True
            self.total_rejected += 1
            return False

    def on_success(self):
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                logger.info(f"[Circuit:{self.name}] HALF_OPEN -> CLOSED, 恢复正常")
                self.state = CircuitState.CLOSED
                self.half_open_inflight = 0
            self.consecutive_failures = 0

    def on_failure(self):
        with self._lock:
            self.consecutive_failures += 1

            if self.state == CircuitState.HALF_OPEN:
                # 试探失败, 再 OPEN 一轮
                logger.warning(f"[Circuit:{self.name}] HALF_OPEN 试探失败, 重新 OPEN")
                self.state = CircuitState.OPEN
                self.opened_at = time.monotonic()
                self.half_open_inflight = 0
                self.total_opened_times += 1
                return

            if (self.state == CircuitState.CLOSED
                    and self.consecutive_failures >= self.failure_threshold):
                logger.error(
                    f"[Circuit:{self.name}] 连续失败 {self.consecutive_failures} 次, "
                    f"熔断 {self.recovery_seconds}s"
                )
                self.state = CircuitState.OPEN
                self.opened_at = time.monotonic()
                self.total_opened_times += 1

    def stats(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "total_rejected": self.total_rejected,
            "total_opened_times": self.total_opened_times,
        }


# ════════════════════════════════════════════════════════════
# 3. 退避重试装饰器
# ════════════════════════════════════════════════════════════
class RateLimitError(Exception):
    """限流相关错误 (熔断拒绝、QPS超时等)"""


def retry_with_backoff(
    func: Callable,
    *args,
    max_attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: tuple = (Exception,),
    no_retry_on: tuple = (RateLimitError,),
    on_failure: Optional[Callable[[Exception], None]] = None,
    name: str = "",
    **kwargs,
):
    """指数退避 + 抖动重试
    
    delay 序列: base, base*2, base*4, base*8 ... (clip max_delay)
    每次再 ×[0.8, 1.2] 抖动, 避免多线程同步雪崩
    
    no_retry_on 优先级高于 retry_on: 命中直接抛错
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except no_retry_on as e:
            raise
        except retry_on as e:
            last_exc = e
            if attempt == max_attempts - 1:
                break
            delay = min(max_delay, base_delay * (2 ** attempt))
            delay = delay * random.uniform(0.8, 1.2)
            logger.warning(
                f"[Retry:{name}] 第 {attempt+1}/{max_attempts} 次失败: {e}, "
                f"{delay:.1f}s 后重试"
            )
            time.sleep(delay)

    if on_failure and last_exc:
        try:
            on_failure(last_exc)
        except Exception:
            pass
    raise last_exc if last_exc else RuntimeError("retry failed without exception")


# ════════════════════════════════════════════════════════════
# 4. 数据源治理网关 - 一个全局单例统管所有源
# ════════════════════════════════════════════════════════════
class SourceGateway:
    """每个数据源 = 一个 (TokenBucket, CircuitBreaker)
    
    使用方式:
        gw = SourceGateway.get()
        with gw.guard("tencent_kline"):
            resp = requests.get(...)
    """

    _instance: Optional["SourceGateway"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._buckets: Dict[str, TokenBucket] = {}
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._register_lock = threading.Lock()

    @classmethod
    def get(cls) -> "SourceGateway":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance._bootstrap_defaults()
        return cls._instance

    def _bootstrap_defaults(self):
        """从 settings 加载默认配置"""
        try:
            from config.settings import settings
            self.register("tencent_qt",
                          qps=settings.rate_limit_tencent_qt,
                          failure_threshold=settings.circuit_failure_threshold,
                          recovery_seconds=settings.circuit_recovery_seconds)
            self.register("tencent_kline",
                          qps=settings.rate_limit_tencent_kline,
                          failure_threshold=settings.circuit_failure_threshold,
                          recovery_seconds=settings.circuit_recovery_seconds)
            self.register("akshare",
                          qps=settings.rate_limit_akshare,
                          failure_threshold=settings.circuit_failure_threshold,
                          recovery_seconds=settings.circuit_recovery_seconds)
            self.register("tushare",
                          qps=settings.rate_limit_tushare,
                          failure_threshold=settings.circuit_failure_threshold,
                          recovery_seconds=settings.circuit_recovery_seconds)
        except Exception as e:
            logger.warning(f"[Gateway] 默认注册失败: {e}")

    def register(
        self,
        source: str,
        qps: float,
        burst: Optional[float] = None,
        failure_threshold: int = 10,
        recovery_seconds: int = 300,
        half_open_max: int = 3,
    ):
        with self._register_lock:
            self._buckets[source] = TokenBucket(qps=qps, burst=burst, name=source)
            self._breakers[source] = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_seconds=recovery_seconds,
                half_open_max=half_open_max,
                name=source,
            )
            logger.info(f"[Gateway] 注册数据源 {source}: qps={qps}")

    def acquire(self, source: str, timeout: Optional[float] = 30.0):
        """获取放行许可: 先过熔断, 再过令牌桶"""
        breaker = self._breakers.get(source)
        bucket = self._buckets.get(source)
        if breaker is None or bucket is None:
            return  # 未注册的源不限流

        if not breaker.allow_request():
            raise RateLimitError(f"[{source}] 熔断中, 拒绝请求")

        if not bucket.acquire(timeout=timeout):
            raise RateLimitError(f"[{source}] 令牌桶等待超时")

    def report_success(self, source: str):
        b = self._breakers.get(source)
        if b:
            b.on_success()

    def report_failure(self, source: str):
        b = self._breakers.get(source)
        if b:
            b.on_failure()

    def guard(self, source: str, timeout: Optional[float] = 30.0):
        """context manager 用法"""
        return _GuardContext(self, source, timeout)

    def stats(self) -> dict:
        return {
            source: {
                "bucket": self._buckets[source].stats(),
                "breaker": self._breakers[source].stats(),
            }
            for source in self._buckets
        }


class _GuardContext:
    def __init__(self, gw: SourceGateway, source: str, timeout: Optional[float]):
        self.gw = gw
        self.source = source
        self.timeout = timeout

    def __enter__(self):
        self.gw.acquire(self.source, timeout=self.timeout)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.gw.report_success(self.source)
        else:
            # 限流错误不计入失败 (是我们自己拒绝的, 不是源挂了)
            if not issubclass(exc_type, RateLimitError):
                self.gw.report_failure(self.source)
        return False  # 不吞异常
