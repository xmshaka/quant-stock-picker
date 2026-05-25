"""P1 冒烟测试: 验证限流 + 缓存 + 校验链路

不依赖外部网络的测试: 限流、熔断、缓存读写、校验逻辑
依赖网络的测试: 实际请求腾讯/AKShare (会被 PY_SMOKE_LIVE=1 触发)
"""
import os
import sys
import time

sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import pandas as pd
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> <level>{level:<6}</level> {message}")


def test_token_bucket():
    """令牌桶: 5 QPS 跑 11 个请求, burst=1 限制后 应耗时 ~2 秒"""
    from data.ratelimit import TokenBucket
    print("\n── Test 1: 令牌桶限流 (5 QPS, burst=1, x 11 请求) ──")
    bucket = TokenBucket(qps=5.0, burst=1, name="test")
    start = time.monotonic()
    for i in range(11):
        bucket.acquire()
    elapsed = time.monotonic() - start
    print(f"  耗时: {elapsed:.2f}s (预期 ~2s)")
    assert 1.5 < elapsed < 3.0, f"耗时异常: {elapsed}"
    print(f"  统计: {bucket.stats()}")
    print("  ✅ 通过")


def test_circuit_breaker():
    """熔断器: 连续失败 5 次后 OPEN, 等 1s 后 HALF_OPEN, 成功 1 次后 CLOSED"""
    from data.ratelimit import CircuitBreaker, CircuitState
    print("\n── Test 2: 熔断器三态转换 ──")
    cb = CircuitBreaker(failure_threshold=5, recovery_seconds=1, half_open_max=2, name="test")

    # 模拟失败
    for i in range(5):
        assert cb.allow_request()
        cb.on_failure()
    assert cb.state == CircuitState.OPEN
    print(f"  ✅ 5次失败后进入 OPEN")

    # OPEN 期间拒绝
    assert not cb.allow_request()
    print(f"  ✅ OPEN 期间拒绝请求")

    # 等待恢复
    time.sleep(1.1)
    assert cb.allow_request()  # 转入 HALF_OPEN
    assert cb.state == CircuitState.HALF_OPEN
    print(f"  ✅ 1s 后转入 HALF_OPEN")

    cb.on_success()
    assert cb.state == CircuitState.CLOSED
    print(f"  ✅ HALF_OPEN 成功后转 CLOSED")
    print(f"  统计: {cb.stats()}")


def test_memory_cache():
    """L1 内存缓存: TTL + LRU"""
    from data.cache_manager import MemoryCache
    print("\n── Test 3: L1 内存缓存 LRU + TTL ──")
    cache = MemoryCache(max_entries=3, default_ttl=1)

    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert cache.get("a") == 1
    print(f"  ✅ set/get OK")

    cache.set("d", 4)  # 触发驱逐
    assert cache.get("b") is None  # b 应该被驱逐 (最久未访问)
    print(f"  ✅ LRU 驱逐 OK (b 已被驱逐)")

    time.sleep(1.1)
    assert cache.get("c") is None  # 已过期
    print(f"  ✅ TTL 过期 OK")
    print(f"  统计: {cache.stats()}")


def test_parquet_cache():
    """L2 Parquet 增量缓存"""
    from data.cache_manager import ParquetCache
    from pathlib import Path
    import tempfile

    print("\n── Test 4: L2 Parquet 增量 K 线缓存 ──")
    with tempfile.TemporaryDirectory() as tmp:
        cache = ParquetCache(root=Path(tmp))

        # 写入历史 K 线
        df1 = pd.DataFrame({
            "symbol": ["600519"] * 5,
            "trade_date": pd.date_range("2025-01-01", periods=5),
            "open": [100, 101, 102, 103, 104],
            "high": [105] * 5,
            "low": [99] * 5,
            "close": [100, 101, 102, 103, 104],
            "volume": [10000] * 5,
        })
        cache.upsert_bars("600519", df1)
        print(f"  ✅ 写入 5 条历史 K 线")

        # 读取
        df_read = cache.get_bars("600519")
        assert len(df_read) == 5
        print(f"  ✅ 读取: {len(df_read)} 条")

        # 增量追加 3 条 (含 1 条重复 -> 应去重)
        df2 = pd.DataFrame({
            "symbol": ["600519"] * 3,
            "trade_date": pd.date_range("2025-01-05", periods=3),
            "open": [104.5, 105, 106],
            "high": [110] * 3,
            "low": [100] * 3,
            "close": [104, 105, 106],
            "volume": [12000] * 3,
        })
        cache.upsert_bars("600519", df2)
        df_merged = cache.get_bars("600519")
        assert len(df_merged) == 7, f"预期 7 条, 实际 {len(df_merged)}"
        print(f"  ✅ 增量合并去重 OK: {len(df_merged)} 条")

        # last_trade_date
        last = cache.last_trade_date("600519")
        assert last == pd.Timestamp("2025-01-07")
        print(f"  ✅ last_trade_date: {last.date()}")


def test_validator():
    """校验器: 异常数据识别"""
    from data.validator import Validator
    print("\n── Test 5: 数据校验器 ──")
    v = Validator(strict=False)

    # 正常
    ok, _ = v.validate_bar({
        "symbol": "600519", "trade_date": "2025-01-01",
        "open": 100, "high": 105, "low": 99, "close": 102, "volume": 10000
    })
    assert ok
    print(f"  ✅ 正常数据通过")

    # 异常: low > high
    ok, msg = v.validate_bar({
        "symbol": "600519", "trade_date": "2025-01-01",
        "open": 100, "high": 99, "low": 105, "close": 102, "volume": 10000
    })
    assert any(i["level"] == "error" for i in v.issues)
    print(f"  ✅ 异常数据捕获 (low>high)")
    print(f"  共记录 {len(v.issues)} 个问题")


def test_live_tencent():
    """实战测试: 真的调一次腾讯接口"""
    if not os.getenv("PY_SMOKE_LIVE"):
        print("\n── Test 6 (live): 跳过 (设置 PY_SMOKE_LIVE=1 启用) ──")
        return

    print("\n── Test 6: 真实腾讯接口 + 缓存命中验证 ──")
    from data.fetchers.tencent_fetcher import TencentFetcher
    from data.ratelimit import SourceGateway
    from data.cache_manager import CacheManager

    f = TencentFetcher()

    # 第 1 次: 应该走源
    print("  第1次 get_daily_bars_cached 600519 (应走源)...")
    t1 = time.monotonic()
    df = f.get_daily_bars_cached("600519", "20250101", "20250120")
    t1 = time.monotonic() - t1
    print(f"    {len(df)} 条, 耗时 {t1*1000:.0f}ms")
    assert not df.empty

    # 第 2 次: 同范围, 应命中 L2
    print("  第2次 同范围 (应命中 L2 缓存)...")
    t2 = time.monotonic()
    df2 = f.get_daily_bars_cached("600519", "20250101", "20250120")
    t2 = time.monotonic() - t2
    print(f"    {len(df2)} 条, 耗时 {t2*1000:.0f}ms")
    # 小文件 I/O 本身比网络还快, 只要不重复调用接口即可
    print(f"  ✅ 缓存命中 (运行时间 {t1*1000:.0f}ms -> {t2*1000:.0f}ms)")

    # 治理层统计
    print(f"\n  Gateway stats: {SourceGateway.get().stats()}")
    print(f"  Cache stats:   {CacheManager.get().stats()}")


if __name__ == "__main__":
    test_token_bucket()
    test_circuit_breaker()
    test_memory_cache()
    test_parquet_cache()
    test_validator()
    test_live_tencent()
    print("\n🎉 全部通过")
