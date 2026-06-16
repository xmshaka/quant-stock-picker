"""三级缓存管理器

L1: 内存 LRU + TTL  -> 进程内热数据 (实时行情、股票列表)
L2: 磁盘 parquet    -> 持久化, 支持增量更新 (按 symbol 分文件存日K)
L3: PostgreSQL      -> 归档与因子结果 (P3 阶段启用, 这里先预留接口)

设计原则:
- 读: L1 -> L2 -> 源, 命中即返回, 不穿透
- 写: 源数据回来后, 同步写 L1 + L2
- 增量: 日K按 symbol 单独存 parquet, 每次只补充新增日期
- 校验: 写入前过 validator (后续接入)
- 容错: 缓存损坏时自动清除并降级到源
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
from loguru import logger

from config.settings import settings
from data.bars_normalizer import normalize_adjust


# ════════════════════════════════════════════════════════════
# L1: 内存 LRU + TTL
# ════════════════════════════════════════════════════════════
@dataclass
class _LRUEntry:
    value: Any
    expire_at: float
    size: int = 1


class MemoryCache:
    """线程安全的 LRU + TTL 内存缓存"""

    def __init__(self, max_entries: int = 1024, default_ttl: int = 600):
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self._data: "OrderedDict[str, _LRUEntry]" = OrderedDict()
        self._lock = threading.Lock()
        # 统计
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            if entry.expire_at < time.time():
                # 过期
                self._data.pop(key, None)
                self.misses += 1
                return None
            # LRU: 命中后移到末尾
            self._data.move_to_end(key)
            self.hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        ttl = ttl or self.default_ttl
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = _LRUEntry(value=value, expire_at=time.time() + ttl)
            # 驱逐
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
                self.evictions += 1

    def delete(self, key: str):
        with self._lock:
            self._data.pop(key, None)

    def clear(self):
        with self._lock:
            self._data.clear()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "entries": len(self._data),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0,
            "evictions": self.evictions,
        }


# ════════════════════════════════════════════════════════════
# L2: Parquet 磁盘缓存 (DataFrame 专用)
# ════════════════════════════════════════════════════════════
class ParquetCache:
    """Parquet 文件缓存
    
    支持两种模式:
    1. snapshot(key, df, ttl): 快照型 (实时行情、股票列表), 整体覆盖
    2. upsert_bars(symbol, df): K线型, 按 trade_date 增量合并
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.snapshot_dir = self.root / "snapshots"
        self.bars_dir = self.root / "bars"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.bars_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # 文件写入锁 (parquet 写不是原子的)
        # 统计
        self.snapshot_hits = 0
        self.snapshot_misses = 0
        self.bar_reads = 0
        self.bar_upserts = 0

    # ── 快照缓存 ──
    def _snapshot_path(self, key: str) -> Path:
        # 用 hash 避免特殊字符问题
        h = hashlib.md5(key.encode()).hexdigest()[:16]
        safe = "".join(c if c.isalnum() else "_" for c in key)[:80]
        return self.snapshot_dir / f"{safe}_{h}.parquet"

    def _snapshot_meta_path(self, p: Path) -> Path:
        return p.with_suffix(".meta")

    def get_snapshot(self, key: str, ttl_seconds: int) -> Optional[pd.DataFrame]:
        p = self._snapshot_path(key)
        meta = self._snapshot_meta_path(p)
        if not p.exists() or not meta.exists():
            self.snapshot_misses += 1
            return None
        try:
            saved_at = float(meta.read_text().strip())
            if time.time() - saved_at > ttl_seconds:
                self.snapshot_misses += 1
                return None
            df = pd.read_parquet(p)
            self.snapshot_hits += 1
            return df
        except Exception as e:
            logger.warning(f"[L2] 读取快照失败 {key}: {e}, 清除")
            try:
                p.unlink(missing_ok=True)
                meta.unlink(missing_ok=True)
            except Exception:
                pass
            self.snapshot_misses += 1
            return None

    def set_snapshot(self, key: str, df: pd.DataFrame):
        if df is None or df.empty:
            return
        p = self._snapshot_path(key)
        meta = self._snapshot_meta_path(p)
        tmp = p.with_suffix(".tmp")
        try:
            with self._lock:
                df.to_parquet(tmp, index=False, compression="snappy")
                os.replace(tmp, p)
                meta.write_text(str(time.time()))
        except Exception as e:
            logger.error(f"[L2] 写入快照失败 {key}: {e}")
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    # ── K线增量缓存 ──
    def _bars_path(self, symbol: str, source: str = "default", adjust: str = "raw") -> Path:
        """K线缓存路径，按 source+adjust 隔离。

        旧版仅按 symbol 存储，容易把 raw/qfq/hfq 混写到同一文件。
        新路径: bars/{source}/{adjust}/{prefix}/{symbol}.parquet
        """
        source_safe = "".join(c if c.isalnum() else "_" for c in str(source or "default").lower())
        adjust_safe = normalize_adjust(adjust)
        # 按交易所前两位分子目录, 避免单目录文件过多
        prefix = symbol[:2] if len(symbol) >= 6 else "xx"
        d = self.bars_dir / source_safe / adjust_safe / prefix
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{symbol}.parquet"

    def get_bars(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        source: str = "default",
        adjust: str = "raw",
    ) -> Optional[pd.DataFrame]:
        """读取本地 K 线. 不存在返回 None"""
        p = self._bars_path(symbol, source=source, adjust=adjust)
        if not p.exists():
            return None
        try:
            df = pd.read_parquet(p)
            self.bar_reads += 1
            if "trade_date" in df.columns:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                if start_date:
                    df = df[df["trade_date"] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df["trade_date"] <= pd.to_datetime(end_date)]
            return df.reset_index(drop=True) if not df.empty else None
        except Exception as e:
            logger.warning(f"[L2] 读取K线失败 {symbol}: {e}, 清除")
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
            return None

    def upsert_bars(
        self,
        symbol: str,
        new_df: pd.DataFrame,
        source: str = "default",
        adjust: str = "raw",
    ):
        """增量合并 K 线: 按 trade_date 去重, 新数据覆盖旧数据
        
        ⚠️ 复权数据有特殊处理: 由于前复权基准会随时间变化,
        如果检测到历史日期价格变动 > 0.1%, 视为基准变化, 整体覆盖
        """
        if new_df is None or new_df.empty:
            return
        if "trade_date" not in new_df.columns or "symbol" not in new_df.columns:
            logger.warning(f"[L2] upsert_bars 数据缺字段, 跳过")
            return

        new_df = new_df.copy()
        new_df["trade_date"] = pd.to_datetime(new_df["trade_date"])
        if "source" not in new_df.columns:
            new_df["source"] = source
        if "adjust" not in new_df.columns:
            new_df["adjust"] = normalize_adjust(adjust)

        p = self._bars_path(symbol, source=source, adjust=adjust)
        with self._lock:
            try:
                if p.exists():
                    old_df = pd.read_parquet(p)
                    old_df["trade_date"] = pd.to_datetime(old_df["trade_date"])

                    # 复权基准漂移检测
                    overlap = pd.merge(
                        old_df[["trade_date", "close"]],
                        new_df[["trade_date", "close"]],
                        on="trade_date",
                        suffixes=("_old", "_new"),
                    )
                    if not overlap.empty:
                        diff = ((overlap["close_old"] - overlap["close_new"]).abs()
                                / overlap["close_old"].clip(lower=1e-6))
                        max_diff = float(diff.max())
                        if max_diff > 0.001:  # 0.1%
                            logger.info(
                                f"[L2] {symbol} 检测到复权基准漂移 (max_diff={max_diff:.4f}), "
                                f"整体重写"
                            )
                            merged = new_df
                        else:
                            # 正常合并: new 覆盖 old
                            merged = pd.concat([old_df, new_df], ignore_index=True)
                            merged = (merged
                                      .drop_duplicates(subset=["trade_date"], keep="last")
                                      .sort_values("trade_date")
                                      .reset_index(drop=True))
                    else:
                        merged = pd.concat([old_df, new_df], ignore_index=True)
                        merged = (merged
                                  .drop_duplicates(subset=["trade_date"], keep="last")
                                  .sort_values("trade_date")
                                  .reset_index(drop=True))
                else:
                    merged = new_df.sort_values("trade_date").reset_index(drop=True)

                tmp = p.with_suffix(".tmp")
                merged.to_parquet(tmp, index=False, compression="snappy")
                os.replace(tmp, p)
                self.bar_upserts += 1
            except Exception as e:
                logger.error(f"[L2] upsert_bars 失败 {symbol}: {e}")

    def last_trade_date(
        self,
        symbol: str,
        source: str = "default",
        adjust: str = "raw",
    ) -> Optional[pd.Timestamp]:
        """返回本地最新的交易日期"""
        df = self.get_bars(symbol, source=source, adjust=adjust)
        if df is None or df.empty:
            return None
        return df["trade_date"].max()

    def stats(self) -> dict:
        total = self.snapshot_hits + self.snapshot_misses
        return {
            "snapshot_hits": self.snapshot_hits,
            "snapshot_misses": self.snapshot_misses,
            "snapshot_hit_rate": round(self.snapshot_hits / total, 3) if total else 0,
            "bar_reads": self.bar_reads,
            "bar_upserts": self.bar_upserts,
            "bar_files": sum(1 for _ in self.bars_dir.rglob("*.parquet")),
        }


# ════════════════════════════════════════════════════════════
# 三级缓存统一入口
# ════════════════════════════════════════════════════════════
class CacheManager:
    """三级缓存协调器 - 全局单例"""

    _instance: Optional["CacheManager"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self.l1 = MemoryCache(
            max_entries=settings.cache_l1_size,
            default_ttl=settings.cache_l1_ttl_seconds,
        )
        self.l2 = ParquetCache(root=settings.parquet_dir)

    @classmethod
    def get(cls) -> "CacheManager":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── 高层封装: 通用 get-or-fetch ──
    def get_or_fetch_snapshot(
        self,
        key: str,
        fetch_fn: Callable[[], pd.DataFrame],
        l2_ttl_seconds: int,
        l1_ttl_seconds: Optional[int] = None,
    ) -> pd.DataFrame:
        """通用读模式: L1 -> L2 -> 源, 任一层命中即返回"""
        # L1
        cached = self.l1.get(key)
        if cached is not None:
            return cached
        # L2
        df = self.l2.get_snapshot(key, ttl_seconds=l2_ttl_seconds)
        if df is not None and not df.empty:
            self.l1.set(key, df, ttl=l1_ttl_seconds)
            return df
        # 源
        df = fetch_fn()
        if df is not None and not df.empty:
            self.l2.set_snapshot(key, df)
            self.l1.set(key, df, ttl=l1_ttl_seconds)
        return df

    def get_or_fetch_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        fetch_fn: Callable[[str, str], pd.DataFrame],
        use_pg: Optional[bool] = None,
        source: str = "default",
        adjust: str = "raw",
    ) -> pd.DataFrame:
        """K线增量获取: 本地有就用本地, 缺失部分才请求源

        fetch_fn(start_date, end_date) -> DataFrame
        use_pg: 是否启用 L3 (PostgreSQL) 级缓存。None 时读取 settings.cache_l3_kline_enabled。
                当前 PG bars 未按 source/adjust 隔离，默认禁用，避免复权口径污染 L2。
        """
        if use_pg is None:
            use_pg = bool(getattr(settings, "cache_l3_kline_enabled", False))

        end_ts = pd.to_datetime(end_date)
        start_ts = pd.to_datetime(start_date)

        cached = self.l2.get_bars(symbol, source=source, adjust=adjust)

        # L2 未命中 -> 尝试 L3 (PG) 预热
        if (cached is None or cached.empty) and use_pg:
            # 现在 PG 已支持 source/adjust，但 _load_from_pg 默认只读旧数据。
            # 为了安全，只对 source='' & adjust='raw' 启用 L3。
            if source == "" and adjust == "raw":
                cached = self._load_from_pg(symbol)
            else:
                logger.debug(f"[Cache] L3 skip for source={source}, adjust={adjust} (source=''&adjust='raw' only)")
            if cached is not None and not cached.empty:
                logger.debug(f"[Cache] {symbol} L3->L2 预热 {len(cached)} 条")
                self.l2.upsert_bars(symbol, cached, source=source, adjust=adjust)

        if cached is not None and not cached.empty:
            last_local = cached["trade_date"].max()
            first_local = cached["trade_date"].min()

            # 本地完全覆盖
            if first_local <= start_ts and last_local >= end_ts:
                return cached[(cached["trade_date"] >= start_ts)
                              & (cached["trade_date"] <= end_ts)].reset_index(drop=True)

            # 只补尾部
            if first_local <= start_ts and last_local < end_ts:
                gap_start = (last_local + pd.Timedelta(days=1)).strftime("%Y%m%d")
                logger.info(f"[Cache] {symbol} 增量补尾: {gap_start} -> {end_date}")
                new_df = fetch_fn(gap_start, end_date)
                if new_df is not None and not new_df.empty:
                    self.l2.upsert_bars(symbol, new_df, source=source, adjust=adjust)
                    if use_pg:
                        self._save_to_pg(new_df)
                    merged = self.l2.get_bars(symbol, start_date, end_date, source=source, adjust=adjust)
                    if merged is not None and not merged.empty:
                        return merged
                    return new_df
                return cached[(cached["trade_date"] >= start_ts)].reset_index(drop=True)

        # 全量拉取
        logger.info(f"[Cache] {symbol} 全量拉取: {start_date} -> {end_date}")
        new_df = fetch_fn(start_date, end_date)
        if new_df is not None and not new_df.empty:
            self.l2.upsert_bars(symbol, new_df, source=source, adjust=adjust)
            if use_pg:
                self._save_to_pg(new_df)
        return new_df

    # ── L3: PostgreSQL 交互 ──
    def _load_from_pg(self, symbol: str) -> Optional[pd.DataFrame]:
        """从 PG 拉一只股票的全部 K 线 (供预热 L2)"""
        try:
            # 默认只读 source='' & adjust='raw' 的记录，兼容旧数据
            from data.storage.repository import StockRepository
            repo = StockRepository()
            df = repo.get_bars(symbol)
            if df is None or df.empty:
                return None
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            return df
        except Exception as e:
            logger.debug(f"[L3] PG 读 {symbol} 失败 (可能未初始化): {e}")
            return None

    def _save_to_pg(self, df: pd.DataFrame):
        """写入 PG (异常不阻断主流程)"""
        try:
            # 写入时会带上 df 中的 source/adjust 字段
            from data.storage.repository import StockRepository
            repo = StockRepository()
            repo.save_bars(df)
        except Exception as e:
            logger.warning(f"[L3] PG 写入失败 (不阻断): {e}")

    def stats(self) -> dict:
        return {"l1": self.l1.stats(), "l2": self.l2.stats()}
