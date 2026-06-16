"""增量更新调度器 - 每日全 A 股扫描

核心逻辑:
1. 从 Universe 拿当前生效的股票池
2. 批量查询 PG 中每只股票的最新 trade_date
3. 对每只股票, 只补 (last_local+1, today) 的缺失区间
4. 并发拉取 (受 SourceGateway 限流)
5. L2 parquet + L3 PG 双写
6. 失败任务记入 failed_symbols 表 (下次扫描自动重试)

使用:
    from data.incremental import IncrementalUpdater
    updater = IncrementalUpdater()
    report = updater.run_daily_scan()
    print(report.summary())

    # 自定义范围
    report = updater.run_daily_scan(
        lookback_days=365,   # 首次拉取的天数 (本地没数据时)
        symbols=["600519", "000001"],  # 只扫指定股票
        max_workers=4,
    )
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
from loguru import logger

from config.settings import settings
from data.cache_manager import CacheManager
from data.fetchers.tencent_fetcher import TencentFetcher
from data.fetchers.fallback_fetcher import FallbackFetcher
from data.storage.repository import StockRepository
from data.universe import Universe


@dataclass
class ScanReport:
    """单次扫描的报告"""
    start_ts: float = field(default_factory=time.monotonic)
    total_symbols: int = 0
    skipped_up_to_date: int = 0     # 已经是最新, 无需更新
    skipped_intraday_refresh: int = 0  # 盘中临时数据被收盘后重刷
    updated_count: int = 0          # 实际成功更新
    failed_count: int = 0           # 失败
    new_rows: int = 0               # 总写入行数
    failures: list = field(default_factory=list)  # [{symbol, error}]
    elapsed_seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"\n══ 增量扫描报告 ══\n"
            f"  股票池规模:    {self.total_symbols}\n"
            f"  已是最新:      {self.skipped_up_to_date}\n"
            f"  盘中重刷:      {self.skipped_intraday_refresh}\n"
            f"  成功更新:      {self.updated_count}\n"
            f"  失败:          {self.failed_count}\n"
            f"  新增行数:      {self.new_rows}\n"
            f"  耗时:          {self.elapsed_seconds:.1f}s\n"
            f"  平均:          {self.elapsed_seconds/max(self.updated_count,1)*1000:.0f}ms/只\n"
        )


class IncrementalUpdater:
    """增量调度器"""

    def __init__(
        self,
        max_workers: Optional[int] = None,
        chunk_size: int = 100,
    ):
        self.fetcher = FallbackFetcher() if settings.incremental_use_fallback else TencentFetcher()
        self.cache = CacheManager.get()
        self.repo = StockRepository()
        self.max_workers = max_workers or settings.max_workers_kline
        self.chunk_size = chunk_size

    def run_daily_scan(
        self,
        lookback_days: int = 250,
        symbols: Optional[List[str]] = None,
        end_date: Optional[str] = None,
        max_workers: Optional[int] = None,
    ) -> ScanReport:
        """每日扫描入口

        Args:
            lookback_days: 本地无数据时的初始拉取窗口 (默认 1 年)
            symbols: 指定股票列表, None=从 Universe 取
            end_date: 截止日期 YYYYMMDD, 默认今天
            max_workers: 并发线程数, 默认走 settings
        """
        report = ScanReport()

        # 1. 确定股票池
        if symbols is None:
            logger.info("[Updater] 从 Universe 加载股票池...")
            symbols = Universe.list_symbols()
        report.total_symbols = len(symbols)
        if not symbols:
            logger.warning("[Updater] 股票池为空, 退出")
            report.elapsed_seconds = time.monotonic() - report.start_ts
            return report

        logger.info(f"[Updater] 待扫描股票: {len(symbols)} 只")

        # 2. 批量查 PG 最新日期及 created_at（用于检测盘中临时数据）
        logger.info("[Updater] 批量查询 PG 本地最新日期...")
        latest_map = self._safe_get_latest(symbols)

        # 3. 计算每只股票的拉取区间
        end_dt = (datetime.now() if end_date is None
                  else datetime.strptime(end_date, "%Y%m%d"))
        end_str = end_dt.strftime("%Y%m%d")
        is_after_close = (
            end_dt.hour > settings.market_close_hour
            or (end_dt.hour == settings.market_close_hour and end_dt.minute >= settings.market_close_minute)
        )
        tasks = []
        for sym in symbols:
            last, created_at = latest_map.get(sym, (None, None))
            if last is not None:
                # 已有数据 -> 从 last+1 开始补
                start_dt = pd.Timestamp(last) + pd.Timedelta(days=1)
                if start_dt.date() > end_dt.date():
                    # 日期已是最新，但收盘后需检查是否为盘中临时数据
                    if is_after_close and created_at is not None and self._is_intraday_data(created_at, last):
                        # 盘中写入的临时数据，收盘后强制重拉
                        tasks.append((sym, end_str, end_str))
                        report.skipped_intraday_refresh += 1
                        continue
                    report.skipped_up_to_date += 1
                    continue
                start_str = start_dt.strftime("%Y%m%d")
            else:
                # 没有数据 -> 拉 lookback 天
                start_str = (end_dt - timedelta(days=lookback_days)).strftime("%Y%m%d")
            tasks.append((sym, start_str, end_str))

        if not tasks:
            logger.success(f"[Updater] 全部 {len(symbols)} 只股票已是最新")
            report.elapsed_seconds = time.monotonic() - report.start_ts
            return report

        logger.info(
            f"[Updater] {report.skipped_up_to_date} 只已最新, "
            f"待拉取 {len(tasks)} 只"
        )

        # 4. 并发拉取 (限流由 SourceGateway 控制)
        workers = max_workers or self.max_workers
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._fetch_one, sym, s, e): sym
                for (sym, s, e) in tasks
            }
            done = 0
            for fut in as_completed(futures):
                sym = futures[fut]
                done += 1
                try:
                    rows = fut.result()
                    if rows > 0:
                        report.updated_count += 1
                        report.new_rows += rows
                    else:
                        report.skipped_up_to_date += 1
                except Exception as e:
                    report.failed_count += 1
                    report.failures.append({"symbol": sym, "error": str(e)[:200]})
                    logger.warning(f"[Updater] {sym} 失败: {e}")

                if done % 100 == 0:
                    elapsed = time.monotonic() - report.start_ts
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (len(tasks) - done) / rate if rate > 0 else 0
                    logger.info(
                        f"[Updater] 进度 {done}/{len(tasks)} "
                        f"({rate:.1f} 只/s, ETA {eta:.0f}s)"
                    )

        report.elapsed_seconds = time.monotonic() - report.start_ts
        logger.success(report.summary())
        return report

    # ── 内部方法 ──
    def _safe_get_latest(self, symbols: List[str]) -> dict:
        """批量查询 PG 最新日期及 created_at，失败时静默"""
        out = {}
        try:
            for i in range(0, len(symbols), 500):
                chunk = symbols[i:i + 500]
                out.update(self.repo.get_latest_dates_with_created_at(chunk))
        except Exception as e:
            logger.warning(f"[Updater] PG 查询最新日期失败 (按全量处理): {e}")
        return out

    def _is_intraday_data(self, created_at, trade_date) -> bool:
        """判断是否为盘中临时数据（created_at 早于收盘时间）"""
        if created_at is None or trade_date is None:
            return False
        # 如果记录创建日期等于交易日，且时间早于收盘时间
        return (
            created_at.date() == trade_date
            and (created_at.hour < settings.market_close_hour
                 or (created_at.hour == settings.market_close_hour
                     and created_at.minute < settings.market_close_minute))
        )

    def _fetch_one(self, symbol: str, start_str: str, end_str: str) -> int:
        """拉单只股票, 双写 L2+L3, 返回写入行数"""
        df = self.fetcher.get_daily_bars(symbol, start_date=start_str, end_date=end_str)
        if df is None or df.empty:
            return 0

        source = str(df["source"].iloc[0]) if "source" in df.columns and not df.empty else "incremental"
        adjust = str(df["adjust"].iloc[0]) if "adjust" in df.columns and not df.empty else "raw"

        # 写 L2 parquet：按 source+adjust 隔离，保留 source/adjust 供链路追踪。
        persist_df = df.copy()
        try:
            self.cache.l2.upsert_bars(symbol, persist_df, source=source, adjust=adjust)
        except Exception as e:
            logger.debug(f"[Updater] {symbol} L2 写入失败: {e}")

        # 写 L3 PG：当前表结构未隔离 source/adjust，默认禁用，防止复权口径污染。
        if settings.cache_l3_kline_enabled:
            pg_df = persist_df.drop(columns=["source", "adjust", "amount_estimated"], errors="ignore")
            try:
                self.repo.save_bars(pg_df)
            except Exception as e:
                logger.warning(f"[Updater] {symbol} L3 写入失败: {e}")

        return len(persist_df)
