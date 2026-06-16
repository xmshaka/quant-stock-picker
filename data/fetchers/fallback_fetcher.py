"""多源降级数据获取器

统一封装 Tencent / Tushare / AKShare / Baostock 的降级链路，供看板、增量扫描、脚本复用。

设计原则：
1. 主源优先：默认 Tencent 免费、快、无需 token。
2. 失败即切源：异常、空 DataFrame、基础校验失败都触发下一源。
3. 保留来源：返回数据增加 ``source`` 字段，便于追踪质量和排障。
4. 不强依赖可选源：Tushare / AKShare / Baostock 初始化失败时自动跳过。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import pandas as pd
from loguru import logger

from .base import BaseFetcher
from .tencent_fetcher import TencentFetcher
from .akshare_fetcher import AKShareFetcher
from .tushare_fetcher import TushareFetcher
from .baostock_fetcher import BaostockFetcher
from data.validator import Validator
from config.settings import settings


@dataclass
class SourceAttempt:
    """单个数据源尝试结果"""

    source: str
    ok: bool
    rows: int = 0
    error: str = ""


@dataclass
class FallbackReport:
    """最近一次多源调用的链路报告"""

    operation: str = ""
    symbol: str = ""
    selected_source: str = ""
    attempts: list[SourceAttempt] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.selected_source)

    def summary(self) -> str:
        parts = []
        for a in self.attempts:
            if a.ok:
                parts.append(f"{a.source}:OK({a.rows})")
            else:
                parts.append(f"{a.source}:FAIL({a.error})")
        selected = self.selected_source or "NONE"
        target = f" {self.symbol}" if self.symbol else ""
        return f"[Fallback] {self.operation}{target} -> {selected} | " + " -> ".join(parts)


class FallbackFetcher(BaseFetcher):
    """统一多源降级 Fetcher。

    默认源顺序来自 ``settings.data_source_order``，推荐：
    ``tencent,tushare,akshare,baostock``。AKShare/baostock 仅作为兜底。
    """

    def __init__(self, source_order: Optional[Iterable[str]] = None):
        super().__init__()
        raw_order = list(source_order) if source_order is not None else self._settings_order()
        self.source_order = self._normalize_order(raw_order)
        self.validator = Validator()
        self.fetchers: dict[str, BaseFetcher] = {}
        self.last_report = FallbackReport()
        self._init_fetchers()

    def _settings_order(self) -> list[str]:
        value = getattr(settings, "data_source_order", "tencent,tushare,akshare,baostock")
        if isinstance(value, str):
            return [x.strip() for x in value.split(",") if x.strip()]
        return list(value or [])

    def _normalize_order(self, order: Iterable[str]) -> list[str]:
        valid = {"tencent", "akshare", "tushare", "baostock"}
        out: list[str] = []
        for src in order:
            s = str(src).strip().lower()
            if s in valid and s not in out:
                out.append(s)
        return out or ["tencent", "tushare", "akshare", "baostock"]

    def _init_fetchers(self) -> None:
        constructors = {
            "tencent": TencentFetcher,
            "akshare": AKShareFetcher,
            "tushare": TushareFetcher,
            "baostock": BaostockFetcher,
        }
        for src in self.source_order:
            try:
                self.fetchers[src] = constructors[src]()
            except Exception as e:
                logger.warning(f"[Fallback] 初始化 {src} 失败，跳过: {e}")

    def _run_with_fallback(
        self,
        operation: str,
        call_factory: Callable[[str, BaseFetcher], pd.DataFrame],
        symbol: str = "",
        min_rows: int = 1,
        validate: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        report = FallbackReport(operation=operation, symbol=symbol)

        for src in self.source_order:
            fetcher = self.fetchers.get(src)
            if fetcher is None:
                report.attempts.append(SourceAttempt(src, False, error="not_initialized"))
                continue

            try:
                df = call_factory(src, fetcher)
                if df is None:
                    df = pd.DataFrame()
                if validate is not None and not df.empty:
                    df = validate(df)
                rows = len(df)
                if rows >= min_rows:
                    df = df.copy()
                    df["source"] = src
                    report.selected_source = src
                    report.attempts.append(SourceAttempt(src, True, rows=rows))
                    self.last_report = report
                    if src != self.source_order[0]:
                        logger.warning(report.summary())
                    else:
                        logger.debug(report.summary())
                    return df
                report.attempts.append(SourceAttempt(src, False, rows=rows, error="empty"))
            except Exception as e:
                report.attempts.append(SourceAttempt(src, False, error=str(e)[:160]))
                logger.debug(f"[Fallback] {operation} {symbol} via {src} 失败: {e}")

        self.last_report = report
        logger.warning(report.summary())
        return pd.DataFrame()

    # ── BaseFetcher 接口 ──
    def get_daily_bars(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = "qfq",
        **kwargs,
    ) -> pd.DataFrame:
        def _call(src: str, fetcher: BaseFetcher) -> pd.DataFrame:
            return fetcher.get_daily_bars(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
                **kwargs,
            )

        return self._run_with_fallback(
            operation="get_daily_bars",
            symbol=symbol,
            call_factory=_call,
            validate=self.validator.validate_bars_df,
        )

    def get_stock_list(self) -> pd.DataFrame:
        def _call(src: str, fetcher: BaseFetcher) -> pd.DataFrame:
            return fetcher.get_stock_list()

        return self._run_with_fallback(
            operation="get_stock_list",
            call_factory=_call,
            validate=self.validator.validate_stock_list,
        )

    def get_sector_list(self) -> pd.DataFrame:
        def _call(src: str, fetcher: BaseFetcher) -> pd.DataFrame:
            return fetcher.get_sector_list()

        return self._run_with_fallback(
            operation="get_sector_list",
            call_factory=_call,
        )

    def get_daily_bars_cached(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """兼容 TencentFetcher 的缓存接口。

        如果主源腾讯可用且启用缓存，优先走腾讯缓存；否则回退到普通多源链路。
        """
        primary = self.fetchers.get(self.source_order[0]) if self.source_order else None
        if hasattr(primary, "get_daily_bars_cached"):
            try:
                df = primary.get_daily_bars_cached(symbol, start_date, end_date, adjust=adjust)
                if df is not None and not df.empty:
                    df = df.copy()
                    df["source"] = self.source_order[0]
                    self.last_report = FallbackReport(
                        operation="get_daily_bars_cached",
                        symbol=symbol,
                        selected_source=self.source_order[0],
                        attempts=[SourceAttempt(self.source_order[0], True, rows=len(df))],
                    )
                    return df
            except Exception as e:
                logger.warning(f"[Fallback] 主源缓存接口失败 {symbol}: {e}")

        return self.get_daily_bars(symbol, start_date=start_date, end_date=end_date, adjust=adjust)
