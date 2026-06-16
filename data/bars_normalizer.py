"""A股日K标准化层。

统一所有数据源输出，避免回测滑点/成交额分层因单位差异失真。

标准约定：
- volume: 股
- amount: 元
- adjust: raw/qfq/hfq
- source: tencent/tushare/akshare/baostock/...
- trade_date: pandas datetime64

重要：回测撮合只能使用 adjust=raw；前复权仅用于趋势图/因子。
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

PRICE_COLS = ["open", "high", "low", "close", "pre_close"]
NUMERIC_COLS = PRICE_COLS + ["volume", "amount", "pct_change", "change", "turnover"]
OUTPUT_COLS = [
    "symbol", "trade_date", "open", "high", "low", "close",
    "volume", "amount", "amount_estimated", "pct_change", "change", "turnover", "adjust", "source",
]


def normalize_adjust(adjust: Optional[str]) -> str:
    value = "raw" if adjust in (None, "", "none", "raw") else str(adjust).lower()
    if value not in {"raw", "qfq", "hfq"}:
        return "qfq"
    return value


def normalize_daily_bars(
    df: pd.DataFrame,
    source: str,
    symbol: Optional[str] = None,
    adjust: Optional[str] = "qfq",
) -> pd.DataFrame:
    """标准化日K DataFrame。

    Args:
        df: 已完成列名映射的日K数据。
        source: 数据源名称。
        symbol: 6位代码；缺失时保留原 df['symbol']。
        adjust: raw/qfq/hfq 或旧式 ""。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    src = str(source).lower()
    adj = normalize_adjust(adjust)

    if symbol is not None:
        out["symbol"] = str(symbol).zfill(6)
    elif "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str).str.zfill(6)

    if "trade_date" in out.columns:
        out["trade_date"] = pd.to_datetime(out["trade_date"])

    for col in NUMERIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # 单位统一：volume=股，amount=元
    if "volume" in out.columns:
        if src in {"tencent", "tushare", "akshare"}:
            # 腾讯K线、Tushare daily、AKShare stock_zh_a_hist 成交量均为“手”。
            out["volume"] = out["volume"] * 100.0
        elif src == "baostock":
            # Baostock query_history_k_data_plus volume 为股。
            out["volume"] = out["volume"]

    if "amount" in out.columns:
        if src == "tushare":
            # Tushare daily amount 单位为千元。
            out["amount"] = out["amount"] * 1000.0
        elif src in {"tencent", "akshare", "baostock"}:
            # 腾讯当前由 volume(手)*close*100 估算后为元；AKShare/Baostock 为元。
            out["amount"] = out["amount"]

    # 缺成交额时用标准化后的 volume*close 兜底估算，只作为流动性分层，不作为成交真实额。
    if "amount" not in out.columns and {"volume", "close"}.issubset(out.columns):
        out["amount"] = out["volume"] * out["close"]
        out["amount_estimated"] = True
        logger.debug(f"[BarsNormalize] {src} 缺 amount，使用 volume*close 估算")
    elif "amount_estimated" not in out.columns:
        out["amount_estimated"] = False

    out["adjust"] = adj
    out["source"] = src

    keep = [c for c in OUTPUT_COLS if c in out.columns]
    out = out[keep]
    if "trade_date" in out.columns:
        out = out.sort_values("trade_date")
    return out.reset_index(drop=True)


def assert_raw_for_execution(df: pd.DataFrame) -> None:
    """回测/实盘撮合前硬校验：禁止用复权K线撮合。"""
    if df is None or df.empty:
        return
    if "adjust" not in df.columns:
        raise ValueError("K线缺少 adjust 字段，禁止用于撮合")
    bad = df["adjust"].fillna("").astype(str).str.lower().ne("raw")
    if bad.any():
        raise ValueError("撮合必须使用不复权K线 adjust=raw，禁止使用 qfq/hfq")
