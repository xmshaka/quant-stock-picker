"""多源日K对照工具测试。"""
from __future__ import annotations

import pandas as pd


def _bars(symbol: str, closes: list[float], amount: bool = True) -> pd.DataFrame:
    df = pd.DataFrame({
        "symbol": [symbol] * len(closes),
        "trade_date": pd.date_range("2026-06-10", periods=len(closes), freq="D"),
        "open": closes,
        "high": [x + 1 for x in closes],
        "low": [x - 1 for x in closes],
        "close": closes,
        "volume": [1000.0] * len(closes),
        "pct_change": [0.0] * len(closes),
        "change": [0.0] * len(closes),
    })
    if amount:
        df["amount"] = df["close"] * df["volume"] * 100
    return df


class _Fetcher:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_daily_bars(self, *args, **kwargs):
        return self.df


class _CachedFetcher(_Fetcher):
    def get_daily_bars_cached(self, *args, **kwargs):
        return self.df


def test_compare_daily_bars_summarizes_and_aligns():
    from data.source_compare import compare_daily_bars

    result = compare_daily_bars(
        "000001", "20260610", "20260612", adjust="",
        sources=["a", "b"], baseline="a",
        fetchers={
            "a": _Fetcher(_bars("000001", [10.0, 10.2, 10.4])),
            "b": _CachedFetcher(_bars("000001", [10.0, 10.3, 10.5])),
        },
    )

    assert result.summary["ok"].tolist() == [True, True]
    assert len(result.aligned) == 3
    assert "close_diff_pct_b_vs_a" in result.aligned.columns


def test_compare_daily_bars_warns_missing_amount():
    from data.source_compare import compare_daily_bars

    result = compare_daily_bars(
        "000001", "20260610", "20260612",
        sources=["baostock", "akshare"], baseline="baostock",
        fetchers={
            "baostock": _Fetcher(_bars("000001", [10.0, 10.2], amount=False)),
            "akshare": _Fetcher(_bars("000001", [10.0, 10.2], amount=True)),
        },
    )

    assert any("baostock" in w for w in result.warnings)
    assert any("缺少有效 amount" in w for w in result.warnings)
