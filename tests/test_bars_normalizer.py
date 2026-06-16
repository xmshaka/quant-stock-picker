"""K线标准化层测试。"""
from __future__ import annotations

import pandas as pd
import pytest


def _raw_df(volume=10.0, amount=20.0):
    return pd.DataFrame({
        "symbol": ["1"],
        "trade_date": ["2026-06-12"],
        "open": [10],
        "high": [11],
        "low": [9],
        "close": [10],
        "volume": [volume],
        "amount": [amount],
        "pct_change": [1.0],
        "change": [0.1],
    })


def test_normalize_tencent_volume_to_shares_amount_yuan():
    from data.bars_normalizer import normalize_daily_bars

    df = normalize_daily_bars(_raw_df(volume=10, amount=10000), source="tencent", symbol="000001", adjust="")

    assert df.iloc[0]["volume"] == 1000
    assert df.iloc[0]["amount"] == 10000
    assert df.iloc[0]["adjust"] == "raw"
    assert df.iloc[0]["source"] == "tencent"


def test_normalize_tushare_amount_thousand_to_yuan():
    from data.bars_normalizer import normalize_daily_bars

    df = normalize_daily_bars(_raw_df(volume=10, amount=20), source="tushare", symbol="000001", adjust="raw")

    assert df.iloc[0]["volume"] == 1000
    assert df.iloc[0]["amount"] == 20000


def test_normalize_baostock_volume_already_shares():
    from data.bars_normalizer import normalize_daily_bars

    df = normalize_daily_bars(_raw_df(volume=1000, amount=10000), source="baostock", symbol="000001", adjust="qfq")

    assert df.iloc[0]["volume"] == 1000
    assert df.iloc[0]["amount"] == 10000
    assert df.iloc[0]["adjust"] == "qfq"


def test_assert_raw_for_execution_blocks_qfq():
    from data.bars_normalizer import assert_raw_for_execution, normalize_daily_bars

    df = normalize_daily_bars(_raw_df(), source="baostock", symbol="000001", adjust="qfq")

    with pytest.raises(ValueError, match="不复权"):
        assert_raw_for_execution(df)


def test_assert_raw_for_execution_accepts_raw():
    from data.bars_normalizer import assert_raw_for_execution, normalize_daily_bars

    df = normalize_daily_bars(_raw_df(), source="baostock", symbol="000001", adjust="")
    assert_raw_for_execution(df)
