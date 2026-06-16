"""回测撮合K线 raw 口径硬校验。"""
from __future__ import annotations

import pandas as pd
import pytest


def _bars(adjust="raw", with_meta=True):
    df = pd.DataFrame({
        "symbol": ["000001", "000001"],
        "trade_date": ["2026-06-11", "2026-06-12"],
        "open": [10.0, 10.2],
        "high": [10.5, 10.4],
        "low": [9.9, 10.0],
        "close": [10.2, 10.3],
        "volume": [1000.0, 1200.0],
        "amount": [10200.0, 12360.0],
    })
    if with_meta:
        df["source"] = "unit"
        df["adjust"] = adjust
    return df


def test_prepare_execution_bars_rejects_qfq():
    from backtest.scheme_backtest import _prepare_execution_bars

    with pytest.raises(ValueError, match="不复权"):
        _prepare_execution_bars(_bars(adjust="qfq"))


def test_prepare_execution_bars_accepts_standardized_raw_without_rescaling():
    from backtest.scheme_backtest import _prepare_execution_bars

    df = _prepare_execution_bars(_bars(adjust="raw"))

    assert df["adjust"].eq("raw").all()
    assert df["source"].eq("unit").all()
    assert df.iloc[0]["volume"] == 1000.0
    assert df.iloc[0]["amount"] == 10200.0


def test_prepare_execution_bars_legacy_missing_meta_defaults_raw():
    from backtest.scheme_backtest import _prepare_execution_bars

    df = _prepare_execution_bars(_bars(with_meta=False), fallback_source="legacy")

    assert df["adjust"].eq("raw").all()
    assert df["source"].eq("legacy").all()
