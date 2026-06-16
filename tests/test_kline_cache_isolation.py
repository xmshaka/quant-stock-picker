"""K线缓存按 source+adjust 隔离，防止复权污染撮合。"""
from __future__ import annotations

import pandas as pd


def _bars(symbol: str, close: float, adjust: str, source: str = "tencent") -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": [symbol],
        "trade_date": [pd.Timestamp("2026-06-12")],
        "open": [close],
        "high": [close + 1],
        "low": [close - 1],
        "close": [close],
        "volume": [1000.0],
        "amount": [close * 1000.0],
        "source": [source],
        "adjust": [adjust],
    })


def test_parquet_cache_separates_raw_qfq_and_source(tmp_path):
    from data.cache_manager import ParquetCache

    cache = ParquetCache(tmp_path)
    cache.upsert_bars("600519", _bars("600519", 100.0, "raw", "tencent"), source="tencent", adjust="raw")
    cache.upsert_bars("600519", _bars("600519", 50.0, "qfq", "tencent"), source="tencent", adjust="qfq")
    cache.upsert_bars("600519", _bars("600519", 99.0, "raw", "baostock"), source="baostock", adjust="raw")

    raw = cache.get_bars("600519", source="tencent", adjust="raw")
    qfq = cache.get_bars("600519", source="tencent", adjust="qfq")
    bs = cache.get_bars("600519", source="baostock", adjust="raw")

    assert float(raw.iloc[0]["close"]) == 100.0
    assert float(qfq.iloc[0]["close"]) == 50.0
    assert float(bs.iloc[0]["close"]) == 99.0
    assert (tmp_path / "bars" / "tencent" / "raw" / "60" / "600519.parquet").exists()
    assert (tmp_path / "bars" / "tencent" / "qfq" / "60" / "600519.parquet").exists()
    assert (tmp_path / "bars" / "baostock" / "raw" / "60" / "600519.parquet").exists()


def test_cache_manager_get_or_fetch_bars_isolated_by_adjust(tmp_path, monkeypatch):
    from data.cache_manager import CacheManager, ParquetCache

    cm = CacheManager.get()
    monkeypatch.setattr(cm, "l2", ParquetCache(tmp_path))
    calls = {"raw": 0, "qfq": 0}

    def fetch_raw(start, end):
        calls["raw"] += 1
        return _bars("600519", 100.0, "raw", "tencent")

    def fetch_qfq(start, end):
        calls["qfq"] += 1
        return _bars("600519", 50.0, "qfq", "tencent")

    raw1 = cm.get_or_fetch_bars("600519", "20260612", "20260612", fetch_raw, source="tencent", adjust="raw", use_pg=False)
    qfq1 = cm.get_or_fetch_bars("600519", "20260612", "20260612", fetch_qfq, source="tencent", adjust="qfq", use_pg=False)
    raw2 = cm.get_or_fetch_bars("600519", "20260612", "20260612", fetch_raw, source="tencent", adjust="raw", use_pg=False)

    assert calls == {"raw": 1, "qfq": 1}
    assert float(raw1.iloc[0]["close"]) == 100.0
    assert float(raw2.iloc[0]["close"]) == 100.0
    assert float(qfq1.iloc[0]["close"]) == 50.0
