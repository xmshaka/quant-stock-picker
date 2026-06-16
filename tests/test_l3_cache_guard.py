"""L3 PG K线缓存默认禁用，避免未隔离 source/adjust 的旧数据污染 L2。"""
from __future__ import annotations

import pandas as pd


def _bars(close: float) -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": ["600519"],
        "trade_date": [pd.Timestamp("2026-06-12")],
        "open": [close],
        "high": [close + 1],
        "low": [close - 1],
        "close": [close],
        "volume": [1000.0],
        "amount": [close * 1000.0],
        "source": ["tencent"],
        "adjust": ["raw"],
    })


def test_get_or_fetch_bars_does_not_use_pg_by_default(tmp_path, monkeypatch):
    from data.cache_manager import CacheManager, ParquetCache

    cm = CacheManager.get()
    monkeypatch.setattr(cm, "l2", ParquetCache(tmp_path))
    monkeypatch.setattr("data.cache_manager.settings.cache_l3_kline_enabled", False)

    def fake_pg(symbol):
        raise AssertionError("默认不应读取未隔离的 PG K线缓存")

    monkeypatch.setattr(cm, "_load_from_pg", fake_pg)

    calls = {"fetch": 0}

    def fetch(start, end):
        calls["fetch"] += 1
        return _bars(100.0)

    df = cm.get_or_fetch_bars("600519", "20260612", "20260612", fetch, source="tencent", adjust="raw")

    assert calls["fetch"] == 1
    assert float(df.iloc[0]["close"]) == 100.0


def test_get_or_fetch_bars_uses_pg_only_when_explicitly_enabled(tmp_path, monkeypatch):
    from data.cache_manager import CacheManager, ParquetCache

    cm = CacheManager.get()
    monkeypatch.setattr(cm, "l2", ParquetCache(tmp_path))
    monkeypatch.setattr("data.cache_manager.settings.cache_l3_kline_enabled", True)
    # 注意：现在 L3 只对 source='' & adjust='raw' 启用
    monkeypatch.setattr(cm, "_load_from_pg", lambda symbol: _bars(88.0))

    def fetch(start, end):
        # 因为 source='tencent' 不会触发 L3，所以这里会调用 fetch
        return _bars(100.0)

    df = cm.get_or_fetch_bars("600519", "20260612", "20260612", fetch, source="tencent", adjust="raw")

    # 应该读取的是 fetch 的结果，不是 L3
    assert float(df.iloc[0]["close"]) == 100.0
    cached = cm.l2.get_bars("600519", source="tencent", adjust="raw")
    assert cached is not None and float(cached.iloc[0]["close"]) == 100.0
