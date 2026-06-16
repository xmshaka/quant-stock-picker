"""Tushare 优先数据源策略回归测试。"""

from __future__ import annotations

import pandas as pd


def test_fallback_default_order_prefers_tushare_before_akshare(monkeypatch):
    from data.fetchers.fallback_fetcher import FallbackFetcher

    monkeypatch.setattr("data.fetchers.fallback_fetcher.settings.data_source_order", "")
    ff = FallbackFetcher(source_order=[])

    assert ff.source_order == ["tencent", "tushare", "akshare", "baostock"]


def test_dashboard_loader_default_order_prefers_tushare_before_akshare(monkeypatch):
    from dashboard.data_loader import DataLoader

    monkeypatch.setattr(DataLoader, "_init_fetchers", lambda self: None)
    loader = DataLoader()

    assert loader.sources == ["tencent", "tushare", "akshare", "baostock", "mock"]


def test_dashboard_loader_baostock_preferred_chain(monkeypatch):
    from dashboard.data_loader import DataLoader

    monkeypatch.setattr(DataLoader, "_init_fetchers", lambda self: None)
    loader = DataLoader(preferred="baostock")

    assert loader.sources == ["baostock", "tencent", "tushare", "akshare", "mock"]


def test_universe_uses_tushare_loader_before_akshare(monkeypatch):
    from data import universe as uni

    calls: list[str] = []

    def fake_tushare():
        calls.append("tushare")
        return pd.DataFrame({
            "symbol": ["000001"],
            "name": ["平安银行"],
            "close": [10.0],
            "pct_change": [0.1],
            "volume": [1000.0],
            "amount": [1_000_000.0],
            "turnover": [1.0],
            "pe_ttm": [6.0],
            "pb": [0.7],
            "total_mv": [100_000_000_000.0],
            "float_mv": [80_000_000_000.0],
            "list_date": [pd.Timestamp("1991-04-03")],
        })

    def fake_akshare():
        calls.append("akshare")
        return pd.DataFrame()

    monkeypatch.setattr(uni, "_load_via_tushare", fake_tushare)
    monkeypatch.setattr(uni, "_load_spot_em_akshare", fake_akshare)

    df = uni._load_spot_em()

    assert not df.empty
    assert calls == ["tushare"]


def test_universe_falls_back_to_akshare_only_when_tushare_empty(monkeypatch):
    from data import universe as uni

    calls: list[str] = []

    def fake_tushare():
        calls.append("tushare")
        return pd.DataFrame()

    def fake_akshare():
        calls.append("akshare")
        return pd.DataFrame({"symbol": ["000001"], "name": ["平安银行"]})

    monkeypatch.setattr(uni, "_load_via_tushare", fake_tushare)
    monkeypatch.setattr(uni, "_load_spot_em_akshare", fake_akshare)

    df = uni._load_spot_em()

    assert not df.empty
    assert calls == ["tushare", "akshare"]


def test_tushare_daily_basic_looks_back_when_default_date_empty(monkeypatch, tmp_path):
    from data.fetchers import tushare_fetcher as mod
    from data.cache_manager import CacheManager, ParquetCache

    calls: list[str] = []

    class FakePro:
        def daily_basic(self, trade_date: str):
            calls.append(trade_date)
            if len(calls) < 3:
                return pd.DataFrame()
            return pd.DataFrame({"ts_code": ["000001.SZ"], "pe_ttm": [5.0]})

    class FakeTs:
        @staticmethod
        def pro_api(token: str):
            return FakePro()

    monkeypatch.setattr(mod, "ts", FakeTs())
    monkeypatch.setattr(mod, "_get_tushare_token", lambda: "token")
    cache = CacheManager.get()
    monkeypatch.setattr(cache, "l2", ParquetCache(tmp_path))
    fetcher = mod.TushareFetcher()

    df = fetcher.get_daily_basic()

    assert not df.empty
    assert len(calls) == 3
    assert df.attrs["trade_date"] == calls[-1]


def test_tushare_daily_basic_uses_l2_cache(monkeypatch, tmp_path):
    from data.fetchers import tushare_fetcher as mod
    from data.cache_manager import CacheManager, ParquetCache

    class FakePro:
        def daily_basic(self, trade_date: str):  # pragma: no cover - 命中缓存时不应调用
            raise AssertionError("不应请求 Tushare API")

    class FakeTs:
        @staticmethod
        def pro_api(token: str):
            return FakePro()

    cache = CacheManager.get()
    monkeypatch.setattr(cache, "l2", ParquetCache(tmp_path))
    cached_df = pd.DataFrame({"ts_code": ["000001.SZ"], "pe_ttm": [5.0]})
    cache.l2.set_snapshot("tushare_daily_basic_20260612", cached_df)

    monkeypatch.setattr(mod, "ts", FakeTs())
    monkeypatch.setattr(mod, "_get_tushare_token", lambda: "token")

    fetcher = mod.TushareFetcher()
    df = fetcher.get_daily_basic(trade_date="20260612")

    assert not df.empty
    assert df.attrs["trade_date"] == "20260612"
    assert df.attrs["source"] == "tushare_daily_basic_cache"


def test_universe_preserves_source_meta(monkeypatch):
    from data import universe as uni

    raw = pd.DataFrame({
        "symbol": ["000001"],
        "name": ["平安银行"],
        "close": [10.0],
        "pct_change": [0.1],
        "volume": [1000.0],
        "amount": [1_000_000.0],
        "turnover": [1.0],
        "pe_ttm": [6.0],
        "pb": [0.7],
        "total_mv": [100_000_000_000.0],
        "float_mv": [80_000_000_000.0],
        "list_date": [pd.Timestamp("1991-04-03")],
    })
    raw.attrs["source_meta"] = {
        "primary_source": "tushare_stock_basic",
        "daily_basic_source": "tushare_daily_basic_cache",
        "daily_basic_date": "20260612",
        "quote_source": "tencent_realtime",
    }

    monkeypatch.setattr(uni, "_load_spot_em", lambda: raw)
    df = uni.Universe.load(use_cache=False)

    assert df.attrs["source_meta"]["primary_source"] == "tushare_stock_basic"
    assert df.attrs["source_meta"]["daily_basic_source"] == "tushare_daily_basic_cache"
