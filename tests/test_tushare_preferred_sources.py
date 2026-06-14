"""Tushare 优先数据源策略回归测试。"""

from __future__ import annotations

import pandas as pd


def test_fallback_default_order_prefers_tushare_before_akshare(monkeypatch):
    from data.fetchers.fallback_fetcher import FallbackFetcher

    monkeypatch.setattr("data.fetchers.fallback_fetcher.settings.data_source_order", "")
    ff = FallbackFetcher(source_order=[])

    assert ff.source_order == ["tencent", "tushare", "akshare"]


def test_dashboard_loader_default_order_prefers_tushare_before_akshare(monkeypatch):
    from dashboard.data_loader import DataLoader

    monkeypatch.setattr(DataLoader, "_init_fetchers", lambda self: None)
    loader = DataLoader()

    assert loader.sources == ["tencent", "tushare", "akshare", "mock"]


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


def test_tushare_daily_basic_looks_back_when_default_date_empty(monkeypatch):
    from data.fetchers import tushare_fetcher as mod

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
    fetcher = mod.TushareFetcher()

    df = fetcher.get_daily_basic()

    assert not df.empty
    assert len(calls) == 3
    assert df.attrs["trade_date"] == calls[-1]
