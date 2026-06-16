"""Baostock fetcher 适配层单元测试。"""
from __future__ import annotations

import pandas as pd


def test_baostock_symbol_format_helpers():
    from data.fetchers.baostock_fetcher import BaostockFetcher

    assert BaostockFetcher._to_bs_code("600519") == "sh.600519"
    assert BaostockFetcher._to_bs_code("000001") == "sz.000001"
    assert BaostockFetcher._to_bs_code("600519.SH") == "sh.600519"
    assert BaostockFetcher._from_bs_code("sh.600519") == "600519"
    assert BaostockFetcher._fmt_date("20260615", "19900101") == "2026-06-15"


def test_baostock_daily_bars_parse_without_network(monkeypatch):
    from data.fetchers import baostock_fetcher as mod

    class FakeResult:
        error_code = "0"
        error_msg = ""
        fields = ["date", "code", "open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]

        def __init__(self):
            self.rows = [
                ["2026-06-12", "sh.600519", "1600", "1610", "1590", "1605", "1598", "100", "16050000", "0.5", "0.44"],
            ]
            self.idx = -1

        def next(self):
            self.idx += 1
            return self.idx < len(self.rows)

        def get_row_data(self):
            return self.rows[self.idx]

    class FakeBs:
        @staticmethod
        def login():
            class Login:
                error_code = "0"
                error_msg = ""
            return Login()

        @staticmethod
        def query_history_k_data_plus(*args, **kwargs):
            return FakeResult()

    monkeypatch.setattr(mod, "bs", FakeBs())
    fetcher = mod.BaostockFetcher()
    df = fetcher.get_daily_bars("600519", "20260601", "20260612", adjust="")

    assert not df.empty
    assert df.iloc[0]["symbol"] == "600519"
    assert pd.api.types.is_datetime64_any_dtype(df["trade_date"])
    assert float(df.iloc[0]["change"]) == 7.0


def test_baostock_daily_bars_cached_uses_snapshot(monkeypatch, tmp_path):
    from data.cache_manager import CacheManager, ParquetCache
    from data.fetchers.baostock_fetcher import BaostockFetcher

    cache = CacheManager.get()
    monkeypatch.setattr(cache, "l2", ParquetCache(tmp_path))

    calls = {"n": 0}

    def fake_get_daily_bars(self, symbol, start_date=None, end_date=None, adjust="qfq", **kwargs):
        calls["n"] += 1
        return pd.DataFrame({
            "symbol": [symbol],
            "trade_date": [pd.Timestamp("2026-06-12")],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.2],
            "volume": [1000.0],
            "amount": [10200.0],
            "pct_change": [1.0],
            "change": [0.1],
        })

    monkeypatch.setattr(BaostockFetcher, "get_daily_bars", fake_get_daily_bars)
    fetcher = BaostockFetcher()

    first = fetcher.get_daily_bars_cached("000001", "20260601", "20260612", adjust="")
    second = fetcher.get_daily_bars_cached("000001", "20260601", "20260612", adjust="")

    assert calls["n"] == 1
    assert not first.empty and not second.empty
    assert second.attrs["source"] == "baostock_cache"


def test_baostock_status_reads_health_and_cache_count(monkeypatch, tmp_path):
    from data.fetchers.baostock_fetcher import BaostockFetcher

    cache_dir = tmp_path / "cache"
    parquet_dir = tmp_path / "parquet"
    snap_dir = parquet_dir / "snapshots"
    snap_dir.mkdir(parents=True)
    (snap_dir / "baostock_daily_000001_x.parquet").write_text("x")

    monkeypatch.setattr("data.fetchers.baostock_fetcher.settings.cache_dir", cache_dir)
    monkeypatch.setattr("data.fetchers.baostock_fetcher.settings.parquet_dir", parquet_dir)

    BaostockFetcher._write_health("unit", True, 3)
    status = BaostockFetcher.status()

    assert status["last_ok"] is True
    assert status["last_rows"] == 3
    assert status["cached_daily_files"] == 1


def test_baostock_stock_list_filters_indices(monkeypatch):
    from data.fetchers import baostock_fetcher as mod

    class FakeResult:
        error_code = "0"
        error_msg = ""
        fields = ["code", "code_name", "ipoDate", "outDate", "type", "status"]

        def __init__(self):
            self.rows = [
                ["sh.000001", "上证综合指数", "1991-07-15", "", "2", "1"],
                ["sh.600519", "贵州茅台", "2001-08-27", "", "1", "1"],
                ["sz.000001", "平安银行", "1991-04-03", "", "1", "1"],
                ["sz.399001", "深证成指", "1995-01-23", "", "2", "1"],
                ["sz.000003", "PT金田A", "1991-07-03", "2002-06-14", "1", "0"],
            ]
            self.idx = -1

        def next(self):
            self.idx += 1
            return self.idx < len(self.rows)

        def get_row_data(self):
            return self.rows[self.idx]

    class FakeBs:
        @staticmethod
        def login():
            class Login:
                error_code = "0"
                error_msg = ""
            return Login()

        @staticmethod
        def query_stock_basic():
            return FakeResult()

    monkeypatch.setattr(mod, "bs", FakeBs())
    fetcher = mod.BaostockFetcher()
    df = fetcher.get_stock_list()

    assert df["symbol"].tolist() == ["600519", "000001"]
