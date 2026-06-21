"""每日全池因子快照模块测试"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def test_snapshot_helpers_handle_empty(tmp_path, monkeypatch):
    """空目录时，latest_snapshot_date 应返回 None，has_daily_factors False。"""
    from data import daily_factors as df_mod

    monkeypatch.setattr(df_mod, "DAILY_FACTOR_DIR", tmp_path)
    assert df_mod.latest_snapshot_date() is None
    assert df_mod.has_daily_factors("20260101") is False
    assert df_mod.load_snapshot_meta() is None


def test_snapshot_helpers_roundtrip(tmp_path, monkeypatch):
    """写入 parquet + meta 后，能正确读回。"""
    from data import daily_factors as df_mod

    monkeypatch.setattr(df_mod, "DAILY_FACTOR_DIR", tmp_path)

    date_str = "20260526"
    factor_df = pd.DataFrame({
        "symbol": ["600519", "000001"],
        "trade_date": ["2026-05-26", "2026-05-26"],
        "momentum_20d": [0.12, -0.03],
        "rsi_14": [55.0, 42.5],
    })
    price_df = pd.DataFrame({
        "symbol": ["600519", "000001"],
        "trade_date": ["2026-05-26", "2026-05-26"],
        "close": [1700.0, 11.5],
    })

    factor_df.to_parquet(tmp_path / f"factors_{date_str}.parquet")
    price_df.to_parquet(tmp_path / f"prices_{date_str}.parquet")
    meta = {
        "date": date_str,
        "universe_size": 2,
        "factor_names": ["momentum_20d", "rsi_14"],
        "elapsed_seconds": 1.23,
        "computed_at": "2026-05-26T16:30:00",
    }
    (tmp_path / f"meta_{date_str}.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )

    assert df_mod.latest_snapshot_date() == date_str
    assert df_mod.has_daily_factors(date_str)

    f_df, p_df, names = df_mod.load_daily_factors(date_str)
    assert len(f_df) == 2 and len(p_df) == 2
    assert set(names) == {"momentum_20d", "rsi_14"}

    loaded_meta = df_mod.load_snapshot_meta(date_str)
    assert loaded_meta["universe_size"] == 2
    assert loaded_meta["factor_names"] == ["momentum_20d", "rsi_14"]


def test_latest_data_source_meta_reads_snapshot_meta(tmp_path, monkeypatch):
    """状态页应能从快照 meta 暴露真实数据来源。"""
    from data import daily_factors as df_mod

    monkeypatch.setattr(df_mod, "DAILY_FACTOR_DIR", tmp_path)
    date_str = "20260612"
    pd.DataFrame({"symbol": ["000001"], "trade_date": ["2026-06-12"]}).to_parquet(
        tmp_path / f"factors_{date_str}.parquet"
    )
    meta = {
        "date": date_str,
        "data_source": "daily_factor_snapshot",
        "universe_source": "tushare_stock_basic",
        "quote_source": "tencent_realtime",
        "daily_basic_source": "tushare_daily_basic_cache",
        "daily_basic_date": "20260612",
        "computed_at": "2026-06-12T16:30:00",
    }
    (tmp_path / f"meta_{date_str}.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )

    source = df_mod.latest_data_source_meta()

    assert source["snapshot_date"] == date_str
    assert source["primary_source"] == "tushare_stock_basic"
    assert source["daily_basic_source"] == "tushare_daily_basic_cache"


def test_load_data_prefers_snapshot(tmp_path, monkeypatch):
    """load_data 命中快照时不应调用真实数据源。"""
    from data import daily_factors as df_mod

    monkeypatch.setattr(df_mod, "DAILY_FACTOR_DIR", tmp_path)

    date_str = "20260526"
    factor_df = pd.DataFrame({
        "symbol": ["600519"],
        "trade_date": ["2026-05-26"],
        "momentum_20d": [0.12],
    })
    price_df = pd.DataFrame({
        "symbol": ["600519"],
        "trade_date": ["2026-05-26"],
        "close": [1700.0],
    })
    factor_df.to_parquet(tmp_path / f"factors_{date_str}.parquet")
    price_df.to_parquet(tmp_path / f"prices_{date_str}.parquet")
    (tmp_path / f"meta_{date_str}.json").write_text("{}", encoding="utf-8")

    # 拦截 DataLoader 防止真实数据源被调用
    import dashboard.data_loader as dl

    calls = {"loader": 0}

    def _boom(*args, **kwargs):
        calls["loader"] += 1
        raise AssertionError("不应回退到 DataLoader.load")

    monkeypatch.setattr(dl.DataLoader, "load", _boom, raising=True)

    f_df, p_df, names = dl.load_data(data_source="tencent", prefer_snapshot=True)
    assert calls["loader"] == 0
    assert len(f_df) == 1
    assert "momentum_20d" in names


def test_enrich_short_term_factors_adds_moneyflow_with_unit_conversion():
    """Tushare moneyflow 金额为万元，资金流占成交额必须换算到元口径。"""
    from data.daily_factors import enrich_short_term_factors

    factor_df = pd.DataFrame({
        "symbol": ["000001", "000002"],
        "trade_date": ["2026-06-18", "2026-06-18"],
        "turnover_ratio": [2.0, 4.0],
    })
    price_df = pd.DataFrame({
        "symbol": ["000001", "000002"],
        "trade_date": ["2026-06-18", "2026-06-18"],
        "amount": [100_000_000.0, 200_000_000.0],  # 元
    })
    moneyflow_df = pd.DataFrame({
        "ts_code": ["000001.SZ", "000002.SZ"],
        "trade_date": ["20260618", "20260618"],
        "net_mf_amount": [1000.0, -2000.0],        # 万元
        "buy_sm_amount": [4000.0, 10000.0],
        "buy_md_amount": [4700.0, 9800.0],
        "buy_lg_amount": [500.0, 100.0],
        "buy_elg_amount": [800.0, 100.0],
        "sell_sm_amount": [4000.0, 10000.0],
        "sell_md_amount": [5300.0, 9000.0],
        "sell_lg_amount": [300.0, 400.0],
        "sell_elg_amount": [200.0, 500.0],
    })

    out = enrich_short_term_factors(factor_df, price_df, moneyflow_df)
    row1 = out[out["symbol"] == "000001"].iloc[0]
    row2 = out[out["symbol"] == "000002"].iloc[0]

    assert row1["main_net_mf_amount"] == pytest.approx(1000.0)
    assert row1["large_net_mf_amount"] == pytest.approx(200.0)
    assert row1["elg_net_mf_amount"] == pytest.approx(600.0)
    assert row1["large_elg_net_mf_amount"] == pytest.approx(800.0)
    assert row1["main_net_mf_pct_amount"] == pytest.approx(0.10)  # 1000万 / 1亿
    assert row1["large_elg_net_mf_pct_amount"] == pytest.approx(0.08)
    assert row1["main_net_mf_rank"] > row2["main_net_mf_rank"]
    assert row1["large_elg_net_mf_rank"] > row2["large_elg_net_mf_rank"]


def test_moneyflow_pct_amount_falls_back_to_moneyflow_turnover_when_price_amount_missing():
    """price_df 缺少成交额时，用 moneyflow 买方拆单总额估算成交额。"""
    from data.daily_factors import add_moneyflow_factors

    factor_df = pd.DataFrame({"symbol": ["000001"], "trade_date": ["2026-06-18"]})
    moneyflow_df = pd.DataFrame({
        "ts_code": ["000001.SZ"],
        "trade_date": ["20260618"],
        "net_mf_amount": [1000.0],
        "buy_sm_amount": [4000.0], "buy_md_amount": [3000.0],
        "buy_lg_amount": [2000.0], "buy_elg_amount": [1000.0],
        "sell_sm_amount": [4000.0], "sell_md_amount": [3000.0],
        "sell_lg_amount": [1500.0], "sell_elg_amount": [1500.0],
    })

    out = add_moneyflow_factors(factor_df, moneyflow_df, price_df=None)

    assert out.iloc[0]["main_net_mf_pct_amount"] == pytest.approx(0.10)  # 1000万 / 1亿


def test_relative_turnover_factors_use_symbol_history_only():
    """相对换手因子只按个股历史窗口计算，不混用其他股票和未来数据。"""
    from data.daily_factors import add_relative_turnover_factors

    dates = pd.date_range("2026-01-01", periods=12, freq="B")
    rows = []
    price_rows = []
    for i, d in enumerate(dates, start=1):
        rows.append({"symbol": "000001", "trade_date": d, "turnover_ratio": float(i)})
        rows.append({"symbol": "000002", "trade_date": d, "turnover_ratio": 100.0})
        price_rows.append({"symbol": "000001", "trade_date": d, "amount": float(i * 1_000_000)})
        price_rows.append({"symbol": "000002", "trade_date": d, "amount": 100_000_000.0})
    factor_df = pd.DataFrame(rows)
    price_df = pd.DataFrame(price_rows)

    out = add_relative_turnover_factors(factor_df, price_df)
    latest = out[(out["symbol"] == "000001") & (pd.to_datetime(out["trade_date"]) == dates[-1])].iloc[0]

    assert latest["relative_turnover_5d"] == pytest.approx(12.0 / 10.0)   # mean(8..12)
    assert latest["relative_turnover_20d"] == pytest.approx(12.0 / 6.5)   # 12日可用历史均值
    assert 0 < latest["turnover_percentile_60d"] <= 1
    assert 0 < latest["amount_percentile_60d"] <= 1


def test_relative_turnover_factors_can_use_price_turnover_snapshot():
    """factor_df 无 turnover 时，允许从真实 price snapshot 的 turnover 计算相对换手。"""
    from data.daily_factors import add_relative_turnover_factors

    dates = pd.date_range("2026-01-01", periods=12, freq="B")
    factor_rows = []
    price_rows = []
    for i, d in enumerate(dates, start=1):
        factor_rows.append({"symbol": "000001", "trade_date": d, "momentum_20d": 1.0})
        price_rows.append({"symbol": "000001", "trade_date": d, "turnover": float(i), "amount": float(i * 1_000_000)})

    out = add_relative_turnover_factors(pd.DataFrame(factor_rows), pd.DataFrame(price_rows))
    latest = out[pd.to_datetime(out["trade_date"]) == dates[-1]].iloc[0]

    assert latest["relative_turnover_5d"] == pytest.approx(12.0 / 10.0)
    assert latest["relative_turnover_20d"] == pytest.approx(12.0 / 6.5)
    assert 0 < latest["turnover_percentile_60d"] <= 1


def test_relative_turnover_factors_can_use_tushare_daily_basic_history():
    """真实全池快照应优先可用 Tushare daily_basic.turnover_rate 历史计算相对换手。"""
    from data.daily_factors import add_relative_turnover_factors

    dates = pd.date_range("2026-01-01", periods=12, freq="B")
    factor_rows = []
    daily_basic_rows = []
    price_rows = []
    for i, d in enumerate(dates, start=1):
        factor_rows.append({"symbol": "000001", "trade_date": d, "momentum_20d": 1.0})
        daily_basic_rows.append({"ts_code": "000001.SZ", "trade_date": d.strftime("%Y%m%d"), "turnover_rate": float(i)})
        price_rows.append({"symbol": "000001", "trade_date": d, "amount": float(i * 1_000_000)})

    out = add_relative_turnover_factors(
        pd.DataFrame(factor_rows),
        pd.DataFrame(price_rows),
        pd.DataFrame(daily_basic_rows),
    )
    latest = out[pd.to_datetime(out["trade_date"]) == dates[-1]].iloc[0]

    assert latest["relative_turnover_5d"] == pytest.approx(12.0 / 10.0)
    assert latest["relative_turnover_20d"] == pytest.approx(12.0 / 6.5)
    assert 0 < latest["turnover_percentile_60d"] <= 1


def test_fetch_tushare_daily_basic_turnover_history_uses_trade_dates(monkeypatch):
    """daily_basic 换手率历史按已有交易日拉取，不依赖非交易日猜测。"""
    from data import daily_factors as df_mod

    calls = []

    class FakeTushareFetcher:
        def get_daily_basic(self, trade_date):
            calls.append(trade_date)
            return pd.DataFrame({
                "ts_code": ["000001.SZ"],
                "trade_date": [trade_date],
                "turnover_rate": [2.5],
                "pb": [1.0],
            })

    monkeypatch.setattr("data.fetchers.tushare_fetcher.TushareFetcher", lambda: FakeTushareFetcher())

    out = df_mod.fetch_tushare_daily_basic_turnover_history([
        "2026-06-17",
        "2026-06-18",
        "2026-06-18",
    ])

    assert calls == ["20260617", "20260618"]
    assert list(out.columns) == ["ts_code", "trade_date", "turnover_rate"]
    assert len(out) == 2


def test_price_snapshot_frame_preserves_amount_for_amount_percentile():
    """daily price snapshot 不能再只保留 close，否则 amount_percentile_60d 永久缺失。"""
    from dashboard.data_loader import _price_snapshot_frame

    raw = pd.DataFrame({
        "symbol": ["000001"],
        "trade_date": ["2026-06-18"],
        "open": [10.0],
        "high": [10.5],
        "low": [9.8],
        "close": [10.2],
        "volume": [1_000_000],
        "amount": [102_000_000.0],
        "turnover": [2.5],
        "adjust": ["raw"],
    })

    out = _price_snapshot_frame(raw)

    assert ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount", "turnover"] == list(out.columns)
    assert out.iloc[0]["amount"] == pytest.approx(102_000_000.0)
    assert out.iloc[0]["turnover"] == pytest.approx(2.5)


def test_scheduler_has_factor_job():
    """scheduler 应注册全池因子 job。"""
    pytest.importorskip("apscheduler")
    from data.scheduler import build_scheduler

    sched = build_scheduler()
    job_ids = {j.id for j in sched.get_jobs()}
    assert "daily_factor_snapshot" in job_ids
