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


def test_scheduler_has_factor_job():
    """scheduler 应注册全池因子 job。"""
    pytest.importorskip("apscheduler")
    from data.scheduler import build_scheduler

    sched = build_scheduler()
    job_ids = {j.id for j in sched.get_jobs()}
    assert "daily_factor_snapshot" in job_ids
