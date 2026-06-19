"""每日因子快照覆盖度测试。"""
from datetime import date

import pandas as pd

from data import daily_factors


def test_snapshot_coverage_report_distinguishes_stale_symbols(tmp_path, monkeypatch):
    monkeypatch.setattr(daily_factors, "DAILY_FACTOR_DIR", tmp_path)
    rows = pd.DataFrame([
        {"symbol": "000001", "trade_date": date(2026, 6, 18)},
        {"symbol": "000002", "trade_date": date(2026, 6, 16)},
        {"symbol": "000003", "trade_date": date(2026, 6, 18)},
    ])
    rows.to_parquet(tmp_path / "factors_20260618.parquet")

    report = daily_factors.snapshot_coverage_report("20260618")

    assert report["snapshot_date"] == "20260618"
    assert report["global_latest_date"] == "2026-06-18"
    assert report["symbols"] == 3
    assert report["fresh_symbols"] == 2
    assert report["stale_symbols"] == 1
    assert report["coverage_pct"] == 66.67
    assert report["date_distribution"]["2026-06-18"] == 2
    assert report["date_distribution"]["2026-06-16"] == 1
