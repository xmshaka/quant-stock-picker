"""P4/P5 冒烟测试: 多源降级 + 每日调度告警"""
import sys
from pathlib import Path

sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import pandas as pd
import pytest


class _FailFetcher:
    def get_daily_bars(self, *args, **kwargs):
        raise RuntimeError("boom")

    def get_stock_list(self):
        return pd.DataFrame()

    def get_sector_list(self):
        return pd.DataFrame()


class _OkFetcher:
    def get_daily_bars(self, symbol, start_date=None, end_date=None, **kwargs):
        return pd.DataFrame({
            "symbol": [symbol],
            "trade_date": [pd.Timestamp("2026-05-22")],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.2],
            "volume": [1000.0],
            "amount": [10200.0],
            "pct_change": [1.0],
            "change": [0.1],
        })

    def get_stock_list(self):
        return pd.DataFrame({
            "symbol": ["000001"],
            "name": ["平安银行"],
            "close": [10.2],
        })

    def get_sector_list(self):
        return pd.DataFrame({
            "sector_name": ["银行"],
            "sector_code": ["BK0475"],
        })


def test_fallback_daily_bars_switches_source():
    from data.fetchers.fallback_fetcher import FallbackFetcher

    ff = FallbackFetcher(source_order=["tencent", "akshare"])
    ff.fetchers = {
        "tencent": _FailFetcher(),
        "akshare": _OkFetcher(),
    }

    df = ff.get_daily_bars("000001", "20260501", "20260522")
    assert not df.empty
    assert df["source"].iloc[0] == "akshare"
    assert ff.last_report.selected_source == "akshare"
    assert [a.source for a in ff.last_report.attempts] == ["tencent", "akshare"]
    assert ff.last_report.attempts[0].ok is False
    assert ff.last_report.attempts[1].ok is True


def test_fallback_stock_list_marks_source():
    from data.fetchers.fallback_fetcher import FallbackFetcher

    ff = FallbackFetcher(source_order=["tencent", "akshare"])
    ff.fetchers = {
        "tencent": _FailFetcher(),
        "akshare": _OkFetcher(),
    }

    df = ff.get_stock_list()
    assert not df.empty
    assert df["source"].iloc[0] == "akshare"
    assert ff.last_report.operation == "get_stock_list"


def test_scheduler_write_scan_alert(tmp_path, monkeypatch):
    from data.incremental import ScanReport
    from data import scheduler

    alert_file = tmp_path / "alerts.jsonl"
    monkeypatch.setattr(scheduler.settings, "daily_scan_alert_file", str(alert_file))

    report = ScanReport(
        total_symbols=2,
        skipped_up_to_date=1,
        updated_count=1,
        failed_count=0,
        new_rows=5,
        elapsed_seconds=1.2,
    )
    path = scheduler.write_scan_alert(report, level="INFO")
    assert path == alert_file
    text = alert_file.read_text(encoding="utf-8")
    assert '"level": "INFO"' in text
    assert '"total_symbols": 2' in text
    assert "增量扫描报告" in text


def test_scheduler_builds_job():
    pytest.importorskip("apscheduler")
    from data.scheduler import build_scheduler

    sched = build_scheduler()
    jobs = sched.get_jobs()
    assert len(jobs) == 2
    job_ids = {j.id for j in jobs}
    assert "daily_incremental_scan" in job_ids
    assert "daily_factor_snapshot" in job_ids
