"""Confidence 分桶绩效审计测试。"""
from __future__ import annotations

import pandas as pd

from backtest.confidence_audit import (
    aggregate_confidence_buckets,
    render_confidence_audit_report,
    persist_confidence_audit,
)


def test_aggregate_confidence_buckets_groups_round_details():
    details = pd.DataFrame([
        {"run_id": "r1", "symbol": "600143", "confidence_bucket": "watch", "confidence_action": "observe_only", "pnl": -10.0, "pnl_pct": -0.01, "holding_days": 3},
        {"run_id": "r1", "symbol": "600143", "confidence_bucket": "standard", "confidence_action": "standard_entry", "pnl": 30.0, "pnl_pct": 0.03, "holding_days": 10},
        {"run_id": "r2", "symbol": "600163", "confidence_bucket": "standard", "confidence_action": "standard_entry", "pnl": -5.0, "pnl_pct": -0.005, "holding_days": 6},
    ])

    agg = aggregate_confidence_buckets(details)

    standard = agg[agg["confidence_bucket"] == "standard"].iloc[0]
    assert standard["完成轮数"] == 2
    assert standard["run_count"] == 2
    assert standard["总盈亏"] == 25.0
    watch = agg[agg["confidence_bucket"] == "watch"].iloc[0]
    assert watch["胜率"] == 0.0


def test_render_confidence_audit_report_includes_constraints():
    details = pd.DataFrame([
        {"run_id": "r1", "symbol": "600143", "confidence_bucket": "watch", "confidence_action": "observe_only", "pnl": -10.0, "pnl_pct": -0.01, "holding_days": 3},
    ])

    report = render_confidence_audit_report(summary_df=pd.DataFrame(), details_df=details, config={"pattern": "20260620_*"})

    assert "Confidence 分桶绩效审计报告" in report
    assert "审计模式" in report
    assert "不能单独决定硬阈值" in report
    assert "watch" in report


def test_persist_confidence_audit_writes_expected_files(tmp_path):
    summary = pd.DataFrame([{"run_id": "r1", "confidence_bucket": "watch", "完成轮数": 1}])
    details = pd.DataFrame([
        {"run_id": "r1", "symbol": "600143", "confidence_bucket": "watch", "confidence_action": "observe_only", "pnl": -10.0, "pnl_pct": -0.01, "holding_days": 3},
    ])

    out = persist_confidence_audit(output_dir=tmp_path / "audit", summary_df=summary, details_df=details, config={"pattern": "r*"})

    assert (out / "confidence_summary.csv").exists()
    assert (out / "confidence_details.csv").exists()
    assert (out / "confidence_bucket_aggregate.csv").exists()
    assert (out / "summary.md").exists()
