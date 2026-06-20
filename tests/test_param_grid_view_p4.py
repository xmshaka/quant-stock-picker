"""P4 参数网格前端辅助函数测试。"""
from __future__ import annotations

import json

import pandas as pd

from dashboard.param_grid_view import (
    format_grid_results_for_display,
    list_grid_audit_runs,
    load_grid_audit_run,
    summarize_grid_run,
)


def _write_run(root, run_id="20250101_000000_pullback"):
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    cfg = {
        "strategy_id": "pullback",
        "created_at": "2025-01-01T00:00:00",
        "max_runs": 2,
        "lookback_days": 20,
        "top_n": 5,
        "initial_capital": 1_000_000,
        "factor_names": ["rsi14"],
        "symbols": ["000001"],
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    (run_dir / "summary.md").write_text("# 参数网格验证摘要\n", encoding="utf-8")
    pd.DataFrame([
        {
            "rank": 1,
            "scheme_id": "pullback",
            "eligible": True,
            "risk_score": 0.03,
            "stability_score": 0.10,
            "total_return": 0.05,
            "annual_return": 0.10,
            "max_drawdown": 0.03,
            "sharpe_ratio": 1.2,
            "win_rate": 0.55,
            "trade_count": 8,
            "avg_holding_days": 5,
            "max_single_pct": 0.18,
            "params_json": '{"stop_loss_atr_mult": 1.8}',
        },
        {
            "rank": 2,
            "scheme_id": "pullback",
            "eligible": False,
            "risk_score": 0.10,
            "stability_score": 0.05,
            "total_return": 0.08,
            "annual_return": 0.15,
            "max_drawdown": 0.10,
            "sharpe_ratio": 0.8,
            "win_rate": 0.50,
            "trade_count": 1,
            "avg_holding_days": 3,
            "max_single_pct": 0.22,
            "params_json": '{"stop_loss_atr_mult": 2.0}',
        },
    ]).to_csv(run_dir / "grid_results.csv", index=False)
    return run_dir


def test_list_grid_audit_runs_reads_config_and_result_summary(tmp_path):
    run_dir = _write_run(tmp_path)

    runs = list_grid_audit_runs(tmp_path)

    assert len(runs) == 1
    row = runs.iloc[0]
    assert row["run_id"] == run_dir.name
    assert row["strategy_id"] == "pullback"
    assert row["row_count"] == 2
    assert row["eligible_count"] == 1
    assert row["best_return"] == 0.08
    assert row["best_drawdown"] == 0.03


def test_load_grid_audit_run_reads_artifacts(tmp_path):
    run_dir = _write_run(tmp_path)

    loaded = load_grid_audit_run(run_dir)

    assert loaded["config"]["strategy_id"] == "pullback"
    assert "参数网格验证摘要" in loaded["summary"]
    assert len(loaded["results"]) == 2
    assert loaded["run_id"] == run_dir.name


def test_format_grid_results_for_display_formats_percent_and_columns():
    df = pd.DataFrame([
        {
            "rank": 1,
            "eligible": True,
            "risk_score": 0.03,
            "total_return": 0.05,
            "max_drawdown": 0.03,
            "sharpe_ratio": 1.2,
            "win_rate": 0.55,
            "trade_count": 8,
            "avg_holding_days": 5,
            "max_single_pct": 0.18,
            "params_json": "{}",
        }
    ])

    display = format_grid_results_for_display(df)

    assert list(display.columns) == ["排名", "合格", "风险分", "总收益", "最大回撤", "夏普", "胜率", "交易数", "平均持仓天数", "最大单票占比", "参数"]
    assert display.iloc[0]["总收益"] == "5.0000%"
    assert display.iloc[0]["最大回撤"] == "3.0000%"
    assert display.iloc[0]["风险分"] == "0.0300"


def test_summarize_grid_run_returns_metric_cards():
    summary = summarize_grid_run(
        {"strategy_id": "breakout"},
        pd.DataFrame([
            {"rank": 1, "eligible": True, "total_return": 0.04, "max_drawdown": 0.02},
            {"rank": 2, "eligible": False, "total_return": 0.08, "max_drawdown": 0.10},
        ]),
    )

    assert summary["策略"] == "breakout"
    assert summary["组合数"] == "2 组"
    assert summary["合格组合"] == "1 组"
    assert summary["Top1收益"] == "4.0000%"
    assert summary["Top1回撤"] == "2.0000%"


def test_list_grid_audit_runs_empty_root_has_standard_columns(tmp_path):
    runs = list_grid_audit_runs(tmp_path / "missing")
    assert runs.empty
    assert "run_id" in runs.columns
    assert "eligible_count" in runs.columns
