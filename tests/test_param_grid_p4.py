"""P4 参数网格验证测试。"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from backtest.param_grid import (
    GRID_RESULT_COLUMNS,
    GridRankingPolicy,
    clone_scheme_with_params,
    default_param_grid,
    iter_param_grid,
    normalize_grid_result,
    rank_grid_results,
    run_parameter_grid,
)
from strategy.schemes import BUILTIN_SCHEMES


def test_iter_param_grid_is_deterministic_and_limited():
    grid = iter_param_grid({"a": [1, 2], "b": ["x", "y"]}, max_runs=3)
    assert grid == [
        {"a": 1, "b": "x"},
        {"a": 1, "b": "y"},
        {"a": 2, "b": "x"},
    ]


def test_clone_scheme_with_params_does_not_mutate_builtin():
    base = BUILTIN_SCHEMES["trend_momentum"]
    original_stop = base.stop_loss_atr_mult
    original_hold = base.exit_config.max_holding_days
    original_min = base.resonance_config.min_confirmations

    cloned = clone_scheme_with_params(base, {
        "stop_loss_atr_mult": 1.8,
        "exit_config.max_holding_days": 8,
        "resonance_config.min_confirmations": 4,
    })

    assert cloned.stop_loss_atr_mult == pytest.approx(1.8)
    assert cloned.exit_config.max_holding_days == 8
    assert cloned.resonance_config.min_confirmations == 4
    assert base.stop_loss_atr_mult == original_stop
    assert base.exit_config.max_holding_days == original_hold
    assert base.resonance_config.min_confirmations == original_min


def test_default_param_grid_contains_p4_core_fields():
    grid = default_param_grid("breakout", max_runs=1)
    assert len(grid) == 1
    params = grid[0]
    assert "stop_loss_atr_mult" in params
    assert "take_profit_atr_mult" in params
    assert "trailing_atr_mult" in params
    assert "exit_config.max_holding_days" in params
    assert "resonance_config.min_confirmations" in params
    assert "exit_config.market_defense_score" in params


def test_default_param_grid_supports_balanced_exit_activation_fields():
    """P4: balanced 也必须纳入小样本网格，验证 10/20 与跟踪止盈激活语义。"""
    grid = default_param_grid("balanced", max_runs=4)
    assert len(grid) == 4
    params = grid[0]
    assert params["exit_config.max_holding_days"] == 15
    assert params["exit_config.time_stop_days"] == 7
    assert "exit_config.trailing_activation_pct" in params
    assert "exit_config.trailing_activation_atr_mult" in params
    combos = {(row["exit_config.max_holding_days"], row["exit_config.time_stop_days"]) for row in grid}
    assert combos == {(15, 7), (15, 10), (20, 7), (20, 10)}


def test_rank_grid_results_prioritizes_lower_drawdown_over_higher_return():
    low_dd = normalize_grid_result(
        scheme_id="trend_momentum",
        params={"case": "low_dd"},
        metrics={
            "total_return": 0.06,
            "annual_return": 0.08,
            "max_drawdown": 0.04,
            "sharpe_ratio": 1.0,
            "win_rate": 0.55,
            "trade_count": 10,
            "max_single_pct": 0.18,
            "turnover_rate": 1.5,
        },
    )
    high_return_high_dd = normalize_grid_result(
        scheme_id="trend_momentum",
        params={"case": "high_return_high_dd"},
        metrics={
            "total_return": 0.18,
            "annual_return": 0.25,
            "max_drawdown": 0.12,
            "sharpe_ratio": 1.4,
            "win_rate": 0.60,
            "trade_count": 12,
            "max_single_pct": 0.18,
            "turnover_rate": 1.5,
        },
    )

    ranked = rank_grid_results([high_return_high_dd, low_dd], GridRankingPolicy(min_trades=3, max_drawdown_limit=0.20))

    first_params = json.loads(ranked.iloc[0]["params_json"])
    assert first_params["case"] == "low_dd"
    assert ranked.iloc[0]["max_drawdown"] == pytest.approx(0.04)


def test_run_parameter_grid_applies_params_and_returns_standard_columns():
    seen = []

    def fake_runner(scheme, params):
        seen.append((scheme.stop_loss_atr_mult, scheme.exit_config.max_holding_days, params))
        return {
            "total_return": 0.05 + (2.0 - scheme.stop_loss_atr_mult) * 0.01,
            "annual_return": 0.07,
            "max_drawdown": 0.05 + (scheme.stop_loss_atr_mult - 1.8) * 0.01,
            "sharpe_ratio": 1.1,
            "win_rate": 0.52,
            "trade_count": 8,
            "avg_holding_days": 6,
            "max_single_pct": 0.20,
            "turnover_rate": 1.2,
        }

    df = run_parameter_grid(
        strategy_id="pullback",
        param_grid=[
            {"stop_loss_atr_mult": 2.0, "exit_config.max_holding_days": 15},
            {"stop_loss_atr_mult": 1.8, "exit_config.max_holding_days": 12},
        ],
        runner=fake_runner,
    )

    assert list(df.columns) == GRID_RESULT_COLUMNS
    assert len(df) == 2
    assert seen[0][0] == pytest.approx(2.0)
    assert seen[0][1] == 15
    assert seen[1][0] == pytest.approx(1.8)
    assert seen[1][1] == 12
    assert set(df["eligible"].tolist()) == {True}


def test_rank_grid_results_empty_has_standard_columns():
    df = rank_grid_results([])
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == GRID_RESULT_COLUMNS
    assert df.empty
