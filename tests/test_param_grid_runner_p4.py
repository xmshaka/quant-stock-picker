"""P4 参数网格 runner / CLI 测试。"""
from __future__ import annotations

import subprocess
import sys
import json

import pandas as pd
import pytest

from backtest.param_grid_runner import (
    build_grid_audit_config,
    infer_factor_names,
    load_grid_input_frames,
    make_scheme_grid_runner,
    metrics_from_scheme_result,
    persist_grid_audit_run,
    render_grid_summary,
    run_scheme_parameter_grid,
    save_grid_results,
)
from backtest.scheme_backtest import SchemeBacktestResult
from strategy.schemes import BUILTIN_SCHEMES


def test_infer_factor_names_uses_numeric_non_meta_columns():
    df = pd.DataFrame({
        "symbol": ["000001"],
        "trade_date": ["2025-01-01"],
        "close": [10.0],
        "rsi14": [55.0],
        "momentum_5d": [0.03],
        "industry": ["银行"],
    })
    assert infer_factor_names(df) == ["rsi14", "momentum_5d"]
    assert infer_factor_names(df, ["rsi14"]) == ["rsi14"]


def test_load_grid_input_frames_normalizes_trade_date(tmp_path):
    factor_path = tmp_path / "factors.csv"
    price_path = tmp_path / "prices.csv"
    pd.DataFrame({"symbol": ["000001"], "trade_date": ["2025-01-01"], "rsi14": [50]}).to_csv(factor_path, index=False)
    pd.DataFrame({"symbol": ["000001"], "trade_date": ["2025-01-01"], "close": [10]}).to_csv(price_path, index=False)

    factor_df, price_df = load_grid_input_frames(factor_path, price_path)

    assert pd.api.types.is_datetime64_any_dtype(factor_df["trade_date"])
    assert pd.api.types.is_datetime64_any_dtype(price_df["trade_date"])
    assert factor_df.iloc[0]["symbol"] == "000001"
    assert price_df.iloc[0]["symbol"] == "000001"


def test_metrics_from_scheme_result_extracts_avg_holding_and_concentration():
    result = SchemeBacktestResult(
        scheme_id="pullback",
        scheme_name="回调低吸",
        start_date="2025-01-01",
        end_date="2025-01-10",
        trade_count=2,
        total_return=0.05,
        annual_return=0.12,
        sharpe_ratio=1.2,
        max_drawdown=0.04,
        win_rate=0.5,
        trade_details=[
            {"action": "BUY", "position_after": 1000, "exec_price": 10.0},
            {"action": "SELL", "position_after": 0, "exec_price": 11.0, "holding_days": 5},
        ],
    )

    metrics = metrics_from_scheme_result(result, initial_capital=100_000)

    assert metrics["avg_holding_days"] == pytest.approx(5.0)
    assert metrics["max_single_pct"] == pytest.approx(0.10)
    assert metrics["total_return"] == pytest.approx(0.05)


def test_make_scheme_grid_runner_calls_scheme_backtester(monkeypatch):
    calls = []

    class FakeBacktester:
        def run(self, **kwargs):
            calls.append(kwargs)
            return SchemeBacktestResult(
                scheme_id=kwargs["scheme"].scheme_id,
                scheme_name=kwargs["scheme"].name,
                start_date="2025-01-01",
                end_date="2025-01-10",
                trade_count=4,
                total_return=0.03,
                annual_return=0.08,
                max_drawdown=0.02,
                sharpe_ratio=1.1,
                win_rate=0.5,
                trade_details=[],
            )

    monkeypatch.setattr("backtest.param_grid_runner.SchemeBacktester", FakeBacktester)
    runner = make_scheme_grid_runner(
        factor_df=pd.DataFrame({"symbol": ["000001"], "trade_date": ["2025-01-01"], "rsi14": [50.0]}),
        price_df=pd.DataFrame({"symbol": ["000001"], "trade_date": ["2025-01-01"], "close": [10.0]}),
        factor_names=["rsi14"],
        symbols=["000001"],
        lookback_days=5,
        top_n=1,
        initial_capital=100_000,
    )
    metrics = runner(BUILTIN_SCHEMES["pullback"], {"stop_loss_atr_mult": 1.8})

    assert metrics["trade_count"] == 4
    assert calls[0]["symbols"] == ["000001"]
    assert calls[0]["lookback_days"] == 5
    assert calls[0]["top_n"] == 1
    assert calls[0]["initial_capital"] == 100_000


def test_run_scheme_parameter_grid_with_stubbed_runner(monkeypatch):
    class FakeBacktester:
        def run(self, **kwargs):
            scheme = kwargs["scheme"]
            return SchemeBacktestResult(
                scheme_id=scheme.scheme_id,
                scheme_name=scheme.name,
                start_date="2025-01-01",
                end_date="2025-01-10",
                trade_count=5,
                total_return=0.04,
                annual_return=0.10,
                max_drawdown=0.03,
                sharpe_ratio=1.0,
                win_rate=0.55,
                trade_details=[],
            )

    monkeypatch.setattr("backtest.param_grid_runner.SchemeBacktester", FakeBacktester)
    df = run_scheme_parameter_grid(
        strategy_id="breakout",
        factor_df=pd.DataFrame({"symbol": ["000001"], "trade_date": ["2025-01-01"], "rsi14": [50.0]}),
        price_df=pd.DataFrame({"symbol": ["000001"], "trade_date": ["2025-01-01"], "close": [10.0]}),
        factor_names=["rsi14"],
        symbols=["000001"],
        max_runs=2,
    )

    assert len(df) == 2
    assert df["eligible"].all()
    assert set(df["scheme_id"]) == {"breakout"}


def test_save_grid_results_csv(tmp_path):
    out = save_grid_results(pd.DataFrame([{"rank": 1, "total_return": 0.1}]), tmp_path / "grid.csv")
    assert out.exists()
    loaded = pd.read_csv(out)
    assert loaded.iloc[0]["rank"] == 1


def test_persist_grid_audit_run_writes_standard_artifacts(tmp_path):
    df = pd.DataFrame([
        {
            "rank": 1,
            "scheme_id": "pullback",
            "params_json": '{"stop_loss_atr_mult": 1.8}',
            "eligible": True,
            "total_return": 0.05,
            "max_drawdown": 0.03,
            "sharpe_ratio": 1.2,
            "trade_count": 8,
        }
    ])
    cfg = build_grid_audit_config(
        strategy_id="pullback",
        max_runs=1,
        lookback_days=20,
        top_n=5,
        initial_capital=1_000_000,
        symbols=["000001"],
        factor_names=["rsi14"],
        extra={"baseline": "unit"},
    )

    run_dir = persist_grid_audit_run(
        df=df,
        root=tmp_path,
        strategy_id="pullback",
        config=cfg,
        run_id="20250101_000000_pullback",
    )

    assert (run_dir / "grid_results.csv").exists()
    assert (run_dir / "grid_results.parquet").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "summary.md").exists()
    loaded_cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert loaded_cfg["strategy_id"] == "pullback"
    assert loaded_cfg["ranking_policy"]["min_trades"] == 3
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "参数网格验证摘要" in summary
    assert "Top 5" in summary


def test_render_grid_summary_handles_empty_result():
    summary = render_grid_summary(pd.DataFrame(), {"strategy_id": "breakout", "max_runs": 0})
    assert "无参数网格结果" in summary


def test_run_param_grid_cli_smoke_uses_local_price_data(tmp_path):
    """真实 CLI smoke：小样本、max_runs=2，不触发外部全A重任务。"""
    dates = pd.date_range("2025-01-01", periods=8, freq="D")
    symbols = ["000001", "000002", "000003", "000004"]
    factors = []
    prices = []
    for i, d in enumerate(dates):
        for j, sym in enumerate(symbols):
            factors.append({
                "symbol": sym,
                "trade_date": d,
                "momentum_20d": j + i * 0.01,
                "momentum_5d": j * 0.5,
                "volume_ratio": 1 + j * 0.1,
                "boll_position": 0.2 + j * 0.1,
                "high_20d_distance": -0.1 * j,
                "rsi14": 40 + j * 5,
            })
            base = 10 + j + i * 0.2
            prices.append({
                "symbol": sym,
                "trade_date": d,
                "open": base,
                "high": base + 0.5,
                "low": base - 0.5,
                "close": base + 0.1,
                "volume": 200_000,
                "amount": 200_000_000,
                "source": "sample",
                "adjust": "raw",
            })
    factor_path = tmp_path / "factors.csv"
    price_path = tmp_path / "prices.csv"
    out_path = tmp_path / "grid.csv"
    pd.DataFrame(factors).to_csv(factor_path, index=False)
    pd.DataFrame(prices).to_csv(price_path, index=False)

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_param_grid.py",
            "--strategy-id", "trend_momentum",
            "--factor-path", str(factor_path),
            "--price-path", str(price_path),
            "--factor-names", "momentum_20d,momentum_5d,volume_ratio,boll_position,high_20d_distance,rsi14",
            "--symbols", ",".join(symbols),
            "--lookback-days", "6",
            "--top-n", "1",
            "--max-runs", "2",
            "--output", str(out_path),
            "--audit-root", str(tmp_path / "audit"),
        ],
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "saved 2 grid rows" in proc.stdout
    assert "audit saved to" in proc.stdout
    df = pd.read_csv(out_path)
    assert len(df) == 2
    assert set(["rank", "params_json", "max_drawdown", "total_return", "trade_count", "eligible"]).issubset(df.columns)
    assert df["trade_count"].max() >= 1
    audit_dirs = list((tmp_path / "audit").glob("*_trend_momentum"))
    assert len(audit_dirs) == 1
    assert (audit_dirs[0] / "config.json").exists()
    assert (audit_dirs[0] / "summary.md").exists()
