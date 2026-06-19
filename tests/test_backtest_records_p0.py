"""P0 回测记录与K线买卖点一致性测试。"""
from datetime import date
import os
import time

import pytest
import pandas as pd

from backtest.engine import BacktestParams, get_liquidity_slippage_rate
from backtest.records import (
    BacktestRunConfig,
    STANDARD_TRADE_COLUMNS,
    list_backtest_runs,
    load_backtest_run,
    delete_backtest_run,
    persist_backtest_run,
    validate_backtest_consistency,
    validate_trade_schema,
    make_run_id,
    summarize_liquidity_slippage,
    trade_details_to_frame,
    trade_points_to_frame,
)
from backtest.scheme_backtest import SchemeBacktestResult
from signals.rules import TradePoint
from strategy.schemes import StrategyScheme


def test_executed_signals_trade_details_consistency():
    """K线点位、统计次数、交易明细必须来自同一执行事件源。"""
    executed = {
        "000001": [
            TradePoint(
                date=date(2025, 5, 6), action="BUY", reason="执行买入",
                confidence=1.0, price=10.05, exec_price=10.05, shares=1000,
                cash_after=989_945, position_shares=1000,
            ),
            TradePoint(
                date=date(2025, 5, 12), action="SELL", reason="执行卖出",
                confidence=1.0, price=10.80, exec_price=10.80, shares=1000,
                cash_after=1_000_710, position_shares=0, pnl=710,
            ),
        ]
    }
    raw = {
        "000001": [
            TradePoint(date=date(2025, 5, 5), action="BUY", reason="原始规则", confidence=0.8, price=10.0),
            TradePoint(date=date(2025, 5, 11), action="SELL", reason="原始规则", confidence=0.8, price=10.9),
        ]
    }
    result = SchemeBacktestResult(
        scheme_id="p0",
        scheme_name="P0测试",
        start_date="2025-05-01",
        end_date="2025-05-15",
        run_id="20250515_000000_p0",
        buy_count=1,
        sell_count=1,
        trade_count=1,
        signals_executed=executed,
        signals_raw=raw,
        trade_details=[
            {"symbol": "000001", "date": date(2025, 5, 6), "action": "BUY", "shares": 1000},
            {"symbol": "000001", "date": date(2025, 5, 12), "action": "SELL", "shares": 1000},
        ],
    )

    check = validate_backtest_consistency(result)
    assert check["ok"] is True
    assert result.stock_signals == executed
    assert result.signals_raw == raw


def test_persist_backtest_run_writes_required_files(tmp_path):
    """run_id 目录必须包含 config/metrics/trades/signals/equity。"""
    executed = {
        "000001": [
            TradePoint(date=date(2025, 5, 6), action="BUY", reason="执行买入", confidence=1, price=10, shares=100),
        ]
    }
    result = SchemeBacktestResult(
        scheme_id="p0",
        scheme_name="P0测试",
        start_date="2025-05-01",
        end_date="2025-05-15",
        run_id="20250515_000000_p0",
        buy_count=1,
        sell_count=0,
        trade_count=0,
        signals_executed=executed,
        trade_details=[{"symbol": "000001", "date": date(2025, 5, 6), "action": "BUY", "shares": 100}],
        equity_curve={"2025-05-06": 1_000_000.0},
    )
    cfg = BacktestRunConfig(
        run_id=result.run_id,
        scheme_id=result.scheme_id,
        scheme_name=result.scheme_name,
        start_date=result.start_date,
        end_date=result.end_date,
        lookback_days=20,
        top_n=1,
        initial_capital=1_000_000,
        cost={"commission": 0.00025, "stamp_duty": 0.001, "transfer_fee": 0.00001, "slippage": 0.002},
        risk={"single_position_cap": 0.20, "total_position_cap": 0.90},
    )
    trades = trade_details_to_frame(result.trade_details, run_id=result.run_id, source="executed")
    run_dir = persist_backtest_run(
        result=result,
        config=cfg,
        trades=trades,
        signals_executed=trades,
        signals_raw=pd.DataFrame(),
        equity=pd.DataFrame([{"run_id": result.run_id, "date": "2025-05-06", "equity": 1_000_000}]),
        root=tmp_path,
    )

    assert (run_dir / "config.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "report.md").exists()
    trades_path = run_dir / "trades.parquet"
    if trades_path.exists():
        saved_trades = pd.read_parquet(trades_path)
    else:
        saved_trades = pd.read_csv(run_dir / "trades.csv")
    assert validate_trade_schema(saved_trades)["ok"] is True
    assert (run_dir / "signals_executed.parquet").exists() or (run_dir / "signals_executed.csv").exists()
    assert (run_dir / "equity.parquet").exists() or (run_dir / "equity.csv").exists()


def test_backtest_params_p0_cost_and_risk_defaults():
    """默认回测参数必须满足A股成本和仓位红线。"""
    params = BacktestParams(start_date=date(2025, 1, 1), end_date=date(2025, 2, 1))
    assert params.initial_capital == 1_000_000.0
    assert params.commission_rate == 0.00025
    assert params.stamp_duty == 0.001
    assert params.transfer_fee == 0.00001
    assert params.slippage >= 0.002
    assert params.position_pct <= 0.90


def test_make_run_id_uses_runtime_date_not_backtest_end_date():
    """run_id 必须表示运行时间，不能用回测结束交易日。"""
    run_id = make_run_id("value", date(1999, 1, 1))
    assert not run_id.startswith("19990101_")
    assert run_id.endswith("_value")


def test_liquidity_tiered_slippage_rate():
    """P1: 滑点必须按成交额分层，避免小票按蓝筹成本美化回测。"""
    assert get_liquidity_slippage_rate(600_000_000) == (0.002, "large_cap_gt_5e")
    assert get_liquidity_slippage_rate(100_000_000) == (0.005, "mid_cap_1e_5e")
    assert get_liquidity_slippage_rate(499_999_999) == (0.005, "mid_cap_1e_5e")
    assert get_liquidity_slippage_rate(99_999_999) == (0.010, "small_cap_lt_1e")
    assert get_liquidity_slippage_rate(0, default_rate=0.002) == (0.002, "unknown_default")


def test_trade_detail_contains_p0_cost_and_pnl_fields():
    """成交明细标准化后必须具备P0审计字段。"""
    rows = [{
        "symbol": "000001",
        "date": date(2025, 5, 6),
        "action": "SELL",
        "shares": 1000,
        "exec_price": 10.8,
        "commission": 5.0,
        "stamp_duty": 10.8,
        "transfer_fee": 0.108,
        "slippage": 21.6,
        "slippage_rate": 0.002,
        "liquidity_bucket": "large_cap_gt_5e",
        "turnover_amount": 600_000_000,
        "avg_cost": 10.0,
        "pnl": 762.492,
        "pnl_pct": 0.0762,
        "holding_days": 6,
    }]
    df = trade_details_to_frame(rows, run_id="run_x", source="executed")
    assert validate_trade_schema(df)["ok"] is True
    assert list(df.columns[:len(STANDARD_TRADE_COLUMNS)]) == STANDARD_TRADE_COLUMNS
    row = df.iloc[0]
    assert row["run_id"] == "run_x"
    assert row["exec_date"] == "2025-05-06"
    assert row["event_type"] == "SELL"
    assert row["amount"] == 10800.0
    assert row["slippage_rate"] == 0.002
    assert row["liquidity_bucket"] == "large_cap_gt_5e"
    assert row["turnover_amount"] == 600_000_000


def test_trade_detail_preserves_p1_liquidity_slippage_fields():
    """trades.parquet schema 必须保留滑点分层审计字段。"""
    rows = [{
        "symbol": "000777",
        "date": date(2025, 5, 7),
        "action": "BUY",
        "shares": 1000,
        "exec_price": 10.10,
        "amount": 10_100,
        "slippage": 100.0,
        "slippage_rate": 0.010,
        "liquidity_bucket": "small_cap_lt_1e",
        "turnover_amount": 80_000_000,
    }]
    df = trade_details_to_frame(rows, run_id="run_p1", source="executed")
    row = df.iloc[0]
    assert validate_trade_schema(df)["ok"] is True
    assert row["slippage_rate"] == pytest.approx(0.0100)
    assert row["liquidity_bucket"] == "small_cap_lt_1e"
    assert row["turnover_amount"] == 80_000_000


def test_summarize_liquidity_slippage_for_ui_audit():
    """历史记录页使用的滑点审计汇总必须给出分层统计。"""
    df = trade_details_to_frame([
        {
            "symbol": "000001", "date": date(2025, 5, 7), "action": "BUY",
            "exec_price": 10.0, "shares": 1000, "amount": 10_000,
            "slippage": 20.0, "slippage_rate": 0.002,
            "liquidity_bucket": "large_cap_gt_5e", "turnover_amount": 600_000_000,
        },
        {
            "symbol": "000777", "date": date(2025, 5, 7), "action": "BUY",
            "exec_price": 10.0, "shares": 1000, "amount": 10_000,
            "slippage": 100.0, "slippage_rate": 0.010,
            "liquidity_bucket": "small_cap_lt_1e", "turnover_amount": 80_000_000,
        },
    ], run_id="run_p1")
    summary = summarize_liquidity_slippage(df)
    assert summary["ok"] is True
    assert summary["rows"] == 2
    assert summary["total_amount"] == 20_000
    assert summary["total_slippage"] == 120
    assert summary["weighted_slippage_rate"] == pytest.approx(0.006)
    assert set(summary["buckets"]["liquidity_bucket"]) == {"large_cap_gt_5e", "small_cap_lt_1e"}


def test_summarize_liquidity_slippage_infers_amount_for_memory_ui_result():
    """回测页内存态 amount=0 时，滑点审计不能显示成交额/加权滑点率为0。"""
    df = pd.DataFrame([
        {
            "symbol": "002145",
            "exec_price": 4.58,
            "shares": 43800,
            "amount": 0.0,
            "slippage": 998.44,
            "slippage_rate": 0.005,
            "liquidity_bucket": "mid_cap_1e_5e",
            "turnover_amount": 261_522_122,
        }
    ])
    summary = summarize_liquidity_slippage(df)
    assert summary["total_amount"] > 0
    assert summary["total_slippage"] == pytest.approx(998.44)
    assert summary["weighted_slippage_rate"] == pytest.approx(0.005, rel=1e-3)


def test_summarize_liquidity_slippage_marks_legacy_audit_records():
    """历史记录缺少新版流动性字段时，只提示旧口径，不自动篡改历史数据。"""
    df = pd.DataFrame([
        {
            "symbol": "002145",
            "exec_price": 4.58,
            "shares": 43800,
            "amount": 200_604.0,
            "slippage": 401.21,
        }
    ])
    summary = summarize_liquidity_slippage(df)
    assert summary["ok"] is True
    assert summary["is_legacy_audit"] is True
    assert set(summary["missing_audit_columns"]) == {"slippage_rate", "liquidity_bucket", "turnover_amount"}
    assert summary["total_amount"] == pytest.approx(200_604.0)


def test_single_stock_signal_executes_next_day_open_and_records_signal_date(monkeypatch):
    """单股信号必须 T+1 开盘成交，并保留 signal_date/exec_date 审计链路。"""
    import backtest.scheme_backtest as sb

    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
    bars = pd.DataFrame({
        "symbol": ["000001"] * 3,
        "trade_date": dates,
        "open": [10.00, 10.50, 10.80],
        "high": [10.20, 10.80, 11.00],
        "low": [9.90, 10.40, 10.70],
        "close": [10.10, 10.70, 10.90],
        "volume": [1_000_000] * 3,
        "amount": [1_000_000_000] * 3,
    })
    monkeypatch.setattr(sb, "_fetch_ohlcv", lambda symbols, lookback_days: bars.copy())
    monkeypatch.setattr(
        sb,
        "evaluate_layered",
        lambda sym_bars, strategy_type="balanced": [
            TradePoint(date=dates[0].date(), action="BUY", reason="测试信号", confidence=1.0, price=10.10, rule_name="测试规则")
        ],
    )
    scheme = StrategyScheme(
        scheme_id="test_open_exec",
        name="T+1开盘成交测试",
        description="",
        factor_weights={},
        signal_rules=[],
        regime_fit=["*"],
        enable_market_timing=False,
        max_add_times=0,
        take_profit_atr_mult=100.0,
        stop_loss_atr_mult=100.0,
        trailing_atr_mult=100.0,
    )
    factor_df = bars[["symbol", "trade_date"]].copy()
    result = sb.SchemeBacktester().run(
        scheme, factor_df=factor_df, price_df=bars.copy(), factor_names=[],
        symbols=["000001"], lookback_days=3, initial_capital=1_000_000,
    )

    buy = next(t for t in result.trade_details if t["action"] == "BUY")
    assert buy["signal_date"] == dates[0].date()
    assert buy["exec_date"] == dates[1].date()
    assert buy["date"] == dates[1].date()
    assert buy["exec_price"] == pytest.approx(10.50 * 1.002)


def test_trailing_take_profit_requires_cost_adjusted_profit(monkeypatch):
    """trailing_stop 未进入扣成本盈利保护区时，不能触发 ATR跟踪止盈。"""
    import backtest.scheme_backtest as sb

    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"])
    bars = pd.DataFrame({
        "symbol": ["000001"] * 4,
        "trade_date": dates,
        "open": [10.00, 10.00, 10.40, 9.95],
        "high": [10.20, 10.20, 10.60, 9.98],
        "low": [9.80, 9.80, 10.00, 9.50],
        "close": [10.00, 10.00, 10.50, 9.70],
        "volume": [1_000_000] * 4,
        "amount": [1_000_000_000] * 4,
    })
    monkeypatch.setattr(sb, "_fetch_ohlcv", lambda symbols, lookback_days: bars.copy())
    monkeypatch.setattr(
        sb,
        "evaluate_layered",
        lambda sym_bars, strategy_type="balanced": [
            TradePoint(date=dates[0].date(), action="BUY", reason="测试买入", confidence=1.0, price=10.00, rule_name="测试规则")
        ],
    )
    scheme = StrategyScheme(
        scheme_id="test_trailing_guard",
        name="跟踪止盈条件测试",
        description="",
        factor_weights={},
        signal_rules=[],
        regime_fit=["*"],
        enable_market_timing=False,
        max_add_times=0,
        stop_loss_atr_mult=10.0,
        take_profit_atr_mult=100.0,
        trailing_atr_mult=1.5,
    )
    factor_df = bars[["symbol", "trade_date"]].copy()
    result = sb.SchemeBacktester().run(
        scheme, factor_df=factor_df, price_df=bars.copy(), factor_names=[],
        symbols=["000001"], lookback_days=4, initial_capital=1_000_000,
    )

    sell_reasons = [t["reason"] for t in result.trade_details if t["action"] == "SELL"]
    assert sell_reasons
    assert all("跟踪止盈" not in reason for reason in sell_reasons)
    assert sell_reasons[-1] == "末日清仓"


def test_list_and_load_backtest_runs(tmp_path):
    """历史回测记录必须可 list/load。"""
    result = SchemeBacktestResult(
        scheme_id="p0",
        scheme_name="P0测试",
        start_date="2025-05-01",
        end_date="2025-05-15",
        run_id="20250515_000001_p0",
        total_return=0.01,
        buy_count=1,
        sell_count=0,
        trade_count=0,
        trade_details=[{"symbol": "000001", "date": date(2025, 5, 6), "action": "BUY", "exec_price": 10, "shares": 100}],
        equity_curve={"2025-05-06": 1_000_000.0},
    )
    cfg = BacktestRunConfig(
        run_id=result.run_id,
        scheme_id=result.scheme_id,
        scheme_name=result.scheme_name,
        start_date=result.start_date,
        end_date=result.end_date,
        lookback_days=20,
        top_n=1,
        initial_capital=1_000_000,
    )
    trades = trade_details_to_frame(result.trade_details, run_id=result.run_id)
    persist_backtest_run(
        result=result,
        config=cfg,
        trades=trades,
        signals_executed=pd.DataFrame(),
        signals_raw=pd.DataFrame(),
        equity=pd.DataFrame([{"run_id": result.run_id, "date": "2025-05-06", "equity": 1_000_000}]),
        root=tmp_path,
    )

    runs = list_backtest_runs(root=tmp_path)
    assert result.run_id in runs["run_id"].tolist()
    loaded = load_backtest_run(result.run_id, root=tmp_path)
    assert loaded["config"]["scheme_name"] == "P0测试"
    assert validate_trade_schema(loaded["trades"])["ok"] is True
    assert len(loaded["equity"]) == 1

    trash_path = delete_backtest_run(result.run_id, root=tmp_path, trash=True)
    assert trash_path.exists()
    assert result.run_id not in list_backtest_runs(root=tmp_path)["run_id"].tolist()


def test_list_backtest_runs_sorted_by_created_time_not_run_id(tmp_path):
    """历史页必须优先展示最新保存记录，不能被 run_id 字符串顺序误导。"""
    older_dir = tmp_path / "20260612_225416_value"
    newer_dir = tmp_path / "20260612_130311_value"
    older_dir.mkdir()
    newer_dir.mkdir()
    for run_dir, run_id in [(older_dir, older_dir.name), (newer_dir, newer_dir.name)]:
        (run_dir / "config.json").write_text(
            '{"run_id":"%s","scheme_name":"低波价值","scheme_id":"value"}' % run_id,
            encoding="utf-8",
        )
        (run_dir / "metrics.json").write_text('{"consistency":{"ok":true}}', encoding="utf-8")

    old_ts = time.time() - 120
    new_ts = time.time()
    os.utime(older_dir, (old_ts, old_ts))
    os.utime(newer_dir, (new_ts, new_ts))

    runs = list_backtest_runs(root=tmp_path)
    assert runs.iloc[0]["run_id"] == newer_dir.name
    assert runs.iloc[1]["run_id"] == older_dir.name
