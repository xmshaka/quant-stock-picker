"""P0 回测记录与K线买卖点一致性测试。"""
from datetime import date
import os
import time

import pytest
import pandas as pd

from backtest.engine import BacktestEngine, BacktestParams, get_liquidity_slippage_rate
from backtest.records import (
    BacktestRunConfig,
    STANDARD_TRADE_COLUMNS,
    list_backtest_runs,
    load_backtest_run,
    delete_backtest_run,
    persist_backtest_run,
    summarize_exit_audit,
    validate_backtest_consistency,
    validate_trade_schema,
    make_run_id,
    scheme_audit_snapshot,
    summarize_liquidity_slippage,
    trade_details_to_frame,
    trade_points_to_frame,
)
from backtest.scheme_backtest import SchemeBacktestResult, classify_exit_reason
from dashboard.kline_events import trade_points_from_executed_frame
from signals.rules import TradePoint
from strategy.schemes import BUILTIN_SCHEMES, ExitConfig, StrategyScheme


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


def test_trade_detail_contains_p0_exit_audit_fields():
    """P0: 卖出成交明细必须具备机器可检索的退出审计字段。"""
    rows = [{
        "symbol": "000001",
        "date": date(2025, 5, 8),
        "action": "SELL",
        "reason": "跟踪止盈(最高11.20)",
        "rule_name": "ATR跟踪止盈",
        "exec_price": 10.8,
        "shares": 1000,
        "avg_cost": 10.0,
        "pnl": 760.0,
        "exit_type": "take_profit",
        "exit_subtype": "atr_trailing_profit",
        "trigger_price": 10.75,
        "projected_pnl": 760.0,
    }]
    df = trade_details_to_frame(rows, run_id="run_exit", source="executed")
    assert validate_trade_schema(df)["ok"] is True
    row = df.iloc[0]
    assert row["exit_type"] == "take_profit"
    assert row["exit_subtype"] == "atr_trailing_profit"
    assert row["trigger_price"] == pytest.approx(10.75)
    assert row["projected_pnl"] == pytest.approx(760.0)


def test_trade_detail_defaults_exit_audit_fields_for_legacy_rows():
    """旧记录缺字段时仅补空字段，不静默改写退出分类。"""
    df = trade_details_to_frame([
        {"symbol": "000001", "date": date(2025, 5, 8), "action": "SELL", "exec_price": 10.0, "shares": 100}
    ], run_id="legacy")
    row = df.iloc[0]
    assert row["exit_type"] == ""
    assert row["exit_subtype"] == ""
    assert row["trigger_price"] == pytest.approx(10.0)
    assert row["projected_pnl"] == pytest.approx(0.0)


def test_classify_exit_reason_maps_common_exit_types():
    assert classify_exit_reason("ATR止损", "止损(10.0)") == ("stop_loss", "atr_hard_stop")
    assert classify_exit_reason("ATR跟踪止盈", "跟踪止盈") == ("take_profit", "atr_trailing_profit")
    assert classify_exit_reason("ATR跟踪回撤止损", "跟踪止盈失效") == ("stop_loss", "atr_trailing_profit_failed")
    assert classify_exit_reason("信号卖出", "规则卖出") == ("signal_exit", "rule_signal")
    assert classify_exit_reason("末日清仓", "末日清仓") == ("final_liquidation", "end_of_backtest")


def test_summarize_exit_audit_for_history_page():
    """历史记录页退出审计必须汇总 exit_type 与触发价/预估盈亏。"""
    df = trade_details_to_frame([
        {
            "symbol": "000001", "date": date(2025, 5, 8), "action": "SELL",
            "exec_price": 10.8, "shares": 1000, "pnl": 760.0,
            "exit_type": "take_profit", "exit_subtype": "atr_trailing_profit",
            "trigger_price": 10.75, "projected_pnl": 760.0,
        },
        {
            "symbol": "000002", "date": date(2025, 5, 9), "action": "SELL",
            "exec_price": 9.8, "shares": 1000, "pnl": -260.0,
            "exit_type": "stop_loss", "exit_subtype": "atr_hard_stop",
            "trigger_price": 9.9, "projected_pnl": -260.0,
        },
        {
            "symbol": "000003", "date": date(2025, 5, 10), "action": "BUY",
            "exec_price": 11.0, "shares": 1000,
        },
    ], run_id="run_exit_summary")

    summary = summarize_exit_audit(df)

    assert summary["ok"] is True
    assert summary["sell_rows"] == 2
    assert summary["take_profit_rows"] == 1
    assert summary["stop_loss_rows"] == 1
    assert summary["total_pnl"] == pytest.approx(500.0)
    assert set(summary["summary"]["exit_type"]) == {"take_profit", "stop_loss"}
    assert {"退出类型", "退出子类"}.issubset(summary["details"].columns)


def test_summarize_exit_audit_marks_legacy_records():
    """旧记录缺退出审计字段时只提示旧口径，不反推篡改历史。"""
    df = pd.DataFrame([
        {"symbol": "000001", "date": "2025-05-08", "action": "SELL", "exec_price": 10.0, "shares": 1000, "pnl": 100.0}
    ])
    summary = summarize_exit_audit(df)
    assert summary["ok"] is True
    assert summary["is_legacy_exit_audit"] is True
    assert set(summary["missing_exit_columns"]) == {"exit_type", "exit_subtype", "trigger_price", "projected_pnl"}
    assert summary["sell_rows"] == 1
    assert summary["details"].iloc[0]["退出类型"] == "未记录"


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
    assert all("ATR跟踪止盈" != t.get("rule_name") for t in result.trade_details if t["action"] == "SELL")


def test_time_stop_exit_records_standard_audit_fields(monkeypatch):
    """P2: 时间止损必须走统一 SELL 审计字段，不等到末日清仓。"""
    import backtest.scheme_backtest as sb

    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"])
    bars = pd.DataFrame({
        "symbol": ["000001"] * 5,
        "trade_date": dates,
        "open": [10.00, 10.00, 10.01, 10.00, 10.00],
        "high": [10.10, 10.10, 10.10, 10.10, 10.10],
        "low": [9.90, 9.90, 9.90, 9.90, 9.90],
        "close": [10.00, 10.00, 10.00, 10.00, 10.00],
        "volume": [1_000_000] * 5,
        "amount": [1_000_000_000] * 5,
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
        scheme_id="test_time_stop",
        name="时间止损测试",
        description="",
        factor_weights={},
        signal_rules=[],
        regime_fit=["*"],
        enable_market_timing=False,
        max_add_times=0,
        stop_loss_atr_mult=100.0,
        take_profit_atr_mult=100.0,
        trailing_atr_mult=100.0,
        exit_config=ExitConfig(max_holding_days=10, time_stop_days=2, time_stop_min_profit_pct=0.02),
    )
    factor_df = bars[["symbol", "trade_date"]].copy()
    result = sb.SchemeBacktester().run(
        scheme, factor_df=factor_df, price_df=bars.copy(), factor_names=[],
        symbols=["000001"], lookback_days=5, initial_capital=1_000_000,
    )

    sells = [t for t in result.trade_details if t["action"] == "SELL"]
    assert sells
    sell = sells[0]
    assert sell["rule_name"] == "时间止损"
    assert sell["exit_type"] == "stop_loss"
    assert sell["exit_subtype"] == "time_stop"
    assert sell["holding_days"] >= 2
    assert sell["projected_pnl"] < 20_000  # 未达到2%收益目标


def test_time_stop_and_max_holding_use_trading_days_not_calendar_days(monkeypatch):
    """时间止损/最长持仓必须按交易日计数，周末自然日不能触发退出。"""
    import backtest.scheme_backtest as sb

    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"])
    bars = pd.DataFrame({
        "symbol": ["000001"] * len(dates),
        "trade_date": dates,
        "open": [10.00] * len(dates),
        "high": [10.10] * len(dates),
        "low": [9.90] * len(dates),
        "close": [10.00] * len(dates),
        "volume": [1_000_000] * len(dates),
        "amount": [1_000_000_000] * len(dates),
    })
    monkeypatch.setattr(sb, "_fetch_ohlcv", lambda symbols, lookback_days: bars.copy())
    monkeypatch.setattr(
        sb,
        "evaluate_layered",
        lambda sym_bars, strategy_type="balanced": [
            # 1/2信号，1/5买入执行；若按自然日，1/7 已是2天；按交易日，1/7 才是第2个持仓交易日。
            TradePoint(date=dates[0].date(), action="BUY", reason="测试买入", confidence=1.0, price=10.00, rule_name="测试规则")
        ],
    )
    scheme = StrategyScheme(
        scheme_id="test_trading_day_stop",
        name="交易日时间止损测试",
        description="",
        factor_weights={},
        signal_rules=[],
        regime_fit=["*"],
        enable_market_timing=False,
        max_add_times=0,
        stop_loss_atr_mult=100.0,
        take_profit_atr_mult=100.0,
        trailing_atr_mult=100.0,
        exit_config=ExitConfig(max_holding_days=2, time_stop_days=2, time_stop_min_profit_pct=0.02, failure_window_days=0),
    )

    result = sb.SchemeBacktester().run(
        scheme, factor_df=bars[["symbol", "trade_date"]].copy(), price_df=bars.copy(),
        factor_names=[], symbols=["000001"], lookback_days=4, initial_capital=1_000_000,
    )

    sells = [t for t in result.trade_details if t["action"] == "SELL"]
    assert sells
    assert sells[0]["date"] == dates[3].date()
    assert sells[0]["holding_days"] == 2
    assert sells[0]["rule_name"] == "最长持仓退出"


def test_trailing_drawdown_exit_preempts_time_stop_when_profit_guard_lost(monkeypatch):
    """跳空跌破跟踪线且扣成本后亏损时，应归因为跟踪回撤止损，而不是时间止损。"""
    import backtest.scheme_backtest as sb

    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06"])
    bars = pd.DataFrame({
        "symbol": ["000001"] * 4,
        "trade_date": dates,
        "open": [10.00, 10.00, 11.20, 9.90],
        "high": [10.10, 10.20, 11.30, 10.00],
        "low": [9.90, 9.90, 11.00, 9.60],
        "close": [10.00, 10.10, 11.10, 9.80],
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
        scheme_id="test_trailing_preempt_time_stop",
        name="跟踪回撤优先级测试",
        description="",
        factor_weights={},
        signal_rules=[],
        regime_fit=["*"],
        enable_market_timing=False,
        max_add_times=0,
        stop_loss_atr_mult=100.0,
        take_profit_atr_mult=100.0,
        trailing_atr_mult=1.0,
        exit_config=ExitConfig(max_holding_days=20, time_stop_days=2, time_stop_min_profit_pct=0.0, failure_window_days=0),
    )

    result = sb.SchemeBacktester().run(
        scheme, factor_df=bars[["symbol", "trade_date"]].copy(), price_df=bars.copy(),
        factor_names=[], symbols=["000001"], lookback_days=4, initial_capital=1_000_000,
    )

    sell = next(t for t in result.trade_details if t["action"] == "SELL")
    assert sell["rule_name"] == "ATR跟踪回撤止损"
    assert sell["exit_subtype"] == "atr_trailing_profit_failed"
    assert sell["projected_pnl"] < 0


def test_trailing_exit_requires_activation_range(monkeypatch):
    """未达到最高浮盈5%或1ATR激活区间时，即使跌破跟踪线也不触发跟踪退出。"""
    import backtest.scheme_backtest as sb

    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06"])
    bars = pd.DataFrame({
        "symbol": ["000001"] * 4,
        "trade_date": dates,
        "open": [10.00, 10.00, 10.20, 9.98],
        "high": [10.10, 10.20, 10.30, 10.00],
        "low": [9.90, 9.90, 10.00, 9.80],
        "close": [10.00, 10.10, 10.20, 9.90],
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
        scheme_id="test_trailing_activation_range",
        name="跟踪止盈激活区间测试",
        description="",
        factor_weights={},
        signal_rules=[],
        regime_fit=["*"],
        enable_market_timing=False,
        max_add_times=0,
        stop_loss_atr_mult=100.0,
        take_profit_atr_mult=100.0,
        trailing_atr_mult=1.0,
        exit_config=ExitConfig(
            max_holding_days=20,
            time_stop_days=20,
            failure_window_days=0,
            trailing_activation_pct=0.05,
            trailing_activation_atr_mult=10.0,
        ),
    )

    result = sb.SchemeBacktester().run(
        scheme, factor_df=bars[["symbol", "trade_date"]].copy(), price_df=bars.copy(),
        factor_names=[], symbols=["000001"], lookback_days=4, initial_capital=1_000_000,
    )

    sells = [t for t in result.trade_details if t["action"] == "SELL"]
    assert sells
    assert sells[0]["rule_name"] == "末日清仓"
    assert all(t.get("exit_subtype") != "atr_trailing_profit_failed" for t in sells)


def test_time_stop_can_be_disabled_by_exit_config(monkeypatch):
    """P2: 用户关闭时间止损后，不应再产生 time_stop 卖点。"""
    import backtest.scheme_backtest as sb

    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"])
    bars = pd.DataFrame({
        "symbol": ["000001"] * 5,
        "trade_date": dates,
        "open": [10.00, 10.00, 10.01, 10.00, 10.00],
        "high": [10.10] * 5,
        "low": [9.90] * 5,
        "close": [10.00] * 5,
        "volume": [1_000_000] * 5,
        "amount": [1_000_000_000] * 5,
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
        scheme_id="test_time_stop_disabled",
        name="禁用时间止损测试",
        description="",
        factor_weights={},
        signal_rules=[],
        regime_fit=["*"],
        enable_market_timing=False,
        max_add_times=0,
        stop_loss_atr_mult=100.0,
        take_profit_atr_mult=100.0,
        trailing_atr_mult=100.0,
        exit_config=ExitConfig(
            enable_time_stop=False,
            max_holding_days=10,
            time_stop_days=2,
            time_stop_min_profit_pct=0.02,
        ),
    )

    result = sb.SchemeBacktester().run(
        scheme,
        factor_df=bars[["symbol", "trade_date"]].copy(),
        price_df=bars.copy(),
        factor_names=[],
        symbols=["000001"],
        lookback_days=5,
        initial_capital=1_000_000,
    )

    assert all(t.get("exit_subtype") != "time_stop" for t in result.trade_details if t["action"] == "SELL")


def test_trend_momentum_failure_exit_when_breaks_ma20(monkeypatch):
    """P2: 强势追涨买入后短期跌破 MA20，应归因为策略失败退出。"""
    import backtest.scheme_backtest as sb

    dates = pd.date_range("2026-01-01", periods=45, freq="D")
    closes = [10 + i * 0.05 for i in range(45)]
    opens = closes.copy()
    opens[42] = 9.50  # T+1 买入
    opens[43] = 9.20  # 持仓1日后跌破前序MA20，触发动量失效
    bars = pd.DataFrame({
        "symbol": ["000001"] * 45,
        "trade_date": dates,
        "open": opens,
        "high": [c * 1.01 for c in closes],
        "low": [min(o, c) * 0.99 for o, c in zip(opens, closes)],
        "close": closes,
        "volume": [1_000_000] * 45,
        "amount": [1_000_000_000] * 45,
    })
    monkeypatch.setattr(sb, "_fetch_ohlcv", lambda symbols, lookback_days: bars.copy())
    monkeypatch.setattr(
        sb,
        "evaluate_layered",
        lambda sym_bars, strategy_type="trend_momentum": [
            TradePoint(date=dates[41].date(), action="BUY", reason="追涨买入", confidence=1.0, price=closes[41], rule_name="测试规则")
        ],
    )
    scheme = StrategyScheme(
        scheme_id="trend_momentum",
        name="强势追涨",
        description="",
        factor_weights={},
        signal_rules=[],
        regime_fit=["*"],
        enable_market_timing=False,
        max_add_times=0,
        stop_loss_atr_mult=100.0,
        take_profit_atr_mult=100.0,
        trailing_atr_mult=100.0,
        exit_config=ExitConfig(max_holding_days=10, time_stop_days=9, time_stop_min_profit_pct=0.02, failure_window_days=3),
    )
    factor_df = bars[["symbol", "trade_date"]].copy()
    result = sb.SchemeBacktester().run(
        scheme, factor_df=factor_df, price_df=bars.copy(), factor_names=[],
        symbols=["000001"], lookback_days=45, initial_capital=1_000_000,
    )

    sell = next(t for t in result.trade_details if t["action"] == "SELL")
    assert sell["rule_name"] == "动量失效退出"
    assert sell["exit_type"] == "strategy_failure"
    assert sell["exit_subtype"] == "trend_momentum_failed"
    assert "MA20" in sell["reason"]


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


def test_persist_load_roundtrip_preserves_full_pool_audit_schema(tmp_path):
    """P0: 全池模式保存/读取后 trades 与 signals_executed 审计字段不能丢。"""
    executed = {
        "000777": [
            TradePoint(
                date=date(2025, 1, 2), action="BUY", reason="调仓买入", confidence=1.0,
                price=10.05, exec_price=10.05, shares=1000,
                signal_date=date(2025, 1, 1), exec_date=date(2025, 1, 2),
            ),
            TradePoint(
                date=date(2025, 1, 4), action="SELL", reason="调仓卖出", confidence=1.0,
                price=10.20, exec_price=10.20, shares=1000, pnl=120.0,
                signal_date=date(2025, 1, 3), exec_date=date(2025, 1, 4),
                exit_type="signal_exit", exit_subtype="rule_signal",
                trigger_price=10.20, projected_pnl=120.0,
            ),
        ]
    }
    trade_details = [
        {
            "symbol": "000777", "date": date(2025, 1, 2), "exec_date": date(2025, 1, 2),
            "signal_date": date(2025, 1, 1), "action": "BUY", "exec_price": 10.05,
            "shares": 1000, "reason": "调仓买入", "rule_name": "Backtrader调仓",
        },
        {
            "symbol": "000777", "date": date(2025, 1, 4), "exec_date": date(2025, 1, 4),
            "signal_date": date(2025, 1, 3), "action": "SELL", "exec_price": 10.20,
            "shares": 1000, "pnl": 120.0, "reason": "调仓卖出", "rule_name": "Backtrader调仓",
            "exit_type": "signal_exit", "exit_subtype": "rule_signal",
            "trigger_price": 10.20, "projected_pnl": 120.0,
        },
    ]
    result = SchemeBacktestResult(
        scheme_id="full_pool",
        scheme_name="全池P0",
        start_date="2025-01-01",
        end_date="2025-01-05",
        run_id="20250105_000000_full_pool",
        buy_count=1,
        sell_count=1,
        trade_count=1,
        signals_executed=executed,
        trade_details=trade_details,
        equity_curve={"2025-01-05": 1_000_120.0},
    )
    cfg = BacktestRunConfig(
        run_id=result.run_id,
        scheme_id=result.scheme_id,
        scheme_name=result.scheme_name,
        start_date=result.start_date,
        end_date=result.end_date,
        lookback_days=20,
        top_n=1,
        initial_capital=100_000,
        pool_mode="full_pool",
    )
    trades = trade_details_to_frame(result.trade_details, run_id=result.run_id, source="executed")
    signals_executed = trade_points_to_frame(result.signals_executed, source="executed")
    persist_backtest_run(
        result=result,
        config=cfg,
        trades=trades,
        signals_executed=signals_executed,
        signals_raw=pd.DataFrame(),
        equity=pd.DataFrame([{"run_id": result.run_id, "date": "2025-01-05", "equity": 1_000_120.0}]),
        root=tmp_path,
    )

    loaded = load_backtest_run(result.run_id, root=tmp_path)
    assert validate_trade_schema(loaded["trades"])["ok"] is True
    sell = loaded["trades"][loaded["trades"]["action"] == "SELL"].iloc[0]
    assert sell["signal_date"] == "2025-01-03"
    assert sell["exec_date"] == "2025-01-04"
    assert sell["exit_type"] == "signal_exit"
    assert sell["exit_subtype"] == "rule_signal"
    assert sell["trigger_price"] == pytest.approx(10.20)
    assert sell["projected_pnl"] == pytest.approx(120.0)

    sx = loaded["signals_executed"]
    sx_sell = sx[sx["action"] == "SELL"].iloc[0]
    assert sx_sell["signal_date"] == "2025-01-03"
    assert sx_sell["exec_date"] == "2025-01-04"
    assert sx_sell["exit_type"] == "signal_exit"
    assert sx_sell["exit_subtype"] == "rule_signal"


def test_full_pool_persisted_run_replays_kline_from_exec_date(tmp_path):
    """P3: 全池执行→落盘→历史页K线事件转换必须闭环使用 exec_date。"""
    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"])
    bars = pd.DataFrame({
        "open": [10.0, 11.0, 12.0, 13.0, 14.0],
        "high": [10.5, 11.5, 12.5, 13.5, 14.5],
        "low": [9.5, 10.5, 11.5, 12.5, 13.5],
        "close": [10.2, 11.2, 12.2, 13.2, 14.2],
        "volume": [200_000] * 5,
        "amount": [200_000_000] * 5,
    }, index=dates)
    engine = BacktestEngine(BacktestParams(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 5),
        initial_capital=100_000,
        position_pct=0.90,
        max_stocks=1,
    ))
    engine.add_data("000777", bars)
    engine.add_signals({date(2025, 1, 1): ["000777"], date(2025, 1, 3): []})

    engine_result = engine.run(verbose=False)
    result = SchemeBacktestResult(
        scheme_id="full_pool",
        scheme_name="全池P3",
        start_date="2025-01-01",
        end_date="2025-01-05",
        run_id="20250105_000001_full_pool",
        buy_count=engine_result["buy_count"],
        sell_count=engine_result["sell_count"],
        trade_count=engine_result["trade_count"],
        signals_executed=engine_result["executed_points"],
        trade_details=engine_result["trade_details"],
        equity_curve=engine_result["equity_curve"],
        final_value=engine_result["final_value"],
        total_return=engine_result["total_return"],
        annual_return=engine_result["annual_return"],
        sharpe_ratio=engine_result["sharpe_ratio"],
        max_drawdown=engine_result["max_drawdown"],
        win_rate=engine_result["win_rate"],
    )
    cfg = BacktestRunConfig(
        run_id=result.run_id,
        scheme_id=result.scheme_id,
        scheme_name=result.scheme_name,
        start_date=result.start_date,
        end_date=result.end_date,
        lookback_days=20,
        top_n=1,
        initial_capital=100_000,
        pool_mode="full_pool",
    )
    persist_backtest_run(
        result=result,
        config=cfg,
        trades=trade_details_to_frame(result.trade_details, run_id=result.run_id, source="executed"),
        signals_executed=trade_points_to_frame(result.signals_executed, source="executed"),
        signals_raw=pd.DataFrame(),
        equity=pd.DataFrame([{"run_id": result.run_id, "date": k, "equity": v} for k, v in result.equity_curve.items()]),
        root=tmp_path,
    )

    loaded = load_backtest_run(result.run_id, root=tmp_path)
    sx = loaded["signals_executed"].sort_values(["symbol", "action"]).reset_index(drop=True)
    buy = sx[sx["action"] == "BUY"].iloc[0]
    sell = sx[sx["action"] == "SELL"].iloc[0]
    assert buy["signal_date"] == "2025-01-01"
    assert buy["exec_date"] == "2025-01-02"
    assert sell["signal_date"] == "2025-01-03"
    assert sell["exec_date"] == "2025-01-04"
    assert sell["exit_type"] == "signal_exit"
    assert sell["exit_subtype"] == "rule_signal"

    points = trade_points_from_executed_frame(loaded["signals_executed"])
    by_action = {p.action: p for p in points}
    assert by_action["BUY"].date == date(2025, 1, 2)
    assert by_action["BUY"].signal_date == date(2025, 1, 1)
    assert by_action["SELL"].date == date(2025, 1, 4)
    assert by_action["SELL"].signal_date == date(2025, 1, 3)


def test_persist_backtest_config_preserves_strategy_resonance_snapshot(tmp_path):
    """P1: config.json/report.md 必须保存策略专属共振配置，便于历史回测审计。"""
    scheme = BUILTIN_SCHEMES["breakout"]
    snapshot = scheme_audit_snapshot(scheme)
    result = SchemeBacktestResult(
        scheme_id=scheme.scheme_id,
        scheme_name=scheme.name,
        start_date="2025-01-01",
        end_date="2025-01-20",
        run_id="20250120_000000_breakout",
        buy_count=0,
        sell_count=0,
        trade_count=0,
        trade_details=[],
        equity_curve={"2025-01-20": 1_000_000.0},
    )
    cfg = BacktestRunConfig(
        run_id=result.run_id,
        scheme_id=scheme.scheme_id,
        scheme_name=scheme.name,
        start_date=result.start_date,
        end_date=result.end_date,
        lookback_days=20,
        top_n=3,
        initial_capital=1_000_000,
        scheme_config=snapshot["scheme_config"],
        resonance_config=snapshot["resonance_config"],
    )

    run_dir = persist_backtest_run(
        result=result,
        config=cfg,
        trades=pd.DataFrame(),
        signals_executed=pd.DataFrame(),
        signals_raw=pd.DataFrame(),
        equity=pd.DataFrame([{"run_id": result.run_id, "date": "2025-01-20", "equity": 1_000_000}]),
        root=tmp_path,
    )
    loaded = load_backtest_run(result.run_id, root=tmp_path)

    assert loaded["config"]["resonance_config"]["min_confirmations"] == 3
    assert "break_platform" in loaded["config"]["resonance_config"]["buy_conditions"]
    assert loaded["config"]["scheme_config"]["scheme_id"] == "breakout"
    assert loaded["config"]["scheme_config"]["exit_config"]["failure_window_days"] == 2
    assert "## 策略共振配置" in (run_dir / "report.md").read_text(encoding="utf-8")


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
