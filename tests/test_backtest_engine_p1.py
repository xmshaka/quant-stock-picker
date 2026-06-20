"""P1 Backtrader 分层滑点撮合测试。"""
from datetime import date

import pandas as pd
import pytest

from backtest.engine import BacktestEngine, BacktestParams, estimate_turnover_amount
from backtest.records import STANDARD_TRADE_COLUMNS, trade_details_to_frame, validate_trade_schema
from strategy.schemes import ExitConfig


def _run_one_buy_with_amount(turnover_amount: float):
    """构造单票固定价格K线；Jan1信号，Jan2开盘实际撮合。"""
    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    bars = pd.DataFrame({
        "open": [10.0, 10.0, 10.0],
        "high": [11.0, 11.0, 11.0],
        "low": [9.0, 9.0, 9.0],
        "close": [10.0, 10.0, 10.0],
        "volume": [int(turnover_amount / 10.0)] * 3,
        "amount": [turnover_amount] * 3,
    }, index=dates)
    engine = BacktestEngine(BacktestParams(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 3),
        initial_capital=100_000,
        position_pct=0.90,
        max_stocks=1,
    ))
    engine.add_data("000777", bars)
    engine.add_signals({date(2025, 1, 1): ["000777"]})
    result = engine.run(verbose=False)
    assert len(result["trade_details"]) == 1
    return result["trade_details"][0]


def test_backtrader_executes_signal_on_next_day_open_not_t2():
    """P3: 全池路径必须是 T日信号、T+1开盘成交，不能再延后一根bar。"""
    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    bars = pd.DataFrame({
        "open": [10.0, 11.0, 13.0],
        "high": [10.5, 11.5, 13.5],
        "low": [9.5, 10.5, 12.5],
        "close": [10.2, 12.0, 14.0],
        "volume": [200_000] * 3,
        "amount": [200_000_000] * 3,
    }, index=dates)
    engine = BacktestEngine(BacktestParams(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 3),
        initial_capital=100_000,
        position_pct=0.90,
        max_stocks=1,
    ))
    engine.add_data("000777", bars)
    engine.add_signals({date(2025, 1, 1): ["000777"]})

    result = engine.run(verbose=False)
    trade = result["trade_details"][0]

    assert trade["signal_date"] == date(2025, 1, 1)
    assert trade["exec_date"] == date(2025, 1, 2)
    assert trade["exec_price"] == pytest.approx(11.0 * 1.005)  # 2亿成交额，中盘滑点0.5%


@pytest.mark.parametrize(
    "turnover_amount, expected_rate, expected_bucket, expected_exec_price",
    [
        (600_000_000, 0.002, "large_cap_gt_5e", 10.02),
        (200_000_000, 0.005, "mid_cap_1e_5e", 10.05),
        (80_000_000, 0.010, "small_cap_lt_1e", 10.10),
    ],
)
def test_backtrader_execution_price_uses_liquidity_tiered_slippage(
    turnover_amount, expected_rate, expected_bucket, expected_exec_price,
):
    """P1.3: Backtrader 实际成交价必须和分层滑点审计字段一致。"""
    trade = _run_one_buy_with_amount(turnover_amount)
    assert trade["exec_price"] == pytest.approx(expected_exec_price, rel=1e-8)
    assert trade["price"] == pytest.approx(expected_exec_price, rel=1e-8)
    assert trade["slippage_rate"] == pytest.approx(expected_rate)
    assert trade["liquidity_bucket"] == expected_bucket
    assert trade["turnover_amount"] == pytest.approx(turnover_amount)
    assert trade["slippage"] == pytest.approx(trade["amount"] * expected_rate)


def test_turnover_amount_fallback_uses_lot_volume_when_amount_is_zero():
    """腾讯日K amount=0 时必须用 volume(手) * close * 100 估算市场成交额。"""
    assert estimate_turnover_amount(amount=0, volume=603_975, close=4.27) == pytest.approx(257_897_325)
    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    bars = pd.DataFrame({
        "open": [10.0, 10.0, 10.0],
        "high": [11.0, 11.0, 11.0],
        "low": [9.0, 9.0, 9.0],
        "close": [10.0, 10.0, 10.0],
        "volume": [200_000] * 3,  # 手，估算成交额=2亿
        "amount": [0.0] * 3,
    }, index=dates)
    engine = BacktestEngine(BacktestParams(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 3),
        initial_capital=100_000,
        position_pct=0.90,
        max_stocks=1,
    ))
    engine.add_data("000777", bars)
    engine.add_signals({date(2025, 1, 1): ["000777"]})
    result = engine.run(verbose=False)
    trade = result["trade_details"][0]
    assert trade["turnover_amount"] == pytest.approx(200_000_000)
    assert trade["slippage_rate"] == pytest.approx(0.005)
    assert trade["liquidity_bucket"] == "mid_cap_1e_5e"
    assert trade["exec_price"] == pytest.approx(10.05)


def test_backtrader_full_pool_trade_schema_matches_single_stock_audit_fields():
    """P0: 全池 Backtrader 路径必须和单股路径保留同一套执行审计字段。"""
    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"])
    bars = pd.DataFrame({
        "open": [10.0, 10.0, 10.0, 10.0, 10.0],
        "high": [11.0, 11.0, 11.0, 11.0, 11.0],
        "low": [9.0, 9.0, 9.0, 9.0, 9.0],
        "close": [10.0, 10.0, 10.0, 10.0, 10.0],
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
    # Jan1 收盘信号买入，Jan3 收盘信号清空目标；均应保留 signal_date/exec_date。
    engine.add_signals({date(2025, 1, 1): ["000777"], date(2025, 1, 3): []})

    result = engine.run(verbose=False)
    trades = result["trade_details"]

    assert [t["action"] for t in trades] == ["BUY", "SELL"]
    buy, sell = trades
    assert buy["signal_date"] == date(2025, 1, 1)
    assert buy["exec_date"] == buy["date"]
    assert sell["signal_date"] == date(2025, 1, 3)
    assert sell["exec_date"] == sell["date"]
    assert sell["exit_type"] == "signal_exit"
    assert sell["exit_subtype"] == "rule_signal"
    assert sell["trigger_price"] == pytest.approx(sell["exec_price"])
    assert sell["projected_pnl"] == pytest.approx(sell["pnl"])

    normalized = trade_details_to_frame(trades, run_id="full_pool")
    assert validate_trade_schema(normalized)["ok"] is True
    assert list(normalized.columns[:len(STANDARD_TRADE_COLUMNS)]) == STANDARD_TRADE_COLUMNS

    point = result["executed_points"]["000777"][0]
    assert getattr(point, "signal_date") == date(2025, 1, 1)
    assert getattr(point, "exec_date") == point.date


def test_backtrader_full_pool_time_stop_uses_p2_exit_audit_fields():
    """P3: 全池路径也要接入 P2 时间止损审计字段。"""
    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06"])
    bars = pd.DataFrame({
        "open": [10.0] * 6,
        "high": [10.1] * 6,
        "low": [9.9] * 6,
        "close": [10.0] * 6,
        "volume": [200_000] * 6,
        "amount": [200_000_000] * 6,
    }, index=dates)
    engine = BacktestEngine(BacktestParams(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 6),
        initial_capital=100_000,
        position_pct=0.90,
        max_stocks=1,
        strategy_id="trend_momentum",
        exit_config=ExitConfig(max_holding_days=10, time_stop_days=1, time_stop_min_profit_pct=0.02, failure_window_days=0),
    ))
    engine.add_data("000777", bars)
    engine.add_signals({date(2025, 1, 1): ["000777"], date(2025, 1, 3): ["000777"]})

    result = engine.run(verbose=False)
    sells = [t for t in result["trade_details"] if t["action"] == "SELL"]

    assert sells
    sell = sells[0]
    assert sell["exit_type"] == "stop_loss"
    assert sell["exit_subtype"] == "time_stop"
    assert sell["rule_name"] == "时间止损"
    assert sell["trigger_price"] > 0
    assert sell["projected_pnl"] == pytest.approx(sell["pnl"])


def test_backtrader_time_stop_uses_trading_days_not_calendar_days():
    """全池路径时间止损/最长持仓按交易日计数，周末不应提前触发。"""
    dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"])
    bars = pd.DataFrame({
        "open": [10.0] * 4,
        "high": [10.1] * 4,
        "low": [9.9] * 4,
        "close": [10.0] * 4,
        "volume": [200_000] * 4,
        "amount": [200_000_000] * 4,
    }, index=dates)
    engine = BacktestEngine(BacktestParams(
        start_date=date(2025, 1, 2),
        end_date=date(2025, 1, 7),
        initial_capital=100_000,
        position_pct=0.90,
        max_stocks=1,
        strategy_id="balanced",
        exit_config=ExitConfig(max_holding_days=2, time_stop_days=2, time_stop_min_profit_pct=0.02, failure_window_days=0),
    ))
    engine.add_data("000777", bars)
    # 1/2信号→1/3买入；1/3信号在1/6处理时自然日=3但交易日=1，不能提前卖；1/6信号在1/7处理时交易日=2才卖。
    engine.add_signals({date(2025, 1, 2): ["000777"], date(2025, 1, 3): ["000777"], date(2025, 1, 6): ["000777"]})

    result = engine.run(verbose=False)
    sells = [t for t in result["trade_details"] if t["action"] == "SELL"]

    assert sells
    assert sells[0]["exec_date"] == date(2025, 1, 7)
    assert sells[0]["holding_days"] == 2
    assert sells[0]["exit_subtype"] in {"time_stop", "max_holding_days"}


def test_backtrader_trailing_exit_requires_activation_range():
    """全池路径跟踪退出也必须先满足激活区间。"""
    dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"])
    bars = pd.DataFrame({
        "open": [10.0, 10.0, 10.20, 9.98],
        "high": [10.1, 10.2, 10.30, 10.0],
        "low": [9.9, 9.9, 10.00, 9.8],
        "close": [10.0, 10.1, 10.20, 9.9],
        "volume": [200_000] * 4,
        "amount": [200_000_000] * 4,
    }, index=dates)
    engine = BacktestEngine(BacktestParams(
        start_date=date(2025, 1, 2),
        end_date=date(2025, 1, 7),
        initial_capital=100_000,
        position_pct=0.90,
        max_stocks=1,
        strategy_id="balanced",
        trailing_atr_mult=1.0,
        exit_config=ExitConfig(
            max_holding_days=20,
            time_stop_days=20,
            failure_window_days=0,
            trailing_activation_pct=0.05,
            trailing_activation_atr_mult=10.0,
        ),
    ))
    engine.add_data("000777", bars)
    engine.add_signals({date(2025, 1, 2): ["000777"], date(2025, 1, 3): ["000777"], date(2025, 1, 6): ["000777"]})

    result = engine.run(verbose=False)
    sells = [t for t in result["trade_details"] if t["action"] == "SELL"]

    assert all(sell.get("exit_subtype") not in {"atr_trailing_profit", "atr_trailing_profit_failed"} for sell in sells)


def test_backtrader_full_pool_time_stop_respects_exit_switch():
    """P3: 全池路径也必须尊重用户关闭时间止损的配置。"""
    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06"])
    bars = pd.DataFrame({
        "open": [10.0] * 6,
        "high": [10.1] * 6,
        "low": [9.9] * 6,
        "close": [10.0] * 6,
        "volume": [200_000] * 6,
        "amount": [200_000_000] * 6,
    }, index=dates)
    engine = BacktestEngine(BacktestParams(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 6),
        initial_capital=100_000,
        position_pct=0.90,
        max_stocks=1,
        strategy_id="trend_momentum",
        exit_config=ExitConfig(
            enable_time_stop=False,
            max_holding_days=10,
            time_stop_days=1,
            time_stop_min_profit_pct=0.02,
            failure_window_days=0,
        ),
    ))
    engine.add_data("000777", bars)
    engine.add_signals({date(2025, 1, 1): ["000777"], date(2025, 1, 3): ["000777"]})

    result = engine.run(verbose=False)
    sells = [t for t in result["trade_details"] if t["action"] == "SELL"]

    assert all(sell.get("exit_subtype") != "time_stop" for sell in sells)


def test_backtrader_full_pool_market_defense_exit_uses_audit_fields():
    """P3: 全池路径遇到大盘防御分数应输出 market_exit 审计。"""
    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06"])
    bars = pd.DataFrame({
        "open": [10.0] * 6,
        "high": [10.2] * 6,
        "low": [9.8] * 6,
        "close": [10.0] * 6,
        "volume": [200_000] * 6,
        "amount": [200_000_000] * 6,
    }, index=dates)
    engine = BacktestEngine(BacktestParams(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 6),
        initial_capital=100_000,
        position_pct=0.90,
        max_stocks=1,
        strategy_id="pullback",
        exit_config=ExitConfig(max_holding_days=10, time_stop_days=9, market_defense_score=20),
        market_scores={date(2025, 1, 4): 10.0},
    ))
    engine.add_data("000777", bars)
    engine.add_signals({date(2025, 1, 1): ["000777"], date(2025, 1, 3): ["000777"]})

    result = engine.run(verbose=False)
    sell = next(t for t in result["trade_details"] if t["action"] == "SELL")

    assert sell["exit_type"] == "market_exit"
    assert sell["exit_subtype"] == "market_defense"
    assert sell["rule_name"] == "大盘防御减仓"
    assert "大盘防御减仓" in sell["reason"]
