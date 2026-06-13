"""P1 Backtrader 分层滑点撮合测试。"""
from datetime import date

import pandas as pd
import pytest

from backtest.engine import BacktestEngine, BacktestParams, estimate_turnover_amount


def _run_one_buy_with_amount(turnover_amount: float):
    """构造单票固定价格K线；Jan1信号，Jan3实际撮合。"""
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
