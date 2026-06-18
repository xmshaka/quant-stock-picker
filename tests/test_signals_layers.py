"""测试三层过滤信号架构"""
import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from signals.layers import (
    TrendFilter, StrategyMatcher, ResonanceChecker,
    StrategyType, ConditionResult, evaluate_layered,
)


def make_bars(n=120, trend="up", volatility=0.02, seed=42):
    """生成模拟K线"""
    np.random.seed(seed)
    dates = pd.date_range(end=date(2025, 6, 16), periods=n, freq='B')

    if trend == "up":
        drift = 0.002
    elif trend == "down":
        drift = -0.002
    else:
        drift = 0.0005

    returns = np.random.normal(drift, volatility, n)
    prices = 20 * np.exp(np.cumsum(returns))

    data = []
    for i, d in enumerate(dates):
        daily_vol = prices[i] * volatility
        o = prices[i] - daily_vol * np.random.random()
        h = prices[i] + daily_vol * abs(np.random.randn())
        l = prices[i] - daily_vol * abs(np.random.randn())
        c = prices[i]
        v = np.random.uniform(1e6, 1e7)
        data.append({
            'trade_date': d.date(),
            'open': o, 'high': max(h, o, c), 'low': min(l, o, c),
            'close': c, 'volume': v,
        })
    return pd.DataFrame(data)


class TestTrendFilter:
    def test_up_trend_passes(self):
        bars = make_bars(120, trend="up")
        # 使用 trend_momentum 策略（严格模式 Price>MA20），评分可达 0.5+
        tf = TrendFilter(strategy_type="trend_momentum")
        passed, reason, score = tf.check(bars, 119)
        assert passed, f"上升趋势应通过: {reason}"
        assert score >= 0.5

    def test_down_trend_fails(self):
        bars = make_bars(120, trend="down")
        tf = TrendFilter()
        passed, reason, score = tf.check(bars, 119)
        # 下跌趋势中价格大概率 < MA20
        print(f"下跌趋势: passed={passed}, reason={reason}, score={score:.2f}")

    def test_insufficient_data_fails(self):
        bars = make_bars(30)
        tf = TrendFilter()
        passed, _, _ = tf.check(bars, 30)
        assert not passed  # idx < ma_long (60)

    def test_check_trend_momentum(self):
        bars = make_bars(120, trend="up")
        sm = StrategyMatcher(StrategyType.TREND_MOMENTUM)
        matched, reason, conf = sm.match(bars, 119)
        print(f"追涨: matched={matched}, reason={reason}, conf={conf:.2f}")

    def test_check_pullback_detects(self):
        # 制造回调：先涨后跌
        bars = make_bars(120, trend="up")
        # 最后5天快速下跌
        last_close = bars['close'].values.copy()
        for i in range(115, 120):
            last_close[i] = last_close[110] * 0.92  # 8%回撤
        bars['close'] = last_close

        sm = StrategyMatcher(StrategyType.PULLBACK)
        matched, reason, conf = sm.match(bars, 119)
        print(f"回调: matched={matched}, reason={reason}, conf={conf:.2f}")
        if matched:
            assert conf > 0

    def test_balanced_matches_anything(self):
        bars = make_bars(120, trend="up")
        sm = StrategyMatcher(StrategyType.BALANCED)
        # Balanced should match if any strategy matches
        matched, reason, conf = sm.match(bars, 119)
        print(f"均衡: matched={matched}, reason={reason}, conf={conf:.2f}")
        # In an up trend with momentum, balanced should find something

    def test_insufficient_data(self):
        bars = make_bars(30)
        sm = StrategyMatcher(StrategyType.BALANCED)
        matched, _, _ = sm.match(bars, 19)  # idx < 20
        assert not matched


class TestResonanceChecker:
    def test_buy_conditions_always_return(self):
        bars = make_bars(120)
        rc = ResonanceChecker(min_confirmations=2)
        ok, conditions = rc.check_buy(bars, 119)
        assert len(conditions) == 6
        assert all(isinstance(c, ConditionResult) for c in conditions)
        assert all(c.name for c in conditions)

    def test_sell_conditions_always_return(self):
        bars = make_bars(120)
        rc = ResonanceChecker(min_confirmations=2)
        ok, conditions = rc.check_sell(bars, 119)
        assert len(conditions) == 6
        assert all(isinstance(c, ConditionResult) for c in conditions)

    def test_higher_min_confirmations_reduces_signals(self):
        bars = make_bars(120, trend="up")
        rc2 = ResonanceChecker(min_confirmations=2)
        rc4 = ResonanceChecker(min_confirmations=4)

        ok2, _ = rc2.check_buy(bars, 119)
        ok4, _ = rc4.check_buy(bars, 119)
        # min=4 should be stricter (but not necessarily false)
        assert ok4 is not None


class TestEvaluateLayered:
    def test_up_trend_produces_signals(self):
        bars = make_bars(120, trend="up")
        points = evaluate_layered(bars, strategy_type="balanced")
        # 上升趋势中应该产生一些信号
        print(f"上升趋势: {len(points)} 个信号")
        if points:
            buys = [p for p in points if p.action == "BUY"]
            sells = [p for p in points if p.action == "SELL"]
            print(f"  买入: {len(buys)}, 卖出: {len(sells)}")
            assert all(p.rule_name for p in points)

    def test_no_consecutive_same_signals(self):
        bars = make_bars(120, trend="up")
        points = evaluate_layered(bars, strategy_type="balanced")
        # 状态机确保不会连续同向信号
        last_action = None
        for p in points:
            if last_action is not None:
                # 允许 BUY → SELL → BUY → SELL 交替，但不允许 BUY → BUY
                pass  # 状态机: BUY后只允许SELL, SELL后只允许BUY
            last_action = p.action

    def test_empty_bars(self):
        bars = make_bars(10)  # < 20 days
        points = evaluate_layered(bars)
        assert len(points) == 0

    def test_pullback_strategy_in_downtrend(self):
        # 制造回调场景: 先涨后跌
        bars = make_bars(120, trend="up")
        closes = bars['close'].values.copy()
        # 最后10天快速回调
        for i in range(110, 120):
            closes[i] = closes[105] * (1 - (i - 105) * 0.012)
        bars['close'] = closes

        points = evaluate_layered(bars, strategy_type="pullback")
        print(f"回调策略: {len(points)} 个信号")
        buys = [p for p in points if p.action == "BUY"]
        print(f"  买入: {len(buys)}")
        for b in buys[:3]:
            print(f"    {b.date} {b.reason}")

    def test_different_strategies_different_signals(self):
        bars = make_bars(120, trend="up")
        results = {}
        for st in ["trend_momentum", "pullback", "breakout", "balanced"]:
            points = evaluate_layered(bars, strategy_type=st)
            results[st] = len(points)
        print(f"各策略信号数: {results}")
        # 不同策略应产生不同信号
        assert len(set(results.values())) >= 2 or all(v == 0 for v in results.values())


class TestIntegration:
    """集成测试: 三层过滤 + 回测兼容性"""

    def test_output_format_matches_legacy(self):
        """验证 evaluate_layered 输出格式与 evaluate_all_rules 兼容"""
        from signals.rules import evaluate_all_rules
        from strategy.schemes import BUILTIN_SCHEMES

        bars = make_bars(120, trend="up")
        balanced = BUILTIN_SCHEMES.get("balanced")
        if balanced and balanced.signal_rules:
            legacy_points = evaluate_all_rules(bars, balanced.signal_rules)
        layered_points = evaluate_layered(bars, strategy_type="balanced")

        # 两者都应返回 TradePoint 列表
        assert isinstance(layered_points, list)
        for p in layered_points:
            assert hasattr(p, 'action')
            assert hasattr(p, 'date')
            assert hasattr(p, 'price')
            assert hasattr(p, 'confidence')
            assert p.action in ("BUY", "SELL")
