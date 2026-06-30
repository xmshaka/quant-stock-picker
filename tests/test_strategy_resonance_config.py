"""P1 策略专属共振配置测试。"""

from strategy.schemes import BUILTIN_SCHEMES, ExitConfig, ResonanceConfig, StrategyScheme
from signals.layers import ResonanceChecker
from signals.scanner import _resonance_config, _check_layer3
from tests.test_signal_scanner_p0 import _bars, _factor_row

from datetime import date, timedelta
import pandas as pd


def test_builtin_strategies_have_distinct_resonance_configs():
    configs = {sid: BUILTIN_SCHEMES[sid].resonance_config for sid in ["trend_momentum", "pullback", "breakout"]}

    assert all(isinstance(cfg, ResonanceConfig) for cfg in configs.values())
    # breakout: min_confirmations=4（提升门槛减少噪音突破）
    assert all(cfg.min_confirmations >= 3 for cfg in configs.values())
    assert configs["trend_momentum"].buy_conditions != configs["pullback"].buy_conditions
    assert configs["pullback"].buy_conditions != configs["breakout"].buy_conditions
    # trend_momentum 应有资金流和相对换手条件
    assert any("mf_positive" in cond for cond in configs["trend_momentum"].buy_conditions)
    assert any("turnover_5d_high" in cond for cond in configs["trend_momentum"].buy_conditions)
    # pullback 应有回调相关的条件
    assert any("rsi_oversold" in cond for cond in configs["pullback"].buy_conditions)
    assert any("turnover_5d_low" in cond for cond in configs["pullback"].buy_conditions)
    # breakout 应有突破相关的条件
    assert any("break_platform" in cond for cond in configs["breakout"].buy_conditions)
    assert any("volume_surge" in cond for cond in configs["breakout"].buy_conditions)


def test_strategy_scheme_roundtrip_preserves_resonance_config():
    scheme = BUILTIN_SCHEMES["pullback"]
    restored = StrategyScheme.from_dict(scheme.to_dict())

    assert restored.resonance_config.min_confirmations == scheme.resonance_config.min_confirmations
    assert restored.resonance_config.buy_conditions == scheme.resonance_config.buy_conditions
    assert restored.resonance_config.sell_conditions == scheme.resonance_config.sell_conditions
    assert restored.exit_config.max_holding_days == scheme.exit_config.max_holding_days
    assert restored.exit_config.time_stop_days == scheme.exit_config.time_stop_days
    assert restored.exit_config.time_stop_min_profit_pct == scheme.exit_config.time_stop_min_profit_pct


def test_builtin_strategies_have_short_term_exit_configs():
    assert BUILTIN_SCHEMES["trend_momentum"].exit_config.max_holding_days == 10
    assert BUILTIN_SCHEMES["trend_momentum"].exit_config.time_stop_days == 5
    assert BUILTIN_SCHEMES["trend_momentum"].exit_config.time_stop_min_profit_pct == 0.02
    assert BUILTIN_SCHEMES["trend_momentum"].exit_config.enable_time_stop is True
    assert BUILTIN_SCHEMES["trend_momentum"].exit_config.enable_max_holding_exit is True
    assert BUILTIN_SCHEMES["trend_momentum"].exit_config.enable_strategy_failure_exit is True
    assert BUILTIN_SCHEMES["trend_momentum"].exit_config.enable_market_defense_exit is True
    assert BUILTIN_SCHEMES["pullback"].exit_config.max_holding_days == 15
    assert BUILTIN_SCHEMES["balanced"].exit_config.max_holding_days == 20
    assert BUILTIN_SCHEMES["balanced"].exit_config.time_stop_days == 10
    assert BUILTIN_SCHEMES["breakout"].exit_config.failure_window_days == 2


def test_exit_config_serializes_rule_switches():
    cfg = ExitConfig(
        enable_market_defense_exit=False,
        enable_strategy_failure_exit=False,
        enable_time_stop=False,
        enable_max_holding_exit=False,
        max_holding_days=9,
        time_stop_days=4,
        time_stop_min_profit_pct=0.01,
        failure_window_days=2,
        market_defense_score=30,
        enable_trailing_exit=False,
        trailing_activation_pct=0.08,
        trailing_activation_atr_mult=1.5,
    )
    restored = ExitConfig.from_dict(cfg.to_dict())

    assert restored.enable_market_defense_exit is False
    assert restored.enable_strategy_failure_exit is False
    assert restored.enable_time_stop is False
    assert restored.enable_max_holding_exit is False
    assert restored.enable_trailing_exit is False
    assert restored.max_holding_days == 9
    assert restored.time_stop_days == 4
    assert restored.trailing_activation_pct == 0.08
    assert restored.trailing_activation_atr_mult == 1.5


def test_layers_resonance_checker_loads_strategy_config():
    rc = ResonanceChecker.from_strategy("breakout")

    # breakout: min_confirmations=4（提升门槛减少噪音突破）
    assert rc.min_confirmations == 4
    assert rc.buy_conditions == set(BUILTIN_SCHEMES["breakout"].resonance_config.buy_conditions)
    assert "break_platform" in rc.buy_conditions


def test_scanner_layer3_uses_resonance_config_condition_subset():
    latest = date(2026, 1, 1) + timedelta(days=49)
    closes = [10 + i * 0.15 for i in range(50)]
    bars = pd.DataFrame(_bars("000001", closes))
    
    # 创建包含资金流和相对换手因子的测试数据
    row = pd.Series(_factor_row(
        "000001", latest,
        momentum_5d=0.08,
        momentum_20d=0.25,
        volume_ratio=1.8,
        rsi14=65,
        boll_position=0.85,
        # 添加资金流和相对换手因子
        main_net_mf_amount=50000.0,
        large_elg_net_mf_amount=80000.0,
        main_net_mf_rank=0.85,
        large_elg_net_mf_rank=0.80,
        relative_turnover_5d=1.3,
        amount_percentile_60d=0.75,
        turnover_percentile_60d=0.60,
    ))

    full_count, _, full_reasons = _check_layer3(bars, row, "trend_momentum")
    # 使用新的条件进行限制测试
    restricted = ResonanceConfig(min_confirmations=1, buy_conditions=["large_elg_net_mf_positive"])
    restricted_count, _, restricted_reasons = _check_layer3(bars, row, "trend_momentum", restricted)

    assert full_count >= restricted_count
    assert restricted_count == 1  # 应该满足超大单净流入条件
    assert len(restricted_reasons) == 1
    assert "超大单" in restricted_reasons[0]


def test_scanner_resonance_config_helper_returns_builtin_config():
    cfg = _resonance_config("pullback")

    assert cfg is BUILTIN_SCHEMES["pullback"].resonance_config
    assert "pullback_range" in cfg.buy_conditions
