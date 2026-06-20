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
    assert all(cfg.min_confirmations == 3 for cfg in configs.values())
    assert configs["trend_momentum"].buy_conditions != configs["pullback"].buy_conditions
    assert configs["pullback"].buy_conditions != configs["breakout"].buy_conditions
    assert "near_high" in configs["trend_momentum"].buy_conditions
    assert "pullback_range" in configs["pullback"].buy_conditions
    assert "break_platform" in configs["breakout"].buy_conditions


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

    assert rc.min_confirmations == 3
    assert rc.buy_conditions == set(BUILTIN_SCHEMES["breakout"].resonance_config.buy_conditions)
    assert "break_platform" in rc.buy_conditions


def test_scanner_layer3_uses_resonance_config_condition_subset():
    latest = date(2026, 1, 1) + timedelta(days=49)
    closes = [10 + i * 0.15 for i in range(50)]
    bars = pd.DataFrame(_bars("000001", closes))
    row = pd.Series(_factor_row(
        "000001", latest,
        momentum_5d=0.08,
        momentum_20d=0.25,
        volume_ratio=1.8,
        rsi14=65,
        boll_position=0.85,
    ))

    full_count, full_reasons = _check_layer3(bars, row, "trend_momentum")
    restricted = ResonanceConfig(min_confirmations=1, buy_conditions=["near_high"])
    restricted_count, restricted_reasons = _check_layer3(bars, row, "trend_momentum", restricted)

    assert full_count >= restricted_count
    assert restricted_count == 1
    assert len(restricted_reasons) == 1
    assert "高点" in restricted_reasons[0]


def test_scanner_resonance_config_helper_returns_builtin_config():
    cfg = _resonance_config("pullback")

    assert cfg is BUILTIN_SCHEMES["pullback"].resonance_config
    assert "pullback_range" in cfg.buy_conditions
