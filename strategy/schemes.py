"""策略方案定义 — 多方案择时选股系统的核心数据结构

每个 StrategyScheme 定义：
1. factor_weights — 截面因子权重（选哪只股票）
2. signal_rules   — 个股K线择时规则（何时买卖）
3. regime_fit     — 适配行情类型
"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type
from enum import Enum


class RuleType(Enum):
    """信号规则类型"""
    RSI_REVERSAL = "rsi_reversal"
    MA_CROSS = "ma_cross"
    MACD_TREND = "macd_trend"
    BOLL_BREAK = "boll_break"
    VOLUME_BREAKOUT = "volume_breakout"
    KDJ_CROSS = "kdj_cross"


@dataclass
class ResonanceConfig:
    """策略专属共振配置。

    min_confirmations：最低确认数；
    buy_conditions/sell_conditions：该策略启用的 L3 条件名称，空列表表示使用默认全集。
    """
    min_confirmations: int = 2
    buy_conditions: List[str] = field(default_factory=list)
    sell_conditions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "min_confirmations": self.min_confirmations,
            "buy_conditions": list(self.buy_conditions),
            "sell_conditions": list(self.sell_conditions),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ResonanceConfig":
        return cls(
            min_confirmations=int(d.get("min_confirmations", 2)),
            buy_conditions=list(d.get("buy_conditions", []) or []),
            sell_conditions=list(d.get("sell_conditions", []) or []),
        )


@dataclass
class ExitConfig:
    """策略专属短线退出配置。"""
    enable_market_defense_exit: bool = True
    enable_strategy_failure_exit: bool = True
    enable_trailing_exit: bool = True
    enable_time_stop: bool = True
    enable_max_holding_exit: bool = True
    max_holding_days: int = 20
    time_stop_days: int = 7
    time_stop_min_profit_pct: float = 0.0
    failure_window_days: int = 3
    market_defense_score: float = 20.0
    trailing_activation_pct: float = 0.05
    trailing_activation_atr_mult: float = 1.0

    def to_dict(self) -> Dict:
        return {
            "enable_market_defense_exit": self.enable_market_defense_exit,
            "enable_strategy_failure_exit": self.enable_strategy_failure_exit,
            "enable_trailing_exit": self.enable_trailing_exit,
            "enable_time_stop": self.enable_time_stop,
            "enable_max_holding_exit": self.enable_max_holding_exit,
            "max_holding_days": self.max_holding_days,
            "time_stop_days": self.time_stop_days,
            "time_stop_min_profit_pct": self.time_stop_min_profit_pct,
            "failure_window_days": self.failure_window_days,
            "market_defense_score": self.market_defense_score,
            "trailing_activation_pct": self.trailing_activation_pct,
            "trailing_activation_atr_mult": self.trailing_activation_atr_mult,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ExitConfig":
        return cls(
            enable_market_defense_exit=bool(d.get("enable_market_defense_exit", True)),
            enable_strategy_failure_exit=bool(d.get("enable_strategy_failure_exit", True)),
            enable_trailing_exit=bool(d.get("enable_trailing_exit", True)),
            enable_time_stop=bool(d.get("enable_time_stop", True)),
            enable_max_holding_exit=bool(d.get("enable_max_holding_exit", True)),
            max_holding_days=int(d.get("max_holding_days", 20)),
            time_stop_days=int(d.get("time_stop_days", 7)),
            time_stop_min_profit_pct=float(d.get("time_stop_min_profit_pct", 0.0)),
            failure_window_days=int(d.get("failure_window_days", 3)),
            market_defense_score=float(d.get("market_defense_score", 20.0)),
            trailing_activation_pct=float(d.get("trailing_activation_pct", 0.05)),
            trailing_activation_atr_mult=float(d.get("trailing_activation_atr_mult", 1.0)),
        )


@dataclass
class SignalRuleConfig:
    """信号规则配置（可序列化）"""
    rule_type: RuleType
    params: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {"rule_type": self.rule_type.value, "params": self.params}

    @classmethod
    def from_dict(cls, d: Dict) -> "SignalRuleConfig":
        return cls(rule_type=RuleType(d["rule_type"]), params=d.get("params", {}))


@dataclass
class StrategyScheme:
    """策略方案"""
    scheme_id: str                     # 唯一标识 "reversal" / "momentum" / "value" / "custom_1"
    name: str                          # 显示名 "反转优选"
    description: str                   # 描述
    factor_weights: Dict[str, float]   # {factor_name: weight}
    signal_rules: List[SignalRuleConfig]  # 信号规则列表
    regime_fit: List[str]              # 适配行情 ["震荡整理", "强势单边上涨", "*"]
    is_builtin: bool = True            # 是否内置
    # ── 信号引擎 ──
    signal_mode: str = "layered"       # "layered" 三层过滤 | "legacy" 旧规则
    # ── 大盘择时 ──
    enable_market_timing: bool = True   # 是否启用大盘择时仓位调制
    # ── 资金管理 ──
    max_add_times: int = 2             # 最大加仓次数（0=不加仓，仅建仓）
    position_pct_per_entry: float = 0.30  # 每次建仓/加仓占可用资金比例
    max_single_pct: float = 0.30       # 单只股票最大仓位占总权益比例
    # ── 止盈止损 ──
    stop_loss_atr_mult: float = 2.0    # 止损 = 买入价 - N×ATR
    take_profit_atr_mult: float = 3.0  # 固定止盈 = 买入价 + N×ATR
    trailing_atr_mult: float = 2.0     # 跟踪止盈 = 持仓最高价 - N×ATR
    atr_period: int = 14               # ATR计算周期
    resonance_config: ResonanceConfig = field(default_factory=ResonanceConfig)  # P1: 策略专属共振配置
    exit_config: ExitConfig = field(default_factory=ExitConfig)                  # P2: 策略专属短线退出配置

    def to_dict(self) -> Dict:
        return {
            "scheme_id": self.scheme_id,
            "name": self.name,
            "description": self.description,
            "factor_weights": self.factor_weights,
            "signal_rules": [r.to_dict() for r in self.signal_rules],
            "regime_fit": self.regime_fit,
            "is_builtin": self.is_builtin,
            "signal_mode": self.signal_mode,
            "enable_market_timing": self.enable_market_timing,
            "max_add_times": self.max_add_times,
            "position_pct_per_entry": self.position_pct_per_entry,
            "max_single_pct": self.max_single_pct,
            "stop_loss_atr_mult": self.stop_loss_atr_mult,
            "take_profit_atr_mult": self.take_profit_atr_mult,
            "trailing_atr_mult": self.trailing_atr_mult,
            "atr_period": self.atr_period,
            "resonance_config": self.resonance_config.to_dict(),
            "exit_config": self.exit_config.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "StrategyScheme":
        return cls(
            scheme_id=d["scheme_id"],
            name=d["name"],
            description=d["description"],
            factor_weights=d["factor_weights"],
            signal_rules=[SignalRuleConfig.from_dict(r) for r in d.get("signal_rules", [])],
            regime_fit=d.get("regime_fit", ["*"]),
            is_builtin=d.get("is_builtin", False),
            signal_mode=d.get("signal_mode", "layered"),
            enable_market_timing=d.get("enable_market_timing", True),
            max_add_times=d.get("max_add_times", 2),
            position_pct_per_entry=d.get("position_pct_per_entry", 0.30),
            max_single_pct=d.get("max_single_pct", 0.30),
            stop_loss_atr_mult=d.get("stop_loss_atr_mult", 2.0),
            take_profit_atr_mult=d.get("take_profit_atr_mult", 3.0),
            trailing_atr_mult=d.get("trailing_atr_mult", 2.0),
            atr_period=d.get("atr_period", 14),
            resonance_config=ResonanceConfig.from_dict(d.get("resonance_config", {})),
            exit_config=ExitConfig.from_dict(d.get("exit_config", {})),
        )


# ============================================================
# 内置方案
# ============================================================

BUILTIN_SCHEMES: Dict[str, StrategyScheme] = {}


def _register(scheme: StrategyScheme):
    BUILTIN_SCHEMES[scheme.scheme_id] = scheme


_register(StrategyScheme(
    scheme_id="trend_momentum",
    name="强势追涨",
    description="适合强势上涨市。高动量 + 高量比 + 布林上轨，顺势追涨。",
    factor_weights={
        'momentum_20d': 0.30, 'momentum_5d': 0.20, 'volume_ratio': 0.20,
        'boll_position': 0.10, 'high_20d_distance': 0.10, 'rsi14': 0.10,
    },
    signal_rules=[
        SignalRuleConfig(RuleType.MA_CROSS, {"short": 5, "long": 20}),
        SignalRuleConfig(RuleType.MACD_TREND, {"fast": 12, "slow": 26, "signal": 9}),
    ],
    regime_fit=["强势单边上涨"],
    resonance_config=ResonanceConfig(
        min_confirmations=3,
        buy_conditions=["volume_expand", "ma5_above_ma20", "near_high", "momentum_5d", "momentum_20d", "rsi_not_extreme"],
        sell_conditions=["ma5_below_ma20", "macd_bearish", "volume_price_down", "kdj_death"],
    ),
    exit_config=ExitConfig(
        max_holding_days=10,
        time_stop_days=5,
        time_stop_min_profit_pct=0.02,
        failure_window_days=3,
        market_defense_score=20.0,
    ),
))

_register(StrategyScheme(
    scheme_id="pullback",
    name="回调低吸",
    description="适合上升趋势中的回调。高反转 + 低RSI + 远离高点，低吸买入。",
    factor_weights={
        'reversal': 0.30, 'rsi14': -0.20, 'high_20d_distance': -0.20,
        'volume_ratio': -0.15, 'volatility_20d': -0.15,
    },
    signal_rules=[
        SignalRuleConfig(RuleType.RSI_REVERSAL, {"oversold": 25, "overbought": 75}),
        SignalRuleConfig(RuleType.BOLL_BREAK, {"period": 20, "std_dev": 2.0}),
    ],
    regime_fit=["震荡整理", "弱势单边下跌"],
    resonance_config=ResonanceConfig(
        min_confirmations=3,
        buy_conditions=["rsi_oversold", "boll_lower", "pullback_range", "not_break_20d_low", "volume_calm", "near_support"],
        sell_conditions=["rsi_overbought", "ma5_below_ma20", "boll_upper", "kdj_death"],
    ),
    exit_config=ExitConfig(
        max_holding_days=15,
        time_stop_days=7,
        time_stop_min_profit_pct=0.0,
        failure_window_days=3,
        market_defense_score=20.0,
    ),
))

_register(StrategyScheme(
    scheme_id="breakout",
    name="横盘突破",
    description="适合横盘整理后的突破。布林收窄 + 放量突破 + 短期强势。",
    factor_weights={
        'boll_position': 0.30, 'volume_ratio': 0.25, 'momentum_5d': 0.20,
        'high_20d_distance': 0.15, 'momentum_20d': 0.10,
    },
    signal_rules=[
        SignalRuleConfig(RuleType.VOLUME_BREAKOUT, {"volume_mult": 1.5, "lookback": 20}),
        SignalRuleConfig(RuleType.BOLL_BREAK, {"period": 20, "std_dev": 2.0}),
    ],
    regime_fit=["震荡整理", "强势单边上涨"],
    resonance_config=ResonanceConfig(
        min_confirmations=3,
        buy_conditions=["break_platform", "volume_surge", "ma5_above_ma20", "narrow_range", "momentum_5d", "boll_upper"],
        sell_conditions=["ma5_below_ma20", "macd_bearish", "volume_price_down", "kdj_death"],
    ),
    exit_config=ExitConfig(
        max_holding_days=10,
        time_stop_days=5,
        time_stop_min_profit_pct=0.0,
        failure_window_days=2,
        market_defense_score=20.0,
    ),
))

_register(StrategyScheme(
    scheme_id="balanced",
    name="均衡择时",
    description="全行情适配。各因子均衡配置，不押注单一风格。",
    factor_weights={
        'momentum_20d': 0.10, 'momentum_5d': 0.10, 'reversal': 0.10,
        'rsi14': 0.10, 'boll_position': 0.10, 'volatility_20d': 0.10,
        'volume_ratio': 0.10, 'high_20d_distance': 0.10,
        'float_market_cap': 0.10, 'pb': 0.10,
    },
    signal_rules=[
        SignalRuleConfig(RuleType.RSI_REVERSAL, {"oversold": 30, "overbought": 70}),
        SignalRuleConfig(RuleType.MA_CROSS, {"short": 5, "long": 20}),
        SignalRuleConfig(RuleType.BOLL_BREAK, {"period": 20, "std_dev": 2.0}),
    ],
    regime_fit=["*"],
    resonance_config=ResonanceConfig(
        min_confirmations=2,
        buy_conditions=[],
        sell_conditions=[],
    ),
    exit_config=ExitConfig(
        max_holding_days=20,
        time_stop_days=10,
        time_stop_min_profit_pct=0.0,
        failure_window_days=3,
        market_defense_score=20.0,
    ),
))
