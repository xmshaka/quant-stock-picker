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
    # ── 资金管理 ──
    max_add_times: int = 2             # 最大加仓次数（0=不加仓，仅建仓）
    position_pct_per_entry: float = 0.30  # 每次建仓/加仓占可用资金比例
    max_single_pct: float = 0.30       # 单只股票最大仓位占总权益比例
    # ── 止盈止损 ──
    stop_loss_atr_mult: float = 2.0    # 止损 = 买入价 - N×ATR
    take_profit_atr_mult: float = 3.0  # 固定止盈 = 买入价 + N×ATR
    trailing_atr_mult: float = 2.0     # 跟踪止盈 = 持仓最高价 - N×ATR
    atr_period: int = 14               # ATR计算周期

    def to_dict(self) -> Dict:
        return {
            "scheme_id": self.scheme_id,
            "name": self.name,
            "description": self.description,
            "factor_weights": self.factor_weights,
            "signal_rules": [r.to_dict() for r in self.signal_rules],
            "regime_fit": self.regime_fit,
            "is_builtin": self.is_builtin,
            "max_add_times": self.max_add_times,
            "position_pct_per_entry": self.position_pct_per_entry,
            "max_single_pct": self.max_single_pct,
            "stop_loss_atr_mult": self.stop_loss_atr_mult,
            "take_profit_atr_mult": self.take_profit_atr_mult,
            "trailing_atr_mult": self.trailing_atr_mult,
            "atr_period": self.atr_period,
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
            max_add_times=d.get("max_add_times", 2),
            position_pct_per_entry=d.get("position_pct_per_entry", 0.30),
            max_single_pct=d.get("max_single_pct", 0.30),
            stop_loss_atr_mult=d.get("stop_loss_atr_mult", 2.0),
            take_profit_atr_mult=d.get("take_profit_atr_mult", 3.0),
            trailing_atr_mult=d.get("trailing_atr_mult", 2.0),
            atr_period=d.get("atr_period", 14),
        )


# ============================================================
# 内置方案
# ============================================================

BUILTIN_SCHEMES: Dict[str, StrategyScheme] = {}


def _register(scheme: StrategyScheme):
    BUILTIN_SCHEMES[scheme.scheme_id] = scheme


_register(StrategyScheme(
    scheme_id="reversal",
    name="反转优选",
    description="适合震荡市。低RSI + 高反转 + 低波动，捕捉超跌反弹。",
    factor_weights={
        'rsi14': -0.3, 'macd_hist': 0.15, 'boll_position': 0.1,
        'volatility_20d': -0.2, 'max_dd_60d': -0.15,
        'north_hold_change': 0.2, 'turnover_ratio': 0.1, 'volume_ratio': 0.1,
        'pe_ttm': -0.15, 'pb': -0.1, 'ep': 0.1,
        'roe': 0.15, 'gross_margin': 0.1, 'revenue_growth': 0.1, 'profit_growth': 0.1,
        'momentum_5d': -0.25, 'momentum_20d': -0.15, 'momentum_60d': 0.0,
        'liquidity': 0.1, 'reversal': 0.4,
    },
    signal_rules=[
        SignalRuleConfig(RuleType.RSI_REVERSAL, {"oversold": 30, "overbought": 70}),
        SignalRuleConfig(RuleType.BOLL_BREAK, {"period": 20, "std_dev": 2.0}),
    ],
    regime_fit=["震荡整理"],
))

_register(StrategyScheme(
    scheme_id="momentum",
    name="趋势动量",
    description="适合强势上涨市。高动量 + 放量 + MACD金叉，顺势追涨。",
    factor_weights={
        'rsi14': 0.15, 'macd_hist': 0.25, 'boll_position': 0.1,
        'volatility_20d': -0.05, 'max_dd_60d': -0.05,
        'north_hold_change': 0.3, 'turnover_ratio': 0.15, 'volume_ratio': 0.2,
        'pe_ttm': -0.05, 'pb': -0.05, 'ep': 0.1,
        'roe': 0.15, 'gross_margin': 0.1, 'revenue_growth': 0.15, 'profit_growth': 0.15,
        'momentum_5d': 0.2, 'momentum_20d': 0.35, 'momentum_60d': 0.2,
        'liquidity': 0.15, 'reversal': -0.1,
    },
    signal_rules=[
        SignalRuleConfig(RuleType.MA_CROSS, {"short": 5, "long": 20}),
        SignalRuleConfig(RuleType.MACD_TREND, {"fast": 12, "slow": 26, "signal": 9}),
    ],
    regime_fit=["强势单边上涨"],
))

_register(StrategyScheme(
    scheme_id="value",
    name="低波价值",
    description="适合弱势/防御。低估值 + 高ROE + 低波动，安全边际优先。",
    factor_weights={
        'rsi14': -0.1, 'macd_hist': 0.05, 'boll_position': 0.05,
        'volatility_20d': -0.25, 'max_dd_60d': -0.2,
        'north_hold_change': 0.15, 'turnover_ratio': -0.05, 'volume_ratio': -0.05,
        'pe_ttm': -0.3, 'pb': -0.25, 'ep': 0.2,
        'roe': 0.3, 'gross_margin': 0.25, 'revenue_growth': 0.1, 'profit_growth': 0.1,
        'momentum_5d': -0.05, 'momentum_20d': -0.05, 'momentum_60d': 0.0,
        'liquidity': 0.05, 'reversal': 0.15,
    },
    signal_rules=[
        SignalRuleConfig(RuleType.BOLL_BREAK, {"period": 20, "std_dev": 2.0}),
        SignalRuleConfig(RuleType.VOLUME_BREAKOUT, {"volume_mult": 1.5, "lookback": 20}),
    ],
    regime_fit=["弱势单边下跌", "震荡整理"],
))

_register(StrategyScheme(
    scheme_id="composite",
    name="均衡复合",
    description="全行情适配。各因子均衡配置，不押注单一风格。",
    factor_weights={
        'rsi14': -0.1, 'macd_hist': 0.2, 'boll_position': 0.1,
        'volatility_20d': -0.1, 'max_dd_60d': -0.1,
        'north_hold_change': 0.4, 'turnover_ratio': 0.15, 'volume_ratio': 0.15,
        'pe_ttm': -0.2, 'pb': -0.1, 'ep': 0.1,
        'roe': 0.3, 'gross_margin': 0.2, 'revenue_growth': 0.2, 'profit_growth': 0.2,
        'momentum_5d': -0.15, 'momentum_20d': -0.1, 'momentum_60d': 0.0,
        'liquidity': 0.1, 'reversal': 0.3,
    },
    signal_rules=[
        SignalRuleConfig(RuleType.RSI_REVERSAL, {"oversold": 30, "overbought": 70}),
        SignalRuleConfig(RuleType.MA_CROSS, {"short": 5, "long": 20}),
        SignalRuleConfig(RuleType.BOLL_BREAK, {"period": 20, "std_dev": 2.0}),
    ],
    regime_fit=["*"],
))
