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
    # ── 开仓执行契约 ──
    min_entry_condition_count: int = 3   # 最小condition_count才开仓（0=关闭entry_contract）
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
            "min_entry_condition_count": self.min_entry_condition_count,
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
            min_entry_condition_count=int(d.get("min_entry_condition_count", 3)),
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
    description="适合强势上涨市。高动量 + 资金净流入 + 高相对换手，顺势追涨。",
    factor_weights={
        'momentum_20d': 0.30, 'momentum_5d': 0.20, 'volume_ratio': 0.20,
        'boll_position': 0.10, 'high_20d_distance': 0.10, 'rsi14': 0.10,
    },
    signal_rules=[
        SignalRuleConfig(RuleType.MA_CROSS, {"short": 5, "long": 20}),
        SignalRuleConfig(RuleType.MACD_TREND, {"fast": 12, "slow": 26, "signal": 9}),
    ],
    regime_fit=["强势单边上涨"],
    min_entry_condition_count=6,  # Stage 2后验: cc≥6胜率64.3%均利+4,311
    resonance_config=ResonanceConfig(
        min_confirmations=3,
        buy_conditions=[
            # 资金流条件
            "large_elg_net_mf_positive",        # 超大单净流入
            "main_net_mf_positive",             # 主力净流入
            "large_elg_net_mf_rank_high",       # 超大单净流入排名前20%
            # 相对换手条件
            "relative_turnover_5d_high",        # 5日相对换手率高
            "amount_percentile_60d_high",       # 60日成交额分位高
            # 技术确认条件
            "momentum_5d_strong",               # 5日动量强
            "momentum_20d_strong",              # 20日动量强
            "volume_expand",                    # 放量
            "ma5_above_ma20",                   # 短期均线上穿
            "rsi_not_extreme",                  # RSI不过热
        ],
        sell_conditions=[
            "main_net_mf_negative",             # 主力净流出
            "large_elg_net_mf_negative",        # 超大单净流出
            "relative_turnover_5d_low",         # 5日相对换手率低
            "ma5_below_ma20",                   # 短期均线下穿
            "macd_bearish",                     # MACD转弱
            "volume_price_down",                # 放量下跌
        ],
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
    description="适合上升趋势中的回调。资金流出减缓 + 相对缩量 + 低RSI，低吸买入。",
    factor_weights={
        'reversal': 0.30, 'rsi14': -0.20, 'high_20d_distance': -0.20,
        'volume_ratio': -0.15, 'volatility_20d': -0.15,
    },
    signal_rules=[
        SignalRuleConfig(RuleType.RSI_REVERSAL, {"oversold": 25, "overbought": 75}),
        SignalRuleConfig(RuleType.BOLL_BREAK, {"period": 20, "std_dev": 2.0}),
    ],
    regime_fit=["震荡整理", "弱势单边下跌"],
    min_entry_condition_count=8,  # Stage 2后验: cc≥8胜率46.2%均利+3,141
    resonance_config=ResonanceConfig(
        min_confirmations=3,
        buy_conditions=[
            # 资金流条件
            "main_net_mf_negative_improving",   # 主力净流出但改善
            "large_elg_net_mf_negative_improving", # 超大单净流出但改善
            # 相对换手条件
            "relative_turnover_5d_low",         # 5日相对换手率低（缩量回调）
            "turnover_percentile_60d_low",      # 60日换手率分位低
            # 技术确认条件
            "rsi_oversold",                     # RSI超卖
            "boll_lower",                       # 布林下轨
            "pullback_range",                   # 回调幅度适中
            "not_break_20d_low",                # 不破20日低点
            "volume_calm",                      # 成交量平稳
            "near_support",                     # 接近支撑位
        ],
        sell_conditions=[
            "main_net_mf_negative_worsening",   # 主力净流出恶化
            "large_elg_net_mf_negative_worsening", # 超大单净流出恶化
            "relative_turnover_5d_high",        # 5日相对换手率高（放量下跌）
            "rsi_overbought",                   # RSI超买
            "ma5_below_ma20",                   # 短期均线下穿
            "boll_upper",                       # 布林上轨（反弹到位）
        ],
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
    description="适合横盘整理后的突破。资金大幅流入 + 高相对换手 + 放量突破。",
    factor_weights={
        'boll_position': 0.30, 'volume_ratio': 0.25, 'momentum_5d': 0.20,
        'high_20d_distance': 0.15, 'momentum_20d': 0.10,
    },
    signal_rules=[
        SignalRuleConfig(RuleType.VOLUME_BREAKOUT, {"volume_mult": 1.5, "lookback": 20}),
        SignalRuleConfig(RuleType.BOLL_BREAK, {"period": 20, "std_dev": 2.0}),
    ],
    regime_fit=["震荡整理", "强势单边上涨"],
    min_entry_condition_count=0,  # Stage 2后验: 全cc组亏损，暂停entry_contract
    resonance_config=ResonanceConfig(
        min_confirmations=4,  # 提高门槛：从3→4，减少噪音突破
        buy_conditions=[
            # 资金流条件（突破v2: 超高门槛）
            "large_elg_net_mf_positive_strong", # 超大单>10万
            "main_net_mf_positive_strong",      # 主力>5万
            "mf_rank_elite",                    # 资金排名前20%
            # 量能条件（突破v2: 量比>2x+实体阳线）
            "volume_surge",                     # 量比>2x
            "relative_turnover_5d_high",        # 换手活跃
            "amount_percentile_60d_high",       # 成交额前25%
            "bullish_body",                     # 实体阳线
            # 价格结构条件（突破v2: 收盘确认+多bar验证）
            "break_platform",                   # 收盘突破平台上沿
            "narrow_range",                     # 平台振幅<8%
            "buildup_signal",                   # 前日蓄势
            "sustained_breakout",               # 连续站稳突破位
            # 趋势背景
            "ma5_above_ma20",                   # 均线多头
            "boll_expanding",                   # 布林中上轨
        ],
        sell_conditions=[
            "main_net_mf_negative",             # 主力净流出
            "large_elg_net_mf_negative",        # 超大单净流出
            "relative_turnover_5d_low",         # 5日相对换手率低
            "ma5_below_ma20",                   # 短期均线下穿
            "macd_bearish",                     # MACD转弱
            "volume_price_down",                # 放量下跌
        ],
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
    description="全行情适配。均衡配置，不押注单一风格。长期应作为组合器使用。",
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
        min_confirmations=3,
        buy_conditions=[],  # balanced v2: 策略路由器，委托给子策略生成条件，不设 whitelist
        sell_conditions=[
            "main_net_mf_negative",             # 主力净流出
            "relative_turnover_5d_low",         # 5日相对换手率低
            "ma5_below_ma20",                   # 短期均线下穿
            "macd_bearish",                     # MACD转弱
            "volume_price_down",                # 放量下跌
        ],
    ),
    exit_config=ExitConfig(
        max_holding_days=20,
        time_stop_days=10,
        time_stop_min_profit_pct=0.0,
        failure_window_days=3,
        market_defense_score=20.0,
    ),
))
