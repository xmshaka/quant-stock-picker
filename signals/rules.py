"""Phase 2: 个股级买卖点信号规则

每个规则接收单只股票的K线 DataFrame，输出 TradePoint 列表。
规则可组合，由 StrategyScheme.signal_rules 驱动。
"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Type
from datetime import date

import pandas as pd
import numpy as np

from strategy.schemes import RuleType, SignalRuleConfig


@dataclass
class TradePoint:
    """单个买卖点"""
    date: date
    action: str          # "BUY" | "SELL"
    reason: str          # "RSI<30 超卖反弹"
    confidence: float    # 0-1
    price: float = 0.0   # 触发价格（收盘价）
    rule_name: str = ""  # 规则名
    # ── 交易执行信息（回测引擎填充） ──
    exec_price: float = 0.0      # 实际成交价（含滑点）
    shares: int = 0              # 交易股数
    cash_after: float = 0.0      # 成交后余额
    position_shares: int = 0     # 成交后总持股
    avg_cost: float = 0.0        # 持仓均价
    stop_loss: float = 0.0       # 止损价
    take_profit: float = 0.0     # 止盈价
    trailing_stop: float = 0.0   # 跟踪止盈价（初始=止损价，随股价上涨上移）
    pnl: float = 0.0             # 盈亏金额（仅SELL）
    pnl_pct: float = 0.0         # 盈亏比例（仅SELL）
    holding_days: int = 0        # 持仓天数（仅SELL）


# ============================================================
# 规则基类
# ============================================================

class SignalRule(ABC):
    """信号规则基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def evaluate(self, bars: pd.DataFrame) -> List[TradePoint]:
        """输入单只股票K线，输出买卖点列表

        bars: DataFrame with columns [trade_date, open, high, low, close, volume]
              sorted by trade_date ascending
        """
        pass


# ============================================================
# RSI 超买超卖反转
# ============================================================

class RSIReversalRule(SignalRule):
    """RSI 超卖买入 / 超买卖出"""

    def __init__(self, oversold: int = 30, overbought: int = 70, period: int = 14):
        self.oversold = oversold
        self.overbought = overbought
        self.period = period

    @property
    def name(self) -> str:
        return f"RSI反转({self.oversold}/{self.overbought})"

    def evaluate(self, bars: pd.DataFrame) -> List[TradePoint]:
        if len(bars) < self.period + 1:
            return []

        close = bars['close'].astype(float)
        rsi = self._calc_rsi(close, self.period)
        points = []
        in_oversold = False   # 状态机：是否处于超卖区
        in_overbought = False  # 状态机：是否处于超买区

        for i in range(1, len(rsi)):
            if pd.isna(rsi.iloc[i]):
                continue

            cur = float(rsi.iloc[i])

            # 进入超卖区 → 买入（只触发一次，直到退出再重新进入）
            if cur < self.oversold and not in_oversold:
                in_oversold = True
                points.append(TradePoint(
                    date=bars.iloc[i]['trade_date'],
                    action="BUY",
                    reason=f"RSI({cur:.0f})进入超卖区",
                    confidence=min(1.0, (self.oversold - cur) / 20 + 0.5),
                    price=float(close.iloc[i]),
                    rule_name=self.name,
                ))
            elif cur >= self.oversold:
                in_oversold = False

            # 进入超买区 → 卖出（只触发一次）
            if cur > self.overbought and not in_overbought:
                in_overbought = True
                points.append(TradePoint(
                    date=bars.iloc[i]['trade_date'],
                    action="SELL",
                    reason=f"RSI({cur:.0f})进入超买区",
                    confidence=min(1.0, (cur - self.overbought) / 20 + 0.5),
                    price=float(close.iloc[i]),
                    rule_name=self.name,
                ))
            elif cur <= self.overbought:
                in_overbought = False

        return points

    @staticmethod
    def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)


# ============================================================
# 均线金叉/死叉
# ============================================================

class MACrossRule(SignalRule):
    """短期均线上穿长期均线 → 买入；下穿 → 卖出"""

    def __init__(self, short: int = 5, long: int = 20):
        self.short = short
        self.long = long

    @property
    def name(self) -> str:
        return f"MA{self.short}/{self.long}交叉"

    def evaluate(self, bars: pd.DataFrame) -> List[TradePoint]:
        if len(bars) < self.long + 1:
            return []

        close = bars['close'].astype(float)
        ma_s = close.rolling(self.short).mean()
        ma_l = close.rolling(self.long).mean()
        points = []

        for i in range(1, len(bars)):
            if pd.isna(ma_s.iloc[i]) or pd.isna(ma_l.iloc[i]):
                continue
            if pd.isna(ma_s.iloc[i - 1]) or pd.isna(ma_l.iloc[i - 1]):
                continue

            # 金叉
            if ma_s.iloc[i - 1] <= ma_l.iloc[i - 1] and ma_s.iloc[i] > ma_l.iloc[i]:
                gap_pct = (ma_s.iloc[i] - ma_l.iloc[i]) / ma_l.iloc[i] * 100
                points.append(TradePoint(
                    date=bars.iloc[i]['trade_date'],
                    action="BUY",
                    reason=f"MA{self.short}上穿MA{self.long}，偏离{gap_pct:.1f}%",
                    confidence=min(1.0, 0.5 + gap_pct / 5),
                    price=float(close.iloc[i]),
                    rule_name=self.name,
                ))
            # 死叉
            elif ma_s.iloc[i - 1] >= ma_l.iloc[i - 1] and ma_s.iloc[i] < ma_l.iloc[i]:
                gap_pct = (ma_l.iloc[i] - ma_s.iloc[i]) / ma_l.iloc[i] * 100
                points.append(TradePoint(
                    date=bars.iloc[i]['trade_date'],
                    action="SELL",
                    reason=f"MA{self.short}下穿MA{self.long}，偏离{gap_pct:.1f}%",
                    confidence=min(1.0, 0.5 + gap_pct / 5),
                    price=float(close.iloc[i]),
                    rule_name=self.name,
                ))

        return points


# ============================================================
# MACD 趋势
# ============================================================

class MACDTrendRule(SignalRule):
    """MACD 柱状线由负转正 → 买入；由正转负 → 卖出"""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    @property
    def name(self) -> str:
        return f"MACD({self.fast},{self.slow},{self.signal})"

    def evaluate(self, bars: pd.DataFrame) -> List[TradePoint]:
        if len(bars) < self.slow + self.signal:
            return []

        close = bars['close'].astype(float)
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=self.signal, adjust=False).mean()
        hist = (dif - dea) * 2  # 柱状线
        points = []

        for i in range(1, len(hist)):
            if pd.isna(hist.iloc[i]) or pd.isna(hist.iloc[i - 1]):
                continue
            # 由负转正
            if hist.iloc[i - 1] <= 0 and hist.iloc[i] > 0:
                points.append(TradePoint(
                    date=bars.iloc[i]['trade_date'],
                    action="BUY",
                    reason=f"MACD柱状线翻红({hist.iloc[i]:.3f})",
                    confidence=min(1.0, 0.5 + abs(float(hist.iloc[i])) / close.iloc[i] * 100),
                    price=float(close.iloc[i]),
                    rule_name=self.name,
                ))
            # 由正转负
            elif hist.iloc[i - 1] >= 0 and hist.iloc[i] < 0:
                points.append(TradePoint(
                    date=bars.iloc[i]['trade_date'],
                    action="SELL",
                    reason=f"MACD柱状线翻绿({hist.iloc[i]:.3f})",
                    confidence=min(1.0, 0.5 + abs(float(hist.iloc[i])) / close.iloc[i] * 100),
                    price=float(close.iloc[i]),
                    rule_name=self.name,
                ))

        return points


# ============================================================
# 布林带突破/回归
# ============================================================

class BollingerBreakRule(SignalRule):
    """价格在布林带下轨附近 → 买入；在上轨附近 → 卖出

    使用"状态机"模式：进入区域触发一次，退出后可再次触发。
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0,
                 buy_zone: float = 0.1, sell_zone: float = 0.9):
        self.period = period
        self.std_dev = std_dev
        self.buy_zone = buy_zone    # 位置 <= 此值视为下轨附近
        self.sell_zone = sell_zone  # 位置 >= 此值视为上轨附近

    @property
    def name(self) -> str:
        return f"布林({self.period},{self.std_dev}σ)"

    def evaluate(self, bars: pd.DataFrame) -> List[TradePoint]:
        if len(bars) < self.period + 1:
            return []

        close = bars['close'].astype(float)
        mid = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()
        upper = mid + self.std_dev * std
        lower = mid - self.std_dev * std
        points = []
        in_buy_zone = False
        in_sell_zone = False

        for i in range(1, len(bars)):
            if pd.isna(upper.iloc[i]) or pd.isna(lower.iloc[i]):
                continue
            band_width = upper.iloc[i] - lower.iloc[i]
            if band_width <= 0:
                continue
            pos = (float(close.iloc[i]) - lower.iloc[i]) / band_width

            # 进入下轨区域 → 买入
            if pos <= self.buy_zone and not in_buy_zone:
                in_buy_zone = True
                points.append(TradePoint(
                    date=bars.iloc[i]['trade_date'],
                    action="BUY",
                    reason=f"布林下轨附近，位置{pos:.0%}",
                    confidence=min(1.0, 0.6 + (self.buy_zone - pos) * 2),
                    price=float(close.iloc[i]),
                    rule_name=self.name,
                ))
            elif pos > self.buy_zone:
                in_buy_zone = False

            # 进入上轨区域 → 卖出
            if pos >= self.sell_zone and not in_sell_zone:
                in_sell_zone = True
                points.append(TradePoint(
                    date=bars.iloc[i]['trade_date'],
                    action="SELL",
                    reason=f"布林上轨附近，位置{pos:.0%}",
                    confidence=min(1.0, 0.6 + (pos - self.sell_zone) * 2),
                    price=float(close.iloc[i]),
                    rule_name=self.name,
                ))
            elif pos < self.sell_zone:
                in_sell_zone = False

        return points


# ============================================================
# 放量突破
# ============================================================

class VolumeBreakoutRule(SignalRule):
    """成交量突破 N 日均量 M 倍 + 价格上涨 → 买入"""

    def __init__(self, volume_mult: float = 1.5, lookback: int = 20, price_threshold: float = 0.01):
        self.volume_mult = volume_mult
        self.lookback = lookback
        self.price_threshold = price_threshold

    @property
    def name(self) -> str:
        return f"放量突破({self.volume_mult}x/{self.lookback}日)"

    def evaluate(self, bars: pd.DataFrame) -> List[TradePoint]:
        if len(bars) < self.lookback + 1:
            return []

        close = bars['close'].astype(float)
        volume = bars['volume'].astype(float)
        avg_vol = volume.rolling(self.lookback).mean()
        points = []

        for i in range(1, len(bars)):
            if pd.isna(avg_vol.iloc[i]) or avg_vol.iloc[i] == 0:
                continue
            vol_ratio = volume.iloc[i] / avg_vol.iloc[i]
            price_chg = (close.iloc[i] - close.iloc[i - 1]) / close.iloc[i - 1]

            # 放量上涨
            if vol_ratio >= self.volume_mult and price_chg > self.price_threshold:
                points.append(TradePoint(
                    date=bars.iloc[i]['trade_date'],
                    action="BUY",
                    reason=f"放量{vol_ratio:.1f}倍+涨{price_chg:.1%}",
                    confidence=min(1.0, 0.4 + vol_ratio / 5 + price_chg * 5),
                    price=float(close.iloc[i]),
                    rule_name=self.name,
                ))
            # 放量下跌
            elif vol_ratio >= self.volume_mult and price_chg < -self.price_threshold:
                points.append(TradePoint(
                    date=bars.iloc[i]['trade_date'],
                    action="SELL",
                    reason=f"放量{vol_ratio:.1f}倍+跌{price_chg:.1%}",
                    confidence=min(1.0, 0.4 + vol_ratio / 5 + abs(price_chg) * 5),
                    price=float(close.iloc[i]),
                    rule_name=self.name,
                ))

        return points


# ============================================================
# KDJ 金叉/死叉
# ============================================================

class KDJCrossRule(SignalRule):
    """KDJ 指标 K 线上穿 D 线（超卖区）→ 买入；下穿（超买区）→ 卖出"""

    def __init__(self, period: int = 9, k_period: int = 3, d_period: int = 3):
        self.period = period
        self.k_period = k_period
        self.d_period = d_period

    @property
    def name(self) -> str:
        return f"KDJ({self.period})"

    def evaluate(self, bars: pd.DataFrame) -> List[TradePoint]:
        if len(bars) < self.period + self.k_period + self.d_period:
            return []

        high = bars['high'].astype(float)
        low = bars['low'].astype(float)
        close = bars['close'].astype(float)

        # 计算 RSV
        lowest = low.rolling(self.period).min()
        highest = high.rolling(self.period).max()
        rsv = (close - lowest) / (highest - lowest).replace(0, np.nan) * 100

        # K, D
        k = rsv.ewm(com=self.k_period - 1, adjust=False).mean()
        d = k.ewm(com=self.d_period - 1, adjust=False).mean()
        j = 3 * k - 2 * d

        points = []
        for i in range(1, len(bars)):
            if pd.isna(k.iloc[i]) or pd.isna(d.iloc[i]):
                continue
            # K 上穿 D 且在超卖区
            if k.iloc[i - 1] <= d.iloc[i - 1] and k.iloc[i] > d.iloc[i] and k.iloc[i] < 30:
                points.append(TradePoint(
                    date=bars.iloc[i]['trade_date'],
                    action="BUY",
                    reason=f"KDJ金叉(K={k.iloc[i]:.0f},D={d.iloc[i]:.0f},J={j.iloc[i]:.0f})",
                    confidence=min(1.0, 0.5 + (30 - k.iloc[i]) / 50),
                    price=float(close.iloc[i]),
                    rule_name=self.name,
                ))
            # K 下穿 D 且在超买区
            elif k.iloc[i - 1] >= d.iloc[i - 1] and k.iloc[i] < d.iloc[i] and k.iloc[i] > 70:
                points.append(TradePoint(
                    date=bars.iloc[i]['trade_date'],
                    action="SELL",
                    reason=f"KDJ死叉(K={k.iloc[i]:.0f},D={d.iloc[i]:.0f},J={j.iloc[i]:.0f})",
                    confidence=min(1.0, 0.5 + (k.iloc[i] - 70) / 50),
                    price=float(close.iloc[i]),
                    rule_name=self.name,
                ))

        return points


# ============================================================
# 规则工厂
# ============================================================

RULE_REGISTRY: Dict[RuleType, Type[SignalRule]] = {
    RuleType.RSI_REVERSAL: RSIReversalRule,
    RuleType.MA_CROSS: MACrossRule,
    RuleType.MACD_TREND: MACDTrendRule,
    RuleType.BOLL_BREAK: BollingerBreakRule,
    RuleType.VOLUME_BREAKOUT: VolumeBreakoutRule,
    RuleType.KDJ_CROSS: KDJCrossRule,
}


def create_rule(config: SignalRuleConfig) -> SignalRule:
    """从配置创建规则实例"""
    cls = RULE_REGISTRY.get(config.rule_type)
    if cls is None:
        raise ValueError(f"未知规则类型: {config.rule_type}")
    return cls(**config.params)


def evaluate_all_rules(bars: pd.DataFrame, configs: List[SignalRuleConfig]) -> List[TradePoint]:
    """对单只股票运行所有规则，合并结果

    保证先买后卖：过滤掉第一个 BUY 之前的 SELL 信号。
    同日同动作去重。
    """
    all_points = []
    for cfg in configs:
        try:
            rule = create_rule(cfg)
            points = rule.evaluate(bars)
            all_points.extend(points)
        except Exception as e:
            print(f"[SignalRule] {cfg.rule_type.value} 执行失败: {e}")
    # 按日期排序
    all_points.sort(key=lambda p: p.date)

    # 先买后卖：找到第一个 BUY，丢弃其之前的所有 SELL
    first_buy_idx = None
    for i, p in enumerate(all_points):
        if p.action == "BUY":
            first_buy_idx = i
            break
    if first_buy_idx is not None:
        all_points = all_points[first_buy_idx:]
    else:
        # 没有任何 BUY 信号 → 全部丢弃（不允许空头卖出）
        return []

    # 同日同动作去重
    seen = set()
    deduped = []
    for p in all_points:
        d = p.date.date() if hasattr(p.date, 'date') else p.date
        key = (d, p.action)
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    return deduped
