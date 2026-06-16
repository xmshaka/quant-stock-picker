"""
三层过滤信号架构 — 替代单指标独立触发

Layer 1: 趋势过滤 — 股票是否处于可交易趋势中？
Layer 2: 策略匹配 — 价格行为是否匹配策略类型？
Layer 3: 多条件共振 — ≥2 个技术条件同时确认？

用法:
    from signals.layers import evaluate_layered
    points = evaluate_layered(bars, strategy_type="pullback")
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from signals.rules import TradePoint


class StrategyType(str, Enum):
    TREND_MOMENTUM = "trend_momentum"
    PULLBACK = "pullback"
    BREAKOUT = "breakout"
    BALANCED = "balanced"


@dataclass
class ConditionResult:
    name: str
    met: bool
    value: float
    threshold: float
    direction: str
    confidence: float


# ═══════════════════════════════════════════
# Layer 1: 趋势过滤器
# ═══════════════════════════════════════════
class TrendFilter:
    """检查：MA20>MA60, Price>MA20, ADX>20, 非持续下跌"""

    def __init__(self, ma_short=20, ma_long=60, adx_period=14, adx_threshold=20.0):
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold

    def check(self, bars: pd.DataFrame, idx: int) -> Tuple[bool, str, float]:
        if idx < self.ma_long:
            return False, "数据不足", 0.0
        window = bars.iloc[:idx + 1]
        close = window['close'].astype(float)
        high = window['high'].astype(float) if 'high' in window.columns else close
        low = window['low'].astype(float) if 'low' in window.columns else close

        ma20 = close.rolling(self.ma_short).mean().iloc[-1]
        ma60 = close.rolling(self.ma_long).mean().iloc[-1] if len(close) >= self.ma_long else np.nan
        score = 0.0
        reasons = []

        if close.iloc[-1] > ma20:
            score += 0.4
            reasons.append("价格>MA20")
        else:
            reasons.append("价格<MA20")

        if not np.isnan(ma60):
            if ma20 > ma60:
                score += 0.3
                reasons.append("MA20>MA60")

        adx_val = self._calc_adx(high, low, close, self.adx_period)
        if adx_val > self.adx_threshold:
            score += 0.2
            reasons.append(f"ADX={adx_val:.0f}")
        else:
            reasons.append(f"ADX弱={adx_val:.0f}")

        if len(close) >= 20 and close.iloc[-1] > low.iloc[-20:].min() * 1.02:
            score += 0.1

        passed = score >= 0.5
        reason = "✓ " + ", ".join(reasons) if passed else "✗ " + ", ".join(reasons)
        return passed, reason, score

    @staticmethod
    def _calc_adx(high, low, close, period=14):
        if len(close) < period + 1:
            return 0.0
        tr = pd.concat([
            high.diff().abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        plus_dm[plus_dm <= minus_dm] = 0
        minus_dm[minus_dm <= plus_dm] = 0
        atr = tr.ewm(span=period, adjust=False).mean()
        pdi = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
        mdi = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
        dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
        return float(dx.ewm(span=period, adjust=False).mean().iloc[-1]) if not pd.isna(dx.iloc[-1]) else 0.0


# ═══════════════════════════════════════════
# Layer 2: 策略匹配
# ═══════════════════════════════════════════
class StrategyMatcher:
    def __init__(self, strategy_type: StrategyType):
        self.strategy_type = strategy_type

    def match(self, bars: pd.DataFrame, idx: int) -> Tuple[bool, str, float]:
        if idx < 20:
            return False, "数据不足", 0.0
        window = bars.iloc[:idx + 1]
        close = window['close'].astype(float)
        volume = window['volume'].astype(float) if 'volume' in window.columns else pd.Series([0] * len(close))

        if self.strategy_type == StrategyType.TREND_MOMENTUM:
            return self._match_trend_momentum(close, idx)
        elif self.strategy_type == StrategyType.PULLBACK:
            return self._match_pullback(close, idx)
        elif self.strategy_type == StrategyType.BREAKOUT:
            return self._match_breakout(close, volume, idx)
        else:
            return self._match_any(close, volume, idx)

    def _match_trend_momentum(self, close, idx):
        hh20 = close.iloc[-20:].max()
        current = close.iloc[-1]
        dist = (hh20 - current) / hh20
        mom5 = (current / close.iloc[-6] - 1) if len(close) >= 6 else 0
        mom20 = (current / close.iloc[-21] - 1) if len(close) >= 21 else 0
        if dist < 0.05 and mom5 > 0.01 and mom20 > 0.02:
            conf = min(1.0, (0.05 - dist) * 10 + mom5 * 5 + 0.3)
            return True, f"强势追涨(距高点{dist:.1%},M5={mom5:.1%})", conf
        return False, "非追涨形态", 0.0

    def _match_pullback(self, close, idx):
        hh20 = close.iloc[-20:].max()
        current = close.iloc[-1]
        pb = (hh20 - current) / hh20
        rsi = self._calc_rsi(close, 14)
        if pb > 0.05 and (rsi < 40 or pb > 0.10):
            conf = min(1.0, pb * 5 + (40 - rsi) / 40 + 0.3) if not np.isnan(rsi) else 0.5
            rsi_str = f"RSI={rsi:.0f}" if not np.isnan(rsi) else ""
            return True, f"回调低吸(回撤{pb:.1%},{rsi_str})", conf
        return False, f"回撤不足{pb:.1%}", 0.0

    def _match_breakout(self, close, volume, idx):
        if len(close) < 20:
            return False, "数据不足", 0.0
        prev_10 = close.iloc[-15:-5]
        if len(prev_10) < 5:
            return False, "横盘数据不足", 0.0
        range_pct = (prev_10.max() - prev_10.min()) / prev_10.mean()
        current = close.iloc[-1]
        breakout = current > prev_10.max() * 1.01
        vol_ratio = 0
        if len(volume) >= 20:
            avg_vol = volume.iloc[-20:].mean()
            vol_ratio = volume.iloc[-1] / avg_vol if avg_vol > 0 else 0
        if range_pct < 0.08 and breakout and vol_ratio > 1.3:
            return True, f"横盘突破(振幅{range_pct:.1%},量比{vol_ratio:.1f}x)", min(1.0, vol_ratio / 3 + 0.3)
        return False, "非突破形态", 0.0

    def _match_any(self, close, volume, idx):
        for fn in [self._match_trend_momentum, self._match_pullback, self._match_breakout]:
            if fn == self._match_breakout:
                matched, reason, conf = fn(close, volume, idx)
            else:
                matched, reason, conf = fn(close, idx)
            if matched:
                return True, reason, conf
        return False, "无匹配策略", 0.0

    @staticmethod
    def _calc_rsi(close, period=14):
        if len(close) < period + 1:
            return np.nan
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0
        return float(100 - 100 / (1 + avg_gain / avg_loss))


# ═══════════════════════════════════════════
# Layer 3: 多条件共振
# ═══════════════════════════════════════════
class ResonanceChecker:
    """6个技术条件 ≥ min_confirmations 个同时满足才触发"""

    def __init__(self, min_confirmations=2):
        self.min_confirmations = min_confirmations

    def check_buy(self, bars: pd.DataFrame, idx: int) -> Tuple[bool, List[ConditionResult]]:
        conditions = []
        met_count = 0

        # 1. RSI < 40
        rsi = self._calc_rsi(bars, idx, 14)
        met = rsi < 40
        conditions.append(ConditionResult(
            "RSI超卖", met, rsi, 40, "below",
            min(1.0, (40 - rsi) / 20 + 0.3) if met else 0,
        ))
        if met: met_count += 1

        # 2. MA5金叉MA20
        ma5_met = self._check_ma_cross(bars, idx, 5, 20, "up")
        ma5, ma20 = self._get_mas(bars, idx, 5, 20)
        conditions.append(ConditionResult(
            "MA金叉", ma5_met, ma5 / ma20 if ma20 > 0 else 0, 1.0, "cross_up",
            min(1.0, (ma5 / ma20 - 1) * 50 + 0.3) if ma5_met else 0,
        ))
        if ma5_met: met_count += 1

        # 3. MACD翻红
        macd_met = self._check_macd(bars, idx, "bullish")
        macd_val = self._get_macd_hist(bars, idx)
        conditions.append(ConditionResult(
            "MACD翻红", macd_met, macd_val, 0, "above",
            min(1.0, abs(macd_val) * 10 + 0.3) if macd_met else 0,
        ))
        if macd_met: met_count += 1

        # 4. 布林下轨 < 0.3
        boll_pos = self._get_boll_position(bars, idx, 20, 2.0)
        met = boll_pos < 0.3
        conditions.append(ConditionResult(
            "布林下轨", met, boll_pos, 0.3, "below",
            min(1.0, (0.3 - boll_pos) * 3 + 0.3) if met else 0,
        ))
        if met: met_count += 1

        # 5. 放量 > 1.2x
        vol_ratio = self._get_volume_ratio(bars, idx, 20)
        met = vol_ratio > 1.2
        conditions.append(ConditionResult(
            "放量", met, vol_ratio, 1.2, "above",
            min(1.0, (vol_ratio - 1) * 0.5 + 0.3) if met else 0,
        ))
        if met: met_count += 1

        # 6. KDJ金叉(K<50)
        kdj_met, k_val, d_val = self._check_kdj(bars, idx, "golden")
        conditions.append(ConditionResult(
            "KDJ金叉", kdj_met, k_val, d_val, "cross_up",
            min(1.0, (d_val - k_val) / 10 + 0.3) if kdj_met else 0,
        ))
        if kdj_met: met_count += 1

        return met_count >= self.min_confirmations, conditions

    def check_sell(self, bars: pd.DataFrame, idx: int) -> Tuple[bool, List[ConditionResult]]:
        conditions = []
        met_count = 0

        # 1. RSI > 70
        rsi = self._calc_rsi(bars, idx, 14)
        met = rsi > 70
        conditions.append(ConditionResult(
            "RSI超买", met, rsi, 70, "above",
            min(1.0, (rsi - 70) / 20 + 0.3) if met else 0,
        ))
        if met: met_count += 1

        # 2. MA5死叉MA20
        ma5, ma20 = self._get_mas(bars, idx, 5, 20)
        ma_met = ma5 < ma20 if ma20 > 0 else False
        conditions.append(ConditionResult(
            "MA死叉", ma_met, ma20 / ma5 if ma5 > 0 else 0, 1.0, "cross_down",
            min(1.0, (ma20 / ma5 - 1) * 50 + 0.3) if ma_met else 0,
        ))
        if ma_met: met_count += 1

        # 3. MACD翻绿
        macd_met = self._check_macd(bars, idx, "bearish")
        macd_val = self._get_macd_hist(bars, idx)
        conditions.append(ConditionResult(
            "MACD翻绿", macd_met, macd_val, 0, "below",
            min(1.0, abs(macd_val) * 10 + 0.3) if macd_met else 0,
        ))
        if macd_met: met_count += 1

        # 4. 布林上轨 > 0.8
        boll_pos = self._get_boll_position(bars, idx, 20, 2.0)
        met = boll_pos > 0.8
        conditions.append(ConditionResult(
            "布林上轨", met, boll_pos, 0.8, "above",
            min(1.0, (boll_pos - 0.8) * 5 + 0.3) if met else 0,
        ))
        if met: met_count += 1

        # 5. 放量下跌
        vol_ratio = self._get_volume_ratio(bars, idx, 20)
        price_down = bars['close'].astype(float).iloc[idx] < bars['close'].astype(float).iloc[idx - 1] if idx >= 1 else False
        met = vol_ratio > 1.2 and price_down
        conditions.append(ConditionResult(
            "放量下跌", met, vol_ratio, 1.2, "above",
            min(1.0, (vol_ratio - 1) * 0.5 + 0.3) if met else 0,
        ))
        if met: met_count += 1

        # 6. KDJ死叉(K>50)
        kdj_met, k_val, d_val = self._check_kdj(bars, idx, "death")
        conditions.append(ConditionResult(
            "KDJ死叉", kdj_met, k_val, d_val, "cross_down",
            min(1.0, abs(k_val - d_val) / 10 + 0.3) if kdj_met else 0,
        ))
        if kdj_met: met_count += 1

        return met_count >= self.min_confirmations, conditions

    # ── 工具方法 ──
    @staticmethod
    def _calc_rsi(bars, idx, period=14):
        if idx < period: return np.nan
        close = bars['close'].astype(float).iloc[:idx + 1]
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
        return 100.0 if avg_loss == 0 else float(100 - 100 / (1 + avg_gain / avg_loss))

    @staticmethod
    def _get_mas(bars, idx, short, long):
        close = bars['close'].astype(float).iloc[:idx + 1]
        return float(close.rolling(short).mean().iloc[-1]), float(close.rolling(long).mean().iloc[-1])

    @staticmethod
    def _check_ma_cross(bars, idx, short, long, direction):
        if idx < long + 1: return False
        close = bars['close'].astype(float).iloc[:idx + 1]
        ma_s = close.rolling(short).mean()
        ma_l = close.rolling(long).mean()
        if pd.isna(ma_s.iloc[-1]) or pd.isna(ma_l.iloc[-1]): return False
        if pd.isna(ma_s.iloc[-2]) or pd.isna(ma_l.iloc[-2]): return False
        if direction == "up":
            return ma_s.iloc[-2] <= ma_l.iloc[-2] and ma_s.iloc[-1] > ma_l.iloc[-1]
        return ma_s.iloc[-2] >= ma_l.iloc[-2] and ma_s.iloc[-1] < ma_l.iloc[-1]

    @staticmethod
    def _check_macd(bars, idx, direction):
        if idx < 35: return False
        close = bars['close'].astype(float).iloc[:idx + 1]
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        if direction == "bullish":
            return dif.iloc[-1] > dea.iloc[-1] and dif.iloc[-2] <= dea.iloc[-2]
        return dif.iloc[-1] < dea.iloc[-1] and dif.iloc[-2] >= dea.iloc[-2]

    @staticmethod
    def _get_macd_hist(bars, idx):
        if idx < 35: return 0.0
        close = bars['close'].astype(float).iloc[:idx + 1]
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        return float((dif.iloc[-1] - dea.iloc[-1]) * 2)

    @staticmethod
    def _get_boll_position(bars, idx, period=20, std_dev=2.0):
        if idx < period: return 0.5
        close = bars['close'].astype(float).iloc[:idx + 1]
        mid = close.rolling(period).mean().iloc[-1]
        std = close.rolling(period).std().iloc[-1]
        upper = mid + std_dev * std
        lower = mid - std_dev * std
        return 0.5 if upper - lower == 0 else float((close.iloc[-1] - lower) / (upper - lower))

    @staticmethod
    def _get_volume_ratio(bars, idx, period=20):
        if idx < period: return 1.0
        volume = bars['volume'].astype(float).iloc[:idx + 1]
        avg_vol = volume.iloc[-period:].mean()
        return 1.0 if avg_vol == 0 else float(volume.iloc[-1] / avg_vol)

    @staticmethod
    def _check_kdj(bars, idx, direction):
        if idx < 12: return False, 0, 0
        high = bars['high'].astype(float).iloc[:idx + 1]
        low = bars['low'].astype(float).iloc[:idx + 1]
        close = bars['close'].astype(float).iloc[:idx + 1]
        lowest = low.rolling(9).min()
        highest = high.rolling(9).max()
        rsv = (close - lowest) / (highest - lowest).replace(0, np.nan) * 100
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        k_val, d_val = float(k.iloc[-1]), float(d.iloc[-1])
        if pd.isna(k.iloc[-1]) or pd.isna(d.iloc[-1]): return False, 0, 0
        if direction == "golden":
            return k.iloc[-1] > d.iloc[-1] and k.iloc[-2] <= d.iloc[-2] and k.iloc[-1] < 50, k_val, d_val
        return k.iloc[-1] < d.iloc[-1] and k.iloc[-2] >= d.iloc[-2] and k.iloc[-1] > 50, k_val, d_val


# ═══════════════════════════════════════════
# 三层过滤主函数
# ═══════════════════════════════════════════
def evaluate_layered(
    bars: pd.DataFrame,
    strategy_type: str = "balanced",
    min_confirmations: int = 2,
    trend_filter: Optional[TrendFilter] = None,
    strategy_matcher: Optional[StrategyMatcher] = None,
    resonance_checker: Optional[ResonanceChecker] = None,
) -> List[TradePoint]:
    """三层过滤信号评估 → List[TradePoint]

    对单只股票的K线逐日遍历，通过三层过滤生成买卖点：
    L1 趋势过滤 → L2 策略匹配 → L3 多条件共振
    卖出信号优先检查，状态机避免连续同向信号。
    """
    if bars.empty or len(bars) < 20:
        return []

    tf = trend_filter or TrendFilter()
    sm = strategy_matcher or StrategyMatcher(StrategyType(strategy_type))
    rc = resonance_checker or ResonanceChecker(min_confirmations)

    close = bars['close'].astype(float)
    points = []
    last_action = None

    for i in range(20, len(bars)):
        current_date = bars.iloc[i]['trade_date']
        if hasattr(current_date, 'date'):
            current_date = current_date.date()

        # ── L1: 趋势过滤 ──
        trend_ok, _, _ = tf.check(bars, i)
        if not trend_ok:
            continue

        # ── L2: 策略匹配 ──
        strategy_ok, _, _ = sm.match(bars, i)
        if not strategy_ok:
            continue

        # ── L3: 多条件共振 — 卖出优先 ──
        sell_ok, sell_conds = rc.check_sell(bars, i)
        if sell_ok and last_action == "BUY":
            sell_met = [c for c in sell_conds if c.met]
            reason = " + ".join([f"{c.name}({c.value:.1f})" for c in sell_met])
            conf = float(np.mean([c.confidence for c in sell_met])) if sell_met else 0.5
            points.append(TradePoint(
                date=current_date, action="SELL",
                reason=f"L3共振卖出({len(sell_met)}/6): {reason}",
                confidence=min(1.0, conf), price=float(close.iloc[i]),
                rule_name=f"三层过滤-{strategy_type}",
            ))
            last_action = "SELL"
            continue

        buy_ok, buy_conds = rc.check_buy(bars, i)
        if buy_ok and last_action != "BUY":
            buy_met = [c for c in buy_conds if c.met]
            reason = " + ".join([f"{c.name}({c.value:.1f})" for c in buy_met])
            conf = float(np.mean([c.confidence for c in buy_met])) if buy_met else 0.5
            points.append(TradePoint(
                date=current_date, action="BUY",
                reason=f"L3共振买入({len(buy_met)}/6): {reason}",
                confidence=min(1.0, conf), price=float(close.iloc[i]),
                rule_name=f"三层过滤-{strategy_type}",
            ))
            last_action = "BUY"

    return points
