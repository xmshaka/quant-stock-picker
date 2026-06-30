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

from signals.rules import TradePoint, apply_confidence_audit
from strategy.schemes import BUILTIN_SCHEMES, ResonanceConfig


class StrategyType(str, Enum):
    TREND_MOMENTUM = "trend_momentum"
    PULLBACK = "pullback"
    BREAKOUT = "breakout"
    BALANCED = "balanced"


@dataclass
class ConditionResult:
    key: str
    name: str
    met: bool
    value: float
    threshold: float
    direction: str
    confidence: float

    def audit_text(self) -> str:
        """面向买卖点复盘的可审计文案，显式展示当前值、阈值和方向。

        旧文案只显示 `RSI超卖(39.6)`，容易让用户误解 39.6 是阈值或
        专业意义上的“超卖”。这里统一输出 `名称：值 < 阈值` 等格式。
        """
        if self.direction in {"below", "above"}:
            op = "<" if self.direction == "below" else ">"
            return f"{self.name}：{self.value:.1f} {op} {self.threshold:g}"
        if self.direction == "cross_up":
            return f"{self.name}：{self.value:.2f} 上穿/高于 {self.threshold:.2f}"
        if self.direction == "cross_down":
            return f"{self.name}：{self.value:.2f} 下穿/低于 {self.threshold:.2f}"
        return f"{self.name}：{self.value:.1f}"


def liquidity_audit_tags(bars: pd.DataFrame, idx: int, *, period: int = 20) -> List[str]:
    """买点审计用量能/成交额/换手率标签。

    第一阶段只作为审计与风险提示，不作为硬过滤，避免一次性改变信号。
    """
    if bars is None or bars.empty or idx < 0 or idx >= len(bars):
        return []
    tags: List[str] = []
    row = bars.iloc[idx]
    volume = pd.to_numeric(bars.get("volume", pd.Series(dtype=float)), errors="coerce")
    if len(volume) > idx and idx >= period:
        avg_vol = float(volume.iloc[max(0, idx - period + 1):idx + 1].mean() or 0.0)
        vol = float(volume.iloc[idx] or 0.0)
        vol_ratio = vol / avg_vol if avg_vol > 0 else 0.0
        if vol_ratio < 0.7:
            level = "明显缩量"
        elif vol_ratio < 1.2:
            level = "温和量"
        elif vol_ratio < 1.8:
            level = "放量"
        else:
            level = "明显放量"
        tags.append(f"量能{level}：量比{vol_ratio:.2f}x")
    amount = row.get("amount", np.nan)
    try:
        amount_val = float(amount)
    except Exception:
        amount_val = np.nan
    if np.isfinite(amount_val) and amount_val > 0:
        tags.append(f"成交额{amount_val / 1e8:.2f}亿")
    turnover = row.get("turnover_rate", row.get("turnover", np.nan))
    try:
        turnover_val = float(turnover)
    except Exception:
        turnover_val = np.nan
    if np.isfinite(turnover_val) and turnover_val > 0:
        # 数据源可能给 0.05 或 5 两种口径；审计显示统一成百分数。
        turnover_pct = turnover_val * 100 if turnover_val <= 1 else turnover_val
        if turnover_pct < 0.5:
            level = "冷清"
        elif turnover_pct < 2:
            level = "温和"
        elif turnover_pct < 8:
            level = "活跃"
        elif turnover_pct < 15:
            level = "高换手"
        else:
            level = "极端换手"
        tags.append(f"换手{level}：{turnover_pct:.2f}%")
    else:
        tags.append("换手率缺失")
    return tags


# ═══════════════════════════════════════════
# Layer 1: 趋势过滤器
# ═══════════════════════════════════════════
class TrendFilter:
    """检查：MA20>MA40, Price>MA20, ADX>20, 非持续下跌

    ma_long=40 适配 ≤20 日短线持仓周期。
    strategy_type 影响评分逻辑：
    - pullback/balanced: 回调策略允许 Price<MA20，重点看 MA20>MA40 确认上升趋势
    - trend_momentum/breakout: 维持原有严格逻辑，必须 Price>MA20
    """

    # 策略类型对评分的影响
    PULLBACK_LIKE = {"pullback", "balanced"}

    def __init__(self, ma_short=20, ma_long=40, adx_period=14, adx_threshold=20.0,
                 strategy_type: str = "balanced"):
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.strategy_type = strategy_type
        self._is_pullback = strategy_type in self.PULLBACK_LIKE

    def check(self, bars: pd.DataFrame, idx: int) -> Tuple[bool, float, str]:
        """L1 趋势过滤：连续型评分，对齐 signals.scanner._check_layer1()。"""
        if idx < self.ma_long:
            return False, 0.0, "数据不足"

        def _clamp01(v: float, lo: float, hi: float) -> float:
            return max(0.0, min(1.0, (v - lo) / max(hi - lo, 1e-9)))

        window = bars.iloc[:idx + 1]
        close = window['close'].astype(float)
        current = float(close.iloc[-1])
        ma20 = float(close.rolling(self.ma_short).mean().iloc[-1])
        ma40 = float(close.rolling(self.ma_long).mean().iloc[-1]) if len(close) >= self.ma_long else np.nan
        low20 = float(close.iloc[-20:].min())
        pos_ma20 = (current / ma20 - 1) if ma20 > 0 else 0.0
        ma20_ma40_ratio = (ma20 / ma40 - 1) if ma40 and ma40 > 0 else 0.0
        above_low20_ratio = (current / low20 - 1) if low20 > 0 else 0.0

        if self._is_pullback:
            recent_ma20 = close.rolling(self.ma_short).mean().iloc[-10:] if len(close) >= 10 else None
            had_uptrend = False
            if recent_ma20 is not None and len(recent_ma20) > 0:
                recent_close = close.iloc[-10:]
                for j in range(len(recent_close)):
                    if not pd.isna(recent_ma20.iloc[j]) and recent_close.iloc[j] > recent_ma20.iloc[j]:
                        had_uptrend = True
                        break
            trend_ok = had_uptrend and not pd.isna(ma40) and ma20 >= ma40 * 0.995 and current > low20 * 1.03
            s_uptrend = 35.0 if had_uptrend else 0.0
            s_ma40 = 30.0 * _clamp01(ma20_ma40_ratio, -0.005, 0.03)
            s_low20 = 35.0 * _clamp01(above_low20_ratio, 0.03, 0.12)
            score = s_uptrend + s_ma40 + s_low20
            return trend_ok, score, "L1上升趋势回调未破位"

        trend_ok = (current > ma20 and not pd.isna(ma40) and ma20 > ma40
                    and current > low20 * 1.03)
        s_ma20 = 40.0 * _clamp01(pos_ma20, 0.0, 0.10)
        s_ma40 = 35.0 * _clamp01(ma20_ma40_ratio, 0.0, 0.05)
        s_low20 = 25.0 * _clamp01(above_low20_ratio, 0.03, 0.15)
        score = s_ma20 + s_ma40 + s_low20
        return trend_ok, score, "L1价格在MA20上方且MA20高于MA40"

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

    def match(self, bars: pd.DataFrame, idx: int) -> Tuple[bool, float, str]:
        """L2 策略匹配：连续评分 0-100，对齐 signals.scanner._check_layer2()。

        返回 (ok, l2_score, reason)。
        """
        if idx < 20:
            return False, 0.0, "数据不足"
        window = bars.iloc[:idx + 1]
        close = window['close'].astype(float)
        volume = window['volume'].astype(float) if 'volume' in window.columns else pd.Series([0] * len(close))

        if self.strategy_type == StrategyType.TREND_MOMENTUM:
            return self._match_trend_momentum(close)
        elif self.strategy_type == StrategyType.PULLBACK:
            return self._match_pullback(close)
        elif self.strategy_type == StrategyType.BREAKOUT:
            return self._match_breakout(close, volume)
        else:
            return self._match_any(close, volume)

    def _match_trend_momentum(self, close) -> Tuple[bool, float, str]:
        current = float(close.iloc[-1])
        hh20 = float(close.iloc[-20:].max())
        dist = (hh20 - current) / hh20 if hh20 > 0 else 1.0
        mom5 = current / float(close.iloc[-6]) - 1 if len(close) >= 6 and close.iloc[-6] > 0 else 0.0
        mom20 = current / float(close.iloc[-21]) - 1 if len(close) >= 21 and close.iloc[-21] > 0 else 0.0
        ok = dist <= 0.05 and mom5 > 0.01 and mom20 > 0.02
        score = min(100.0, max(0.0, (0.05 - dist) * 600 + mom5 * 600 + mom20 * 300))
        return ok, score, f"强势追涨：距20日高点{dist:.1%}，M5={mom5:.1%}，M20={mom20:.1%}"

    def _match_pullback(self, close) -> Tuple[bool, float, str]:
        hh20 = float(close.iloc[-20:].max())
        current = float(close.iloc[-1])
        low20 = float(close.iloc[-20:].min())
        pb = (hh20 - current) / hh20 if hh20 > 0 else 0.0
        rsi = self._calc_rsi(close, 14)
        # FIX: 增加 pb<=0.15 上界 + current>low20*1.03 支撑检查，对齐 scanner
        ok = 0.05 <= pb <= 0.15 and current > low20 * 1.03 and (rsi < 45 or pb >= 0.08)
        rsi_val = rsi if not np.isnan(rsi) else 45
        score = min(100.0, max(0.0, 45 + pb * 220 + max(0, 45 - rsi_val) * 1.2))
        return ok, score, f"回调低吸：回撤{pb:.1%}，RSI={rsi_val:.0f}"

    def _match_breakout(self, close, volume) -> Tuple[bool, float, str]:
        if len(close) < 20:
            return False, 0.0, "数据不足"
        prev_10 = close.iloc[-15:-5]
        if len(prev_10) < 5:
            return False, 0.0, "横盘数据不足"
        range_pct = (prev_10.max() - prev_10.min()) / prev_10.mean() if prev_10.mean() > 0 else 1.0
        current = float(close.iloc[-1])
        breakout = current > float(prev_10.max()) * 1.01
        vol_ratio = 0.0
        if len(volume) >= 20:
            avg_vol = volume.iloc[-20:].mean()
            vol_ratio = volume.iloc[-1] / avg_vol if avg_vol > 0 else 0.0
        ok = range_pct < 0.08 and breakout and vol_ratio > 1.3
        score = min(100.0, max(0.0, (0.08 - range_pct) * 500 + max(0, vol_ratio - 1.0) * 35 + (20 if breakout else 0)))
        return ok, score, f"横盘突破：振幅{range_pct:.1%}，量比{vol_ratio:.1f}x"

    def _match_any(self, close, volume) -> Tuple[bool, float, str]:
        """balanced 策略：逐个尝试三个子策略，返回首个匹配。"""
        for fn in [self._match_trend_momentum, self._match_pullback, self._match_breakout]:
            if fn == self._match_breakout:
                matched, score, reason = fn(close, volume)
            else:
                matched, score, reason = fn(close)
            if matched:
                return True, score, reason
        return False, 0.0, "无匹配策略"

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

    def __init__(self, min_confirmations=2, sell_min_confirmations=2,
                 buy_conditions: Optional[List[str]] = None, sell_conditions: Optional[List[str]] = None):
        self.min_confirmations = min_confirmations
        self.sell_min_confirmations = sell_min_confirmations
        self.buy_conditions = set(buy_conditions or [])
        self.sell_conditions = set(sell_conditions or [])

    @classmethod
    def from_strategy(cls, strategy_type: str, fallback_min_confirmations: int = 2) -> "ResonanceChecker":
        scheme = BUILTIN_SCHEMES.get(str(strategy_type))
        cfg = getattr(scheme, "resonance_config", None) if scheme else None
        if cfg is None:
            cfg = ResonanceConfig(min_confirmations=fallback_min_confirmations)
        return cls(
            min_confirmations=int(getattr(cfg, "min_confirmations", fallback_min_confirmations) or fallback_min_confirmations),
            sell_min_confirmations=int(getattr(cfg, "sell_min_confirmations", 2) or 2),
            buy_conditions=list(getattr(cfg, "buy_conditions", []) or []),
            sell_conditions=list(getattr(cfg, "sell_conditions", []) or []),
        )

    def _filter_conditions(self, conditions: List[ConditionResult], side: str) -> List[ConditionResult]:
        enabled = self.buy_conditions if side == "buy" else self.sell_conditions
        if not enabled:
            return conditions
        return [c for c in conditions if c.key in enabled]

    def _scheme_id(self) -> str:
        """根据配置 key 反推当前 L3 使用的策略族。

        P0 FIX: 旧版单股 L3 固定生成 RSI/MA/MACD/BOLL/量能/KDJ 六个通用 key，
        但 `strategy.schemes` 已改为策略专属 key，导致 buy_conditions 过滤后只剩
        0~1 个条件，trend/pullback/breakout 实际几乎无法触发。这里在单股路径中
        改为和 signal scanner 一致的策略专属条件集合。
        """
        keys = set(self.buy_conditions or [])
        if {"near_high", "momentum_20d", "rsi_not_extreme"} & keys:
            return "trend_momentum"
        if {"pullback_range", "not_break_20d_low", "volume_calm", "near_support"} & keys:
            return "pullback"
        if {"break_platform", "volume_surge", "narrow_range", "boll_upper"} & keys:
            return "breakout"
        return "balanced"

    def _strategy_buy_conditions(self, bars: pd.DataFrame, idx: int) -> Optional[List[ConditionResult]]:
        """生成策略专属 BUY 条件，与 `signals.scanner._check_layer3` key 对齐。"""
        scheme_id = self._scheme_id()
        if idx < 20 or bars is None or bars.empty or "close" not in bars:
            return []

        close = pd.to_numeric(bars["close"], errors="coerce")
        current = float(close.iloc[idx])
        if not np.isfinite(current) or current <= 0:
            return []
        volume = pd.to_numeric(bars.get("volume", pd.Series(dtype=float)), errors="coerce")
        close_window = close.iloc[:idx + 1]
        ma5 = float(close_window.rolling(5).mean().iloc[-1]) if len(close_window) >= 5 else np.nan
        ma20 = float(close_window.rolling(20).mean().iloc[-1]) if len(close_window) >= 20 else np.nan
        high20 = float(close_window.iloc[-20:].max())
        low20 = float(close_window.iloc[-20:].min())
        prev = close_window.iloc[-15:-5]
        rsi = self._calc_rsi(bars, idx, 14)
        boll_pos = self._get_boll_position(bars, idx, 20, 2.0)
        vol_ratio = self._get_volume_ratio(bars, idx, 20)
        mom5 = current / float(close_window.iloc[-6]) - 1 if len(close_window) >= 6 and close_window.iloc[-6] > 0 else 0.0
        mom20 = current / float(close_window.iloc[-21]) - 1 if len(close_window) >= 21 and close_window.iloc[-21] > 0 else 0.0
        pullback = 1 - current / high20 if high20 > 0 else 0.0
        range_pct = (float(prev.max()) - float(prev.min())) / float(prev.mean()) if len(prev) >= 5 and float(prev.mean()) > 0 else 1.0

        # 获取资金流和相对换手数据
        mf_vals = self._get_moneyflow_values(bars, idx)
        to_vals = self._get_turnover_values(bars, idx)

        def cr(key: str, name: str, met: bool, value: float, threshold: float, direction: str, conf: float) -> ConditionResult:
            return ConditionResult(key, name, bool(met), float(value), float(threshold), direction, max(0.0, min(1.0, float(conf))))

        if scheme_id == "balanced":
            # balanced v2: 策略路由器 → 评分最优子策略 → 委托其条件生成
            # 旧版问题：仅生成6个通用条件(MA5>MA20/RSI<70/放量等)，
            # 从不读取资金流/换手数据，无法区分信号质量，始终cc=3。
            # v2改进：评估三个子策略适用性 → 选最优 → 生成该策略的专属条件。
            strategy_scores = {
                'trend_momentum': 0,
                'pullback': 0,
                'breakout': 0
            }
            # 趋势动量适用性
            if mf_vals['large_elg_net_mf_amount'] > 50000 and mf_vals['main_net_mf_amount'] > 25000:
                strategy_scores['trend_momentum'] += 2
            if mf_vals.get('large_elg_net_mf_rank', 0) > 0.7:
                strategy_scores['trend_momentum'] += 1
            if to_vals['relative_turnover_5d'] > 1.1 and to_vals['amount_percentile_60d'] > 0.6:
                strategy_scores['trend_momentum'] += 1
            if mom5 > 0.025 and mom20 > 0.04:
                strategy_scores['trend_momentum'] += 1
            if ma5 > ma20 * 1.02:
                strategy_scores['trend_momentum'] += 1
            if 55 < rsi < 68:
                strategy_scores['trend_momentum'] += 1
            # 回调低吸适用性
            if mf_vals['main_net_mf_amount'] > -50000 and mf_vals['large_elg_net_mf_amount'] > -100000:
                strategy_scores['pullback'] += 2
            if to_vals['relative_turnover_5d'] < 0.9 and to_vals['turnover_percentile_60d'] < 0.4:
                strategy_scores['pullback'] += 1
            if vol_ratio < 1.0:
                strategy_scores['pullback'] += 1
            if 0.08 <= pullback <= 0.18:
                strategy_scores['pullback'] += 1
            if current > low20 * 1.05:
                strategy_scores['pullback'] += 1
            if ma20 > 0 and current / ma20 < 1.08:
                strategy_scores['pullback'] += 1
            if rsi < 50:
                strategy_scores['pullback'] += 1
            # 横盘突破适用性
            if mf_vals['large_elg_net_mf_amount'] > 50000 and mf_vals['main_net_mf_amount'] > 30000:
                strategy_scores['breakout'] += 2
            if to_vals['relative_turnover_5d'] > 1.2 and to_vals['amount_percentile_60d'] > 0.7:
                strategy_scores['breakout'] += 1
            if vol_ratio > 1.4:
                strategy_scores['breakout'] += 1
            if len(prev) >= 5 and current > float(prev.max()) * 1.015:
                strategy_scores['breakout'] += 1
            if range_pct < 0.10:
                strategy_scores['breakout'] += 1
            if mom5 > 0.03:
                strategy_scores['breakout'] += 1
            if ma5 > ma20 * 1.02:
                strategy_scores['breakout'] += 1
            if np.isfinite(boll_pos) and boll_pos > 0.7:
                strategy_scores['breakout'] += 1
            # 选最优，委托其条件生成
            best_strategy_name, best_score = max(strategy_scores.items(), key=lambda x: x[1])
            if best_score <= 2:
                return []  # 无子策略明确适用
            # 保存原始 buy_conditions，委托后恢复（避免污染 check_buy 的 fallback 路径）
            delegate_id = best_strategy_name
            self._original_buy_conditions = self.buy_conditions
            delegate_scheme = BUILTIN_SCHEMES.get(delegate_id)
            if delegate_scheme:
                delegate_cfg = getattr(delegate_scheme, "resonance_config", None)
                if delegate_cfg:
                    self.buy_conditions = set(getattr(delegate_cfg, "buy_conditions", []) or [])
            scheme_id = delegate_id
            # 策略路由审计条件
            self._balanced_route_audit = cr(
                "strategy_selection", f"策略路由:{best_strategy_name}({best_score})",
                True, best_score, 0, "above", min(1.0, best_score / 8))
        else:
            self._balanced_route_audit = None

        if scheme_id == "trend_momentum":
            
            # 1. 资金流确定性判断（不只是净流入，还要有质量）
            mf_strong = mf_vals['large_elg_net_mf_amount'] > 50000  # 需要显著流入
            mf_positive = mf_vals['main_net_mf_amount'] > 10000
            mf_rank_good = mf_vals['large_elg_net_mf_rank'] > 0.7
            
            # 资金流综合置信度：强流入+高排名 = 高确定性
            mf_confidence = (
                0.4 * (1 if mf_strong else 0.2) +
                0.3 * (1 if mf_positive else 0.2) +
                0.3 * min(1.0, (mf_vals['large_elg_net_mf_rank'] - 0.5) * 2)
            )
            
            # 2. 量能确定性判断（活跃但不异常）
            turnover_normal = 1.0 < to_vals['relative_turnover_5d'] < 1.4
            amount_healthy = 0.6 < to_vals['amount_percentile_60d'] < 0.85
            volume_healthy = 1.1 < vol_ratio < 1.6
            
            # 量能综合置信度
            volume_confidence = (
                0.4 * (1 if turnover_normal else 0.3) +
                0.3 * (1 if amount_healthy else 0.3) +
                0.3 * (1 if volume_healthy else 0.3)
            )
            
            # 3. 趋势确定性判断
            momentum_strong = mom5 > 0.025 and mom20 > 0.04
            ma_aligned = ma5 > ma20 * 1.02  # 显著高于
            rsi_optimal = 55 < rsi < 68  # 强势但不超买
            
            trend_confidence = (
                0.4 * min(1.0, mom5 * 15) +
                0.3 * (1 if ma_aligned else 0.2) +
                0.3 * (0.8 if rsi_optimal else 0.3)
            )
            
            conditions = [
                # 资金流条件（保持原key，优化逻辑）
                cr("large_elg_net_mf_positive", "超大单显著流入", mf_strong,
                   mf_vals['large_elg_net_mf_amount'], 50000, "above", mf_confidence * 0.4),
                
                cr("main_net_mf_positive", "主力净流入", mf_positive,
                   mf_vals['main_net_mf_amount'], 10000, "above", mf_confidence * 0.3),
                
                cr("large_elg_net_mf_rank_high", "超大单流入排名高", mf_rank_good,
                   mf_vals['large_elg_net_mf_rank'], 0.7, "above", mf_confidence * 0.3),
                
                # 量能条件
                cr("relative_turnover_5d_high", "相对换手活跃", turnover_normal,
                   to_vals['relative_turnover_5d'], 1.0, "above", volume_confidence * 0.4),
                
                cr("amount_percentile_60d_high", "成交额分位健康", amount_healthy,
                   to_vals['amount_percentile_60d'], 0.6, "above", volume_confidence * 0.3),
                
                cr("volume_expand", "温和放量", volume_healthy,
                   vol_ratio, 1.1, "above", volume_confidence * 0.3),
                
                # 趋势条件
                cr("momentum_5d_strong", "5日动量强劲", momentum_strong,
                   mom5, 0.025, "above", trend_confidence * 0.4),
                
                cr("momentum_20d_strong", "20日动量强劲", mom20 > 0.04,
                   mom20, 0.04, "above", trend_confidence * 0.3),
                
                cr("ma5_above_ma20", "MA5显著高于MA20", ma_aligned,
                   ma5 / ma20 if ma20 > 0 else 0, 1.02, "above", trend_confidence * 0.15),
                
                cr("rsi_not_extreme", "RSI强势区间", rsi_optimal,
                   rsi, 68, "below", trend_confidence * 0.15),
            ]
        elif scheme_id == "pullback":
            # 回调低吸策略：聚焦于高确定性健康回调
            
            # 1. 资金流确定性：回调中流出放缓或转正
            mf_improving = mf_vals['main_net_mf_amount'] > -50000  # 流出不超过5万
            elg_improving = mf_vals['large_elg_net_mf_amount'] > -100000  # 流出不超过10万
            
            # 资金流改善置信度
            mf_confidence = (
                0.5 * (1 if mf_improving else 0.3) +
                0.5 * (1 if elg_improving else 0.3)
            )
            
            # 2. 量能确定性：缩量回调，抛压减轻
            turnover_calm = to_vals['relative_turnover_5d'] < 0.9
            turnover_low_percentile = to_vals['turnover_percentile_60d'] < 0.4
            volume_calm = vol_ratio < 1.0
            
            # 量能置信度
            volume_confidence = (
                0.4 * (1 if turnover_calm else 0.2) +
                0.3 * (1 if turnover_low_percentile else 0.2) +
                0.3 * (1 if volume_calm else 0.2)
            )
            
            # 3. 技术确定性：关键支撑有效
            healthy_pullback = 0.08 <= pullback <= 0.18  # 健康回调幅度
            support_holding = current > low20 * 1.05  # 不破关键支撑
            near_support_ma = ma20 > 0 and current / ma20 < 1.08  # 接近均线支撑
            rsi_weak = np.isfinite(rsi) and rsi < 50  # RSI偏弱但不极端
            
            # 技术置信度
            tech_confidence = (
                0.3 * (1 if healthy_pullback else 0.2) +
                0.3 * (1 if support_holding else 0.2) +
                0.2 * (1 if near_support_ma else 0.2) +
                0.2 * (0.8 if rsi_weak else 0.3)
            )
            
            conditions = [
                # 资金流条件（保持原key，优化逻辑）
                cr("main_net_mf_negative_improving", "主力流出改善", mf_improving,
                   mf_vals['main_net_mf_amount'], -50000, "above", mf_confidence * 0.5),
                
                cr("large_elg_net_mf_negative_improving", "超大单流出改善", elg_improving,
                   mf_vals['large_elg_net_mf_amount'], -100000, "above", mf_confidence * 0.5),
                
                # 量能条件
                cr("relative_turnover_5d_low", "相对换手缩量", turnover_calm,
                   to_vals['relative_turnover_5d'], 0.9, "below", volume_confidence * 0.4),
                
                cr("turnover_percentile_60d_low", "换手率分位低", turnover_low_percentile,
                   to_vals['turnover_percentile_60d'], 0.4, "below", volume_confidence * 0.3),
                
                cr("volume_calm", "成交量温和", volume_calm,
                   vol_ratio, 1.0, "below", volume_confidence * 0.3),
                
                # 技术条件
                cr("pullback_range", "健康回调幅度", healthy_pullback,
                   pullback, 0.08, "above", tech_confidence * 0.3),
                
                cr("not_break_20d_low", "不破关键支撑", support_holding,
                   current / low20 if low20 > 0 else 0, 1.05, "above", tech_confidence * 0.3),
                
                cr("near_support", "接近均线支撑", near_support_ma,
                   current / ma20 if ma20 > 0 else 0, 1.08, "below", tech_confidence * 0.2),
                
                cr("rsi_oversold", "RSI偏弱回调", rsi_weak,
                   rsi, 50, "below", tech_confidence * 0.2),
                
                # 布林位置作为辅助确认
                cr("boll_lower", "布林下轨附近", np.isfinite(boll_pos) and boll_pos < 0.4,
                   boll_pos, 0.4, "below", 0.4 if np.isfinite(boll_pos) and boll_pos < 0.4 else 0.1),
            ]
        else:
            # ================================================================
            # 横盘突破策略 v2 — 高确定性真突破
            #
            # 旧版问题：
            #   1. 阈值过低 (vol_ratio>1.4, mom5>0.03) → 噪音信号多
            #   2. 无假突破过滤 → 日内冲高回落仍触发
            #   3. 无多时间框架确认 → 单一bar突破不可靠
            #   4. 无量价关系验证 → 缩量突破无意义
            #
            # v2改进：
            #   1. 四维确认体系：资金流→量能→价格结构→持续性
            #   2. 假突破检测：要求 close > platform_high (非intraday high)
            #   3. 多bar确认：突破需前1-2日已有蓄势信号
            #   4. 量价协同：放量+实体阳线才有效
            # ================================================================
            platform_high = float(prev.max()) if len(prev) >= 5 else high20
            platform_low = float(prev.min()) if len(prev) >= 5 else low20

            # ── 维度1: 资金流 (权重35%) ──
            # 突破必须伴随显著资金流入，不是散户行为
            mf_strong = mf_vals['large_elg_net_mf_amount'] > 100000   # 超大单>10万
            mf_good = mf_vals['large_elg_net_mf_amount'] > 50000      # 超大单>5万
            mf_main_good = mf_vals['main_net_mf_amount'] > 50000      # 主力>5万
            mf_rank_elite = mf_vals.get('large_elg_net_mf_rank', 0) > 0.80  # 排名前20%

            # ── 维度2: 量能 (权重30%) ──
            # 突破量必须显著放大，缩量突破=假突破
            vol_surge = vol_ratio > 2.0        # 量比>2x (旧1.4太宽松)
            vol_strong = vol_ratio > 1.6        # 量比>1.6x
            turnover_hot = to_vals['relative_turnover_5d'] > 1.3
            amount_top = to_vals['amount_percentile_60d'] > 0.75  # 成交额前25%
            # 量价协同：当日是实体阳线(非十字星)
            bar_open = float(bars.iloc[idx].get('open', current)) if 'open' in bars else current
            body_pct = abs(current - bar_open) / bar_open if bar_open > 0 else 0.0
            bullish_body = current > bar_open and body_pct > 0.008  # 实体>0.8%

            # ── 维度3: 价格结构 (权重25%) ──
            # 突破确认：close突破前高，不是盘中冲高
            close_breakout = current > platform_high * 1.01   # 收盘确认突破
            price_breakout = current > platform_high * 1.015  # 显著突破
            consolidation = range_pct < 0.08                  # 振幅<8% (旧10%太宽)
            # 前1-2日蓄势信号：前日量比>0.8(不冷)、前日收阳或接近平台
            if len(close_window) >= 2:
                prev_close = float(close_window.iloc[-2])
                prev_vol = float(volume.iloc[-2]) if len(volume) >= 2 else 0
                avg_vol_20 = float(volume.iloc[max(0, idx-20):idx].mean()) if len(volume) > idx else 1
                prev_vol_ratio = prev_vol / avg_vol_20 if avg_vol_20 > 0 else 0
                buildup = (prev_vol_ratio > 0.8 and prev_close / platform_high > 0.95)
            else:
                buildup = False
            # 突破后站稳：连续2日以上收盘在突破位上方
            sustained_breakout = (len(close_window) >= 3 and
                                  float(close_window.iloc[-2]) > platform_high and
                                  float(close_window.iloc[-3]) > platform_high * 0.98)

            # ── 维度4: 趋势背景 (权重10%) ──
            # 突破在上升趋势中更可靠
            ma_aligned = ma5 > ma20 * 1.01
            boll_expanding = np.isfinite(boll_pos) and 0.5 < boll_pos < 0.95  # 布林中上轨，非极端

            # ── 综合置信度 ──
            mf_conf = (
                0.35 * (1.0 if mf_strong else (0.6 if mf_good else 0.2)) +
                0.35 * (1.0 if mf_main_good else 0.3) +
                0.30 * (1.0 if mf_rank_elite else 0.3)
            )
            vol_conf = (
                0.30 * (1.0 if vol_surge else (0.5 if vol_strong else 0.1)) +
                0.25 * (1.0 if turnover_hot else 0.3) +
                0.25 * (1.0 if amount_top else 0.3) +
                0.20 * (1.0 if bullish_body else 0.2)
            )
            struct_conf = (
                0.40 * (1.0 if close_breakout else (0.5 if price_breakout else 0.1)) +
                0.25 * (1.0 if consolidation else 0.3) +
                0.20 * (1.0 if buildup else 0.2) +
                0.15 * (1.0 if sustained_breakout else 0.3)
            )
            trend_conf = (
                0.60 * (1.0 if ma_aligned else 0.3) +
                0.40 * (1.0 if boll_expanding else 0.3)
            )

            conditions = [
                # 资金流
                cr("large_elg_net_mf_positive_strong", "超大单>10万(突破)", mf_strong,
                   mf_vals['large_elg_net_mf_amount'], 100000, "above", mf_conf * 0.35),
                cr("main_net_mf_positive_strong", "主力>5万(突破)", mf_main_good,
                   mf_vals['main_net_mf_amount'], 50000, "above", mf_conf * 0.35),
                cr("mf_rank_elite", "资金排名前20%", mf_rank_elite,
                   mf_vals.get('large_elg_net_mf_rank', 0), 0.80, "above", mf_conf * 0.30),
                # 量能
                cr("volume_surge", "量比>2x(突破)", vol_surge,
                   vol_ratio, 2.0, "above", vol_conf * 0.30),
                cr("relative_turnover_5d_high", "换手活跃(突破)", turnover_hot,
                   to_vals['relative_turnover_5d'], 1.3, "above", vol_conf * 0.25),
                cr("amount_percentile_60d_high", "成交额前25%", amount_top,
                   to_vals['amount_percentile_60d'], 0.75, "above", vol_conf * 0.25),
                cr("bullish_body", "实体阳线(突破)", bullish_body,
                   body_pct, 0.008, "above", vol_conf * 0.20),
                # 价格结构
                cr("break_platform", "收盘突破平台上沿", close_breakout,
                   current / platform_high if platform_high > 0 else 0, 1.01, "above", struct_conf * 0.40),
                cr("narrow_range", "平台振幅<8%", consolidation,
                   range_pct, 0.08, "below", struct_conf * 0.25),
                cr("buildup_signal", "前日蓄势", buildup,
                   1.0, 0.5, "above", struct_conf * 0.20),
                cr("sustained_breakout", "连续站稳突破位", sustained_breakout,
                   1.0, 0.5, "above", struct_conf * 0.15),
                # 趋势背景
                cr("ma5_above_ma20", "均线多头(突破)", ma_aligned,
                   ma5 / ma20 if ma20 > 0 else 0, 1.01, "above", trend_conf * 0.60),
                cr("boll_expanding", "布林中上轨", boll_expanding,
                   boll_pos, 0.5, "above", trend_conf * 0.40),
            ]
        filtered = self._filter_conditions(conditions, "buy")
        # balanced 路由审计：不在任何策略 whitelist 中，跳过 filter 直接追加
        if getattr(self, '_balanced_route_audit', None) is not None:
            filtered.append(self._balanced_route_audit)
            self._balanced_route_audit = None
        # balanced 委托后恢复原始 buy_conditions（避免污染 check_buy fallback）
        if hasattr(self, '_original_buy_conditions'):
            self.buy_conditions = self._original_buy_conditions
            delattr(self, '_original_buy_conditions')
        return filtered

    def check_buy(self, bars: pd.DataFrame, idx: int) -> Tuple[bool, float, List[ConditionResult]]:
        """检查 BUY 条件，返回 (ok, l3_score, conditions)。

        l3_score: 0-100 连续评分，对齐 scanner._check_layer3() 的输出。
        sum(condition.confidence) / total × 100
        """
        strategy_conditions = self._strategy_buy_conditions(bars, idx)
        if strategy_conditions and len(strategy_conditions) > 0:
            met_count = sum(1 for c in strategy_conditions if c.met)
            total = len(strategy_conditions) or 1
            l3_score = sum(c.confidence for c in strategy_conditions) / total * 100.0
            ok = met_count >= self.min_confirmations
            return ok, l3_score, strategy_conditions

        conditions = []
        met_count = 0

        # 1. RSI < 40：A股短线里这只能称为“偏弱回调”，不是标准超卖。
        # 标准超卖通常应按 RSI<30，极端超卖 RSI<20；后续进入参数化/网格验证。
        rsi = self._calc_rsi(bars, idx, 14)
        met = rsi < 40
        conditions.append(ConditionResult(
            "rsi_weak_pullback", "RSI偏弱回调", met, rsi, 40, "below",
            min(1.0, (40 - rsi) / 20 + 0.3) if met else 0,
        ))
        if met: met_count += 1

        # 2. MA5金叉MA20
        ma5_met = self._check_ma_cross(bars, idx, 5, 20, "up")
        ma5, ma20 = self._get_mas(bars, idx, 5, 20)
        conditions.append(ConditionResult(
            "ma_golden", "MA金叉", ma5_met, ma5 / ma20 if ma20 > 0 else 0, 1.0, "cross_up",
            min(1.0, (ma5 / ma20 - 1) * 50 + 0.3) if ma5_met else 0,
        ))
        if ma5_met: met_count += 1

        # 3. MACD翻红
        macd_met = self._check_macd(bars, idx, "bullish")
        macd_val = self._get_macd_hist(bars, idx)
        conditions.append(ConditionResult(
            "macd_bullish", "MACD翻红", macd_met, macd_val, 0, "above",
            min(1.0, abs(macd_val) * 10 + 0.3) if macd_met else 0,
        ))
        if macd_met: met_count += 1

        # 4. 布林下轨 < 0.3
        boll_pos = self._get_boll_position(bars, idx, 20, 2.0)
        met = boll_pos < 0.3
        conditions.append(ConditionResult(
            "boll_lower", "布林下轨", met, boll_pos, 0.3, "below",
            min(1.0, (0.3 - boll_pos) * 3 + 0.3) if met else 0,
        ))
        if met: met_count += 1

        # 5. 放量 > 1.2x
        vol_ratio = self._get_volume_ratio(bars, idx, 20)
        met = vol_ratio > 1.2
        conditions.append(ConditionResult(
            "volume_expand", "放量", met, vol_ratio, 1.2, "above",
            min(1.0, (vol_ratio - 1) * 0.5 + 0.3) if met else 0,
        ))
        if met: met_count += 1

        # 6. KDJ金叉(K<50)
        kdj_met, k_val, d_val = self._check_kdj(bars, idx, "golden")
        conditions.append(ConditionResult(
            "kdj_golden", "KDJ金叉", kdj_met, k_val, d_val, "cross_up",
            min(1.0, (d_val - k_val) / 10 + 0.3) if kdj_met else 0,
        ))
        if kdj_met: met_count += 1

        active = self._filter_conditions(conditions, "buy")
        met_count = sum(1 for c in active if c.met)
        l3_score = sum(c.confidence for c in active) / max(len(active), 1) * 100.0
        return met_count >= self.min_confirmations, l3_score, active

    def check_sell(self, bars: pd.DataFrame, idx: int) -> Tuple[bool, List[ConditionResult]]:
        conditions = []
        met_count = 0

        def cr(key: str, name: str, met: bool, value: float, threshold: float,
               direction: str, conf: float) -> ConditionResult:
            return ConditionResult(key, name, bool(met), float(value),
                                   float(threshold), direction,
                                   max(0.0, min(1.0, float(conf))))

        # ── 资金流/换手数据（从 bars 列读取，_merge_factor_columns 确保存在）──
        mf_vals = self._get_moneyflow_values(bars, idx)
        to_vals = self._get_turnover_values(bars, idx)

        # 1. RSI > 70
        rsi = self._calc_rsi(bars, idx, 14)
        met = rsi > 70
        conditions.append(cr(
            "rsi_overbought", "RSI超买", met, rsi, 70, "above",
            min(1.0, (rsi - 70) / 20 + 0.3) if met else 0,
        ))
        if met: met_count += 1

        # 2. MA5死叉MA20
        ma5, ma20 = self._get_mas(bars, idx, 5, 20)
        ma_met = ma5 < ma20 if ma20 > 0 else False
        conditions.append(cr(
            "ma5_below_ma20", "MA死叉", ma_met, ma20 / ma5 if ma5 > 0 else 0, 1.0, "cross_down",
            min(1.0, (ma20 / ma5 - 1) * 50 + 0.3) if ma_met else 0,
        ))
        if ma_met: met_count += 1

        # 3. MACD翻绿
        macd_met = self._check_macd(bars, idx, "bearish")
        macd_val = self._get_macd_hist(bars, idx)
        conditions.append(cr(
            "macd_bearish", "MACD翻绿", macd_met, macd_val, 0, "below",
            min(1.0, abs(macd_val) * 10 + 0.3) if macd_met else 0,
        ))
        if macd_met: met_count += 1

        # 4. 布林上轨 > 0.8
        boll_pos = self._get_boll_position(bars, idx, 20, 2.0)
        met = boll_pos > 0.8
        conditions.append(cr(
            "boll_upper", "布林上轨", met, boll_pos, 0.8, "above",
            min(1.0, (boll_pos - 0.8) * 5 + 0.3) if met else 0,
        ))
        if met: met_count += 1

        # 5. 放量下跌
        vol_ratio = self._get_volume_ratio(bars, idx, 20)
        price_down = bars['close'].astype(float).iloc[idx] < bars['close'].astype(float).iloc[idx - 1] if idx >= 1 else False
        met = vol_ratio > 1.2 and price_down
        conditions.append(cr(
            "volume_price_down", "放量下跌", met, vol_ratio, 1.2, "above",
            min(1.0, (vol_ratio - 1) * 0.5 + 0.3) if met else 0,
        ))
        if met: met_count += 1

        # 6. KDJ死叉(K>50)
        kdj_met, k_val, d_val = self._check_kdj(bars, idx, "death")
        conditions.append(cr(
            "kdj_death", "KDJ死叉", kdj_met, k_val, d_val, "cross_down",
            min(1.0, abs(k_val - d_val) / 10 + 0.3) if kdj_met else 0,
        ))
        if kdj_met: met_count += 1

        # ── 7-12: 资金流/换手卖出条件（P0 新增，对齐 sell_conditions 白名单）──

        # 7. 主力净流出
        mf_main_out = mf_vals['main_net_mf_amount'] < -10000
        conditions.append(cr(
            "main_net_mf_negative", "主力净流出>1万", mf_main_out,
            mf_vals['main_net_mf_amount'], -10000, "below",
            min(1.0, abs(mf_vals['main_net_mf_amount']) / 50000) if mf_main_out else 0,
        ))
        if mf_main_out: met_count += 1

        # 8. 超大单净流出
        mf_elg_out = mf_vals['large_elg_net_mf_amount'] < -50000
        conditions.append(cr(
            "large_elg_net_mf_negative", "超大单净流出>5万", mf_elg_out,
            mf_vals['large_elg_net_mf_amount'], -50000, "below",
            min(1.0, abs(mf_vals['large_elg_net_mf_amount']) / 200000) if mf_elg_out else 0,
        ))
        if mf_elg_out: met_count += 1

        # 9. 5日相对换手率过低（交投降温）
        turnover_low = to_vals['relative_turnover_5d'] < 0.7
        conditions.append(cr(
            "relative_turnover_5d_low", "交投降温<0.7x", turnover_low,
            to_vals['relative_turnover_5d'], 0.7, "below",
            min(1.0, (1.0 - to_vals['relative_turnover_5d']) * 3) if turnover_low else 0,
        ))
        if turnover_low: met_count += 1

        # 10. 5日相对换手率过高（恐慌抛售，回调策略专用）
        turnover_high = to_vals['relative_turnover_5d'] > 2.0
        conditions.append(cr(
            "relative_turnover_5d_high", "恐慌放量>2.0x", turnover_high,
            to_vals['relative_turnover_5d'], 2.0, "above",
            min(1.0, (to_vals['relative_turnover_5d'] - 2.0) * 0.5) if turnover_high else 0,
        ))
        if turnover_high: met_count += 1

        # 11. 主力净流出恶化（较5日前更差，回调策略专用）
        if idx >= 5:
            mf_main_5d_ago = self._get_factor_value(bars, idx - 5, 'main_net_mf_amount', 0.0)
        else:
            mf_main_5d_ago = 0.0
        mf_main_worsening = mf_vals['main_net_mf_amount'] < mf_main_5d_ago and mf_vals['main_net_mf_amount'] < 0
        mf_worsening_ratio = (mf_vals['main_net_mf_amount'] / mf_main_5d_ago
                              if mf_main_5d_ago != 0 else 0.0)
        conditions.append(cr(
            "main_net_mf_negative_worsening", "主力流出恶化", mf_main_worsening,
            mf_vals['main_net_mf_amount'], mf_main_5d_ago, "below",
            min(1.0, abs(mf_worsening_ratio) * 0.5) if mf_main_worsening else 0,
        ))
        if mf_main_worsening: met_count += 1

        # 12. 超大单净流出恶化（较5日前更差，回调策略专用）
        if idx >= 5:
            mf_elg_5d_ago = self._get_factor_value(bars, idx - 5, 'large_elg_net_mf_amount', 0.0)
        else:
            mf_elg_5d_ago = 0.0
        mf_elg_worsening = mf_vals['large_elg_net_mf_amount'] < mf_elg_5d_ago and mf_vals['large_elg_net_mf_amount'] < 0
        mf_elg_worsening_ratio = (mf_vals['large_elg_net_mf_amount'] / mf_elg_5d_ago
                                  if mf_elg_5d_ago != 0 else 0.0)
        conditions.append(cr(
            "large_elg_net_mf_negative_worsening", "超大单流出恶化", mf_elg_worsening,
            mf_vals['large_elg_net_mf_amount'], mf_elg_5d_ago, "below",
            min(1.0, abs(mf_elg_worsening_ratio) * 0.5) if mf_elg_worsening else 0,
        ))
        if mf_elg_worsening: met_count += 1

        active = self._filter_conditions(conditions, "sell")
        met_count = sum(1 for c in active if c.met)
        return met_count >= self.sell_min_confirmations, active

    # ── 工具方法 ──
    @staticmethod
    def _get_factor_value(bars: pd.DataFrame, idx: int, factor_name: str, default: float = np.nan) -> float:
        """从 bars 获取指定因子值，支持新因子字段。"""
        if factor_name in bars.columns:
            try:
                val = float(bars.iloc[idx][factor_name])
                return val if np.isfinite(val) else default
            except Exception:
                return default
        return default

    @staticmethod
    def _get_moneyflow_values(bars: pd.DataFrame, idx: int) -> dict:
        """获取资金流相关因子值。"""
        return {
            'main_net_mf_amount': ResonanceChecker._get_factor_value(bars, idx, 'main_net_mf_amount', 0.0),
            'large_elg_net_mf_amount': ResonanceChecker._get_factor_value(bars, idx, 'large_elg_net_mf_amount', 0.0),
            'main_net_mf_pct_amount': ResonanceChecker._get_factor_value(bars, idx, 'main_net_mf_pct_amount', 0.0),
            'large_elg_net_mf_pct_amount': ResonanceChecker._get_factor_value(bars, idx, 'large_elg_net_mf_pct_amount', 0.0),
            'main_net_mf_rank': ResonanceChecker._get_factor_value(bars, idx, 'main_net_mf_rank', 0.5),
            'large_elg_net_mf_rank': ResonanceChecker._get_factor_value(bars, idx, 'large_elg_net_mf_rank', 0.5),
        }

    @staticmethod
    def _get_turnover_values(bars: pd.DataFrame, idx: int) -> dict:
        """获取相对换手相关因子值。"""
        return {
            'relative_turnover_5d': ResonanceChecker._get_factor_value(bars, idx, 'relative_turnover_5d', 1.0),
            'relative_turnover_20d': ResonanceChecker._get_factor_value(bars, idx, 'relative_turnover_20d', 1.0),
            'turnover_percentile_60d': ResonanceChecker._get_factor_value(bars, idx, 'turnover_percentile_60d', 0.5),
            'amount_percentile_60d': ResonanceChecker._get_factor_value(bars, idx, 'amount_percentile_60d', 0.5),
        }

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


def _entry_model_for_strategy(strategy_type: str) -> str:
    mapping = {
        "trend_momentum": "trend_continuation",
        "pullback": "pullback_reversal",
        "breakout": "consolidation_breakout",
        "balanced": "balanced_route_unclassified",
    }
    return mapping.get(str(strategy_type), "unknown")


def _structured_entry_audit(strategy_type: str, buy_met: List[ConditionResult], liq_tags: List[str], bars: pd.DataFrame, idx: int) -> dict:
    """生成买点结构化审计字段。

    第一阶段只落盘审计，不改变信号过滤。资金流/市场上下文若未传入当前单股K线，
    必须写入 missing_fields，不能假装已验证。
    """
    keys = [c.key for c in buy_met]
    names = [c.audit_text() for c in buy_met]
    missing = []
    row = bars.iloc[idx] if bars is not None and not bars.empty and 0 <= idx < len(bars) else pd.Series(dtype=object)
    for field in [
        "market_score", "main_net_mf_pct_amount", "large_elg_net_mf_pct_amount",
        "relative_turnover_20d", "turnover_percentile_60d",
    ]:
        if field not in row or pd.isna(row.get(field)):
            missing.append(field)
    technical = [n for n in names if any(k in n for k in ["RSI", "MA", "MACD", "布林", "KDJ", "动量", "量比", "平台"])]
    risk_tags = list(liq_tags or [])
    return {
        "entry_model": _entry_model_for_strategy(strategy_type),
        "main_trigger": keys[0] if keys else "unclassified",
        "confirmations": ";".join(keys),
        "factor_evidence": "audit_pending_factor_context",
        "market_context": "audit_pending_market_context" if "market_score" in missing else f"market_score={row.get('market_score')}",
        "fund_flow_context": "audit_pending_fund_flow_context" if any(f in missing for f in ["main_net_mf_pct_amount", "large_elg_net_mf_pct_amount"]) else f"main={row.get('main_net_mf_pct_amount')};large_elg={row.get('large_elg_net_mf_pct_amount')}",
        "technical_confirmations": ";".join(technical),
        "veto_checks": "audit_pending_veto_checks",
        "risk_tags": ";".join(risk_tags),
        "missing_fields": ";".join(missing),
    }


# ═══════════════════════════════════════════
# 三层过滤主函数
# ═══════════════════════════════════════════
def check_overheat(bars: pd.DataFrame, idx: int, row: Optional[pd.Series] = None) -> Tuple[float, List[str]]:
    """动量/量能/连板/资金背离 过热检测（统一实现）。

    返回 (penalty_coeff, risk_tags)，coeff ∈ [0.0, 1.0]，乘入信号置信度。

    Args:
        bars: OHLCV K线数据（可能含因子列如 main_net_mf_rank）
        idx: 当前 bar 索引
        row: 可选，scanner 路径提供的因子快照行（含 main_net_mf_rank 等）
             回测路径不传，直接从 bars.columns 读取
    """
    close = bars["close"].astype(float)
    current = float(close.iloc[idx])

    def _sqrt_decay(v: float, threshold: float) -> float:
        if v <= threshold:
            return 1.0
        return 1.0 / max(v / threshold, 1.0) ** 0.5

    tags: List[str] = []

    # ── 1. 价量背离：价格高位 + 大单资金不参与 ──
    lo = max(0, idx - 19)
    hh20 = float(close.iloc[lo:idx + 1].max()) if lo < idx else current
    hh_dist = (current / hh20 - 1) if hh20 > 0 else 0.0

    # 资金排名数据源：row（scanner 快照）优先，否则 bars.columns（回测合并列）
    def _get_rank(col: str, fallback: float = 0.5) -> float:
        if row is not None and col in (row.index if hasattr(row, 'index') else []):
            v = _num_safe(row.get(col, fallback), fallback)
            return float(v) if not pd.isna(v) else fallback
        if col in bars.columns:
            v = bars[col].astype(float).iloc[idx]
            return float(v) if not pd.isna(v) else fallback
        return fallback

    main_mf_rank = _get_rank("main_net_mf_rank", 0.5)
    large_elg_mf_rank = _get_rank("large_elg_net_mf_rank", 0.5)
    has_moneyflow = (row is not None and "main_net_mf_rank" in (row.index if hasattr(row, 'index') else [])) or \
                    "main_net_mf_rank" in bars.columns

    divergence_coeff = 1.0
    if hh_dist >= -0.02 and has_moneyflow:
        smart_money = main_mf_rank * 0.5 + large_elg_mf_rank * 0.5
        if smart_money < 0.20:
            divergence_coeff = 0.40
            tags.append(f"资金严重背离(主力{main_mf_rank:.0%}大单{large_elg_mf_rank:.0%})")
        elif smart_money < 0.35:
            divergence_coeff = 0.65
            tags.append(f"资金背离(主力{main_mf_rank:.0%}大单{large_elg_mf_rank:.0%})")
        elif smart_money < 0.50:
            divergence_coeff = 0.80
            tags.append(f"高位资金偏弱(主力{main_mf_rank:.0%})")
        elif hh_dist >= 0.0 and smart_money >= 0.50:
            tags.append(f"新高+资金确认(主力{main_mf_rank:.0%})")
    elif hh_dist >= 0.0 and not has_moneyflow:
        # 无资金流数据时的回退：仅标记，轻微惩罚
        divergence_coeff = 0.85
        tags.append(f"20日新高追高{hh_dist:.0%}(缺资金流)")
    elif hh_dist >= -0.02 and not has_moneyflow:
        divergence_coeff = 0.90

    # ── 2. 动量过热 ──
    mom5 = current / float(close.iloc[idx - 5]) - 1 if idx >= 5 and close.iloc[idx - 5] > 0 else 0.0
    mom20 = current / float(close.iloc[idx - 20]) - 1 if idx >= 20 and close.iloc[idx - 20] > 0 else 0.0
    mom5_coeff = _sqrt_decay(mom5, 0.25)
    mom20_coeff = _sqrt_decay(mom20, 0.50)
    if mom5_coeff < 1.0:
        tags.append(f"5日动量过热{mom5:.0%}")
    if mom20_coeff < 1.0:
        tags.append(f"20日动量过热{mom20:.0%}")

    # ── 3. 天量出货 ──
    volumes = bars["volume"].astype(float).iloc[max(0, idx - 5):idx + 1]
    vol_ratio = float(volumes.iloc[-1]) / float(volumes.iloc[:5].mean()) if len(volumes) >= 6 and volumes.iloc[:5].mean() > 0 else 1.0
    vol_coeff = _sqrt_decay(vol_ratio, 10.0)
    if vol_coeff < 1.0:
        tags.append(f"天量出货{vol_ratio:.0f}x")

    # ── 4. 连板开板 ──
    if idx >= 5:
        pct_changes = close.iloc[idx - 4:idx + 1].pct_change().dropna()
    else:
        pct_changes = close.iloc[:idx + 1].pct_change().dropna().iloc[-5:]
    limit_up_count = int((pct_changes >= 0.098).sum())
    lu_coeff = 1.0
    if limit_up_count >= 5:
        lu_coeff = 0.0
    elif limit_up_count >= 4:
        lu_coeff = 0.25
    elif limit_up_count >= 3:
        lu_coeff = 0.50
    elif limit_up_count >= 2:
        lu_coeff = 0.75
    if lu_coeff < 1.0:
        tags.append(f"近5日{limit_up_count}板开板风险")

    penalty = divergence_coeff * mom5_coeff * mom20_coeff * vol_coeff * lu_coeff
    return penalty, tags


def _num_safe(value, default: float) -> float:
    """安全提取数值。"""
    try:
        v = float(value)
        return v if pd.notna(v) else default
    except (ValueError, TypeError):
        return default


# 兼容旧调用名
_check_overheat_for_layers = check_overheat


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

    tf = trend_filter or TrendFilter(strategy_type=strategy_type)
    sm = strategy_matcher or StrategyMatcher(StrategyType(strategy_type))
    rc = resonance_checker or ResonanceChecker.from_strategy(strategy_type, min_confirmations)

    close = bars['close'].astype(float)
    points = []
    last_action = None
    entry_close = 0.0  # 记录开仓价，用于加仓盈利判定

    for i in range(20, len(bars)):
        current_date = bars.iloc[i]['trade_date']
        if hasattr(current_date, 'date'):
            current_date = current_date.date()

        # ── L1: 趋势过滤 ──
        trend_ok, l1_score, _ = tf.check(bars, i)
        if not trend_ok:
            continue

        # ── L3: 卖出信号（不依赖 L2 策略匹配，价格/资金流条件独立判断）──
        if last_action == "BUY":
            sell_ok, sell_conds = rc.check_sell(bars, i)
            if sell_ok:
                sell_met = [c for c in sell_conds if c.met]
                reason = " + ".join([c.audit_text() for c in sell_met])
                conf = float(np.mean([c.confidence for c in sell_met])) if sell_met else 0.5
                points.append(apply_confidence_audit(TradePoint(
                    date=current_date, action="SELL",
                    reason=f"L3共振卖出({len(sell_met)}/6): {reason}",
                    confidence=min(1.0, conf), price=float(close.iloc[i]),
                    rule_name=f"三层过滤-{strategy_type}",
                    condition_count=len(sell_met),
                )))
                last_action = "SELL"
                continue

        # ── L2: 策略匹配 ──
        strategy_ok, l2_score, _ = sm.match(bars, i)
        if not strategy_ok:
            continue

        # ── L3: 买入/加仓信号 ──
        buy_ok, l3_score, buy_conds = rc.check_buy(bars, i)
        if buy_ok:
            # 共享的评分计算
            buy_met = [c for c in buy_conds if c.met]
            reason = " + ".join([c.audit_text() for c in buy_met])
            liq_tags = liquidity_audit_tags(bars, i)
            if liq_tags:
                reason = f"{reason}；审计：{'，'.join(liq_tags)}"
            audit_fields = _structured_entry_audit(strategy_type, buy_met, liq_tags, bars, i)
            overheat_coeff, overheat_tags = _check_overheat_for_layers(bars, i)
            if overheat_tags:
                reason = f"{reason}；⚠️ 过热: {'、'.join(overheat_tags)}"
            composite = (l1_score * 0.25 + l2_score * 0.35 + l3_score * 0.40) / 100.0
            composite = min(1.0, composite * overheat_coeff)
            current_price = float(close.iloc[i])

            # ── 加仓判断（已持仓 + BUY 再次触发）──
            if last_action == "BUY" and entry_close > 0:
                pnl_pct = current_price / entry_close - 1
                cost = 0.003  # 佣金万2.5×2 + 印花税0.001 = 0.0015 + 0.001 = 0.0025, 取0.003留余量
                if pnl_pct > cost and composite >= 0.70:
                    points.append(apply_confidence_audit(TradePoint(
                        date=current_date, action="ADD",
                        reason=f"加仓信号({len(buy_met)}/6): {reason}（盈利{pnl_pct*100:+.1f}%）",
                        confidence=min(1.0, composite * 0.5), price=current_price,
                        rule_name=f"三层过滤-{strategy_type}",
                        condition_count=len(buy_met),
                        **audit_fields,
                    ), action="ADD"))
                    # 不改变 last_action，允许后续再次 ADD
                # else: 条件不满足，忽略，保持 BUY 状态等待 SELL

            elif last_action != "BUY":
                points.append(apply_confidence_audit(TradePoint(
                    date=current_date, action="BUY",
                    reason=f"L3共振买入({len(buy_met)}/6): {reason}",
                    confidence=min(1.0, composite), price=current_price,
                    rule_name=f"三层过滤-{strategy_type}",
                    condition_count=len(buy_met),
                    **audit_fields,
                )))
                last_action = "BUY"
                entry_close = current_price

    return points
