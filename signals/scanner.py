"""新方案信号扫描器：大盘择时 + 策略专属条件 + 三层过滤摘要。

用于信号页展示每日候选，不替代回测撮合引擎。
扫描输出只代表 T 日收盘后的计划信号，实际成交必须在 T+1 开盘撮合。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from market.timing import POSITION_BRACKETS, PositionBracket
from strategy.schemes import BUILTIN_SCHEMES, StrategyScheme


@dataclass
class ScanSignal:
    """信号页使用的结构化扫描结果。"""

    symbol: str
    signal_type: str  # BUY / SELL
    scheme_id: str
    strategy_name: str
    signal_date: date
    suggested_exec_date: Optional[date]
    market_score: float
    market_bracket: str
    market_position_pct: float
    factor_score: float
    layer1_score: float
    layer2_score: float
    layer3_confirmations: int
    total_score: float
    entry_reason: str
    risk_tags: List[str] = field(default_factory=list)
    suggested_position_pct: float = 0.0

    # 兼容旧信号卡片字段
    @property
    def strength(self) -> float:
        return round(max(0.0, min(10.0, self.total_score / 10.0)), 2)

    @property
    def score(self) -> float:
        return self.total_score

    @property
    def regime(self) -> str:
        return self.market_bracket


STRATEGY_MARKET_FLOOR = {
    "trend_momentum": 60.0,
    "pullback": 40.0,
    "breakout": 50.0,
}

STRATEGY_MIN_CONFIRMATIONS = {
    "trend_momentum": 3,
    "pullback": 3,
    "breakout": 3,
}


def scan_signals(
    factor_df: pd.DataFrame,
    price_df: pd.DataFrame,
    factor_names: Iterable[str],
    scheme_id: str = "balanced",
    top_n: int = 20,
    market_score: Optional[float] = None,
    include_sell_symbols: Optional[Iterable[str]] = None,
) -> Tuple[List[ScanSignal], List[ScanSignal]]:
    """按 DAILY_START_PLAN 新方案扫描信号。

    - BUY：按 trend_momentum / pullback / breakout 独立扫描。
    - balanced：不直接造信号，只汇总三类子策略的最优结果。
    - SELL：仅对传入的持仓/关注股票做风险退出提示，避免对全市场生成无意义卖出榜。
    """
    if factor_df.empty or "trade_date" not in factor_df.columns or "symbol" not in factor_df.columns:
        return [], []

    latest_date = _to_date(factor_df["trade_date"].max())
    day_data = factor_df[factor_df["trade_date"].map(_to_date) == latest_date].copy()
    if day_data.empty:
        return [], []

    market_score_val = 50.0 if market_score is None else float(market_score)
    bracket = _market_bracket(market_score_val)
    price_map = _build_price_map(price_df)
    next_exec_date = _next_trade_date(price_df, latest_date)
    factor_names_set = set(factor_names or [])

    target_schemes = _target_schemes(scheme_id)
    buy_candidates: List[ScanSignal] = []

    for sid in target_schemes:
        scheme = BUILTIN_SCHEMES[sid]
        floor = STRATEGY_MARKET_FLOOR[sid]
        if market_score_val < floor:
            continue

        scored = _factor_scores(day_data, scheme.factor_weights, factor_names_set)
        for symbol, factor_score in scored.items():
            bars = price_map.get(symbol)
            row = day_data[day_data["symbol"] == symbol]
            if bars is None or bars.empty or row.empty:
                continue
            if len(bars) < 40:
                continue
            latest_row = row.iloc[0]
            l1_ok, l1_score, l1_reason = _check_layer1(bars, sid)
            if not l1_ok:
                continue
            l2_ok, l2_score, l2_reason = _check_layer2(bars, latest_row, sid)
            if not l2_ok:
                continue
            confirmations, l3_reasons = _check_layer3(bars, latest_row, sid)
            if confirmations < STRATEGY_MIN_CONFIRMATIONS[sid]:
                continue

            total_score = round(
                factor_score * 0.35 + l1_score * 0.20 + l2_score * 0.25 + min(confirmations / 6, 1.0) * 100 * 0.20,
                4,
            )
            reason = f"{l2_reason}；{l1_reason}；共振{confirmations}/6：{'、'.join(l3_reasons)}"
            risk_tags = _build_risk_tags(bars, latest_row, market_score_val, sid)
            buy_candidates.append(ScanSignal(
                symbol=symbol,
                signal_type="BUY",
                scheme_id=sid,
                strategy_name=scheme.name,
                signal_date=latest_date,
                suggested_exec_date=next_exec_date,
                market_score=market_score_val,
                market_bracket=bracket.label,
                market_position_pct=bracket.position_pct,
                factor_score=round(float(factor_score), 4),
                layer1_score=round(float(l1_score), 4),
                layer2_score=round(float(l2_score), 4),
                layer3_confirmations=confirmations,
                total_score=total_score,
                entry_reason=reason,
                risk_tags=risk_tags,
                suggested_position_pct=round(min(0.20, scheme.position_pct_per_entry * bracket.per_entry_mult), 4),
            ))

    # balanced 作为组合器：同一股票只保留最高分策略；单策略页面则正常保留该策略结果。
    if scheme_id == "balanced":
        best_by_symbol: Dict[str, ScanSignal] = {}
        for sig in buy_candidates:
            old = best_by_symbol.get(sig.symbol)
            if old is None or sig.total_score > old.total_score:
                best_by_symbol[sig.symbol] = sig
        buy_candidates = list(best_by_symbol.values())

    buy_candidates.sort(key=lambda s: s.total_score, reverse=True)
    buy_candidates = buy_candidates[:top_n]

    sell_signals = _scan_sell_signals(
        include_sell_symbols or [], price_map, day_data, latest_date,
        next_exec_date, market_score_val, bracket,
    )
    return buy_candidates, sell_signals[:top_n]


def _target_schemes(scheme_id: str) -> List[str]:
    if scheme_id == "balanced":
        return ["trend_momentum", "pullback", "breakout"]
    if scheme_id in {"trend_momentum", "pullback", "breakout"}:
        return [scheme_id]
    return ["trend_momentum", "pullback", "breakout"]


def _factor_scores(day_data: pd.DataFrame, weights: Dict[str, float], factor_names: set) -> pd.Series:
    symbols = day_data["symbol"].astype(str)
    score = pd.Series(0.0, index=symbols.values)
    weight_sum = 0.0
    for factor, weight in weights.items():
        if factor_names and factor not in factor_names:
            continue
        if factor not in day_data.columns:
            continue
        vals = pd.to_numeric(day_data[factor], errors="coerce")
        if vals.notna().sum() < 3:
            continue
        pct = vals.rank(pct=True, method="average")
        directional = pct if weight >= 0 else 1 - pct
        score = score.add(pd.Series(directional.values * abs(weight) * 100, index=symbols.values), fill_value=0.0)
        weight_sum += abs(weight)
    if weight_sum <= 0:
        return pd.Series(dtype=float)
    return (score / weight_sum).sort_values(ascending=False)


def _build_price_map(price_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if price_df.empty or "symbol" not in price_df.columns or "trade_date" not in price_df.columns:
        return {}
    result = {}
    for symbol, bars in price_df.groupby("symbol"):
        b = bars.sort_values("trade_date").copy()
        b["trade_date"] = b["trade_date"].map(_to_date)
        if "close" not in b.columns:
            continue
        # 信号扫描允许 close-only 数据；缺失 OHLCV 用保守默认值补齐。
        for col in ["open", "high", "low"]:
            if col not in b.columns:
                b[col] = b["close"]
        if "volume" not in b.columns:
            b["volume"] = 0.0
        result[str(symbol)] = b.reset_index(drop=True)
    return result


def _check_layer1(bars: pd.DataFrame, scheme_id: str) -> Tuple[bool, float, str]:
    close = bars["close"].astype(float)
    if len(close) < 40:
        return False, 0.0, "L1数据不足"
    ma20 = close.rolling(20).mean().iloc[-1]
    ma40 = close.rolling(40).mean().iloc[-1]
    current = close.iloc[-1]
    low20 = close.iloc[-20:].min()

    if scheme_id == "pullback":
        ma20_series = close.rolling(20).mean().iloc[-10:]
        recent_close = close.iloc[-10:]
        had_uptrend = bool((recent_close.values > ma20_series.values).any())
        trend_ok = had_uptrend and ma20 >= ma40 * 0.995 and current > low20 * 1.03
        score = 0.0
        score += 35 if had_uptrend else 0
        score += 30 if ma20 >= ma40 * 0.995 else 0
        score += 35 if current > low20 * 1.03 else 0
        return trend_ok, score, "L1上升趋势回调未破位"

    trend_ok = current > ma20 and ma20 > ma40 and current > low20 * 1.03
    score = 0.0
    score += 40 if current > ma20 else 0
    score += 35 if ma20 > ma40 else 0
    score += 25 if current > low20 * 1.03 else 0
    return trend_ok, score, "L1价格在MA20上方且MA20高于MA40"


def _check_layer2(bars: pd.DataFrame, row: pd.Series, scheme_id: str) -> Tuple[bool, float, str]:
    close = bars["close"].astype(float)
    current = float(close.iloc[-1])
    hh20 = float(close.iloc[-20:].max())
    low20 = float(close.iloc[-20:].min())
    pb = (hh20 - current) / hh20 if hh20 > 0 else 0.0
    mom5 = current / float(close.iloc[-6]) - 1 if len(close) >= 6 and close.iloc[-6] > 0 else 0.0
    mom20 = current / float(close.iloc[-21]) - 1 if len(close) >= 21 and close.iloc[-21] > 0 else 0.0
    rsi = _num(row.get("rsi14"), np.nan)
    vol_ratio = _num(row.get("volume_ratio"), 1.0)

    if scheme_id == "trend_momentum":
        ok = pb <= 0.05 and mom5 > 0.01 and mom20 > 0.02
        score = min(100.0, max(0.0, (0.05 - pb) * 600 + mom5 * 600 + mom20 * 300))
        return ok, score, f"强势追涨：距20日高点{pb:.1%}，M5={mom5:.1%}，M20={mom20:.1%}"

    if scheme_id == "pullback":
        ok = 0.05 <= pb <= 0.15 and current > low20 * 1.03 and (rsi < 45 or pb >= 0.08)
        score = min(100.0, max(0.0, 45 + pb * 220 + max(0, 45 - (rsi if not np.isnan(rsi) else 45)) * 1.2))
        return ok, score, f"回调低吸：回撤{pb:.1%}，RSI={rsi:.0f}"

    prev = close.iloc[-15:-5]
    range_pct = (prev.max() - prev.min()) / prev.mean() if len(prev) >= 5 and prev.mean() > 0 else 1.0
    breakout = current > float(prev.max()) * 1.01 if len(prev) >= 5 else False
    ok = range_pct < 0.08 and breakout and vol_ratio > 1.3
    score = min(100.0, max(0.0, (0.08 - range_pct) * 500 + max(0, vol_ratio - 1.0) * 35 + (20 if breakout else 0)))
    return ok, score, f"横盘突破：振幅{range_pct:.1%}，量比{vol_ratio:.1f}x"


def _check_layer3(bars: pd.DataFrame, row: pd.Series, scheme_id: str) -> Tuple[int, List[str]]:
    close = bars["close"].astype(float)
    current = float(close.iloc[-1])
    ma5 = float(close.rolling(5).mean().iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    hh20 = float(close.iloc[-20:].max())
    pb = (hh20 - current) / hh20 if hh20 > 0 else 0.0
    rsi = _num(row.get("rsi14"), np.nan)
    boll_pos = _num(row.get("boll_position"), np.nan)
    # 快照中 boll_position 可能是百分制，也可能是 0-1，统一成 0-1。
    boll01 = boll_pos / 100.0 if boll_pos > 1 else boll_pos
    vol_ratio = _num(row.get("volume_ratio"), 1.0)
    mom5 = current / float(close.iloc[-6]) - 1 if len(close) >= 6 and close.iloc[-6] > 0 else 0.0
    mom20 = current / float(close.iloc[-21]) - 1 if len(close) >= 21 and close.iloc[-21] > 0 else 0.0

    checks: List[Tuple[bool, str]]
    if scheme_id == "trend_momentum":
        checks = [
            (vol_ratio > 1.3, f"放量{vol_ratio:.1f}x"),
            (ma5 > ma20, "MA5高于MA20"),
            (pb <= 0.05, f"接近20日高点{pb:.1%}"),
            (mom5 > 0.01, f"M5={mom5:.1%}"),
            (mom20 > 0.02, f"M20={mom20:.1%}"),
            (np.isnan(rsi) or rsi < 85, f"RSI未极端{rsi:.0f}" if not np.isnan(rsi) else "RSI缺失"),
        ]
    elif scheme_id == "pullback":
        checks = [
            (not np.isnan(rsi) and rsi < 45, f"RSI={rsi:.0f}"),
            (not np.isnan(boll01) and boll01 < 0.35, f"布林位置{boll01:.2f}"),
            (0.05 <= pb <= 0.15, f"回撤{pb:.1%}"),
            (current > close.iloc[-20:].min() * 1.03, "不破20日低点"),
            (vol_ratio < 1.1, f"缩量/温和量比{vol_ratio:.1f}x"),
            (current >= close.iloc[-1] * 0.98, "当日未明显破位"),
        ]
    else:
        prev = close.iloc[-15:-5]
        range_pct = (prev.max() - prev.min()) / prev.mean() if len(prev) >= 5 and prev.mean() > 0 else 1.0
        checks = [
            (len(prev) >= 5 and current > float(prev.max()) * 1.01, "突破平台上沿"),
            (vol_ratio > 1.5, f"量比{vol_ratio:.1f}x"),
            (ma5 > ma20, "MA5高于MA20"),
            (range_pct < 0.08, f"平台振幅{range_pct:.1%}"),
            (mom5 > 0.01, f"M5={mom5:.1%}"),
            (not np.isnan(boll01) and boll01 > 0.6, f"布林上沿{boll01:.2f}"),
        ]
    reasons = [label for ok, label in checks if ok]
    return len(reasons), reasons


def _scan_sell_signals(
    symbols: Iterable[str],
    price_map: Dict[str, pd.DataFrame],
    day_data: pd.DataFrame,
    latest_date: date,
    next_exec_date: Optional[date],
    market_score: float,
    bracket: PositionBracket,
) -> List[ScanSignal]:
    results = []
    symbol_set = [str(s) for s in symbols]
    for symbol in symbol_set:
        bars = price_map.get(symbol)
        row = day_data[day_data["symbol"] == symbol]
        if bars is None or len(bars) < 20:
            continue
        close = bars["close"].astype(float)
        current = float(close.iloc[-1])
        ma5 = float(close.rolling(5).mean().iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma40 = float(close.rolling(40).mean().iloc[-1]) if len(close) >= 40 else ma20
        rsi = _num(row.iloc[0].get("rsi14"), np.nan) if not row.empty else np.nan
        reasons = []
        if market_score < 20:
            reasons.append("大盘防御档")
        if ma5 < ma20:
            reasons.append("MA5低于MA20")
        if current < ma40:
            reasons.append("跌破MA40")
        if not np.isnan(rsi) and rsi > 70:
            reasons.append(f"RSI超买{rsi:.0f}")
        if not reasons:
            continue
        total = min(100.0, 45 + len(reasons) * 15)
        results.append(ScanSignal(
            symbol=symbol,
            signal_type="SELL",
            scheme_id="risk_exit",
            strategy_name="风险退出",
            signal_date=latest_date,
            suggested_exec_date=next_exec_date,
            market_score=market_score,
            market_bracket=bracket.label,
            market_position_pct=bracket.position_pct,
            factor_score=0.0,
            layer1_score=0.0,
            layer2_score=0.0,
            layer3_confirmations=len(reasons),
            total_score=total,
            entry_reason="；".join(reasons),
            risk_tags=reasons,
            suggested_position_pct=0.0,
        ))
    results.sort(key=lambda s: s.total_score, reverse=True)
    return results


def _build_risk_tags(bars: pd.DataFrame, row: pd.Series, market_score: float, scheme_id: str) -> List[str]:
    tags = []
    if market_score < STRATEGY_MARKET_FLOOR.get(scheme_id, 50):
        tags.append("大盘偏弱")
    close = bars["close"].astype(float)
    vol20 = close.pct_change().iloc[-20:].std() * np.sqrt(252)
    if pd.notna(vol20) and vol20 > 0.45:
        tags.append("高波动")
    vol_ratio = _num(row.get("volume_ratio"), 1.0)
    if vol_ratio > 3:
        tags.append("异常放量")
    return tags


def _market_bracket(score: float) -> PositionBracket:
    score = max(0.0, min(100.0, float(score)))
    for bracket in POSITION_BRACKETS:
        if bracket.min_score <= score < bracket.max_score or (score == 100 and bracket.max_score == 100):
            return bracket
    return POSITION_BRACKETS[2]


def _next_trade_date(price_df: pd.DataFrame, latest_date: date) -> Optional[date]:
    if price_df.empty or "trade_date" not in price_df.columns:
        return None
    dates = sorted({_to_date(d) for d in price_df["trade_date"].dropna().tolist()})
    for d in dates:
        if d > latest_date:
            return d
    return None


def _to_date(value) -> date:
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    return pd.to_datetime(value).date()


def _num(value, default: float) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default
