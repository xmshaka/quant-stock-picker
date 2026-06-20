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
from strategy.schemes import BUILTIN_SCHEMES, StrategyScheme, ResonanceConfig


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
    layer3_total: int = 0
    layer3_min_confirmations: int = 0
    layer3_condition_keys: List[str] = field(default_factory=list)
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

DEFAULT_MIN_CONFIRMATIONS = 3


def scan_signals(
    factor_df: pd.DataFrame,
    price_df: pd.DataFrame,
    factor_names: Iterable[str],
    scheme_id: str = "balanced",
    top_n: int = 20,
    market_score: Optional[float] = None,
    include_sell_symbols: Optional[Iterable[str]] = None,
    include_sell_context: Optional[Dict[str, Dict]] = None,
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
            tradable_ok, tradable_reason, tradable_tags = _check_l4_tradability(bars, latest_row, sid)
            if not tradable_ok:
                continue
            l1_ok, l1_score, l1_reason = _check_layer1(bars, sid)
            if not l1_ok:
                continue
            l2_ok, l2_score, l2_reason = _check_layer2(bars, latest_row, sid)
            if not l2_ok:
                continue
            resonance_cfg = _resonance_config(sid)
            confirmations, l3_reasons = _check_layer3(bars, latest_row, sid, resonance_cfg)
            min_confirmations = int(resonance_cfg.min_confirmations or DEFAULT_MIN_CONFIRMATIONS)
            if confirmations < min_confirmations:
                continue

            layer3_total = len(resonance_cfg.buy_conditions) or 6
            total_score = round(
                factor_score * 0.35 + l1_score * 0.20 + l2_score * 0.25 + min(confirmations / max(layer3_total, 1), 1.0) * 100 * 0.20,
                4,
            )
            reason = f"{l2_reason}；{l1_reason}；共振{confirmations}/{layer3_total}：{'、'.join(l3_reasons)}"
            risk_tags = _build_risk_tags(bars, latest_row, market_score_val, sid)
            risk_tags.extend([t for t in tradable_tags if t not in risk_tags])
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
                layer3_total=layer3_total,
                layer3_min_confirmations=min_confirmations,
                layer3_condition_keys=list(resonance_cfg.buy_conditions or []),
                total_score=total_score,
                entry_reason=f"{reason}；L4可交易性：{tradable_reason}",
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
        next_exec_date, market_score_val, bracket, include_sell_context or {},
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


def _resonance_config(scheme_id: str) -> ResonanceConfig:
    scheme = BUILTIN_SCHEMES.get(str(scheme_id))
    cfg = getattr(scheme, "resonance_config", None) if scheme else None
    if cfg is None:
        return ResonanceConfig(min_confirmations=DEFAULT_MIN_CONFIRMATIONS)
    return cfg


def _check_layer3(bars: pd.DataFrame, row: pd.Series, scheme_id: str, resonance_config: Optional[ResonanceConfig] = None) -> Tuple[int, List[str]]:
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

    checks: List[Tuple[str, bool, str]]
    if scheme_id == "trend_momentum":
        checks = [
            ("volume_expand", vol_ratio > 1.3, f"放量{vol_ratio:.1f}x"),
            ("ma5_above_ma20", ma5 > ma20, "MA5高于MA20"),
            ("near_high", pb <= 0.05, f"接近20日高点{pb:.1%}"),
            ("momentum_5d", mom5 > 0.01, f"M5={mom5:.1%}"),
            ("momentum_20d", mom20 > 0.02, f"M20={mom20:.1%}"),
            ("rsi_not_extreme", np.isnan(rsi) or rsi < 85, f"RSI未极端{rsi:.0f}" if not np.isnan(rsi) else "RSI缺失"),
        ]
    elif scheme_id == "pullback":
        checks = [
            ("rsi_oversold", not np.isnan(rsi) and rsi < 45, f"RSI={rsi:.0f}"),
            ("boll_lower", not np.isnan(boll01) and boll01 < 0.35, f"布林位置{boll01:.2f}"),
            ("pullback_range", 0.05 <= pb <= 0.15, f"回撤{pb:.1%}"),
            ("not_break_20d_low", current > close.iloc[-20:].min() * 1.03, "不破20日低点"),
            ("volume_calm", vol_ratio < 1.1, f"缩量/温和量比{vol_ratio:.1f}x"),
            ("near_support", current >= close.iloc[-1] * 0.98, "当日未明显破位"),
        ]
    else:
        prev = close.iloc[-15:-5]
        range_pct = (prev.max() - prev.min()) / prev.mean() if len(prev) >= 5 and prev.mean() > 0 else 1.0
        checks = [
            ("break_platform", len(prev) >= 5 and current > float(prev.max()) * 1.01, "突破平台上沿"),
            ("volume_surge", vol_ratio > 1.5, f"量比{vol_ratio:.1f}x"),
            ("ma5_above_ma20", ma5 > ma20, "MA5高于MA20"),
            ("narrow_range", range_pct < 0.08, f"平台振幅{range_pct:.1%}"),
            ("momentum_5d", mom5 > 0.01, f"M5={mom5:.1%}"),
            ("boll_upper", not np.isnan(boll01) and boll01 > 0.6, f"布林上沿{boll01:.2f}"),
        ]
    cfg = resonance_config or _resonance_config(scheme_id)
    enabled = set(cfg.buy_conditions or [])
    active = [item for item in checks if not enabled or item[0] in enabled]
    reasons = [label for _, ok, label in active if ok]
    return len(reasons), reasons


def _check_l4_tradability(bars: pd.DataFrame, row: pd.Series, scheme_id: str) -> Tuple[bool, str, List[str]]:
    """Layer 4：风险可交易性检查。

    信号页只生成 T 日收盘后的候选计划，L4 用于剔除明显不可成交/不适合短线执行的票：
    - 价格/高低开收数据异常；
    - 一字涨跌停或涨停封死，T+1 实际可成交性差；
    - 成交额过低，滑点容易被低估。

    注意：这里不使用未来 T+1 数据，只检查截至信号日的行情与快照字段。
    """
    if bars.empty:
        return False, "行情缺失", ["数据缺失"]
    latest = bars.iloc[-1]
    try:
        close = float(latest.get("close", np.nan))
        high = float(latest.get("high", close))
        low = float(latest.get("low", close))
        open_ = float(latest.get("open", close))
    except Exception:
        return False, "价格字段异常", ["数据异常"]
    prices = [open_, high, low, close]
    if any((not np.isfinite(v)) or v <= 0 for v in prices) or high < low:
        return False, "OHLC异常", ["数据异常"]

    prev_close = 0.0
    if len(bars) >= 2:
        try:
            prev_close = float(bars["close"].astype(float).iloc[-2])
        except Exception:
            prev_close = 0.0
    pct = _pct_change(row, close, prev_close)
    limit_pct = _limit_threshold(row)
    sealed_limit_up = pct >= limit_pct * 0.98 and close >= high * 0.999
    sealed_limit_down = pct <= -limit_pct * 0.98 and close <= low * 1.001
    one_price_limit = abs(high - low) / max(close, 1e-9) <= 0.001 and (sealed_limit_up or sealed_limit_down)
    if one_price_limit:
        return False, "一字涨跌停不可可靠成交", ["涨跌停不可成交"]
    if sealed_limit_up:
        return False, "涨停封死不追买", ["涨停封死"]
    if sealed_limit_down:
        return False, "跌停封死流动性风险", ["跌停封死"]

    turnover = _latest_turnover_amount(latest, row, close)
    min_turnover = 100_000_000.0 if scheme_id == "trend_momentum" else 50_000_000.0
    if 0 < turnover < min_turnover:
        return False, f"成交额不足{turnover / 100_000_000:.2f}亿", ["成交额不足"]

    tags: List[str] = []
    if turnover <= 0:
        tags.append("成交额缺失")
        turnover_text = "成交额缺失"
    else:
        turnover_text = f"成交额{turnover / 100_000_000:.2f}亿"
    return True, turnover_text, tags


def _pct_change(row: pd.Series, close: float, prev_close: float) -> float:
    """返回小数形式涨跌幅，兼容 pct_change/pct_chg 为小数或百分数。"""
    for key in ("pct_change", "pct_chg", "change_pct"):
        val = _num(row.get(key), np.nan)
        if not np.isnan(val):
            return val / 100.0 if abs(val) > 1 else val
    if prev_close > 0:
        return close / prev_close - 1
    return 0.0


def _limit_threshold(row: pd.Series) -> float:
    """A股普通票 10%，ST/风险警示票 5%。"""
    raw_name = str(row.get("name", "") or row.get("stock_name", ""))
    is_st = bool(row.get("is_st", False)) if "is_st" in row.index else False
    return 0.05 if is_st or "ST" in raw_name.upper() else 0.10


def _latest_turnover_amount(latest: pd.Series, row: pd.Series, close: float) -> float:
    """估算信号日市场成交额，优先用标准化 amount，缺失时用 volume*close*100 兜底。"""
    amount = _context_float(latest.get("amount", 0.0)) or _context_float(row.get("amount", 0.0))
    if amount > 0:
        return amount
    volume = _context_float(latest.get("volume", 0.0)) or _context_float(row.get("volume", 0.0))
    if volume > 0 and close > 0:
        return volume * close * 100.0
    return 0.0


def _scan_sell_signals(
    symbols: Iterable[str],
    price_map: Dict[str, pd.DataFrame],
    day_data: pd.DataFrame,
    latest_date: date,
    next_exec_date: Optional[date],
    market_score: float,
    bracket: PositionBracket,
    sell_context: Dict[str, Dict],
) -> List[ScanSignal]:
    results = []
    symbol_set = [str(s) for s in symbols]
    for symbol in symbol_set:
        bars = price_map.get(symbol)
        row = day_data[day_data["symbol"] == symbol]
        if bars is None or len(bars) < 20:
            continue
        context = sell_context.get(symbol, {}) or {}
        scheme_id = _infer_scheme_id(context)
        exit_cfg = getattr(BUILTIN_SCHEMES.get(scheme_id), "exit_config", None)
        close = bars["close"].astype(float)
        current = float(close.iloc[-1])
        ma5 = float(close.rolling(5).mean().iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma40 = float(close.rolling(40).mean().iloc[-1]) if len(close) >= 40 else ma20
        low20 = float(close.iloc[-20:].min())
        rsi = _num(row.iloc[0].get("rsi14"), np.nan) if not row.empty else np.nan
        reasons = []
        exit_reason_keys = []
        market_floor = float(getattr(exit_cfg, "market_defense_score", 20.0) or 20.0)
        if bool(getattr(exit_cfg, "enable_market_defense_exit", True)) and market_score < market_floor:
            reasons.append("大盘防御减仓")
            exit_reason_keys.append("market_defense")

        entry_date = _context_date(context.get("entry_date") or context.get("add_date") or context.get("signal_date"))
        entry_price = _context_float(context.get("entry_price") or context.get("avg_cost"))
        if entry_date is not None:
            holding_days = _trading_days_between(bars, entry_date, latest_date)
            if entry_price <= 0:
                entry_price = _entry_close_from_bars(bars, entry_date)
            pnl_pct = current / entry_price - 1 if entry_price > 0 else 0.0
            max_holding_days = int(getattr(exit_cfg, "max_holding_days", 20) or 20)
            time_stop_days = int(getattr(exit_cfg, "time_stop_days", 7) or 7)
            min_profit = float(getattr(exit_cfg, "time_stop_min_profit_pct", 0.0) or 0.0)
            if bool(getattr(exit_cfg, "enable_max_holding_exit", True)) and holding_days >= max_holding_days:
                reasons.append(f"最长持仓退出{holding_days}日")
                exit_reason_keys.append("max_holding_days")
            elif bool(getattr(exit_cfg, "enable_time_stop", True)) and holding_days >= time_stop_days and pnl_pct < min_profit:
                reasons.append(f"时间止损{holding_days}日收益{pnl_pct:.1%}")
                exit_reason_keys.append("time_stop")

        failure_window = int(getattr(exit_cfg, "failure_window_days", 3) or 3)
        in_failure_window = True
        if entry_date is not None:
            in_failure_window = _trading_days_between(bars, entry_date, latest_date) <= failure_window
        if bool(getattr(exit_cfg, "enable_strategy_failure_exit", True)) and in_failure_window:
            if scheme_id == "trend_momentum" and current < ma20:
                reasons.append("动量失效退出跌破MA20")
                exit_reason_keys.append("trend_momentum_failed")
            elif scheme_id == "pullback" and current < low20 * 1.01:
                reasons.append("回调破位退出接近20日低点")
                exit_reason_keys.append("pullback_breakdown")
            elif scheme_id == "breakout":
                platform_high = _breakout_platform_high(bars, entry_date)
                if platform_high > 0 and current < platform_high:
                    reasons.append("突破失败退出跌回平台")
                    exit_reason_keys.append("breakout_failed")

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
            layer3_total=len(reasons),
            layer3_min_confirmations=1 if reasons else 0,
            layer3_condition_keys=exit_reason_keys or reasons,
            total_score=total,
            entry_reason="；".join(reasons),
            risk_tags=reasons,
            suggested_position_pct=0.0,
        ))
    results.sort(key=lambda s: s.total_score, reverse=True)
    return results


def _infer_scheme_id(context: Dict) -> str:
    raw = " ".join(str(context.get(k, "")) for k in ("scheme_id", "strategy_id", "strategy_name", "add_reason", "note"))
    if "trend_momentum" in raw or "强势" in raw or "追涨" in raw or "动量" in raw:
        return "trend_momentum"
    if "pullback" in raw or "回调" in raw or "低吸" in raw:
        return "pullback"
    if "breakout" in raw or "突破" in raw:
        return "breakout"
    return "balanced"


def _context_date(value) -> Optional[date]:
    if value in (None, ""):
        return None
    try:
        return _to_date(value)
    except Exception:
        return None


def _context_float(value) -> float:
    try:
        if value in (None, "") or pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _entry_close_from_bars(bars: pd.DataFrame, entry_date: date) -> float:
    matched = bars[bars["trade_date"].map(_to_date) <= entry_date]
    if matched.empty:
        return 0.0
    return float(matched.iloc[-1]["close"])


def _trading_days_between(bars: pd.DataFrame, start_date: date, end_date: date) -> int:
    """按K线交易日计算持仓天数，买入执行日为第0天。"""
    if bars.empty or start_date is None or end_date is None:
        return 0
    try:
        unique_dates = sorted({_to_date(d) for d in bars["trade_date"].dropna().tolist()})
        if start_date in unique_dates and end_date in unique_dates:
            return max(unique_dates.index(end_date) - unique_dates.index(start_date), 0)
        between = [d for d in unique_dates if start_date < d <= end_date]
        return max(len(between), 0)
    except Exception:
        return max(0, (end_date - start_date).days)


def _breakout_platform_high(bars: pd.DataFrame, entry_date: Optional[date]) -> float:
    b = bars.copy()
    if entry_date is not None:
        b = b[b["trade_date"].map(_to_date) < entry_date]
    prev = b.tail(15)
    if len(prev) < 5:
        return 0.0
    return float(prev["close"].astype(float).max())


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
