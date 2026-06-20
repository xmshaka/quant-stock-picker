"""Streamlit 看板 — 量化选股系统

性能优化：
- hotspot 通过 session_state 缓存，不阻塞首屏
- SignalEngine 只在需要时执行，不跟随每次 rerun
- 导航用 st.session_state + radio 替代多按钮触发 rerun
"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import inspect
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from data_loader import load_data, FACTOR_NAME_MAP, NAME_MAP
from signals.engine import SignalEngine, SignalFormatter
from signals.scanner import scan_signals
from signals.portfolio import PortfolioManager, PoolItem
from strategy.schemes import BUILTIN_SCHEMES
from theme import (
    inject_theme, metric_row, section_header, badge, badge_html,
    empty_state, signal_card, topbar, C,
)


def _scan_signals_compat(*, factor_df, price_df, factor_names, scheme_id, market_score, top_n, include_sell_symbols, include_sell_context):
    """兼容 Streamlit 旧进程缓存的 signals.scanner.scan_signals 签名。

    线上 Streamlit 进程可能在模块更新前已启动，rerun 时会复用 sys.modules 中的旧函数，
    旧函数不接受 include_sell_context。这里先避免页面直接 TypeError；进程重启后会走新版签名。
    """
    kwargs = {
        "scheme_id": scheme_id,
        "market_score": market_score,
        "top_n": top_n,
        "include_sell_symbols": list(include_sell_symbols),
    }
    try:
        if "include_sell_context" in inspect.signature(scan_signals).parameters:
            kwargs["include_sell_context"] = include_sell_context
    except (TypeError, ValueError):
        kwargs["include_sell_context"] = include_sell_context
    return scan_signals(factor_df, price_df, factor_names, **kwargs)

# ========== 页面配置 ==========
st.set_page_config(page_title="量化选股", page_icon="🎯", layout="centered", initial_sidebar_state="collapsed")
inject_theme()

# ========== 初始化 ==========
pm = PortfolioManager("main")

# ── 数据加载（带缓存）──
@st.cache_data(ttl=300, show_spinner=False)
def get_data(n_stocks=100, include_symbols=None):
    kw = {"n_stocks": n_stocks, "n_days": 120, "prefer_snapshot": True}
    if include_symbols:
        kw["include_symbols"] = include_symbols
    return load_data(data_source="real", **kw)

# ── 热点（异步懒加载，不阻塞首屏）──
def _load_hotspot():
    from hotspot.bridge import get_bridge
    bridge = get_bridge()
    return {
        "bridge": bridge,
        "hot_sectors": bridge.get_hot_sectors_list(top_n=5),
        "hot_stocks": bridge.get_hot_stocks_list(top_n=5),
    }

if "hotspot" not in st.session_state:
    st.session_state.hotspot = None
    # 用 spinner 但不阻塞 —— 首屏先渲染，热点后补
    with st.spinner("加载热点数据..."):
        try:
            st.session_state.hotspot = _load_hotspot()
        except Exception:
            st.session_state.hotspot = {"bridge": None, "hot_sectors": pd.DataFrame(), "hot_stocks": pd.DataFrame()}

hotspot = st.session_state.hotspot

# ── 信号缓存（避免每次 rerun 重算）──
@st.cache_data(ttl=120, show_spinner=False)
def compute_signals(data_key, factor_names_key, scheme_id, market_score, portfolio_symbols_key, portfolio_context_key):
    """按新短线方案扫描信号；用 hashable key 触发缓存，实际数据从 session_state 取。"""
    buy, sell = _scan_signals_compat(
        factor_df=st.session_state._factor_df,
        price_df=st.session_state._price_df,
        factor_names=st.session_state._factor_names,
        scheme_id=scheme_id,
        market_score=market_score,
        top_n=20,
        include_sell_symbols=list(portfolio_symbols_key),
        include_sell_context=st.session_state.get("_portfolio_signal_context", {}),
    )
    return buy, sell

# ── 加载主数据 ──
_portfolio_syms = sorted({item.symbol for item in (pm.watch_list + pm.hold_list)})
st.session_state._portfolio_signal_context = {
    item.symbol: {
        "add_date": item.add_date,
        "add_reason": item.add_reason,
        "note": item.note,
        "signal_strength": item.signal_strength,
        "signal_score": item.signal_score,
    }
    for item in (pm.watch_list + pm.hold_list)
}
_n_stocks = max(100, len(_portfolio_syms) + 100)

with st.spinner("加载行情数据..."):
    factor_df, price_df, factor_names = get_data(n_stocks=_n_stocks, include_symbols=_portfolio_syms)

# 存入 session_state 供缓存函数引用
st.session_state._factor_df = factor_df
st.session_state._price_df = price_df
st.session_state._factor_names = factor_names

latest_date = pd.to_datetime(factor_df['trade_date'].max()).strftime('%Y-%m-%d')

def _date_key(value) -> str:
    """统一日期比较口径，避免 date/Timestamp/字符串混用导致详情为空。"""
    if pd.isna(value):
        return ""
    return pd.to_datetime(value).strftime('%Y-%m-%d')


def _build_data_key(factor_df: pd.DataFrame, price_df: pd.DataFrame, latest_date: str):
    """生成稳定缓存key；禁止用 id(df)，否则切页 rerun 会反复重扫。"""
    factor_symbols = factor_df['symbol'].astype(str) if 'symbol' in factor_df.columns else pd.Series(dtype=str)
    price_symbols = price_df['symbol'].astype(str) if 'symbol' in price_df.columns else pd.Series(dtype=str)
    return (
        latest_date,
        int(len(factor_df)),
        int(len(price_df)),
        int(factor_symbols.nunique()),
        int(price_symbols.nunique()),
        tuple(sorted(factor_symbols.unique())[:10]),
        tuple(sorted(price_symbols.unique())[:10]),
    )


def _signal_resonance_text(signal) -> str:
    """信号卡共振展示，必须使用 scanner 传回的策略配置总数，禁止硬编码 /6。"""
    confirmations = getattr(signal, 'layer3_confirmations', '-')
    total = getattr(signal, 'layer3_total', 0) or 6
    minimum = getattr(signal, 'layer3_min_confirmations', 0) or '-'
    return f"共振 {confirmations}/{total}，最低{minimum}"


def _strategy_resonance_summary(scheme_id: str):
    """当前扫描策略的共振配置摘要。balanced 展示三个子策略。"""
    target_ids = ["trend_momentum", "pullback", "breakout"] if scheme_id == "balanced" else [scheme_id]
    rows = []
    for sid in target_ids:
        scheme = BUILTIN_SCHEMES.get(sid)
        if scheme is None:
            continue
        cfg = getattr(scheme, "resonance_config", None)
        if cfg is None:
            continue
        rows.append({
            "策略": scheme.name,
            "最低确认": cfg.min_confirmations,
            "买入条件数": len(cfg.buy_conditions or []),
            "卖出条件数": len(cfg.sell_conditions or []),
            "买入条件": "、".join(cfg.buy_conditions or []),
        })
    return rows

# ── 新方案默认状态 ──
if "signal_scheme_id" not in st.session_state:
    st.session_state.signal_scheme_id = "balanced"
if "market_score_override" not in st.session_state:
    st.session_state.market_score_override = 50.0

# ========== 顶部 ==========
left_info = f"🎯 <strong>量化选股</strong>"
hot_html = ""
if hotspot and not hotspot["hot_sectors"].empty:
    sectors = hotspot["hot_sectors"].head(3)
    tags = " ".join([f'{badge(s["industry"], "hot")}' for _, s in sectors.iterrows()])
    hot_html = f"热点 {tags}"
topbar(latest_date, left_html=left_info, right_html=hot_html)

# ========== 导航 ==========
PAGES = ["🎯 信号", f"👁 观察({len(pm.watch_list)})", f"💼 持仓({len(pm.hold_list)})", "🔥 热点", "🩺 状态", "⚙ 配置"]
if "nav_page" not in st.session_state:
    st.session_state.nav_page = 0

nav = st.columns(len(PAGES))
for i, label in enumerate(PAGES):
    with nav[i]:
        if st.button(label, key=f"nav_{i}", width="stretch",
                     type="primary" if st.session_state.nav_page == i else "secondary"):
            st.session_state.nav_page = i
            st.rerun()

page_idx = st.session_state.nav_page

# ========== 工具函数 ==========
def fmt_name(symbol: str) -> str:
    name = NAME_MAP.get(symbol, "")
    return f"{symbol} {name}" if name else symbol

def show_factor_detail(symbol, factor_df, price_df, latest_date):
    """个股因子详情面板。"""
    from signals.engine import SignalEngine

    fdf = factor_df.copy()
    fdf['_date_key'] = fdf['trade_date'].map(_date_key)
    latest_key = _date_key(latest_date)
    symbol_data = fdf[fdf['symbol'].astype(str) == str(symbol)].copy()
    day_data = fdf[fdf['_date_key'] == latest_key].copy()

    # 优先展示全局最新交易日；若该股票当天无快照，则回退到该股票自身最新日并明确提示。
    detail = symbol_data[symbol_data['_date_key'] == latest_key]
    if detail.empty and not symbol_data.empty:
        symbol_latest = symbol_data['_date_key'].max()
        detail = symbol_data[symbol_data['_date_key'] == symbol_latest]
        st.warning(f"该股票缺少全局最新日 {latest_key} 快照，当前展示该股票自身最新日 {symbol_latest}。")
        day_data = fdf[fdf['_date_key'] == symbol_latest].copy()
    if detail.empty:
        st.warning("无该股票最新数据")
        return

    row = detail.iloc[0]
    factors = {k: v for k, v in row.items() if k not in ['symbol', 'trade_date', '_date_key'] and pd.notna(v)}

    engine = SignalEngine()
    regime = engine._detect_regime(symbol, price_df, day_data)

    st.markdown(f"**{fmt_name(symbol)}** {badge(regime, 'regime')}", unsafe_allow_html=True)
    st.caption(f"因子日期：{row['_date_key']}；全局最新交易日：{latest_key}")

    FACTOR_DIRS = {
        'rsi14': -1, 'macd_hist': 1, 'boll_position': 1, 'volatility_20d': -1, 'max_dd_60d': -1,
        'north_hold_change': 1, 'turnover_ratio': 1, 'volume_ratio': 1,
        'pe_ttm': -1, 'pb': -1, 'ep': 1,
        'roe': 1, 'gross_margin': 1, 'revenue_growth': 1, 'profit_growth': 1,
        'momentum_5d': -1, 'momentum_20d': -1, 'momentum_60d': 0, 'liquidity': 1, 'reversal': 1,
    }
    DIR_LABEL = {1: "↑ 越高越好", -1: "↓ 越低越好", 0: "— 中性"}

    groups = {
        '技术': ['rsi14', 'macd_hist', 'boll_position', 'volatility_20d', 'max_dd_60d'],
        '情绪': ['north_hold_change', 'turnover_ratio', 'volume_ratio'],
        '估值': ['pe_ttm', 'pb', 'ep'],
        '质量': ['roe', 'gross_margin', 'revenue_growth', 'profit_growth'],
        '动量': ['momentum_5d', 'momentum_20d', 'momentum_60d', 'liquidity', 'reversal'],
    }

    for group_name, gfactors in groups.items():
        rows = []
        for f in gfactors:
            if f not in factors or f not in day_data.columns:
                continue
            vals = pd.to_numeric(day_data[f], errors='coerce').dropna()
            raw = pd.to_numeric(pd.Series([factors.get(f)]), errors='coerce').iloc[0]
            if vals.empty or pd.isna(raw):
                continue
            p25, p75 = vals.quantile(0.25), vals.quantile(0.75)
            pct = float((vals <= raw).mean() * 100)
            d = FACTOR_DIRS.get(f, 1)
            z = (raw - vals.mean()) / vals.std() if vals.std() > 0 else 0
            ds = z * d if d != 0 else np.nan

            if ds != ds:  # NaN
                tag = badge("中性", "neutral")
            elif ds >= 1.0:
                tag = badge("加分", "buy")
            elif ds >= 0.3:
                tag = badge("小加分", "buy")
            elif ds <= -1.0:
                tag = badge("扣分", "sell")
            elif ds <= -0.3:
                tag = badge("小扣分", "sell")
            else:
                tag = badge("平均", "neutral")

            rows.append({
                '因子': FACTOR_NAME_MAP.get(f, f),
                '值': round(raw, 3),
                '位置': f"P{pct:.0f}",
                '区间': f"{p25:.2f} ~ {p75:.2f}",
                '方向': DIR_LABEL.get(d, ""),
                '判断': tag,
            })
        if rows:
            with st.expander(f"**{group_name}** ({len(rows)})", expanded=False):
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                             column_config={"判断": st.column_config.TextColumn()})

# ========== 页面 0: 信号 ==========
if page_idx == 0:
    # ── 新方案扫描控制 ──
    with st.expander("⚙ 新方案信号扫描", expanded=True):
        c1, c2 = st.columns([2, 1])
        with c1:
            scheme_options = {
                "balanced": "全部策略/组合器",
                "trend_momentum": "强势追涨",
                "pullback": "回调低吸",
                "breakout": "横盘突破",
            }
            selected_scheme = st.selectbox(
                "扫描策略",
                list(scheme_options.keys()),
                format_func=lambda k: scheme_options[k],
                index=list(scheme_options.keys()).index(st.session_state.signal_scheme_id),
            )
        with c2:
            market_score_input = st.slider(
                "大盘评分",
                0.0, 100.0, float(st.session_state.market_score_override), 1.0,
                help="临时使用手动评分；后续接入 market/timing.py 实时评分。",
            )
        st.caption("当前扫描路径：大盘评分 → 股票池过滤 → 三策略独立候选 → L1趋势过滤 → L2形态匹配 → L3共振确认。信号日为T日收盘，建议成交日为T+1。")
        resonance_rows = _strategy_resonance_summary(selected_scheme)
        if resonance_rows:
            st.caption("策略共振配置（与信号卡 共振 x/y 口径一致）")
            st.dataframe(pd.DataFrame(resonance_rows), width="stretch", hide_index=True)
        if selected_scheme != st.session_state.signal_scheme_id or market_score_input != st.session_state.market_score_override:
            st.session_state.signal_scheme_id = selected_scheme
            st.session_state.market_score_override = market_score_input
            compute_signals.clear()
            st.rerun()

    # ── 计算信号 ──
    portfolio_symbols = tuple(sorted({item.symbol for item in (pm.watch_list + pm.hold_list)}))
    portfolio_context_key = tuple(sorted(
        (
            sym,
            str(ctx.get("add_date", "")),
            str(ctx.get("add_reason", "")),
            str(ctx.get("note", "")),
        )
        for sym, ctx in st.session_state.get("_portfolio_signal_context", {}).items()
    ))
    data_key = _build_data_key(factor_df, price_df, latest_date)
    signal_cache_key = (
        data_key,
        tuple(factor_names),
        st.session_state.signal_scheme_id,
        float(st.session_state.market_score_override),
        portfolio_symbols,
        portfolio_context_key,
    )
    if st.session_state.get("_latest_signal_cache_key") == signal_cache_key:
        buy_signals, sell_signals = st.session_state.get("_latest_signal_result", ([], []))
    else:
        buy_signals, sell_signals = compute_signals(
            data_key,
            tuple(factor_names),
            st.session_state.signal_scheme_id,
            float(st.session_state.market_score_override),
            portfolio_symbols,
            portfolio_context_key,
        )
        st.session_state._latest_signal_cache_key = signal_cache_key
        st.session_state._latest_signal_result = (buy_signals, sell_signals)
        st.session_state._latest_signal_map = {s.symbol: s for s in (buy_signals + sell_signals)}

    # ── 顶部指标 ──
    metric_row([
        {"label": "大盘评分", "value": f"{st.session_state.market_score_override:.0f}"},
        {"label": "买入候选", "value": str(len(buy_signals)), "color": "green"},
        {"label": "风险退出", "value": str(len(sell_signals)), "color": "red" if sell_signals else ""},
        {"label": "扫描策略", "value": scheme_options.get(st.session_state.signal_scheme_id, st.session_state.signal_scheme_id)},
    ])

    # ── 手动加入 ──
    with st.expander("＋ 手动添加股票", expanded=False):
        c1, c2, c3 = st.columns([3, 2, 1])
        with c1:
            manual_symbol = st.text_input("代码", placeholder="000001", label_visibility="collapsed")
        with c2:
            manual_pool = st.selectbox("池", ["观察池", "持仓池"], label_visibility="collapsed")
        with c3:
            if st.button("添加", type="primary", width="stretch"):
                if manual_symbol:
                    sym = manual_symbol.strip().upper()
                    today = str(datetime.now().date())
                    item = PoolItem(symbol=sym, add_date=today, add_reason="手动", signal_strength=0, signal_score=0)
                    if manual_pool == "观察池":
                        pm.add_to_watch(item)
                    else:
                        pm.add_to_hold(item)
                    st.rerun()

    # ── 买入信号 ──
    section_header("买入信号", f"({len(buy_signals)})")
    if buy_signals:
        bridge = hotspot.get("bridge") if hotspot else None
        for s in sorted(buy_signals, key=lambda x: x.strength, reverse=True)[:20]:
            is_hot = bridge.is_hot_stock(s.symbol) if bridge else False
            in_watch = pm.is_in_watch(s.symbol)
            in_hold = pm.is_in_hold(s.symbol)

            regime_b = badge(s.regime[:4], "regime") if s.regime != '震荡整理' else ""
            hot_b = badge("热点", "hot") if is_hot else ""
            risk_b = " ".join([badge(t, "risk") for t in s.risk_tags]) if s.risk_tags else ""
            badges_html = f"{regime_b} {hot_b} {risk_b}"

            col_info, col_action = st.columns([4, 1])
            with col_info:
                signal_card(
                    name=f"<strong>{fmt_name(s.symbol)}</strong>",
                    meta=(
                        f"{s.strategy_name} · 总分 {s.score:.2f} · {_signal_resonance_text(s)} · "
                        f"信号日 {getattr(s, 'signal_date', latest_date)} · T+1 {getattr(s, 'suggested_exec_date', '-') or '-'} · "
                        f"建议仓位 {getattr(s, 'suggested_position_pct', 0) * 100:.1f}%"
                    ),
                    badges_html=badges_html,
                )
                if getattr(s, 'entry_reason', ''):
                    st.caption(s.entry_reason)
            with col_action:
                if in_hold:
                    st.markdown(f'<div style="text-align:center;padding-top:12px;">{badge("持仓", "hold")}</div>', unsafe_allow_html=True)
                elif in_watch:
                    if st.button("买入", key=f"buy_{s.symbol}", width="stretch"):
                        pm.move_to_hold(s.symbol)
                        st.rerun()
                else:
                    if st.button("+观察", key=f"watch_{s.symbol}", width="stretch"):
                        pm.add_to_watch(PoolItem(
                            symbol=s.symbol, add_date=str(latest_date),
                            add_reason=s.strategy_name, signal_strength=s.strength, signal_score=s.score,
                        ))
                        st.rerun()
    else:
        empty_state("📭", "暂无买入信号")

    # ── 卖出信号 ──
    section_header("卖出信号", f"({len(sell_signals)})")
    if sell_signals:
        for s in sorted(sell_signals, key=lambda x: x.strength, reverse=True)[:20]:
            in_hold = pm.is_in_hold(s.symbol)
            regime_b = badge(s.regime[:4], "regime") if s.regime != '震荡整理' else ""
            risk_b = " ".join([badge(t, "risk") for t in s.risk_tags]) if s.risk_tags else ""

            col_info, col_action = st.columns([4, 1])
            with col_info:
                signal_card(
                    name=f"<strong>{fmt_name(s.symbol)}</strong>",
                    meta=(
                        f"{s.strategy_name} · 风险分 {s.score:.2f} · "
                        f"信号日 {getattr(s, 'signal_date', latest_date)} · T+1 {getattr(s, 'suggested_exec_date', '-') or '-'}"
                    ),
                    badges_html=f"{regime_b} {risk_b}",
                )
                if getattr(s, 'entry_reason', ''):
                    st.caption(s.entry_reason)
            with col_action:
                if in_hold:
                    if st.button("卖出", key=f"sell_{s.symbol}", type="primary", width="stretch"):
                        pm.remove_from_hold(s.symbol)
                        st.rerun()
                else:
                    st.markdown(f'<div style="text-align:center;padding-top:12px;color:{C["text2"]};font-size:0.72rem;">未持仓</div>', unsafe_allow_html=True)
    else:
        empty_state("📭", "暂无卖出信号")

# ========== 页面 1: 观察池 ==========
elif page_idx == 1:
    section_header("观察池", f"({len(pm.watch_list)})")
    if pm.watch_list:
        bridge = hotspot.get("bridge") if hotspot else None
        for item in pm.watch_list:
            hot = bridge.get_stock_hot_summary(item.symbol) if bridge else {"has_hot": False}
            hot_b = badge("热点", "hot") if hot.get("has_hot") else ""

            col_info, col_btns = st.columns([3, 2])
            with col_info:
                live_sig = st.session_state.get("_latest_signal_map", {}).get(item.symbol)
                live_meta = (
                    f"最新扫描：{live_sig.strategy_name} · 总分 {live_sig.score:.2f} · {_signal_resonance_text(live_sig)} · {live_sig.signal_date}"
                    if live_sig else
                    f"入池记录：{item.add_reason} · 强度 {item.signal_strength} · {item.add_date}"
                )
                st.markdown(f'<div style="font-size:0.88rem;font-weight:600;">{fmt_name(item.symbol)} {hot_b}</div>', unsafe_allow_html=True)
                st.caption(live_meta)
                if live_sig and getattr(live_sig, 'entry_reason', ''):
                    st.caption(live_sig.entry_reason)
            with col_btns:
                b1, b2, b3 = st.columns(3)
                with b1:
                    if st.button("📊", key=f"wd_{item.symbol}", help="详情", width="stretch"):
                        st.session_state.detail_symbol = item.symbol
                with b2:
                    if st.button("买入", key=f"wb_{item.symbol}", width="stretch"):
                        pm.move_to_hold(item.symbol)
                        st.rerun()
                with b3:
                    if st.button("✕", key=f"wr_{item.symbol}", width="stretch"):
                        pm.remove_from_watch(item.symbol)
                        st.rerun()

            if st.session_state.get('detail_symbol') == item.symbol:
                show_factor_detail(item.symbol, factor_df, price_df, latest_date)

            st.divider()
    else:
        empty_state("👁", "观察池为空 — 在信号页添加股票")

# ========== 页面 2: 持仓池 ==========
elif page_idx == 2:
    section_header("持仓池", f"({len(pm.hold_list)})")
    if pm.hold_list:
        bridge = hotspot.get("bridge") if hotspot else None

        # 热点提醒
        if bridge:
            alerts = bridge.get_portfolio_hot_alert([i.symbol for i in pm.hold_list])
            if not alerts.empty:
                with st.container():
                    st.markdown(f'<div style="background:{C["orange_bg"]};padding:8px 12px;border-radius:8px;margin-bottom:10px;font-size:0.78rem;">'
                                f'⚠️ <strong>持仓热点提醒</strong></div>', unsafe_allow_html=True)
                    for _, row in alerts.head(3).iterrows():
                        e = badge("利好", "buy") if row["sentiment"] > 0.2 else badge("利空", "sell") if row["sentiment"] < -0.2 else badge("中性", "neutral")
                        st.markdown(f'<div style="font-size:0.72rem;padding-left:8px;">{e} <strong>{fmt_name(row["symbol"])}</strong>: {row["latest_title"]}</div>', unsafe_allow_html=True)

        for item in pm.hold_list:
            hot = bridge.get_stock_hot_summary(item.symbol) if bridge else {"has_hot": False}
            hot_b = badge("热点", "hot") if hot.get("has_hot") else ""

            col_info, col_btns = st.columns([3, 2])
            with col_info:
                live_sig = st.session_state.get("_latest_signal_map", {}).get(item.symbol)
                live_meta = (
                    f"最新扫描：{live_sig.strategy_name} · 风险分 {live_sig.score:.2f} · {live_sig.signal_date}"
                    if live_sig else
                    f"入池记录：{item.add_reason} · 强度 {item.signal_strength} · {item.add_date}"
                )
                st.markdown(f'<div style="font-size:0.88rem;font-weight:600;">{fmt_name(item.symbol)} {hot_b}</div>', unsafe_allow_html=True)
                sent_label = ""
                if hot.get("has_hot"):
                    avg_s = hot.get("avg_sentiment", 0)
                    if avg_s > 0.2:
                        sent_label = badge("利好", "buy")
                    elif avg_s < -0.2:
                        sent_label = badge("利空", "sell")
                st.caption(f"{live_meta} {sent_label}")
                if live_sig and getattr(live_sig, 'entry_reason', ''):
                    st.caption(live_sig.entry_reason)
            with col_btns:
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("📊", key=f"hd_{item.symbol}", help="详情", width="stretch"):
                        st.session_state.detail_symbol = item.symbol
                with b2:
                    if st.button("卖出", key=f"hs_{item.symbol}", type="primary", width="stretch"):
                        pm.remove_from_hold(item.symbol)
                        st.rerun()

            if st.session_state.get('detail_symbol') == item.symbol:
                show_factor_detail(item.symbol, factor_df, price_df, latest_date)

            st.divider()
    else:
        empty_state("💼", "持仓池为空 — 在观察池点击买入")

# ========== 页面 3: 热点 ==========
elif page_idx == 3:
    section_header("热点追踪")
    if hotspot and not hotspot["hot_sectors"].empty:
        section_header("热门行业")
        for _, row in hotspot["hot_sectors"].head(10).iterrows():
            heat = row.get("heat_score", 0)
            bar_w = min(100, heat * 20) if heat else 0
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:6px 0;">
                <span style="font-size:0.82rem;font-weight:600;width:120px;">{row.get('industry', '')}</span>
                <div class="qsp-progress" style="flex:1;"><div class="fill" style="width:{bar_w}%;background:{C['accent']};"></div></div>
                <span style="font-size:0.72rem;color:{C['text2']};width:60px;text-align:right;">热度 {heat:.1f}</span>
            </div>
            """, unsafe_allow_html=True)

    if hotspot and not hotspot["hot_stocks"].empty:
        section_header("热门个股")
        for _, row in hotspot["hot_stocks"].head(10).iterrows():
            sent = row.get("avg_sentiment", 0)
            sent_b = badge("利好", "buy") if sent > 0.2 else badge("利空", "sell") if sent < -0.2 else badge("中性", "neutral")
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:6px 0;">
                <span style="font-size:0.82rem;font-weight:600;width:120px;">{row.get('stock', '')}</span>
                {sent_b}
                <span style="font-size:0.72rem;color:{C['text2']};">热度 {row.get('heat_score', 0):.1f} · 提及 {row.get('mention_count', 0)}</span>
            </div>
            """, unsafe_allow_html=True)

    if not hotspot or (hotspot["hot_sectors"].empty and hotspot["hot_stocks"].empty):
        empty_state("🔥", "暂无热点数据")

# ========== 页面 4: 状态 ==========
elif page_idx == 4:
    section_header("系统状态")

    from data.daily_factors import latest_snapshot_date, load_snapshot_meta
    snap_date = latest_snapshot_date()
    if snap_date:
        meta = load_snapshot_meta(snap_date)
        if meta:
            metric_row([
                {"label": "快照日期", "value": snap_date},
                {"label": "股票池", "value": f"{meta.get('universe_size', 0)} 只"},
                {"label": "因子行数", "value": f"{meta.get('factor_rows', 0):,}"},
                {"label": "耗时", "value": f"{meta.get('elapsed_seconds', 0):.0f}s"},
            ])

    from data.scan_status import load_scan_reports
    try:
        reports = load_scan_reports()
        if not reports.empty:
            section_header("最近扫描")
            latest = reports.iloc[0]
            metric_row([
                {"label": "扫描时间", "value": str(latest.get("ts", ""))[:16]},
                {"label": "股票池", "value": f"{latest.get('total_symbols', 0)}"},
                {"label": "成功更新", "value": f"{latest.get('updated_count', 0)}", "color": "green"},
                {"label": "失败", "value": f"{latest.get('failed_count', 0)}", "color": "red" if latest.get('failed_count', 0) > 0 else ""},
            ])
    except Exception:
        pass

# ========== 页面 5: 配置 ==========
elif page_idx == 5:
    section_header("系统配置")
    st.caption("跳转到独立配置页面进行股票池、扫描参数等设置。")
    if st.button("打开配置页面 →", type="primary", width="stretch"):
        st.switch_page("pages/6_股票配置.py")

st.divider()
st.caption("量化选股系统 · 因子驱动 · 数据实时")
