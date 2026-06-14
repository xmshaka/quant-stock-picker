"""Streamlit 看板 — 量化选股系统

性能优化：
- hotspot 通过 session_state 缓存，不阻塞首屏
- SignalEngine 只在需要时执行，不跟随每次 rerun
- 导航用 st.session_state + radio 替代多按钮触发 rerun
"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from data_loader import load_data, FACTOR_NAME_MAP, NAME_MAP
from signals.engine import SignalEngine, SignalFormatter
from signals.portfolio import PortfolioManager, PoolItem
from theme import (
    inject_theme, metric_row, section_header, badge, badge_html,
    empty_state, signal_card, topbar, C,
)

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
def compute_signals(factor_df_key, price_df_key, factor_names_key, weights_key):
    """用 hashable key 触发缓存，实际数据从 session_state 取。"""
    engine = SignalEngine(buy_threshold=0.7, sell_threshold=2.0, min_strength=2.0)
    buy, sell = engine.generate_signals(
        st.session_state._factor_df, st.session_state._price_df,
        st.session_state._factor_names, st.session_state._signal_weights, top_n=20,
    )
    return buy, sell

# ── 加载主数据 ──
_portfolio_syms = sorted({item.symbol for item in (pm.watch_list + pm.hold_list)})
_n_stocks = max(100, len(_portfolio_syms) + 100)

with st.spinner("加载行情数据..."):
    factor_df, price_df, factor_names = get_data(n_stocks=_n_stocks, include_symbols=_portfolio_syms)

# 存入 session_state 供缓存函数引用
st.session_state._factor_df = factor_df
st.session_state._price_df = price_df
st.session_state._factor_names = factor_names

latest_date = pd.to_datetime(factor_df['trade_date'].max()).strftime('%Y-%m-%d')

# ── 默认权重 ──
DEFAULT_WEIGHTS = {
    'rsi14': -0.1, 'macd_hist': 0.2, 'boll_position': 0.1,
    'volatility_20d': -0.1, 'max_dd_60d': -0.1,
    'north_hold_change': 0.4, 'turnover_ratio': 0.15, 'volume_ratio': 0.15,
    'pe_ttm': -0.2, 'pb': -0.1, 'ep': 0.1,
    'roe': 0.3, 'gross_margin': 0.2, 'revenue_growth': 0.2, 'profit_growth': 0.2,
    'momentum_5d': -0.15, 'momentum_20d': -0.1, 'momentum_60d': 0.0, 'liquidity': 0.1, 'reversal': 0.3,
}
if "signal_weights" not in st.session_state:
    st.session_state._signal_weights = DEFAULT_WEIGHTS.copy()

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

    detail = factor_df[(factor_df['symbol'] == symbol) & (factor_df['trade_date'] == latest_date)]
    day_data = factor_df[factor_df['trade_date'] == latest_date].copy()
    if detail.empty:
        st.warning("无该股票最新数据")
        return

    row = detail.iloc[0]
    factors = {k: v for k, v in row.items() if k not in ['symbol', 'trade_date'] and pd.notna(v)}

    engine = SignalEngine()
    regime = engine._detect_regime(symbol, price_df, day_data)

    st.markdown(f"**{fmt_name(symbol)}** {badge(regime, 'regime')}", unsafe_allow_html=True)

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
    # ── 权重面板 ──
    with st.expander("⚙ 因子权重调节", expanded=False):
        FACTOR_GROUPS = {
            '技术': ['rsi14', 'macd_hist', 'boll_position', 'volatility_20d', 'max_dd_60d'],
            '情绪': ['north_hold_change', 'turnover_ratio', 'volume_ratio'],
            '估值': ['pe_ttm', 'pb', 'ep'],
            '质量': ['roe', 'gross_margin', 'revenue_growth', 'profit_growth'],
            '动量': ['momentum_5d', 'momentum_20d', 'momentum_60d', 'liquidity', 'reversal'],
        }
        weights = {}
        cols = st.columns(2)
        col_idx = 0
        for gname, gfactors in FACTOR_GROUPS.items():
            available = [f for f in gfactors if f in factor_names]
            if not available:
                continue
            with cols[col_idx % 2]:
                st.caption(f"**{gname}**")
                for f in available:
                    default = st.session_state._signal_weights.get(f, DEFAULT_WEIGHTS.get(f, 0.0))
                    cn = FACTOR_NAME_MAP.get(f, f)
                    weights[f] = st.slider(cn, -1.0, 1.0, default, 0.05, key=f"ws_{f}")
            col_idx += 1

        if st.button("应用权重", type="primary", width="stretch"):
            st.session_state._signal_weights = weights
            # 清除信号缓存
            compute_signals.clear()
            st.rerun()

    # ── 计算信号 ──
    buy_signals, sell_signals = compute_signals(
        id(factor_df), id(price_df), tuple(factor_names),
        tuple(sorted(st.session_state._signal_weights.items())),
    )

    # ── 顶部指标 ──
    metric_row([
        {"label": "强力买入", "value": str(len([s for s in buy_signals if s.strength >= 4])), "color": "green"},
        {"label": "一般买入", "value": str(len([s for s in buy_signals if s.strength < 4])), "color": "yellow"},
        {"label": "强力卖出", "value": str(len([s for s in sell_signals if s.strength >= 4])), "color": "red"},
        {"label": "一般卖出", "value": str(len([s for s in sell_signals if s.strength < 4])), "color": "orange"},
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
                    meta=f"{s.strategy_name} · 强度 {s.strength} · 得分 {s.score:.2f}",
                    badges_html=badges_html,
                )
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
                    meta=f"{s.strategy_name} · 强度 {s.strength}",
                    badges_html=f"{regime_b} {risk_b}",
                )
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
                st.markdown(f'<div style="font-size:0.88rem;font-weight:600;">{fmt_name(item.symbol)} {hot_b}</div>', unsafe_allow_html=True)
                st.caption(f"{item.add_reason} · 强度 {item.signal_strength} · {item.add_date}")
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
                st.markdown(f'<div style="font-size:0.88rem;font-weight:600;">{fmt_name(item.symbol)} {hot_b}</div>', unsafe_allow_html=True)
                sent_label = ""
                if hot.get("has_hot"):
                    avg_s = hot.get("avg_sentiment", 0)
                    if avg_s > 0.2:
                        sent_label = badge("利好", "buy")
                    elif avg_s < -0.2:
                        sent_label = badge("利空", "sell")
                st.caption(f"{item.add_reason} · 强度 {item.signal_strength} · {item.add_date} {sent_label}")
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
