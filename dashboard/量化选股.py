"""Streamlit 看板 - 移动端适配版"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import streamlit as st
import pandas as pd
import numpy as np

from data_loader import load_data, FACTOR_NAME_MAP, NAME_MAP
from signals.engine import SignalEngine, SignalFormatter
from signals.portfolio import PortfolioManager, PoolItem
from hotspot.bridge import get_bridge
from datetime import datetime

# ========== 页面配置 ==========
st.set_page_config(
    page_title="选股信号",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ========== 移动端优化CSS ==========
st.markdown("""
<style>
    html { font-size: 14px; }
    h1 { font-size: 1.3rem !important; margin-top: 1.0rem !important; margin-bottom: 0.3rem !important; }
    h2 { font-size: 1.1rem !important; margin-top: 0.6rem !important; margin-bottom: 0.3rem !important; }
    h3 { font-size: 1rem !important; margin-top: 0.4rem !important; }
    .block-container { padding-top: 3.5rem; padding-left: 0.8rem; padding-right: 0.8rem; max-width: 100%; }
    .stMetric { background: #f8f9fa; border-radius: 6px; padding: 6px 4px; }
    .stMetric label { font-size: 0.7rem !important; color: #666; }
    .stMetric div[data-testid="stMetricValue"] { font-size: 1rem !important; font-weight: 600; }
    .stButton > button { padding: 0.3rem 0.6rem; font-size: 0.8rem; min-height: 32px; }
    .stDataFrame td { font-size: 0.8rem; padding: 4px 6px !important; }
    .stDataFrame th { font-size: 0.75rem; padding: 4px 6px !important; }
    .stSlider { padding-bottom: 0.3rem; }
    hr { margin: 0.5rem 0 !important; }
    .factor-detail { background: #f0f2f6; padding: 8px; border-radius: 6px; margin: 4px 0; }
    .factor-detail td { font-size: 0.75rem !important; }
    
    @media (max-width: 768px) {
        .row-widget.stHorizontalBlock { flex-direction: column !important; }
        .row-widget.stHorizontalBlock > div { width: 100% !important; }
    }
</style>
""", unsafe_allow_html=True)

# ========== 初始化 Portfolio ==========
pm = PortfolioManager("main")

# ========== 数据加载 ==========
@st.cache_data(ttl=300)
def get_data(_version=3):
    # _version 用于强制刷新 Streamlit 缓存
    return load_data(data_source="real", n_stocks=100, n_days=120)

@st.cache_data(ttl=300)
def get_hotspot_summary():
    bridge = get_bridge()
    return {
        "hot_sectors": bridge.get_hot_sectors_list(top_n=5),
        "hot_stocks": bridge.get_hot_stocks_list(top_n=5),
    }

factor_df, price_df, factor_names = get_data()
hotspot_summary = get_hotspot_summary()
latest_date = pd.to_datetime(factor_df['trade_date'].max()).strftime('%Y-%m-%d')

# ========== 辅助函数 ==========
def fmt_name(symbol: str) -> str:
    """返回 '代码 名称' 格式"""
    from data_loader import NAME_MAP
    name = NAME_MAP.get(symbol, "")
    return f"{symbol} {name}" if name else symbol

def show_factor_detail(symbol, factor_df, latest_date):
    """显示个股因子详情"""
    detail_data = factor_df[(factor_df['symbol'] == symbol) & (factor_df['trade_date'] == latest_date)]
    if detail_data.empty:
        st.warning("无该股票最新数据")
        return
    
    row = detail_data.iloc[0]
    factors = {k: v for k, v in row.items() if k not in ['symbol', 'trade_date'] and pd.notna(v)}
    
    st.markdown(f"**📊 因子详情 — {fmt_name(symbol)}**")
    
    groups = {
        '技术因子': ['rsi14', 'macd_hist', 'boll_position', 'boll_width', 'volatility_20d', 'max_dd_60d'],
        '情绪因子': ['north_hold_change', 'margin_change', 'turnover_ratio'],
        '估值因子': ['pe_ttm', 'pb', 'ep'],
        '质量因子': ['roe', 'gross_margin', 'revenue_growth', 'profit_growth'],
        '动量/流动性': ['momentum_20d', 'momentum_60d', 'liquidity', 'reversal'],
    }
    
    for group_name, group_factors in groups.items():
        group_data = {FACTOR_NAME_MAP.get(f, f): round(factors.get(f, np.nan), 3) 
                     for f in group_factors if f in factors}
        if group_data:
            with st.expander(f"**{group_name}**", expanded=False):
                df = pd.DataFrame([group_data]).T.reset_index()
                df.columns = ['因子', '值']
                st.dataframe(df, use_container_width=True, hide_index=True,
                            column_config={"因子": st.column_config.TextColumn(width="medium"),
                                         "值": st.column_config.NumberColumn(width="small", format="%.3f")})
    
    if st.button("关闭详情", key=f"close_{symbol}"):
        st.session_state.detail_symbol = None
        st.rerun()

# ========== 页面标题 ==========
st.title(f"🎯 量化选股 | {latest_date}")

# ========== 热点概览条 ==========
if not hotspot_summary["hot_sectors"].empty or not hotspot_summary["hot_stocks"].empty:
    with st.container():
        hot_html = '<div style="background:#fff3e0;padding:6px 10px;border-radius:6px;margin-bottom:8px;font-size:0.75rem;">'
        hot_html += '<span style="color:#e65100;font-weight:600;">🔥 今日热点:</span> '
        sectors = hotspot_summary["hot_sectors"]
        if not sectors.empty:
            sector_tags = " ".join([f'<span style="background:#ffcc80;padding:1px 5px;border-radius:8px;margin-right:4px;">{row["industry"]}</span>' for _, row in sectors.head(3).iterrows()])
            hot_html += sector_tags
        stocks = hotspot_summary["hot_stocks"]
        if not stocks.empty:
            stock_tags = " ".join([f'<span style="background:#c8e6c9;padding:1px 5px;border-radius:8px;margin-right:4px;">{row["stock"]}</span>' for _, row in stocks.head(3).iterrows()])
            hot_html += " | " + stock_tags
        hot_html += '</div>'
        st.markdown(hot_html, unsafe_allow_html=True)

# ========== 顶部导航 ==========
nav_col1, nav_col2, nav_col3, nav_col4, nav_col5, nav_col6 = st.columns(6)
with nav_col1:
    if st.button("🎯 信号", use_container_width=True, type="primary" if st.session_state.get('page', '信号') == '信号' else "secondary"):
        st.session_state.page = '信号'
        st.rerun()
with nav_col2:
    if st.button(f"👁️ 观察({len(pm.watch_list)})", use_container_width=True, type="primary" if st.session_state.get('page') == '观察' else "secondary"):
        st.session_state.page = '观察'
        st.rerun()
with nav_col3:
    if st.button(f"💼 持仓({len(pm.hold_list)})", use_container_width=True, type="primary" if st.session_state.get('page') == '持仓' else "secondary"):
        st.session_state.page = '持仓'
        st.rerun()
with nav_col4:
    if st.button("🔥 热点", use_container_width=True):
        st.switch_page("pages/4_热点追踪.py")
with nav_col5:
    if st.button("🩺 状态", use_container_width=True):
        st.switch_page("pages/5_数据状态.py")
with nav_col6:
    if st.button("⚙️ 配置", use_container_width=True):
        st.switch_page("pages/6_股票配置.py")

if 'page' not in st.session_state:
    st.session_state.page = '信号'

page = st.session_state.page

# ========== 信号页面 ==========
if page == '信号':
    with st.expander("⚙️ 因子权重", expanded=False):
        FACTOR_GROUPS = {
            '技术': ['rsi14', 'macd_hist', 'boll_position', 'boll_width', 'volatility_20d', 'max_dd_60d'],
            '情绪': ['north_hold_change', 'margin_change', 'turnover_ratio', 'volume_ratio'],
            '估值': ['pe_ttm', 'pb', 'ep'],
            '质量': ['roe', 'gross_margin', 'revenue_growth', 'profit_growth'],
            '动量': ['momentum_20d', 'momentum_60d', 'liquidity', 'reversal'],
        }
        DEFAULT_WEIGHTS = {
            'rsi14': -0.1, 'macd_hist': 0.2, 'boll_position': 0.1, 'boll_width': 0.0,
            'volatility_20d': -0.1, 'max_dd_60d': -0.1,
            'north_hold_change': 0.3, 'margin_change': 0.2, 'turnover_ratio': 0.1, 'volume_ratio': 0.1,
            'pe_ttm': -0.2, 'pb': -0.1, 'ep': 0.1,
            'roe': 0.3, 'gross_margin': 0.2, 'revenue_growth': 0.2, 'profit_growth': 0.2,
            'momentum_20d': -0.1, 'momentum_60d': 0.0, 'liquidity': 0.1, 'reversal': 0.3,
        }
        factor_weights = {}
        for group_name, group_factors in FACTOR_GROUPS.items():
            available = [f for f in group_factors if f in factor_names]
            if not available:
                continue
            st.caption(f"**{group_name}**")
            for f in available:
                default = DEFAULT_WEIGHTS.get(f, 0.0)
                cn = FACTOR_NAME_MAP.get(f, f)
                factor_weights[f] = st.slider(cn, -1.0, 1.0, default, 0.05, key=f"w_{f}")

    engine = SignalEngine(buy_threshold=0.7, sell_threshold=0.3, min_strength=2.0)
    buy_signals, sell_signals = engine.generate_signals(
        factor_df, price_df, factor_names, factor_weights if 'factor_weights' in locals() else {}, top_n=20
    )

    c1, c2 = st.columns(2)
    c1.metric("强力买入", len([s for s in buy_signals if s.strength >= 4]))
    c2.metric("强力卖出", len([s for s in sell_signals if s.strength >= 4]))

    with st.expander("➕ 手动加入", expanded=False):
        col1, col2 = st.columns([3, 1])
        with col1:
            manual_symbol = st.text_input("股票代码", placeholder="如 000001", label_visibility="collapsed")
        with col2:
            manual_pool = st.selectbox("池子", ["观察池", "持仓池"], label_visibility="collapsed")
        if st.button("加入", type="primary", use_container_width=True):
            if manual_symbol:
                symbol = manual_symbol.strip().upper()
                today = str(datetime.now().date())
                if manual_pool == "观察池":
                    pm.add_to_watch(PoolItem(symbol=symbol, add_date=today, add_reason="手动", signal_strength=0, signal_score=0))
                    st.success(f"{symbol} → 观察池")
                else:
                    pm.add_to_hold(PoolItem(symbol=symbol, add_date=today, add_reason="手动", signal_strength=0, signal_score=0))
                    st.success(f"{symbol} → 持仓池")
                st.rerun()
            else:
                st.warning("请输入代码")

    st.subheader(f"🟢 买入 ({len(buy_signals)})")
    if buy_signals:
        bridge = get_bridge()
        for s in sorted(buy_signals, key=lambda x: x.strength, reverse=True)[:10]:
            in_watch = pm.is_in_watch(s.symbol)
            in_hold = pm.is_in_hold(s.symbol)
            hot_badge = bridge.get_hot_badge(s.symbol)
            is_hot = bridge.is_hot_stock(s.symbol)
            with st.container():
                col_info, col_btn = st.columns([3, 1])
                with col_info:
                    hot_tag = f" <span style='background:#ff9800;color:white;padding:1px 5px;border-radius:8px;font-size:0.65rem;'>热点</span>" if is_hot else ""
                    st.markdown(f"**{fmt_name(s.symbol)}** {s.emoji}{hot_badge} | {s.strategy_name}{hot_tag}", unsafe_allow_html=True)
                    caption = f"强度:{s.strength} 得分:{s.score:.2f}"
                    if is_hot:
                        hot_news = bridge.get_stock_hot_summary(s.symbol)
                        if hot_news["has_hot"]:
                            caption += f" | 热度+{hot_news['heat_score']:.0f}"
                    st.caption(caption)
                with col_btn:
                    if in_hold:
                        st.success("持仓")
                    elif in_watch:
                        if st.button("买入", key=f"buy_{s.symbol}", use_container_width=True):
                            pm.move_to_hold(s.symbol)
                            st.rerun()
                    else:
                        if st.button("+观察", key=f"watch_{s.symbol}", use_container_width=True):
                            pm.add_to_watch(PoolItem(
                                symbol=s.symbol, add_date=str(latest_date),
                                add_reason=s.strategy_name, signal_strength=s.strength, signal_score=s.score,
                            ))
                            st.rerun()
                st.divider()
    else:
        st.info("暂无")

    st.subheader(f"🔴 卖出 ({len(sell_signals)})")
    if sell_signals:
        for s in sorted(sell_signals, key=lambda x: x.strength, reverse=True)[:10]:
            in_hold = pm.is_in_hold(s.symbol)
            with st.container():
                col_info, col_btn = st.columns([3, 1])
                with col_info:
                    st.markdown(f"**{fmt_name(s.symbol)}** {s.emoji} | {s.strategy_name}")
                    st.caption(f"强度:{s.strength}")
                with col_btn:
                    if in_hold:
                        if st.button("卖出", key=f"sell_{s.symbol}", type="primary", use_container_width=True):
                            pm.remove_from_hold(s.symbol)
                            st.rerun()
                    else:
                        st.caption("未持仓")
                st.divider()
    else:
        st.info("暂无")

# ========== 观察池页面 ==========
elif page == '观察':
    st.subheader(f"👁️ 观察池 ({len(pm.watch_list)})")
    if pm.watch_list:
        bridge = get_bridge()
        for item in pm.watch_list:
            hot = bridge.get_stock_hot_summary(item.symbol)
            hot_badge = bridge.get_hot_badge(item.symbol)
            with st.container():
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**{fmt_name(item.symbol)}** {hot_badge} | {item.add_reason}")
                    caption = f"加入:{item.add_date} | 强度:{item.signal_strength}"
                    if hot["has_hot"]:
                        caption += f" | 📰{hot['news_count']}条"
                    st.caption(caption)
                with col2:
                    c2a, c2b = st.columns(2)
                    with c2a:
                        if st.button("买入", key=f"wb_{item.symbol}", use_container_width=True):
                            pm.move_to_hold(item.symbol)
                            st.rerun()
                    with c2b:
                        if st.button("×", key=f"wr_{item.symbol}", use_container_width=True):
                            pm.remove_from_watch(item.symbol)
                            st.rerun()
                
                if st.button(f"📊 详情", key=f"wd_{item.symbol}", use_container_width=True):
                    st.session_state.detail_symbol = item.symbol
                    st.session_state.detail_page = '观察'
                    st.rerun()
                
                if st.session_state.get('detail_symbol') == item.symbol and st.session_state.get('detail_page') == '观察':
                    show_factor_detail(item.symbol, factor_df, latest_date)
                
                st.divider()
    else:
        st.info("暂无观察标的")

# ========== 持仓池页面 ==========
elif page == '持仓':
    st.subheader(f"💼 持仓池 ({len(pm.hold_list)})")
    if pm.hold_list:
        bridge = get_bridge()
        hot_alerts = bridge.get_portfolio_hot_alert([item.symbol for item in pm.hold_list])
        if not hot_alerts.empty:
            with st.container():
                st.markdown("<div style='background:#fff3e0;padding:6px 10px;border-radius:6px;margin:4px 0;font-size:0.75rem;'>⚠️ <b>持仓热点提醒</b></div>", unsafe_allow_html=True)
                for _, row in hot_alerts.head(3).iterrows():
                    emoji = "🟢" if row["sentiment"] > 0.2 else "🔴" if row["sentiment"] < -0.2 else "⚪"
                    st.markdown(f"<div style='font-size:0.7rem;padding-left:10px;'>{emoji} <b>{fmt_name(row['symbol'])}</b>: {row['latest_title']} ({row['news_count']}条)</div>", unsafe_allow_html=True)

        for item in pm.hold_list:
            hot = bridge.get_stock_hot_summary(item.symbol)
            hot_badge = bridge.get_hot_badge(item.symbol)
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{fmt_name(item.symbol)}** {hot_badge} | {item.add_reason}")
                    caption = f"买入:{item.add_date} | 强度:{item.signal_strength}"
                    if hot["has_hot"]:
                        caption += f" | 📰{hot['news_count']}条"
                        if hot['avg_sentiment'] > 0.2:
                            caption += " 🟢利好"
                        elif hot['avg_sentiment'] < -0.2:
                            caption += " 🔴利空"
                    st.caption(caption)
                with col2:
                    if st.button("卖出", key=f"hs_{item.symbol}", type="primary", use_container_width=True):
                        pm.remove_from_hold(item.symbol)
                        st.rerun()
                
                if st.button(f"📊 详情", key=f"hd_{item.symbol}", use_container_width=True):
                    st.session_state.detail_symbol = item.symbol
                    st.session_state.detail_page = '持仓'
                    st.rerun()
                
                if st.session_state.get('detail_symbol') == item.symbol and st.session_state.get('detail_page') == '持仓':
                    show_factor_detail(item.symbol, factor_df, latest_date)
                
                st.divider()
    else:
        st.info("暂无持仓")

st.divider()
st.caption("💡 菜单→策略排行榜")
