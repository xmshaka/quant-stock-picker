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
def get_data(_version=4, _n_stocks=100, _include_symbols=None):
    # _version / _n_stocks / _include_symbols 用于强制刷新 Streamlit 缓存
    kwargs = {"n_stocks": _n_stocks, "n_days": 120, "prefer_snapshot": True}
    if _include_symbols:
        kwargs["include_symbols"] = _include_symbols
    return load_data(data_source="real", **kwargs)

@st.cache_data(ttl=300)
def get_hotspot_summary():
    bridge = get_bridge()
    return {
        "hot_sectors": bridge.get_hot_sectors_list(top_n=5),
        "hot_stocks": bridge.get_hot_stocks_list(top_n=5),
    }

# 确保 portfolio 里的股票优先被加载
_portfolio_syms = sorted({item.symbol for item in (pm.watch_list + pm.hold_list)})
_n_stocks = max(100, len(_portfolio_syms) + 100)

factor_df, price_df, factor_names = get_data(_n_stocks=_n_stocks, _include_symbols=_portfolio_syms)
hotspot_summary = get_hotspot_summary()
latest_date = pd.to_datetime(factor_df['trade_date'].max()).strftime('%Y-%m-%d')

# 自动刷新观察池/持仓池的最新信号
_refresh_engine = SignalEngine(buy_threshold=0.7, sell_threshold=0.3, min_strength=2.0)
_changed_symbols = pm.refresh_signals(_refresh_engine, factor_df, price_df, factor_names, {})
if _changed_symbols:
    st.toast(f"📡 {len(_changed_symbols)} 只股票信号已更新", icon="🔄")

# ========== 辅助函数 ==========
def fmt_name(symbol: str) -> str:
    """返回 '代码 名称' 格式"""
    from data_loader import NAME_MAP
    name = NAME_MAP.get(symbol, "")
    return f"{symbol} {name}" if name else symbol


# 因子解释方向：1=越高越好，-1=越低越好，0=中性/仅观察。
# 与默认信号权重保持一致，用于详情页解释“这个值是加分还是扣分”。
FACTOR_DIRECTIONS = {
    'rsi14': -1,                 # 当前策略偏反转：RSI低更容易加分
    'macd_hist': 1,
    'boll_position': 1,
    'volatility_20d': -1,
    'max_dd_60d': -1,
    'north_hold_change': 1,
    'margin_change': 1,
    'turnover_ratio': 1,
    'volume_ratio': 1,
    'pe_ttm': -1,
    'pb': -1,
    'ep': 1,
    'roe': 1,
    'gross_margin': 1,
    'revenue_growth': 1,
    'profit_growth': 1,
    'momentum_5d': -1,           # 5日动量偏反转
    'momentum_20d': -1,          # 当前策略偏短期反转
    'momentum_60d': 0,
    'liquidity': 1,
    'reversal': 1,
}

FACTOR_DIRECTION_LABEL = {1: "越高越好", -1: "越低越好", 0: "中性观察"}


def _factor_judgement(score: float) -> str:
    """把方向调整后的截面 z-score 转为直观标签。"""
    if pd.isna(score):
        return "—"
    if score >= 1.0:
        return "🟢 明显加分"
    if score >= 0.3:
        return "🟡 小幅加分"
    if score <= -1.0:
        return "🔴 明显扣分"
    if score <= -0.3:
        return "🟠 小幅扣分"
    return "⚪ 接近平均"


def _get_rsi_dynamic_direction(day_data: pd.DataFrame, symbol: str) -> tuple[float, str]:
    """返回 (direction, 模式标签)，和信号引擎逻辑保持一致（浮点方向）"""
    if 'momentum_20d' not in day_data.columns:
        return FACTOR_DIRECTIONS.get('rsi14', -1), ""
    m20 = day_data.set_index('symbol')['momentum_20d']
    m20_abs = m20.abs()
    threshold = m20_abs.quantile(0.8)
    if threshold == 0:
        return FACTOR_DIRECTIONS.get('rsi14', -1), ""
    ts = min(m20_abs.get(symbol, 0) / threshold, 1.0)
    if ts > 0.5:
        base_dir = FACTOR_DIRECTIONS.get('rsi14', -1)
        dynamic_dir = base_dir * (1 - 2 * ts)
        return dynamic_dir, "趋势"
    return FACTOR_DIRECTIONS.get('rsi14', -1), ""


def _get_rsi_regime_label(regime: str, rsi: float) -> tuple[float, str]:
    """返回 (direction, 标签)，和信号引擎三态逻辑一致"""
    if regime == '震荡整理':
        return -1, "反转（低RSI加分）"
    elif regime == '强势单边上涨':
        if rsi >= 75:
            return 0.5, "跟随·追高风险衰减50%"
        return 1, "跟随（高RSI加分）"
    elif regime == '弱势单边上涨':
        return 0, "中性（不参与打分）"
    elif regime == '强势单边下跌':
        if rsi <= 25:
            return 0.5, "跟随·超卖保护衰减50%"
        return 1, "跟随（高RSI扣分）"
    elif regime == '弱势单边下跌':
        return 0, "中性（不参与打分）"
    return -1, "反转（低RSI加分）"


def show_factor_detail(symbol, factor_df, price_df, latest_date):
    """显示个股因子详情：原始值 + 截面正常区间 + 方向调整后的相对评分（趋势感知）。"""
    from signals.engine import SignalEngine

    detail_data = factor_df[(factor_df['symbol'] == symbol) & (factor_df['trade_date'] == latest_date)]
    day_data = factor_df[factor_df['trade_date'] == latest_date].copy()
    if detail_data.empty:
        st.warning("无该股票最新数据")
        return

    row = detail_data.iloc[0]
    factors = {k: v for k, v in row.items() if k not in ['symbol', 'trade_date'] and pd.notna(v)}

    # 行情分类（和信号引擎一致）
    engine = SignalEngine()
    regime = engine._detect_regime(symbol, price_df, day_data)
    regime_badge = f" <span style='background:#e3f2fd;color:#1565c0;padding:2px 6px;border-radius:6px;font-size:0.75rem;'>📊 {regime}</span>"

    st.markdown(f"**📊 因子详情 — {fmt_name(symbol)}{regime_badge}**", unsafe_allow_html=True)
    st.caption("说明：RSI根据行情分类动态调整方向——震荡市反转、强趋势跟随、极端值衰减。")

    groups = {
        '技术因子': ['rsi14', 'macd_hist', 'boll_position', 'volatility_20d', 'max_dd_60d'],
        '情绪因子': ['north_hold_change', 'turnover_ratio', 'volume_ratio'],
        '估值因子': ['pe_ttm', 'pb', 'ep'],
        '质量因子': ['roe', 'gross_margin', 'revenue_growth', 'profit_growth'],
        '动量/流动性': ['momentum_5d', 'momentum_20d', 'momentum_60d', 'liquidity', 'reversal'],
    }

    for group_name, group_factors in groups.items():
        rows = []
        for f in group_factors:
            if f not in factors or f not in day_data.columns:
                continue
            vals = pd.to_numeric(day_data[f], errors='coerce').dropna()
            raw = pd.to_numeric(pd.Series([factors.get(f)]), errors='coerce').iloc[0]
            if vals.empty or pd.isna(raw):
                continue

            mean = vals.mean()
            std = vals.std()
            p25 = vals.quantile(0.25)
            p75 = vals.quantile(0.75)
            raw_pct = float((vals <= raw).mean() * 100)

            # RSI 三态标签（和信号引擎保持一致）
            if f == 'rsi14':
                direction, dir_label = _get_rsi_regime_label(regime, raw)
            else:
                direction = FACTOR_DIRECTIONS.get(f, 1)
                dir_label = FACTOR_DIRECTION_LABEL.get(direction, "越高越好")

            z = 0.0 if std == 0 or pd.isna(std) else (raw - mean) / std
            directional_score = z * direction if direction != 0 else np.nan

            if direction == 1:
                relative_pos = f"高于 {raw_pct:.0f}%"
            elif direction == -1:
                relative_pos = f"低于 {100 - raw_pct:.0f}%"
            else:
                relative_pos = f"百分位 {raw_pct:.0f}%"

            rows.append({
                '因子': FACTOR_NAME_MAP.get(f, f),
                '原值': raw,
                '判断': _factor_judgement(directional_score),
                '相对评分': directional_score,
                '相对位置': relative_pos,
                '正常区间(P25~P75)': f"{p25:.3f} ~ {p75:.3f}",
                '方向': dir_label,
            })

        if rows:
            with st.expander(f"**{group_name}**", expanded=False):
                df = pd.DataFrame(rows)
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "因子": st.column_config.TextColumn(width="medium"),
                        "原值": st.column_config.NumberColumn(width="small", format="%.3f"),
                        "正常区间(P25~P75)": st.column_config.TextColumn(width="medium"),
                        "方向": st.column_config.TextColumn(width="small"),
                        "相对位置": st.column_config.TextColumn(width="small"),
                        "相对评分": st.column_config.NumberColumn(width="small", format="%.2f"),
                        "判断": st.column_config.TextColumn(width="medium"),
                    },
                )
    
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
            '技术': ['rsi14', 'macd_hist', 'boll_position', 'volatility_20d', 'max_dd_60d'],
            '情绪': ['north_hold_change', 'turnover_ratio', 'volume_ratio'],
            '估值': ['pe_ttm', 'pb', 'ep'],
            '质量': ['roe', 'gross_margin', 'revenue_growth', 'profit_growth'],
            '动量': ['momentum_5d', 'momentum_20d', 'momentum_60d', 'liquidity', 'reversal'],
        }
        DEFAULT_WEIGHTS = {
            'rsi14': -0.1, 'macd_hist': 0.2, 'boll_position': 0.1,
            'volatility_20d': -0.1, 'max_dd_60d': -0.1,
            'north_hold_change': 0.4, 'turnover_ratio': 0.15, 'volume_ratio': 0.15,
            'pe_ttm': -0.2, 'pb': -0.1, 'ep': 0.1,
            'roe': 0.3, 'gross_margin': 0.2, 'revenue_growth': 0.2, 'profit_growth': 0.2,
            'momentum_5d': -0.15, 'momentum_20d': -0.1, 'momentum_60d': 0.0, 'liquidity': 0.1, 'reversal': 0.3,
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
        for s in sorted(buy_signals, key=lambda x: x.strength, reverse=True)[:20]:
            in_watch = pm.is_in_watch(s.symbol)
            in_hold = pm.is_in_hold(s.symbol)
            hot_badge = bridge.get_hot_badge(s.symbol)
            is_hot = bridge.is_hot_stock(s.symbol)
            with st.container():
                col_info, col_btn = st.columns([3, 1])
                with col_info:
                    hot_tag = f" <span style='background:#ff9800;color:white;padding:1px 5px;border-radius:8px;font-size:0.65rem;'>热点</span>" if is_hot else ""
                    regime_tag = f" <span style='background:#e3f2fd;color:#1565c0;padding:1px 5px;border-radius:8px;font-size:0.65rem;'>{s.regime[:4]}</span>" if s.regime != '震荡整理' else ""
                    risk_tag = s.risk_badge if hasattr(s, 'risk_badge') else ""
                    st.markdown(f"**{fmt_name(s.symbol)}** {s.emoji}{hot_badge}{regime_tag}{risk_tag} | {s.strategy_name}{hot_tag}", unsafe_allow_html=True)
                    caption = f"强度:{s.strength} 得分:{s.score:.2f}"
                    if s.risk_tags:
                        caption += f" | ⚠️{','.join(s.risk_tags)}"
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
        for s in sorted(sell_signals, key=lambda x: x.strength, reverse=True)[:20]:
            in_hold = pm.is_in_hold(s.symbol)
            with st.container():
                col_info, col_btn = st.columns([3, 1])
                with col_info:
                    regime_tag = f" <span style='background:#e3f2fd;color:#1565c0;padding:1px 5px;border-radius:8px;font-size:0.65rem;'>{s.regime[:4]}</span>" if s.regime != '震荡整理' else ""
                    risk_tag = s.risk_badge if hasattr(s, 'risk_badge') else ""
                    st.markdown(f"**{fmt_name(s.symbol)}** {s.emoji}{regime_tag}{risk_tag} | {s.strategy_name}")
                    caption = f"强度:{s.strength}"
                    if s.risk_tags:
                        caption += f" | ⚠️{','.join(s.risk_tags)}"
                    st.caption(caption)
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
                    show_factor_detail(item.symbol, factor_df, price_df, latest_date)
                
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
                    show_factor_detail(item.symbol, factor_df, price_df, latest_date)
                
                st.divider()
    else:
        st.info("暂无持仓")

st.divider()
st.caption("💡 菜单→策略排行榜")
