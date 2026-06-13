"""策略排行榜页面"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_loader import load_data, FACTOR_NAME_MAP
from signals.tracker import StrategyTracker
from theme import inject_theme, metric_row, section_header, badge, empty_state, C

st.set_page_config(page_title="策略排行榜", page_icon="🏆", layout="wide")
inject_theme()

# ========== 数据 ==========
@st.cache_data(ttl=300, show_spinner=False)
def get_data():
    return load_data(data_source="real", n_stocks=100, n_days=120)

factor_df, price_df, factor_names = get_data()

# ========== 参数 ==========
section_header("策略排行榜")

c1, c2, c3 = st.columns(3)
with c1:
    top_n = st.slider("选股数量", 5, 50, 20)
with c2:
    hold_days = st.selectbox("持有天数", [1, 5, 10, 20], index=1)
with c3:
    lookback = st.slider("回溯交易日", 10, 60, 30)

# ========== 计算 ==========
tracker = StrategyTracker(hold_days=hold_days)
with st.spinner("回测各策略..."):
    stats_df = tracker.track_all_strategies(factor_df, price_df, factor_names, top_n=top_n, lookback_days=lookback)

if stats_df.empty:
    empty_state("📊", "策略数据不足，请调整参数")
    st.stop()

# ========== 指标概览 ==========
best = stats_df.iloc[0]
metric_row([
    {"label": "最佳策略", "value": FACTOR_NAME_MAP.get(best['strategy_name'], best['strategy_name'])},
    {"label": "5日均涨", "value": f"{best['avg_return_5d']:+.2%}", "color": "green" if best['avg_return_5d'] > 0 else "red"},
    {"label": "胜率", "value": f"{best['win_rate']:.0%}", "color": "green" if best['win_rate'] > 0.5 else "red"},
    {"label": "夏普比率", "value": f"{best['sharpe']:.2f}", "color": "green" if best['sharpe'] > 0 else "red"},
    {"label": "评分", "value": f"{best['score']:.1f}", "color": "green" if best['score'] >= 8 else "yellow" if best['score'] >= 5 else "red"},
], cols=5)

# ========== 排行榜表 ==========
section_header("策略表现排行")
display = stats_df.copy()
display['strategy_name'] = display['strategy_name'].apply(lambda x: FACTOR_NAME_MAP.get(x, x))
display['avg_return_5d'] = display['avg_return_5d'].apply(lambda x: f"{x:+.2%}")
display['avg_return_10d'] = display['avg_return_10d'].apply(lambda x: f"{x:+.2%}")
display['win_rate'] = display['win_rate'].apply(lambda x: f"{x:.0%}")
display['sharpe'] = display['sharpe'].apply(lambda x: f"{x:.2f}")
display['score'] = display['score'].apply(lambda x: f"{x:.1f}")
display.columns = ['策略名称', '5日均涨', '10日均涨', '胜率', '夏普', '评分', '交易数']
st.dataframe(display, use_container_width=True, hide_index=True)

# ========== 评分图 ==========
section_header("策略评分对比")
colors = [C['green'] if s >= 8 else C['yellow'] if s >= 5 else C['red'] for s in stats_df['score']]
fig = go.Figure()
fig.add_trace(go.Bar(
    x=[FACTOR_NAME_MAP.get(x, x) for x in stats_df['strategy_name']],
    y=stats_df['score'], marker_color=colors,
    text=[f"{s:.1f}" for s in stats_df['score']], textposition='outside',
    textfont=dict(size=10),
))
fig.update_layout(height=380, template="plotly_dark",
                  paper_bgcolor=C['bg'], plot_bgcolor=C['surface'],
                  font=dict(color=C['text']),
                  xaxis_title="策略", yaxis_title="评分")
fig.add_hline(y=8, line_dash="dot", line_color=C['green'], annotation_text="优秀")
fig.add_hline(y=5, line_dash="dot", line_color=C['yellow'], annotation_text="一般")
st.plotly_chart(fig, use_container_width=True)

# ========== 收益 vs 胜率 ==========
section_header("收益 vs 胜率分布")
stats_df['marker_size'] = stats_df['score'].abs().clip(lower=1)
fig2 = px.scatter(stats_df, x='win_rate', y='avg_return_5d',
                  size='marker_size', color='score',
                  color_continuous_scale=[[0, C['red']], [0.5, C['yellow']], [1, C['green']]],
                  hover_name='strategy_name',
                  labels={'win_rate': '胜率', 'avg_return_5d': '5日平均收益'})
fig2.update_layout(height=400, template="plotly_dark",
                   paper_bgcolor=C['bg'], plot_bgcolor=C['surface'],
                   font=dict(color=C['text']))
fig2.add_hline(y=0, line_dash="dash", line_color=C['border'])
fig2.add_vline(x=0.5, line_dash="dash", line_color=C['border'])
st.plotly_chart(fig2, use_container_width=True)

# ========== 文本 ==========
with st.expander("📝 文本版"):
    st.code(tracker.format_ranking(stats_df), language=None)
