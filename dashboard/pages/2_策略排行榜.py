"""策略排行榜页面"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_loader import load_data, FACTOR_NAME_MAP

st.set_page_config(page_title="策略排行榜", page_icon="🏆", layout="wide")

# ========== 移动端适配CSS ==========
st.markdown("""
<style>
    .block-container { padding-top: 3.5rem !important; padding-left: 0.8rem; padding-right: 0.8rem; }
    h1 { font-size: 1.3rem !important; margin-top: 0.5rem !important; }
    h2 { font-size: 1.1rem !important; margin-top: 0.6rem !important; }
    h3 { font-size: 1rem !important; margin-top: 0.4rem !important; }
</style>
""", unsafe_allow_html=True)

st.title("🏆 策略排行榜")

# ========== 数据加载 ==========
@st.cache_data(ttl=300)
def get_data(_version=3):
    # _version 用于强制刷新 Streamlit 缓存
    return load_data(data_source="real", n_stocks=100, n_days=120)

with st.spinner("加载数据..."):
    factor_df, price_df, factor_names = get_data()

# ========== 参数配置 ==========
col1, col2, col3 = st.columns(3)
with col1:
    top_n = st.slider("选股数量", 5, 50, 20)
with col2:
    hold_days = st.selectbox("持有天数", [1, 5, 10, 20], index=1)
with col3:
    lookback = st.slider("回溯交易日", 10, 60, 30)

# ========== 计算策略表现 ==========
from signals.tracker import StrategyTracker

tracker = StrategyTracker(hold_days=hold_days)

with st.spinner("回测各策略表现..."):
    stats_df = tracker.track_all_strategies(
        factor_df, price_df, factor_names, top_n=top_n, lookback_days=lookback
    )

if stats_df.empty:
    st.warning("策略数据不足，请调整参数")
    st.stop()

# ========== 排行榜表格 ==========
st.subheader("📊 策略表现")

display = stats_df.copy()
display['strategy_name'] = display['strategy_name'].apply(lambda x: FACTOR_NAME_MAP.get(x, x))
display['avg_return_5d'] = display['avg_return_5d'].apply(lambda x: f"{x:+.2%}")
display['avg_return_10d'] = display['avg_return_10d'].apply(lambda x: f"{x:+.2%}")
display['win_rate'] = display['win_rate'].apply(lambda x: f"{x:.0%}")
display['sharpe'] = display['sharpe'].apply(lambda x: f"{x:.2f}")
display['score'] = display['score'].apply(lambda x: f"{x:.1f}")
display.columns = ['策略名称', '5日均涨', '10日均涨', '胜率', '夏普比率', '评分', '交易次数']

st.dataframe(display, use_container_width=True, hide_index=True)

# ========== 评分排名图 ==========
st.subheader("📈 策略评分")

fig = go.Figure()
colors = ['#27ae60' if s >= 8 else '#f39c12' if s >= 5 else '#e74c3c' for s in stats_df['score']]

fig.add_trace(go.Bar(
    x=[FACTOR_NAME_MAP.get(x, x) for x in stats_df['strategy_name']],
    y=stats_df['score'],
    marker_color=colors,
    text=[f"{s:.1f}" for s in stats_df['score']],
    textposition='outside',
))

fig.update_layout(
    title="策略综合评分",
    xaxis_title="策略",
    yaxis_title="评分",
    height=400,
    template="plotly_white",
    showlegend=False,
)
fig.add_hline(y=8, line_dash="dot", line_color="green", annotation_text="优秀")
fig.add_hline(y=5, line_dash="dot", line_color="orange", annotation_text="一般")
st.plotly_chart(fig, use_container_width=True)

# ========== 收益vs胜率散点图 ==========
st.subheader("🔍 收益 vs 胜率")

# Plotly marker size 不支持负数，创建非负辅助列
stats_df['marker_size'] = stats_df['score'].abs().clip(lower=1)

fig2 = px.scatter(
    stats_df,
    x='win_rate',
    y='avg_return_5d',
    size='marker_size',
    color='score',
    color_continuous_scale='RdYlGn',
    hover_name='strategy_name',
    labels={'win_rate': '胜率', 'avg_return_5d': '5日平均收益'},
    title="策略收益-胜率分布"
)
fig2.update_layout(height=400, template="plotly_white")
fig2.add_hline(y=0, line_dash="dash", line_color="gray")
fig2.add_vline(x=0.5, line_dash="dash", line_color="gray")
st.plotly_chart(fig2, use_container_width=True)

# ========== 文本输出 ==========
st.subheader("📝 文本版")
with st.expander("复制文本"):
    text = tracker.format_ranking(stats_df)
    st.code(text, language=None)
