"""单因子深度分析页面"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.ic_analysis import ICAnalyzer
from analysis.report import ICReport
from data_loader import load_data, FACTOR_NAME_MAP

st.set_page_config(page_title="单因子分析", page_icon="🔬", layout="wide")

# ========== 移动端适配CSS ==========
st.markdown("""
<style>
    .block-container { padding-top: 3.5rem !important; padding-left: 0.8rem; padding-right: 0.8rem; }
    h1 { font-size: 1.3rem !important; margin-top: 0.5rem !important; }
    h2 { font-size: 1.1rem !important; margin-top: 0.6rem !important; }
    h3 { font-size: 1rem !important; margin-top: 0.4rem !important; }
</style>
""", unsafe_allow_html=True)

st.title("🔬 单因子深度分析")

# ========== 侧边栏配置 ==========
st.sidebar.header("⚙️ 配置")

data_source = st.sidebar.radio("数据源", ["模拟数据", "真实数据(AKShare)"], index=1)
source_key = "mock" if data_source == "模拟数据" else "real"

n_stocks = st.sidebar.slider("股票数量", 10, 200, 50)
n_days = st.sidebar.slider("交易日数量", 20, 252, 60)

# 加载数据
@st.cache_data(ttl=300)
def get_data(source, n_stocks, n_days, _version=3):
    # _version 用于强制刷新 Streamlit 缓存
    return load_data(data_source=source, n_stocks=n_stocks, n_days=n_days)

with st.spinner("加载数据..."):
    factor_df, price_df, factor_names = get_data(source_key, n_stocks, n_days)

# ========== 因子选择 ==========
st.subheader("📌 选择因子")

factor_options = {FACTOR_NAME_MAP.get(f, f): f for f in factor_names}
selected_factor_cn = st.selectbox("因子", list(factor_options.keys()), index=0)
selected_factor = factor_options[selected_factor_cn]

horizons = st.multiselect(
    "IC衰减预测期数",
    options=[1, 5, 10, 20, 60],
    default=[1, 5, 10, 20],
    help="选择多个预测期数进行IC衰减分析"
)

n_groups = st.slider("分组数量", 3, 10, 5)

# ========== 生成报告 ==========
analyzer = ICAnalyzer(min_stocks=10)
report_gen = ICReport(analyzer)

with st.spinner(f"分析因子 '{selected_factor}' 中..."):
    report = report_gen.generate(
        factor_df, price_df, selected_factor, horizons=horizons
    )

# ========== 指标卡片 ==========
st.subheader("📊 核心指标")

s = report['summary']
cols = st.columns(5)

with cols[0]:
    st.metric("IC均值", f"{s['ic_mean']:+.4f}")
with cols[1]:
    st.metric("IR", f"{s['ir']:+.4f}")
with cols[2]:
    st.metric("Rank IC均值", f"{s['rank_ic_mean']:+.4f}")
with cols[3]:
    st.metric("IC>0占比", f"{s['positive_ratio']:.1%}")
with cols[4]:
    st.metric("有效天数", f"{s['valid_days']}")

# ========== IC时间序列 ==========
st.subheader("📈 IC时间序列")

ic_series = report['ic_series']
if not ic_series.empty:
    ic_series['trade_date'] = pd.to_datetime(ic_series['trade_date'])
    
    fig_ts = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("IC 时间序列", "Rank IC 时间序列")
    )
    
    # IC
    fig_ts.add_trace(
        go.Scatter(x=ic_series['trade_date'], y=ic_series['ic'],
                   mode='lines+markers', name='IC',
                   marker=dict(size=4), line=dict(width=1.5)),
        row=1, col=1
    )
    fig_ts.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    fig_ts.add_hline(y=s['ic_mean'], line_dash="dot", line_color="red",
                     annotation_text=f"均值={s['ic_mean']:+.4f}", row=1, col=1)
    
    # Rank IC
    fig_ts.add_trace(
        go.Scatter(x=ic_series['trade_date'], y=ic_series['rank_ic'],
                   mode='lines+markers', name='Rank IC',
                   marker=dict(size=4), line=dict(width=1.5, color='orange')),
        row=2, col=1
    )
    fig_ts.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    fig_ts.add_hline(y=s['rank_ic_mean'], line_dash="dot", line_color="red",
                     annotation_text=f"均值={s['rank_ic_mean']:+.4f}", row=2, col=1)
    
    fig_ts.update_layout(
        height=500,
        template="plotly_white",
        showlegend=False,
        title_text=f"因子 '{selected_factor}' IC 时间序列"
    )
    st.plotly_chart(fig_ts, use_container_width=True)
else:
    st.warning("IC序列数据为空")

# ========== IC衰减分析 ==========
st.subheader("📉 IC衰减分析")

ic_decay = report['ic_decay']
if not ic_decay.empty:
    fig_decay = make_subplots(
        rows=1, cols=2,
        subplot_titles=("IC均值衰减", "IR衰减"),
        horizontal_spacing=0.1
    )
    
    # IC均值
    fig_decay.add_trace(
        go.Bar(x=ic_decay['horizon'], y=ic_decay['ic_mean'],
               name='IC均值', marker_color='#3498db'),
        row=1, col=1
    )
    fig_decay.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    
    # IR
    fig_decay.add_trace(
        go.Bar(x=ic_decay['horizon'], y=ic_decay['ir'],
               name='IR', marker_color='#e74c3c'),
        row=1, col=2
    )
    fig_decay.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
    fig_decay.add_hline(y=0.3, line_dash="dot", line_color="green", row=1, col=2,
                        annotation_text="有效阈值")
    
    fig_decay.update_layout(height=350, template="plotly_white", showlegend=False)
    fig_decay.update_xaxes(title_text="预测期(日)", row=1, col=1)
    fig_decay.update_xaxes(title_text="预测期(日)", row=1, col=2)
    fig_decay.update_yaxes(title_text="IC均值", row=1, col=1)
    fig_decay.update_yaxes(title_text="IR", row=1, col=2)
    
    st.plotly_chart(fig_decay, use_container_width=True)
    
    # 衰减表格
    st.caption("IC衰减详情")
    decay_display = ic_decay.copy()
    decay_display['ic_mean'] = decay_display['ic_mean'].apply(lambda x: f"{x:+.4f}")
    decay_display['rank_ic_mean'] = decay_display['rank_ic_mean'].apply(lambda x: f"{x:+.4f}")
    decay_display['ir'] = decay_display['ir'].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else "N/A")
    decay_display.columns = ['预测期(日)', 'IC均值', 'Rank IC均值', 'IC标准差', 'IR', '有效天数']
    st.dataframe(decay_display, hide_index=True, use_container_width=True)
else:
    st.warning("IC衰减数据为空")

# ========== 分组收益分析 ==========
st.subheader("📊 分组收益分析")

group_return = report['group_return']
if not group_return.empty:
    fig_group = go.Figure()
    
    colors = px.colors.diverging.RdYlGn
    bar_colors = [colors[int(i * (len(colors)-1) / max(len(group_return)-1, 1))] 
                  for i in range(len(group_return))]
    
    fig_group.add_trace(go.Bar(
        x=[f"组{int(g)}" for g in group_return['group']],
        y=group_return['mean_return'],
        error_y=dict(type='data', array=group_return['std']/np.sqrt(group_return['count'])),
        marker_color=bar_colors,
        text=[f"{v:+.4f}" for v in group_return['mean_return']],
        textposition='outside',
    ))
    
    fig_group.update_layout(
        title=f"分组收益 (预测期={horizons[0] if horizons else 5}日)",
        xaxis_title="分组 (组0=因子值最小, 组末=因子值最大)",
        yaxis_title="平均收益",
        height=400,
        template="plotly_white"
    )
    fig_group.add_hline(y=0, line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig_group, use_container_width=True)
    
    # 单调性判断
    returns = group_return['mean_return'].values
    monotonic = all(returns[i] <= returns[i+1] for i in range(len(returns)-1)) or \
                all(returns[i] >= returns[i+1] for i in range(len(returns)-1))
    
    if monotonic:
        direction = "递增" if returns[-1] > returns[0] else "递减"
        st.success(f"✅ 分组收益呈现**单调{direction}**趋势，因子有效")
    else:
        st.info("ℹ️ 分组收益未呈现明显单调趋势")
    
    # 分组详情表
    st.caption("分组详情")
    group_display = group_return.copy()
    group_display['mean_return'] = group_display['mean_return'].apply(lambda x: f"{x:+.4f}")
    group_display['std'] = group_display['std'].apply(lambda x: f"{x:.4f}")
    group_display['count'] = group_display['count'].astype(int)
    group_display.columns = ['分组', '平均收益', '标准差', '样本数']
    st.dataframe(group_display, hide_index=True, use_container_width=True)
else:
    st.warning("分组收益数据为空")

# ========== 文本报告 ==========
st.subheader("📝 文本报告")
with st.expander("查看文本报告"):
    st.text(report_gen.to_text(report))
