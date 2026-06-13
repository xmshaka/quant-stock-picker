"""单因子深度分析页面"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.ic_analysis import ICAnalyzer
from analysis.report import ICReport
from data_loader import load_data, FACTOR_NAME_MAP
from theme import inject_theme, metric_row, section_header, badge, empty_state, C

st.set_page_config(page_title="单因子分析", page_icon="🔬", layout="wide")
inject_theme()

# ========== 数据加载 ==========
@st.cache_data(ttl=300, show_spinner=False)
def get_data(source, n_stocks, n_days):
    return load_data(data_source=source, n_stocks=n_stocks, n_days=n_days)

# ========== 侧边栏 ==========
with st.sidebar:
    st.markdown("### ⚙ 配置")
    data_source = st.radio("数据源", ["真实数据", "模拟数据"], index=0)
    source_key = "real" if data_source == "真实数据" else "mock"
    n_stocks = st.slider("股票数量", 10, 200, 50)
    n_days = st.slider("交易日数", 20, 252, 60)

factor_df, price_df, factor_names = get_data(source_key, n_stocks, n_days)

# ========== 因子选择 ==========
section_header("单因子深度分析")

col_sel1, col_sel2, col_sel3 = st.columns([2, 1, 1])
with col_sel1:
    factor_options = {FACTOR_NAME_MAP.get(f, f): f for f in factor_names}
    selected_cn = st.selectbox("因子", list(factor_options.keys()), index=0)
    selected_factor = factor_options[selected_cn]
with col_sel2:
    horizons = st.multiselect("预测期", [1, 5, 10, 20, 60], default=[1, 5, 10, 20])
with col_sel3:
    n_groups = st.slider("分组数", 3, 10, 5)

# ========== 分析 ==========
analyzer = ICAnalyzer(min_stocks=10)
report_gen = ICReport(analyzer)

with st.spinner(f"分析 {selected_cn}..."):
    report = report_gen.generate(factor_df, price_df, selected_factor, horizons=horizons)

s = report['summary']

# ========== 核心指标 ==========
metric_row([
    {"label": "IC 均值", "value": f"{s['ic_mean']:+.4f}", "color": "green" if s['ic_mean'] > 0 else "red"},
    {"label": "IR", "value": f"{s['ir']:+.4f}", "color": "green" if s['ir'] > 0.3 else "red" if s['ir'] < -0.3 else ""},
    {"label": "Rank IC", "value": f"{s['rank_ic_mean']:+.4f}", "color": "green" if s['rank_ic_mean'] > 0 else "red"},
    {"label": "IC>0 占比", "value": f"{s['positive_ratio']:.1%}", "color": "green" if s['positive_ratio'] > 0.5 else "red"},
    {"label": "有效天数", "value": str(s['valid_days'])},
], cols=5)

# ========== IC 时间序列 ==========
section_header("IC 时间序列")
ic_series = report['ic_series']
if not ic_series.empty:
    ic_series['trade_date'] = pd.to_datetime(ic_series['trade_date'])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("IC", "Rank IC"))
    fig.add_trace(go.Scatter(x=ic_series['trade_date'], y=ic_series['ic'],
                             mode='lines+markers', name='IC', marker=dict(size=3),
                             line=dict(width=1.5, color=C['accent'])), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color=C['border'], row=1, col=1)
    fig.add_hline(y=s['ic_mean'], line_dash="dot", line_color=C['red'],
                  annotation_text=f"均值={s['ic_mean']:+.4f}", row=1, col=1)

    fig.add_trace(go.Scatter(x=ic_series['trade_date'], y=ic_series['rank_ic'],
                             mode='lines+markers', name='Rank IC', marker=dict(size=3),
                             line=dict(width=1.5, color=C['yellow'])), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color=C['border'], row=2, col=1)
    fig.add_hline(y=s['rank_ic_mean'], line_dash="dot", line_color=C['red'],
                  annotation_text=f"均值={s['rank_ic_mean']:+.4f}", row=2, col=1)

    fig.update_layout(height=480, template="plotly_dark", showlegend=False,
                      paper_bgcolor=C['bg'], plot_bgcolor=C['surface'],
                      font=dict(color=C['text']))
    st.plotly_chart(fig, use_container_width=True)
else:
    empty_state("📈", "IC 序列数据为空")

# ========== IC 衰减 ==========
section_header("IC 衰减分析")
ic_decay = report['ic_decay']
if not ic_decay.empty:
    fig_d = make_subplots(rows=1, cols=2, subplot_titles=("IC 均值衰减", "IR 衰减"), horizontal_spacing=0.1)
    fig_d.add_trace(go.Bar(x=ic_decay['horizon'], y=ic_decay['ic_mean'],
                           marker_color=C['accent'], name='IC均值'), row=1, col=1)
    fig_d.add_hline(y=0, line_dash="dash", line_color=C['border'], row=1, col=1)
    fig_d.add_trace(go.Bar(x=ic_decay['horizon'], y=ic_decay['ir'],
                           marker_color=C['red'], name='IR'), row=1, col=2)
    fig_d.add_hline(y=0, line_dash="dash", line_color=C['border'], row=1, col=2)
    fig_d.add_hline(y=0.3, line_dash="dot", line_color=C['green'], row=1, col=2, annotation_text="有效")
    fig_d.update_layout(height=340, template="plotly_dark", showlegend=False,
                        paper_bgcolor=C['bg'], plot_bgcolor=C['surface'],
                        font=dict(color=C['text']))
    fig_d.update_xaxes(title_text="预测期(日)", row=1, col=1)
    fig_d.update_xaxes(title_text="预测期(日)", row=1, col=2)
    st.plotly_chart(fig_d, use_container_width=True)

    with st.expander("衰减详情表"):
        ddisp = ic_decay.copy()
        ddisp.columns = ['预测期(日)', 'IC均值', 'Rank IC均值', 'IC标准差', 'IR', '有效天数']
        for c in ['IC均值', 'Rank IC均值', 'IR']:
            ddisp[c] = ddisp[c].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else "N/A")
        st.dataframe(ddisp, hide_index=True, use_container_width=True)
else:
    empty_state("📉", "IC 衰减数据为空")

# ========== 分组收益 ==========
section_header("分组收益分析")
group_return = report['group_return']
if not group_return.empty:
    horizon_label = horizons[0] if horizons else 5
    colors_seq = [C['red'], C['orange'], C['yellow'], C['green'], C['accent']]
    bar_colors = [colors_seq[min(i, len(colors_seq)-1)] for i in range(len(group_return))]

    fig_g = go.Figure()
    fig_g.add_trace(go.Bar(
        x=[f"G{int(g)}" for g in group_return['group']],
        y=group_return['mean_return'],
        error_y=dict(type='data', array=group_return['std']/np.sqrt(group_return['count'])),
        marker_color=bar_colors,
        text=[f"{v:+.3f}" for v in group_return['mean_return']],
        textposition='outside', textfont=dict(size=10),
    ))
    fig_g.update_layout(
        title=f"分组收益 (预测期={horizon_label}日)",
        xaxis_title="分组 (G0=最小, G末=最大)", yaxis_title="平均收益",
        height=380, template="plotly_dark",
        paper_bgcolor=C['bg'], plot_bgcolor=C['surface'],
        font=dict(color=C['text']),
    )
    fig_g.add_hline(y=0, line_dash="dash", line_color=C['border'])
    st.plotly_chart(fig_g, use_container_width=True)

    returns = group_return['mean_return'].values
    mono = all(returns[i] <= returns[i+1] for i in range(len(returns)-1)) or \
           all(returns[i] >= returns[i+1] for i in range(len(returns)-1))
    if mono:
        direction = "递增" if returns[-1] > returns[0] else "递减"
        st.success(f"✅ 分组收益单调{direction}，因子有效")
    else:
        st.info("ℹ️ 分组收益未呈现明显单调趋势")

    with st.expander("分组详情表"):
        gdisp = group_return.copy()
        gdisp.columns = ['分组', '平均收益', '标准差', '样本数']
        gdisp['平均收益'] = gdisp['平均收益'].apply(lambda x: f"{x:+.4f}")
        gdisp['标准差'] = gdisp['标准差'].apply(lambda x: f"{x:.4f}")
        st.dataframe(gdisp, hide_index=True, use_container_width=True)
else:
    empty_state("📊", "分组收益数据为空")

# ========== 文本报告 ==========
with st.expander("📝 文本报告"):
    st.code(report_gen.to_text(report), language=None)
