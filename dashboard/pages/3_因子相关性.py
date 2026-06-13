"""因子相关性分析页面"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data_loader import load_data, FACTOR_NAME_MAP
from theme import inject_theme, section_header, empty_state, C

st.set_page_config(page_title="因子相关性", page_icon="🔗", layout="wide")
inject_theme()

# ========== 侧边栏 ==========
with st.sidebar:
    st.markdown("### ⚙ 配置")
    data_source = st.radio("数据源", ["真实数据", "模拟数据"], index=0)
    source_key = "real" if data_source == "真实数据" else "mock"
    n_stocks = st.slider("股票数量", 10, 200, 50)
    n_days = st.slider("交易日数", 20, 252, 60)
    corr_method = st.selectbox("相关系数", ["spearman", "pearson"], index=0)

@st.cache_data(ttl=300, show_spinner=False)
def get_data(source, n_stocks, n_days):
    return load_data(data_source=source, n_stocks=n_stocks, n_days=n_days)

factor_df, price_df, factor_names = get_data(source_key, n_stocks, n_days)
cn_map = {f: FACTOR_NAME_MAP.get(f, f) for f in factor_names}

# ========== 截面相关性 ==========
section_header("截面因子相关性")

latest_date = factor_df['trade_date'].max()
latest_factors = factor_df[factor_df['trade_date'] == latest_date]

if not latest_factors.empty:
    pivot = latest_factors.set_index('symbol')[factor_names].dropna()

    if len(pivot) > 5:
        corr_matrix = pivot.corr(method=corr_method)
        corr_cn = corr_matrix.rename(columns=cn_map, index=cn_map)

        fig = px.imshow(corr_cn, text_auto=".2f", aspect="equal",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig.update_layout(height=520, template="plotly_dark",
                          paper_bgcolor=C['bg'], plot_bgcolor=C['surface'],
                          font=dict(color=C['text'], size=11))
        st.plotly_chart(fig, use_container_width=True)

        # 高相关警告
        high_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                v = corr_matrix.iloc[i, j]
                if abs(v) > 0.7:
                    high_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], v))

        if high_pairs:
            st.warning("⚠️ 高相关因子对（建议去重）:")
            for f1, f2, c in high_pairs:
                st.write(f"  • **{cn_map.get(f1, f1)}** ↔ **{cn_map.get(f2, f2)}**: {c:+.3f}")
        else:
            st.success("✅ 各因子间相关性适中，无明显冗余")
    else:
        empty_state("📊", "截面样本不足")
else:
    empty_state("📊", "无最新截面数据")

# ========== IC 序列相关性 ==========
section_header("IC 序列相关性")

from analysis.ic_analysis import ICAnalyzer
analyzer = ICAnalyzer(min_stocks=10)

ic_dict = {}
for name in factor_names:
    ic_df = analyzer.analyze_single_factor(factor_df, price_df, name, horizon=5)
    if not ic_df.empty:
        ic_dict[name] = ic_df.set_index('trade_date')['ic']

if len(ic_dict) >= 2:
    ic_combined = pd.DataFrame(ic_dict)
    ic_corr = ic_combined.corr(method=corr_method)
    ic_corr_cn = ic_corr.rename(columns=cn_map, index=cn_map)

    fig_ic = px.imshow(ic_corr_cn, text_auto=".2f", aspect="equal",
                       color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig_ic.update_layout(height=520, template="plotly_dark",
                         paper_bgcolor=C['bg'], plot_bgcolor=C['surface'],
                         font=dict(color=C['text'], size=11),
                         title_text="IC 序列相关性（因子预测力的时间一致性）")
    st.plotly_chart(fig_ic, use_container_width=True)

    # 散点
    section_header("IC 散点矩阵")
    selected_cn = st.multiselect("选择因子 (2-4)", list(cn_map.values()),
                                 default=list(cn_map.values())[:2], max_selections=4)
    selected = [k for k, v in cn_map.items() if v in selected_cn]

    if len(selected) >= 2:
        ic_sel = ic_combined[selected].rename(columns=cn_map)
        fig_sc = px.scatter_matrix(ic_sel.reset_index(), dimensions=list(ic_sel.columns))
        fig_sc.update_layout(height=550, template="plotly_dark",
                             paper_bgcolor=C['bg'], plot_bgcolor=C['surface'],
                             font=dict(color=C['text']))
        st.plotly_chart(fig_sc, use_container_width=True)
else:
    empty_state("📈", "IC 序列数据不足")

# ========== 分组收益对比 ==========
section_header("分组收益对比")
selected_h = st.selectbox("预测期", [1, 5, 10, 20], index=1)

fig_grp = go.Figure()
colors_seq = px.colors.qualitative.Set2
for idx, name in enumerate(factor_names):
    group_df = analyzer.group_return_analysis(factor_df, price_df, name, n_groups=5, horizon=selected_h)
    if not group_df.empty:
        fig_grp.add_trace(go.Scatter(
            x=[f"G{int(g)}" for g in group_df['group']], y=group_df['mean_return'],
            mode='lines+markers', name=cn_map.get(name, name),
            line=dict(color=colors_seq[idx % len(colors_seq)], width=1.5),
            marker=dict(size=6),
        ))

fig_grp.update_layout(
    title=f"各因子分组收益 (预测期={selected_h}日)",
    xaxis_title="分组", yaxis_title="平均收益", height=420,
    template="plotly_dark", paper_bgcolor=C['bg'], plot_bgcolor=C['surface'],
    font=dict(color=C['text']),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
)
fig_grp.add_hline(y=0, line_dash="dash", line_color=C['border'])
st.plotly_chart(fig_grp, use_container_width=True)
st.caption("💡 有效因子的分组收益应呈现单调趋势")
