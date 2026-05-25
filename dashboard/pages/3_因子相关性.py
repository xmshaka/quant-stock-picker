"""因子相关性分析页面"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_loader import load_data, FACTOR_NAME_MAP

st.set_page_config(page_title="因子相关性", page_icon="🔗", layout="wide")

# ========== 移动端适配CSS ==========
st.markdown("""
<style>
    .block-container { padding-top: 3.5rem !important; padding-left: 0.8rem; padding-right: 0.8rem; }
    h1 { font-size: 1.3rem !important; margin-top: 0.5rem !important; }
    h2 { font-size: 1.1rem !important; margin-top: 0.6rem !important; }
    h3 { font-size: 1rem !important; margin-top: 0.4rem !important; }
</style>
""", unsafe_allow_html=True)

st.title("🔗 因子相关性分析")

# ========== 侧边栏 ==========
st.sidebar.header("⚙️ 配置")

data_source = st.sidebar.radio("数据源", ["模拟数据", "真实数据(AKShare)"], index=1)
source_key = "mock" if data_source == "模拟数据" else "real"

n_stocks = st.sidebar.slider("股票数量", 10, 200, 50)
n_days = st.sidebar.slider("交易日数量", 20, 252, 60)

corr_method = st.sidebar.selectbox("相关系数方法", ["pearson", "spearman"], index=1)

# 加载数据
@st.cache_data(ttl=300)
def get_data(source, n_stocks, n_days, _version=3):
    # _version 用于强制刷新 Streamlit 缓存
    return load_data(data_source=source, n_stocks=n_stocks, n_days=n_days)

with st.spinner("加载数据..."):
    factor_df, price_df, factor_names = get_data(source_key, n_stocks, n_days)

# ========== 截面因子相关性 ==========
st.subheader("📊 截面因子相关性")

# 中文映射
cn_map = {f: FACTOR_NAME_MAP.get(f, f) for f in factor_names}

# 取最新一个截面
latest_date = pd.to_datetime(factor_df['trade_date'].max()).strftime('%Y-%m-%d')
latest_factors = factor_df[factor_df['trade_date'] == factor_df['trade_date'].max()]

if len(latest_factors) > 0:
    # 构造因子矩阵
    pivot = latest_factors.pivot_table(
        index='symbol', 
        columns='trade_date',
        values=factor_names
    )
    # 取第一层column（只有一个日期）
    if isinstance(pivot.columns, pd.MultiIndex):
        pivot = pivot.xs(factor_df['trade_date'].max(), level='trade_date', axis=1)
    
    pivot = pivot.dropna()
    
    if len(pivot) > 5:
        corr_matrix = pivot.corr(method=corr_method)
        # 用中文重命名
        corr_matrix = corr_matrix.rename(columns=cn_map, index=cn_map)
        
        # 热力图
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="equal",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title=f"因子相关性热力图 ({latest_date})"
        )
        fig_heatmap.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # 相关性解读
        st.caption("相关性解读")
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_val
                    ))
        
        if high_corr_pairs:
            st.warning("⚠️ 发现高相关因子对（建议去重）:")
            for f1, f2, c in high_corr_pairs:
                st.write(f"  • **{f1}** ↔ **{f2}**: {c:+.3f}")
        else:
            st.success("✅ 各因子间相关性适中，无明显冗余")
    else:
        st.warning("截面数据样本不足，无法计算相关性")
else:
    st.warning("无最新截面数据")

# ========== 时间序列相关性 ==========
st.subheader("📈 因子IC序列相关性")

import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")
from analysis.ic_analysis import ICAnalyzer

analyzer = ICAnalyzer(min_stocks=10)

# 计算每个因子的IC序列
ic_series_dict = {}
for name in factor_names:
    ic_df = analyzer.analyze_single_factor(factor_df, price_df, name, horizon=5)
    if not ic_df.empty:
        ic_series_dict[name] = ic_df.set_index('trade_date')['ic']

if len(ic_series_dict) >= 2:
    # 合并为DataFrame
    ic_df_combined = pd.DataFrame(ic_series_dict)
    ic_corr = ic_df_combined.corr(method=corr_method)
    
    fig_ic_heatmap = px.imshow(
        ic_corr,
        text_auto=".2f",
        aspect="equal",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="IC序列相关性（反映因子预测力的时间一致性）"
    )
    fig_ic_heatmap.update_layout(height=500, template="plotly_white")
    st.plotly_chart(fig_ic_heatmap, use_container_width=True)
    
    # 散点矩阵
    st.subheader("🔍 IC序列散点矩阵")
    
    if len(factor_names) <= 4:
        ic_df_display = ic_df_combined.rename(columns=cn_map)
        fig_scatter = px.scatter_matrix(
            ic_df_display.reset_index(),
            dimensions=list(ic_df_display.columns),
            title="IC序列散点矩阵"
        )
        fig_scatter.update_layout(height=600, template="plotly_white")
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("因子数量超过4个，散点矩阵暂略。请选择2-4个因子查看。")
        
        factor_options = {FACTOR_NAME_MAP.get(f, f): f for f in factor_names}
        selected_pair_cn = st.multiselect(
            "选择两个因子查看IC散点图",
            list(factor_options.keys()),
            default=list(factor_options.keys())[:2],
            max_selections=2
        )
        selected_pair = [factor_options[f] for f in selected_pair_cn]
        
        if len(selected_pair) == 2:
            fig_pair = px.scatter(
                ic_df_combined.reset_index(),
                x=selected_pair[0],
                y=selected_pair[1],
                title=f"{factor_options.get(selected_pair[0], selected_pair[0])} vs {factor_options.get(selected_pair[1], selected_pair[1])} IC散点",
                trendline="ols"
            )
            fig_pair.update_layout(height=450, template="plotly_white")
            fig_pair.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_pair.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_pair, use_container_width=True)
else:
    st.warning("IC序列数据不足，无法计算相关性")

# ========== 因子收益率贡献 ==========
st.subheader("💰 分组收益对比")

# 取一个预测期，对比各因子的分组收益
selected_h = st.selectbox("预测期数", [1, 5, 10, 20], index=1, key="corr_h")

fig_group_compare = go.Figure()
colors = px.colors.qualitative.Set1

for idx, name in enumerate(factor_names):
    group_df = analyzer.group_return_analysis(
        factor_df, price_df, name, n_groups=5, horizon=selected_h
    )
    if not group_df.empty:
        fig_group_compare.add_trace(go.Scatter(
            x=[f"组{int(g)}" for g in group_df['group']],
            y=group_df['mean_return'],
            mode='lines+markers',
            name=FACTOR_NAME_MAP.get(name, name),
            line=dict(color=colors[idx % len(colors)], width=2),
            marker=dict(size=8)
        ))

fig_group_compare.update_layout(
    title=f"各因子分组收益对比 (预测期={selected_h}日)",
    xaxis_title="分组",
    yaxis_title="平均收益",
    height=450,
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig_group_compare.add_hline(y=0, line_dash="dash", line_color="gray")
st.plotly_chart(fig_group_compare, use_container_width=True)

st.caption("💡 有效因子的分组收益应呈现单调趋势（递增或递减）")
