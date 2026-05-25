"""热点追踪页面 - 实时新闻 + NLP分析"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import streamlit as st
import pandas as pd

from hotspot.aggregator import HotspotAggregator
from data_loader import NAME_MAP
from datetime import datetime

# ========== 页面配置 ==========
st.set_page_config(page_title="热点追踪", page_icon="🔥", layout="centered")

# ========== 移动端适配CSS ==========
st.markdown("""
<style>
    .block-container { padding-top: 3.5rem !important; padding-left: 0.8rem; padding-right: 0.8rem; }
    h1 { font-size: 1.3rem !important; margin-top: 0.5rem !important; }
    h2 { font-size: 1.1rem !important; margin-top: 0.6rem !important; }
    h3 { font-size: 1rem !important; margin-top: 0.4rem !important; }
</style>
""", unsafe_allow_html=True)


def fmt_name(symbol: str) -> str:
    """返回 '代码 名称' 格式"""
    name = NAME_MAP.get(symbol, "")
    return f"{symbol} {name}" if name else symbol

# 移动端CSS
st.markdown("""
<style>
    html { font-size: 14px; }
    h1 { font-size: 1.3rem !important; }
    .news-item { background: #f8f9fa; padding: 8px 10px; border-radius: 6px; margin: 4px 0; }
    .news-item .title { font-size: 0.85rem; font-weight: 500; }
    .news-item .meta { font-size: 0.7rem; color: #666; margin-top: 2px; }
    .news-item .tags { margin-top: 4px; }
    .tag-stock { background: #e3f2fd; color: #1565c0; padding: 1px 6px; border-radius: 10px; font-size: 0.65rem; }
    .tag-industry { background: #e8f5e9; color: #2e7d32; padding: 1px 6px; border-radius: 10px; font-size: 0.65rem; }
    .tag-sentiment-pos { background: #c8e6c9; color: #1b5e20; }
    .tag-sentiment-neg { background: #ffcdd2; color: #b71c1c; }
    .tag-sentiment-neu { background: #f5f5f5; color: #616161; }
    .stMetric { background: #f8f9fa; border-radius: 6px; padding: 6px 4px; }
    .stMetric label { font-size: 0.7rem !important; }
    .stMetric div[data-testid="stMetricValue"] { font-size: 1rem !important; }
</style>
""", unsafe_allow_html=True)

st.title("🔥 热点追踪")

# ========== 数据加载（带缓存） ==========
@st.cache_data(ttl=300)
def get_hotspot_data():
    """获取热点数据，缓存5分钟"""
    agg = HotspotAggregator(max_news=80)
    return agg.run()

# 刷新按钮
col1, col2 = st.columns([3, 1])
with col1:
    st.caption(f"最后更新: {datetime.now().strftime('%H:%M:%S')}")
with col2:
    if st.button("🔄 刷新", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with st.spinner("正在抓取新闻..."):
    result = get_hotspot_data()

if result["news"].empty:
    st.warning("暂无新闻数据，请检查网络或稍后刷新")
    st.stop()

# ========== 情感统计卡片 ==========
ss = result["sentiment_summary"]
st.subheader("📊 市场情绪")
c1, c2, c3, c4 = st.columns(4)
c1.metric("总新闻", ss["total"])
c2.metric("🟢 利好", ss["positive"], delta=f"{ss['positive']/max(ss['total'],1)*100:.0f}%")
c3.metric("🔴 利空", ss["negative"], delta=f"{ss['negative']/max(ss['total'],1)*100:.0f}%")
c4.metric("平均情感", f"{ss['avg_score']:+.2f}")

# ========== 热门行业 ==========
st.subheader("🏭 热门行业")
sectors = result["hot_sectors"]
if not sectors.empty:
    # 柱状图
    chart_data = sectors.head(10).copy()
    chart_data["行业"] = chart_data["industry"]
    chart_data["热度"] = chart_data["heat_score"]
    st.bar_chart(chart_data.set_index("行业")["热度"], use_container_width=True, height=200)

    # 表格
    display_sectors = sectors.head(15).copy()
    display_sectors.columns = ["行业", "热度", "平均情感", "新闻数"]
    st.dataframe(display_sectors, use_container_width=True, hide_index=True,
                column_config={
                    "热度": st.column_config.NumberColumn(format="%.1f"),
                    "平均情感": st.column_config.NumberColumn(format="%.2f"),
                })
else:
    st.info("暂无行业热点数据")

# ========== 热门股票 ==========
st.subheader("📈 热门股票")
hot_stocks = result["hot_stocks"]
if not hot_stocks.empty:
    display_stocks = hot_stocks.head(15).copy()
    display_stocks.columns = ["股票代码", "热度", "平均情感", "提及次数"]
    st.dataframe(display_stocks, use_container_width=True, hide_index=True,
                column_config={
                    "热度": st.column_config.NumberColumn(format="%.1f"),
                    "平均情感": st.column_config.NumberColumn(format="%.2f"),
                })
else:
    st.info("暂无股票热点数据")

# ========== 热门新闻列表 ==========
st.subheader("📰 热门新闻")
scored = result["scored"]
if not scored.empty:
    # 筛选器
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        sentiment_filter = st.multiselect("情感筛选", ["positive", "negative", "neutral"],
                                           default=["positive", "negative", "neutral"])
    with filter_col2:
        stock_filter = st.text_input("股票筛选", placeholder="输入股票名称或代码")

    filtered = scored[scored["sentiment_label"].isin(sentiment_filter)]
    if stock_filter:
        filtered = filtered[filtered["stocks"].str.contains(stock_filter, na=False) |
                            filtered["title"].str.contains(stock_filter, na=False)]

    for _, row in filtered.head(30).iterrows():
        # 情感标签样式
        if row["sentiment_label"] == "positive":
            sentiment_class = "tag-sentiment-pos"
            sentiment_text = "利好"
        elif row["sentiment_label"] == "negative":
            sentiment_class = "tag-sentiment-neg"
            sentiment_text = "利空"
        else:
            sentiment_class = "tag-sentiment-neu"
            sentiment_text = "中性"

        # 股票标签
        stock_tags = ""
        if row["stocks"]:
            for s in row["stocks"].split(",")[:3]:
                stock_tags += f'<span class="tag-stock">{fmt_name(s)}</span> '

        # 行业标签
        industry_tags = ""
        if row["industries"]:
            for i in row["industries"].split(",")[:2]:
                industry_tags += f'<span class="tag-industry">{i}</span> '

        st.markdown(f"""
        <div class="news-item">
            <div class="title">{row['title']}</div>
            <div class="meta">{row['source']} | 热度:{row['heat_score']:.1f} | 强度:{row['intensity']:.2f}</div>
            <div class="tags">
                <span class="{sentiment_class}" style="padding:1px 6px;border-radius:10px;font-size:0.65rem;">{sentiment_text}</span>
                {stock_tags}{industry_tags}
            </div>
        </div>
        """, unsafe_allow_html=True)

    if len(filtered) == 0:
        st.info("无匹配新闻")
else:
    st.info("暂无新闻数据")

st.divider()
st.caption("💡 数据来源: 新浪财经RSS | 每5分钟自动缓存")
