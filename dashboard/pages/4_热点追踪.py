"""热点追踪页面"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import streamlit as st
import pandas as pd
from datetime import datetime

from hotspot.aggregator import HotspotAggregator
from hotspot.nlp import SentimentAnalyzer
from theme import inject_theme, metric_row, section_header, badge, empty_state, progress_bar, C

st.set_page_config(page_title="热点追踪", page_icon="🔥", layout="wide")
inject_theme()

section_header("热点追踪")

# ========== 加载 ==========
@st.cache_data(ttl=600, show_spinner=False)
def load_hotspot():
    agg = HotspotAggregator(max_news=80)
    return agg.run()

with st.spinner("抓取热点新闻..."):
    result = load_hotspot()

if not result or result.get("news", pd.DataFrame()).empty:
    empty_state("🔥", "暂未抓取到热点新闻")
    st.stop()

news_df = result["news"]
scored_df = result["scored"]
hot_sectors = result["hot_sectors"]
hot_stocks = result["hot_stocks"]
sentiment = result["sentiment_summary"]

# ========== 概览 ==========
metric_row([
    {"label": "新闻总数", "value": str(sentiment['total'])},
    {"label": "利好", "value": str(sentiment['positive']), "color": "green"},
    {"label": "利空", "value": str(sentiment['negative']), "color": "red"},
    {"label": "中性", "value": str(sentiment['neutral'])},
    {"label": "均分", "value": f"{sentiment['avg_score']:+.3f}", "color": "green" if sentiment['avg_score'] > 0 else "red"},
], cols=5)

# ========== 热门行业 ==========
if not hot_sectors.empty:
    section_header("热门行业")
    for _, row in hot_sectors.head(10).iterrows():
        heat = row.get("heat_score", 0)
        bar_w = min(100, heat * 20)
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:7px 0;">
            <span style="font-size:0.82rem;font-weight:600;width:140px;">{row.get('industry', '')}</span>
            <div class="qsp-progress" style="flex:1;"><div class="fill" style="width:{bar_w}%;background:{C['accent']};"></div></div>
            <span style="font-size:0.72rem;color:{C['text2']};width:80px;text-align:right;">热度 {heat:.1f}</span>
        </div>
        """, unsafe_allow_html=True)

# ========== 热门个股 ==========
if not hot_stocks.empty:
    section_header("热门个股")
    for _, row in hot_stocks.head(10).iterrows():
        sent = row.get("avg_sentiment", 0)
        sent_b = badge("利好", "buy") if sent > 0.2 else badge("利空", "sell") if sent < -0.2 else badge("中性", "neutral")
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:7px 0;">
            <span style="font-size:0.82rem;font-weight:600;width:140px;">{row.get('stock', '')}</span>
            {sent_b}
            <span style="font-size:0.72rem;color:{C['text2']};">热度 {row.get('heat_score', 0):.1f} · 提及 {row.get('mention_count', 0)}</span>
        </div>
        """, unsafe_allow_html=True)

# ========== 新闻列表 ==========
section_header("新闻列表")
if not scored_df.empty:
    for _, row in scored_df.head(30).iterrows():
        title = row.get("title", "")
        source = row.get("source", "")
        heat = row.get("heat_score", 0)
        sent_score = row.get("sentiment_score", 0)
        sent_label = row.get("sentiment_label", "neutral")
        sent_b = badge("利好", "buy") if sent_label == "positive" else badge("利空", "sell") if sent_label == "negative" else badge("中性", "neutral")

        st.markdown(f"""
        <div style="padding:6px 0;border-bottom:1px solid {C['border']};">
            <div style="font-size:0.82rem;color:{C['text']};">{title}</div>
            <div style="font-size:0.68rem;color:{C['text2']};margin-top:2px;">
                {source} · 热度 {heat:.1f} · {sent_b}
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    empty_state("📰", "暂无新闻")
