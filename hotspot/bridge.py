"""热点桥接模块 - 关联热点新闻与个股/持仓/信号

为看板提供热点关联查询能力：
- 某股票是否出现在今日热点中
- 持仓股票的热点提醒
- 买入信号的热点加成
"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import pandas as pd
from typing import Dict, List, Optional
from functools import lru_cache
from datetime import datetime

from hotspot.aggregator import HotspotAggregator
from hotspot.nlp import EntityExtractor


class HotspotBridge:
    """热点桥接器 - 连接热点数据与个股"""

    def __init__(self):
        self._scored_df: Optional[pd.DataFrame] = None
        self._hot_stocks: Optional[pd.DataFrame] = None
        self._hot_sectors: Optional[pd.DataFrame] = None
        self._entity = EntityExtractor()
        self._load_time: Optional[datetime] = None

    def refresh(self, max_news: int = 80):
        """刷新热点数据"""
        agg = HotspotAggregator(max_news=max_news)
        result = agg.run()
        self._scored_df = result["scored"]
        self._hot_stocks = result["hot_stocks"]
        self._hot_sectors = result["hot_sectors"]
        self._load_time = datetime.now()

    def _ensure_loaded(self):
        if self._scored_df is None:
            self.refresh()

    # ── 查询接口 ──
    def get_stock_hot_news(self, symbol_or_name: str) -> pd.DataFrame:
        """获取某股票关联的热点新闻"""
        self._ensure_loaded()
        if self._scored_df is None or self._scored_df.empty:
            return pd.DataFrame()

        symbol = symbol_or_name.strip()
        # 匹配股票名称或代码
        mask = self._scored_df["stocks"].str.contains(symbol, na=False)
        return self._scored_df[mask].sort_values("heat_score", ascending=False).head(10)

    def get_stock_hot_summary(self, symbol_or_name: str) -> Dict:
        """获取某股票的热点摘要"""
        news = self.get_stock_hot_news(symbol_or_name)
        if news.empty:
            return {"has_hot": False, "news_count": 0, "avg_sentiment": 0, "latest_title": ""}

        return {
            "has_hot": True,
            "news_count": len(news),
            "avg_sentiment": round(news["sentiment_score"].mean(), 3),
            "latest_title": news.iloc[0]["title"] if len(news) > 0 else "",
            "heat_score": round(news["heat_score"].sum(), 2),
            "sentiment_label": news.iloc[0]["sentiment_label"] if len(news) > 0 else "neutral",
        }

    def get_hot_stocks_list(self, top_n: int = 10) -> pd.DataFrame:
        """获取热门股票列表"""
        self._ensure_loaded()
        if self._hot_stocks is None or self._hot_stocks.empty:
            return pd.DataFrame()
        return self._hot_stocks.head(top_n)

    def get_hot_sectors_list(self, top_n: int = 10) -> pd.DataFrame:
        """获取热门行业列表"""
        self._ensure_loaded()
        if self._hot_sectors is None or self._hot_sectors.empty:
            return pd.DataFrame()
        return self._hot_sectors.head(top_n)

    def is_hot_stock(self, symbol_or_name: str) -> bool:
        """判断某股票是否为热门"""
        self._ensure_loaded()
        if self._hot_stocks is None or self._hot_stocks.empty:
            return False
        return symbol_or_name in self._hot_stocks["stock"].values

    def get_hot_badge(self, symbol_or_name: str) -> str:
        """获取热点徽章emoji"""
        summary = self.get_stock_hot_summary(symbol_or_name)
        if not summary["has_hot"]:
            return ""
        if summary["sentiment_label"] == "positive":
            return "🔥"
        elif summary["sentiment_label"] == "negative":
            return "⚠️"
        return "📰"

    def get_portfolio_hot_alert(self, symbols: List[str]) -> pd.DataFrame:
        """获取持仓股票的热点提醒"""
        rows = []
        for sym in symbols:
            summary = self.get_stock_hot_summary(sym)
            if summary["has_hot"]:
                rows.append({
                    "symbol": sym,
                    "news_count": summary["news_count"],
                    "sentiment": summary["avg_sentiment"],
                    "label": summary["sentiment_label"],
                    "latest_title": summary["latest_title"][:50],
                    "heat_score": summary["heat_score"],
                })

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("heat_score", ascending=False)


# 全局单例
_hotspot_bridge: Optional[HotspotBridge] = None

def get_bridge() -> HotspotBridge:
    """获取热点桥接器单例"""
    global _hotspot_bridge
    if _hotspot_bridge is None:
        _hotspot_bridge = HotspotBridge()
    return _hotspot_bridge
