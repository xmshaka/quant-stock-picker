"""热点聚合器 - 爬虫 + NLP 流水线

Usage:
    from hotspot.aggregator import HotspotAggregator
    agg = HotspotAggregator()
    result = agg.run()
    # result["news"]      -> 原始新闻 DataFrame
    # result["scored"]    -> 评分后新闻 DataFrame
    # result["hot_sectors"] -> 热门行业 DataFrame
    # result["hot_stocks"]  -> 热门股票 DataFrame
"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import pandas as pd
from typing import Dict, Optional
from datetime import datetime
from loguru import logger

from hotspot.crawler import HotspotCrawler
from hotspot.nlp import SentimentAnalyzer, EntityExtractor, HotspotScorer


class HotspotAggregator:
    """热点聚合器 - 一键获取今日热点"""

    def __init__(self, max_news: int = 100):
        self.max_news = max_news
        self.crawler = HotspotCrawler()
        self.scorer = HotspotScorer()

    def run(self, save_cache: bool = False) -> Dict:
        """
        运行完整流水线

        Returns:
            {
                "timestamp": 运行时间,
                "news": 原始新闻 DataFrame,
                "scored": 评分后新闻 DataFrame,
                "hot_sectors": 热门行业 DataFrame,
                "hot_stocks": 热门股票 DataFrame,
                "sentiment_summary": 情感统计,
            }
        """
        logger.info("[Aggregator] 开始热点聚合...")
        t0 = datetime.now()

        # 1. 爬取新闻
        news_items = self.crawler.fetch_all(max_items=self.max_news)
        if not news_items:
            logger.warning("[Aggregator] 未抓取到任何新闻")
            return self._empty_result()

        news_df = self.crawler.to_dataframe(news_items)
        logger.info(f"[Aggregator] 抓取 {len(news_items)} 条新闻")

        # 2. NLP分析
        news_dicts = [item.to_dict() for item in news_items]

        # 情感分析
        sentiment_df = self.scorer.sentiment.analyze_news(news_dicts)

        # 实体提取
        entity_df = self.scorer.entity.extract_from_news(news_dicts)

        # 3. 热点评分
        scored_df = self.scorer.score_news(news_dicts)

        # 4. 热门行业聚合
        hot_sectors = self.scorer.get_daily_hotspots(news_dicts)

        # 5. 热门股票统计
        hot_stocks = self._aggregate_hot_stocks(scored_df)

        # 6. 情感摘要
        sentiment_summary = {
            "total": len(sentiment_df),
            "positive": len(sentiment_df[sentiment_df["label"] == "positive"]),
            "negative": len(sentiment_df[sentiment_df["label"] == "negative"]),
            "neutral": len(sentiment_df[sentiment_df["label"] == "neutral"]),
            "avg_score": round(sentiment_df["score"].mean(), 4) if not sentiment_df.empty else 0,
        }

        result = {
            "timestamp": t0.isoformat(),
            "news": news_df,
            "scored": scored_df,
            "hot_sectors": hot_sectors,
            "hot_stocks": hot_stocks,
            "sentiment_summary": sentiment_summary,
        }

        elapsed = (datetime.now() - t0).total_seconds()
        logger.info(f"[Aggregator] 完成，耗时 {elapsed:.1f}s | "
                    f"新闻{len(news_items)}条 | 热门行业{len(hot_sectors)}个 | 热门股票{len(hot_stocks)}只")

        if save_cache:
            self._save_cache(result)

        return result

    def _aggregate_hot_stocks(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """从评分后的新闻聚合热门股票"""
        if scored_df.empty or "stocks" not in scored_df.columns:
            return pd.DataFrame()

        stock_scores = {}
        for _, row in scored_df.iterrows():
            if not row["stocks"]:
                continue
            for stock in row["stocks"].split(","):
                stock = stock.strip()
                if not stock:
                    continue
                if stock not in stock_scores:
                    stock_scores[stock] = {"heat_score": 0, "sentiment_score": 0, "mention_count": 0}
                stock_scores[stock]["heat_score"] += row["heat_score"]
                stock_scores[stock]["sentiment_score"] += row["sentiment_score"]
                stock_scores[stock]["mention_count"] += 1

        if not stock_scores:
            return pd.DataFrame()

        rows = []
        for stock, data in stock_scores.items():
            rows.append({
                "stock": stock,
                "heat_score": round(data["heat_score"], 2),
                "avg_sentiment": round(data["sentiment_score"] / data["mention_count"], 4),
                "mention_count": data["mention_count"],
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("heat_score", ascending=False).reset_index(drop=True)
        return df

    def _empty_result(self) -> Dict:
        """空结果"""
        return {
            "timestamp": datetime.now().isoformat(),
            "news": pd.DataFrame(),
            "scored": pd.DataFrame(),
            "hot_sectors": pd.DataFrame(),
            "hot_stocks": pd.DataFrame(),
            "sentiment_summary": {"total": 0, "positive": 0, "negative": 0, "neutral": 0, "avg_score": 0},
        }

    def _save_cache(self, result: Dict):
        """保存缓存到文件（可选）"""
        try:
            import pickle
            from pathlib import Path
            cache_dir = Path("/root/.openclaw/workspace/quant-stock-picker/data/hotspot_cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"hotspot_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
            logger.info(f"[Aggregator] 缓存已保存: {cache_file}")
        except Exception as e:
            logger.warning(f"[Aggregator] 缓存保存失败: {e}")
