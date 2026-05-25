"""热点模块 - 财经爬虫 + NLP情感分析 + 热点聚合"""
from .crawler import HotspotCrawler
from .nlp import SentimentAnalyzer, EntityExtractor, HotspotScorer
from .aggregator import HotspotAggregator

__all__ = [
    "HotspotCrawler",
    "SentimentAnalyzer",
    "EntityExtractor",
    "HotspotScorer",
    "HotspotAggregator",
]
