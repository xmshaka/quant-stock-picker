"""数据获取器"""
from .base import BaseFetcher
from .akshare_fetcher import AKShareFetcher
from .tencent_fetcher import TencentFetcher
from .tushare_fetcher import TushareFetcher
from .fallback_fetcher import FallbackFetcher

__all__ = ["BaseFetcher", "AKShareFetcher", "TencentFetcher", "TushareFetcher", "FallbackFetcher"]
