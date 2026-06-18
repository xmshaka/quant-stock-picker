"""数据获取器"""
from .base import BaseFetcher

# 惰性导入：各 fetcher 的第三方依赖可能缺失（如 akshare/baostock）
# 缺失时对应符号设为 None，调用方自行判空

try:
    from .akshare_fetcher import AKShareFetcher
except ImportError:
    AKShareFetcher = None  # type: ignore

try:
    from .baostock_fetcher import BaostockFetcher
except ImportError:
    BaostockFetcher = None  # type: ignore

from .tencent_fetcher import TencentFetcher

try:
    from .tushare_fetcher import TushareFetcher
except ImportError:
    TushareFetcher = None  # type: ignore

try:
    from .fallback_fetcher import FallbackFetcher
except ImportError:
    FallbackFetcher = None  # type: ignore

__all__ = [
    "BaseFetcher", "AKShareFetcher", "BaostockFetcher", "TencentFetcher",
    "TushareFetcher", "FallbackFetcher",
]
