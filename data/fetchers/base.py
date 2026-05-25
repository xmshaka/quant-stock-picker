"""数据获取基类"""
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
from loguru import logger


class BaseFetcher(ABC):
    """数据获取器基类"""

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def get_daily_bars(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """获取日线行情"""
        pass

    @abstractmethod
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        pass

    @abstractmethod
    def get_sector_list(self) -> pd.DataFrame:
        """获取板块/行业列表"""
        pass

    def _safe_fetch(self, func, *args, **kwargs) -> Optional[pd.DataFrame]:
        """安全获取，自动重试"""
        max_retries = 3
        for i in range(max_retries):
            try:
                result = func(*args, **kwargs)
                if result is not None and not result.empty:
                    return result
            except Exception as e:
                logger.warning(f"[{self.name}] 第{i+1}次获取失败: {e}")
        logger.error(f"[{self.name}] 获取失败，已达最大重试次数")
        return None
