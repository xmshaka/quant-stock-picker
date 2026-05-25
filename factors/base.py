"""因子基类与注册机制"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type
from dataclasses import dataclass

import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger


@dataclass
class FactorResult:
    """因子计算结果"""
    name: str                    # 因子名称
    values: pd.Series            # 因子值 (index=symbol)
    direction: int = 1           # 1=正向因子（越大越好），-1=反向因子
    group: str = "custom"        # 因子分组
    
    @property
    def ranked(self) -> pd.Series:
        """返回排名得分 (0-1标准化)"""
        return self._normalize(self.values * self.direction)
    
    @staticmethod
    def _normalize(s: pd.Series) -> pd.Series:
        """Min-Max归一化到0-1"""
        s = s.dropna()
        if s.empty:
            return s
        min_val, max_val = s.min(), s.max()
        if max_val > min_val:
            return (s - min_val) / (max_val - min_val)
        return pd.Series(0.5, index=s.index)


class Factor(ABC):
    """因子基类"""
    
    name: str = ""
    group: str = "custom"
    direction: int = 1
    
    def __init__(self):
        if not self.name:
            self.name = self.__class__.__name__
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> FactorResult:
        """
        计算因子值
        
        Args:
            df: 股票数据DataFrame，包含多只股票的数据
                必须包含: symbol, trade_date, open, high, low, close, volume
        
        Returns:
            FactorResult
        """
        pass
    
    def __repr__(self):
        return f"Factor({self.name}, group={self.group}, direction={self.direction})"


class FactorRegistry:
    """因子注册中心 - 管理所有因子"""
    
    _factors: Dict[str, Type[Factor]] = {}
    
    @classmethod
    def register(cls, factor_class: Type[Factor]) -> Type[Factor]:
        """注册因子类（装饰器）"""
        instance = factor_class()
        cls._factors[instance.name] = factor_class
        logger.debug(f"注册因子: {instance.name}")
        return factor_class
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[Factor]]:
        """获取因子类"""
        return cls._factors.get(name)
    
    @classmethod
    def list_factors(cls, group: Optional[str] = None) -> List[str]:
        """列出所有已注册因子"""
        if group:
            return [n for n, fc in cls._factors.items() 
                   if fc().group == group]
        return list(cls._factors.keys())
    
    @classmethod
    def build_all(cls) -> List[Factor]:
        """实例化所有因子"""
        return [fc() for fc in cls._factors.values()]


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """缩尾处理 - 去除极端值"""
    q_low, q_high = s.quantile(lower), s.quantile(upper)
    return s.clip(lower=q_low, upper=q_high)


def zscore(s: pd.Series) -> pd.Series:
    """Z-score标准化"""
    mean, std = s.mean(), s.std()
    if std > 0:
        return (s - mean) / std
    return pd.Series(0, index=s.index)


def rank_ic(factor_values: pd.Series, forward_returns: pd.Series) -> float:
    """
    计算Rank IC（斯皮尔曼相关系数）
    
    Args:
        factor_values: 因子值序列
        forward_returns: 未来收益序列
    
    Returns:
        Rank IC值
    """
    common_idx = factor_values.dropna().index.intersection(
        forward_returns.dropna().index
    )
    if len(common_idx) < 10:
        return np.nan
    
    f = factor_values.loc[common_idx]
    r = forward_returns.loc[common_idx]
    
    corr, _ = stats.spearmanr(f, r)
    return corr
