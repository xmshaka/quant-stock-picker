"""选股模型

多因子打分 + 板块过滤 + 热点加权
所有数据必须来自真实 API，严禁编造。
"""
from typing import Dict, List, Optional, Set
import pandas as pd
import numpy as np
from loguru import logger


class MultiFactorScorer:
    """多因子选股打分器"""
    
    def __init__(
        self,
        factor_weights: Optional[Dict[str, float]] = None,
        sector_whitelist: Optional[Set[str]] = None,
        sector_blacklist: Optional[Set[str]] = None,
        min_mv: Optional[float] = None,      # 最小市值（万元）
        max_mv: Optional[float] = None,      # 最大市值（万元）
        min_turnover: Optional[float] = None,  # 最小换手率
        hotspot_weight: float = 0.0,         # 热点板块加分权重
    ):
        """
        Args:
            factor_weights: 因子名→权重，如 {"pe_ttm": -0.3, "roe": 0.4}
                若为空，默认等权
            sector_whitelist: 板块白名单，只选这些板块
            sector_blacklist: 板块黑名单，排除这些板块
            min_mv: 最小流通市值（万元），过滤小盘股
            max_mv: 最大流通市值（万元），过滤超大盘
            min_turnover: 最小换手率（如 0.01=1%），过滤流动性差的
            hotspot_weight: 热点板块额外加分权重（0~1）
        """
        self.factor_weights = factor_weights or {}
        self.sector_whitelist = sector_whitelist
        self.sector_blacklist = sector_blacklist
        self.min_mv = min_mv
        self.max_mv = max_mv
        self.min_turnover = min_turnover
        self.hotspot_weight = hotspot_weight
    
    def score(
        self,
        factor_matrix: pd.DataFrame,
        sector_map: Optional[pd.Series] = None,
        hotspot_sectors: Optional[Set[str]] = None,
        mv_series: Optional[pd.Series] = None,
        turnover_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        计算综合得分并排序
        
        Args:
            factor_matrix: DataFrame, index=symbol, columns=factor_names, values=0~1排名得分
            sector_map: Series, index=symbol, values=sector_name
            hotspot_sectors: 当前热点板块集合
            mv_series: Series, index=symbol, values=流通市值
            turnover_series: Series, index=symbol, values=换手率
        
        Returns:
            DataFrame, index=symbol, columns=[factor1, factor2, ..., total_score]
            按 total_score 降序排列
        """
        if factor_matrix.empty:
            logger.warning("因子矩阵为空")
            return pd.DataFrame()
        
        df = factor_matrix.copy()
        
        # === 1. 过滤 ===
        mask = pd.Series(True, index=df.index)
        
        # 板块过滤
        if sector_map is not None and (self.sector_whitelist or self.sector_blacklist):
            sectors = sector_map.reindex(df.index)
            if self.sector_whitelist:
                mask &= sectors.isin(self.sector_whitelist)
            if self.sector_blacklist:
                mask &= ~sectors.isin(self.sector_blacklist)
        
        # 市值过滤
        if mv_series is not None and (self.min_mv or self.max_mv):
            mv = mv_series.reindex(df.index)
            if self.min_mv:
                mask &= mv >= self.min_mv
            if self.max_mv:
                mask &= mv <= self.max_mv
        
        # 换手率过滤
        if turnover_series is not None and self.min_turnover:
            turn = turnover_series.reindex(df.index)
            mask &= turn >= self.min_turnover
        
        df = df[mask]
        if df.empty:
            logger.warning("过滤后无股票")
            return pd.DataFrame()
        
        # === 2. 多因子加权得分 ===
        available_factors = [c for c in df.columns if c != "total_score"]
        
        if not self.factor_weights:
            # 默认等权
            weights = {f: 1.0 / len(available_factors) for f in available_factors}
        else:
            weights = self.factor_weights
        
        # 只使用存在的因子
        weights = {k: v for k, v in weights.items() if k in available_factors}
        
        if not weights:
            logger.warning("没有可用因子权重")
            return pd.DataFrame()
        
        # 归一化权重
        total_w = sum(abs(v) for v in weights.values())
        if total_w == 0:
            total_w = 1
        weights = {k: v / total_w for k, v in weights.items()}
        
        # 计算加权得分
        scores = pd.Series(0.0, index=df.index)
        for factor, weight in weights.items():
            scores += df[factor].fillna(0.5) * weight
        
        df["total_score"] = scores
        
        # === 3. 热点板块加权 ===
        if self.hotspot_weight > 0 and hotspot_sectors and sector_map is not None:
            sectors = sector_map.reindex(df.index)
            is_hotspot = sectors.isin(hotspot_sectors)
            df.loc[is_hotspot, "total_score"] += self.hotspot_weight
            logger.info(f"热点板块股票: {is_hotspot.sum()} 只")
        
        # 最终排序
        df = df.sort_values("total_score", ascending=False)
        
        logger.info(f"选股完成: {len(df)} 只股票, 最高分={df['total_score'].max():.4f}")
        return df
    
    def get_top_n(
        self,
        factor_matrix: pd.DataFrame,
        n: int = 20,
        **kwargs
    ) -> pd.DataFrame:
        """获取前N只股票"""
        result = self.score(factor_matrix, **kwargs)
        return result.head(n)


class SectorRotator:
    """板块轮动筛选器
    
    基于板块动量和资金流向筛选强势板块
    """
    
    def __init__(
        self,
        momentum_window: int = 20,
        min_stocks: int = 3,
    ):
        self.momentum_window = momentum_window
        self.min_stocks = min_stocks
    
    def get_hot_sectors(
        self,
        sector_df: pd.DataFrame,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        获取热点板块
        
        Args:
            sector_df: DataFrame, columns=[sector_code, sector_name, trade_date, 
                                          close, pct_change, turnover, up_count, down_count]
        
        Returns:
            DataFrame, 按热度排序的前N板块
        """
        if sector_df.empty:
            return pd.DataFrame()
        
        # 取最新日期
        latest_date = sector_df["trade_date"].max()
        latest = sector_df[sector_df["trade_date"] == latest_date].copy()
        
        # 计算热度分 = 涨幅排名 + 上涨家数比
        latest["up_ratio"] = latest["up_count"] / (latest["up_count"] + latest["down_count"] + 1)
        latest["heat_score"] = latest["pct_change"].rank(pct=True) * 0.6 + \
                               latest["up_ratio"].rank(pct=True) * 0.4
        
        result = latest.sort_values("heat_score", ascending=False).head(top_n)
        logger.info(f"热点板块 Top{top_n}: {result['sector_name'].tolist()}")
        return result
