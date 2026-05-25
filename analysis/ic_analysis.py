"""因子IC分析 - 检验因子预测有效性"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import date

import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger


@dataclass
class ICResult:
    """单日IC结果"""
    trade_date: date
    ic: float           # Pearson IC
    rank_ic: float      # Spearman Rank IC
    
    @property
    def valid(self) -> bool:
        return not (pd.isna(self.ic) or pd.isna(self.rank_ic))


class ICAnalyzer:
    """因子IC分析器"""
    
    def __init__(self, min_stocks: int = 10):
        self.min_stocks = min_stocks
    
    def calc_ic(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series
    ) -> Tuple[float, float]:
        """
        计算IC和Rank IC
        
        Returns:
            (IC, Rank_IC)
        """
        common = factor_values.dropna().index.intersection(
            forward_returns.dropna().index
        )
        if len(common) < self.min_stocks:
            return np.nan, np.nan
        
        f = factor_values.loc[common]
        r = forward_returns.loc[common]
        
        # Pearson IC
        if f.std() == 0 or r.std() == 0:
            return np.nan, np.nan
        ic = np.corrcoef(f, r)[0, 1]
        
        # Rank IC (Spearman)
        rank_ic, _ = stats.spearmanr(f, r)
        
        return float(ic), float(rank_ic)
    
    def calc_forward_return(
        self,
        price_df: pd.DataFrame,
        horizon: int = 5
    ) -> pd.DataFrame:
        """
        计算未来N期收益率
        
        Args:
            price_df: DataFrame with columns [symbol, trade_date, close]
            horizon: 未来期数(交易日)
        
        Returns:
            DataFrame with columns [symbol, trade_date, forward_return]
        """
        df = price_df.sort_values(['symbol', 'trade_date']).copy()
        df['forward_return'] = df.groupby('symbol')['close'].shift(-horizon) / df['close'] - 1
        return df[['symbol', 'trade_date', 'forward_return']].dropna()
    
    def analyze_single_factor(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_name: str,
        horizon: int = 5
    ) -> pd.DataFrame:
        """
        对单个因子做IC序列分析
        
        Args:
            factor_df: DataFrame [symbol, trade_date, {factor_name}]
            price_df: DataFrame [symbol, trade_date, close]
            factor_name: 因子列名
            horizon: 预测期数
        
        Returns:
            DataFrame [trade_date, ic, rank_ic]
        """
        # 计算未来收益
        fwd = self.calc_forward_return(price_df, horizon)
        
        # 统一 trade_date 类型，避免 object vs datetime64 的 merge 报错
        factor_df = factor_df.copy()
        factor_df['trade_date'] = pd.to_datetime(factor_df['trade_date'])
        fwd['trade_date'] = pd.to_datetime(fwd['trade_date'])
        
        # 合并
        merged = factor_df.merge(
            fwd, on=['symbol', 'trade_date'], how='inner'
        )
        
        if merged.empty:
            logger.warning(f"因子 {factor_name} 无有效数据")
            return pd.DataFrame(columns=['trade_date', 'ic', 'rank_ic'])
        
        results = []
        for td, group in merged.groupby('trade_date'):
            ic, rank_ic = self.calc_ic(
                group[factor_name],
                group['forward_return']
            )
            results.append({
                'trade_date': td,
                'ic': ic,
                'rank_ic': rank_ic
            })
        
        return pd.DataFrame(results)
    
    def calc_ic_stats(self, ic_df: pd.DataFrame) -> Dict[str, float]:
        """
        计算IC统计指标
        
        Returns:
            {
                'ic_mean': IC均值,
                'ic_std': IC标准差,
                'ir': 信息比率,
                'rank_ic_mean': Rank IC均值,
                'rank_ic_std': Rank IC标准差,
                'positive_ratio': IC>0占比,
                'valid_days': 有效天数
            }
        """
        ic = ic_df['ic'].dropna()
        rank_ic = ic_df['rank_ic'].dropna()
        
        if len(ic) == 0:
            return {
                'ic_mean': np.nan, 'ic_std': np.nan, 'ir': np.nan,
                'rank_ic_mean': np.nan, 'rank_ic_std': np.nan,
                'positive_ratio': 0.0, 'valid_days': 0
            }
        
        return {
            'ic_mean': float(ic.mean()),
            'ic_std': float(ic.std()),
            'ir': float(ic.mean() / ic.std()) if ic.std() > 0 else np.nan,
            'rank_ic_mean': float(rank_ic.mean()),
            'rank_ic_std': float(rank_ic.std()),
            'positive_ratio': float((ic > 0).mean()),
            'valid_days': len(ic)
        }
    
    def calc_ic_decay(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_name: str,
        horizons: List[int] = [1, 5, 10, 20]
    ) -> pd.DataFrame:
        """
        计算IC衰减 - 不同预测期的IC表现
        
        Returns:
            DataFrame [horizon, ic_mean, rank_ic_mean, ic_std, ir]
        """
        results = []
        for h in horizons:
            ic_df = self.analyze_single_factor(factor_df, price_df, factor_name, h)
            stats_dict = self.calc_ic_stats(ic_df)
            results.append({
                'horizon': h,
                'ic_mean': stats_dict['ic_mean'],
                'rank_ic_mean': stats_dict['rank_ic_mean'],
                'ic_std': stats_dict['ic_std'],
                'ir': stats_dict['ir'],
                'valid_days': stats_dict['valid_days']
            })
        return pd.DataFrame(results)
    
    def group_return_analysis(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_name: str,
        n_groups: int = 5,
        horizon: int = 5
    ) -> pd.DataFrame:
        """
        分组收益分析 - 按因子分n组，看每组未来收益
        
        Returns:
            DataFrame [group, mean_return, std, count]
        """
        fwd = self.calc_forward_return(price_df, horizon)
        
        # 统一 trade_date 类型
        factor_df = factor_df.copy()
        factor_df['trade_date'] = pd.to_datetime(factor_df['trade_date'])
        fwd['trade_date'] = pd.to_datetime(fwd['trade_date'])
        
        merged = factor_df.merge(fwd, on=['symbol', 'trade_date'], how='inner')
        
        if merged.empty:
            return pd.DataFrame(columns=['group', 'mean_return', 'std', 'count'])
        
        # 按日期分组，每组内分位数分组
        def assign_group(x):
            try:
                return pd.qcut(x.rank(method='first'), n_groups, labels=False, duplicates='drop')
            except ValueError:
                return pd.Series(np.nan, index=x.index)
        
        merged['group'] = merged.groupby('trade_date')[factor_name].transform(assign_group)
        merged = merged.dropna(subset=['group'])
        
        if merged.empty:
            return pd.DataFrame(columns=['group', 'mean_return', 'std', 'count'])
        
        return merged.groupby('group')['forward_return'].agg(
            mean_return='mean',
            std='std',
            count='count'
        ).reset_index()
