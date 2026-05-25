"""IC分析报告生成"""
from typing import Dict, List
from datetime import date

import pandas as pd
import numpy as np
from loguru import logger

from .ic_analysis import ICAnalyzer


class ICReport:
    """IC分析报告"""
    
    def __init__(self, analyzer: ICAnalyzer = None):
        self.analyzer = analyzer or ICAnalyzer()
    
    def generate(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_name: str,
        horizons: List[int] = [1, 5, 10, 20]
    ) -> Dict:
        """
        生成完整IC分析报告
        
        Returns:
            {
                'factor_name': 因子名,
                'summary': IC统计摘要,
                'ic_series': IC时间序列DataFrame,
                'ic_decay': IC衰减DataFrame,
                'group_return': 分组收益DataFrame
            }
        """
        # IC序列 (默认5日)
        ic_series = self.analyzer.analyze_single_factor(
            factor_df, price_df, factor_name, horizon=5
        )
        
        # 统计摘要
        summary = self.analyzer.calc_ic_stats(ic_series)
        
        # IC衰减
        ic_decay = self.analyzer.calc_ic_decay(
            factor_df, price_df, factor_name, horizons
        )
        
        # 分组收益
        group_return = self.analyzer.group_return_analysis(
            factor_df, price_df, factor_name, n_groups=5, horizon=5
        )
        
        return {
            'factor_name': factor_name,
            'summary': summary,
            'ic_series': ic_series,
            'ic_decay': ic_decay,
            'group_return': group_return
        }
    
    def to_text(self, report: Dict) -> str:
        """转为文本报告"""
        s = report['summary']
        lines = [
            f"\n{'='*50}",
            f"因子IC分析报告: {report['factor_name']}",
            f"{'='*50}",
            f"IC均值:       {s['ic_mean']:.4f}",
            f"IC标准差:     {s['ic_std']:.4f}",
            f"IR(信息比率): {s['ir']:.4f}",
            f"Rank IC均值:  {s['rank_ic_mean']:.4f}",
            f"IC>0占比:     {s['positive_ratio']:.1%}",
            f"有效天数:     {s['valid_days']}",
            f"{'-'*50}",
            "IC衰减分析:",
        ]
        
        for _, row in report['ic_decay'].iterrows():
            lines.append(
                f"  {int(row['horizon']):3d}日 | IC={row['ic_mean']:+.4f} | "
                f"RankIC={row['rank_ic_mean']:+.4f} | IR={row['ir']:+.4f}"
            )
        
        lines.append(f"{'-'*50}")
        lines.append("分组收益(5日):")
        for _, row in report['group_return'].iterrows():
            lines.append(
                f"  组{int(row['group'])} | 均值={row['mean_return']:+.4f} | "
                f"标准差={row['std']:.4f} | 样本={int(row['count'])}"
            )
        lines.append(f"{'='*50}\n")
        
        return "\n".join(lines)
    
    def multi_factor_report(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_names: List[str]
    ) -> pd.DataFrame:
        """
        多因子对比报告
        
        Returns:
            DataFrame [factor_name, ic_mean, ir, rank_ic_mean, positive_ratio]
        """
        results = []
        for name in factor_names:
            ic_series = self.analyzer.analyze_single_factor(
                factor_df, price_df, name, horizon=5
            )
            stats = self.analyzer.calc_ic_stats(ic_series)
            results.append({
                'factor_name': name,
                **stats
            })
        
        df = pd.DataFrame(results)
        # 按IR排序
        return df.sort_values('ir', ascending=False).reset_index(drop=True)
