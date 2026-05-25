"""质量因子"""
import pandas as pd
import numpy as np

from .base import Factor, FactorRegistry, FactorResult, winsorize


@FactorRegistry.register
class ROE(Factor):
    """净资产收益率 - 正向因子"""
    name = "roe"
    group = "quality"
    direction = 1
    
    def calculate(self, df: pd.DataFrame) -> FactorResult:
        latest = df.groupby("symbol").tail(1)
        values = latest.set_index("symbol")["roe"]
        values = winsorize(values, 0.01, 0.99).clip(-1, 1)
        return FactorResult(name=self.name, values=values,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class ROA(Factor):
    """总资产收益率 - 正向因子"""
    name = "roa"
    group = "quality"
    direction = 1
    
    def calculate(self, df: pd.DataFrame) -> FactorResult:
        latest = df.groupby("symbol").tail(1)
        values = latest.set_index("symbol")["roa"]
        values = winsorize(values, 0.01, 0.99).clip(-0.5, 0.5)
        return FactorResult(name=self.name, values=values,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class GrossMargin(Factor):
    """毛利率 - 正向因子"""
    name = "gross_margin"
    group = "quality"
    direction = 1
    
    def calculate(self, df: pd.DataFrame) -> FactorResult:
        latest = df.groupby("symbol").tail(1)
        values = latest.set_index("symbol")["gross_margin"]
        values = winsorize(values, 0.01, 0.99).clip(0, 1)
        return FactorResult(name=self.name, values=values,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class NetMargin(Factor):
    """净利率 - 正向因子"""
    name = "net_margin"
    group = "quality"
    direction = 1
    
    def calculate(self, df: pd.DataFrame) -> FactorResult:
        latest = df.groupby("symbol").tail(1)
        values = latest.set_index("symbol")["net_margin"]
        values = winsorize(values, 0.01, 0.99).clip(-1, 1)
        return FactorResult(name=self.name, values=values,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class RevenueGrowth(Factor):
    """营收增长率 - 正向因子"""
    name = "revenue_growth"
    group = "quality"
    direction = 1
    
    def calculate(self, df: pd.DataFrame) -> FactorResult:
        latest = df.groupby("symbol").tail(1)
        values = latest.set_index("symbol")["revenue_growth"]
        values = winsorize(values, 0.01, 0.99).clip(-2, 2)
        return FactorResult(name=self.name, values=values,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class ProfitGrowth(Factor):
    """净利润增长率 - 正向因子"""
    name = "profit_growth"
    group = "quality"
    direction = 1
    
    def calculate(self, df: pd.DataFrame) -> FactorResult:
        latest = df.groupby("symbol").tail(1)
        values = latest.set_index("symbol")["profit_growth"]
        values = winsorize(values, 0.01, 0.99).clip(-5, 5)
        return FactorResult(name=self.name, values=values,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class QualityComposite(Factor):
    """质量综合因子 = ROE + 毛利率 + 营收增长 等权合成"""
    name = "quality_composite"
    group = "quality"
    direction = 1
    
    def calculate(self, df: pd.DataFrame) -> FactorResult:
        latest = df.groupby("symbol").tail(1).set_index("symbol")
        
        # 取各质量指标
        roe = latest.get("roe", pd.Series(np.nan, index=latest.index))
        gm = latest.get("gross_margin", pd.Series(np.nan, index=latest.index))
        rg = latest.get("revenue_growth", pd.Series(np.nan, index=latest.index))
        ng = latest.get("net_margin", pd.Series(np.nan, index=latest.index))
        
        # Z-score标准化后等权合成
        from .base import zscore
        composite = (zscore(roe.fillna(roe.median())) + 
                    zscore(gm.fillna(gm.median())) + 
                    zscore(rg.fillna(rg.median())) + 
                    zscore(ng.fillna(ng.median()))) / 4
        
        composite = winsorize(composite, 0.01, 0.99)
        return FactorResult(name=self.name, values=composite,
                          direction=self.direction, group=self.group)
