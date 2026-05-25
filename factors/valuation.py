"""估值因子"""
import pandas as pd
import numpy as np

from .base import Factor, FactorRegistry, FactorResult, winsorize


@FactorRegistry.register
class PE_TTM(Factor):
    """市盈率TTM - 反向因子，越低越好"""
    name = "pe_ttm"
    group = "valuation"
    direction = -1  # 反向因子
    
    def calculate(self, df: pd.DataFrame) -> FactorResult:
        """需要pe_ttm列"""
        latest = df.groupby("symbol").tail(1)
        values = latest.set_index("symbol")["pe_ttm"]
        values = winsorize(values, 0.01, 0.99)
        return FactorResult(name=self.name, values=values, 
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class PB(Factor):
    """市净率 - 反向因子"""
    name = "pb"
    group = "valuation"
    direction = -1
    
    def calculate(self, df: pd.DataFrame) -> FactorResult:
        latest = df.groupby("symbol").tail(1)
        values = latest.set_index("symbol")["pb"]
        values = winsorize(values, 0.01, 0.99)
        return FactorResult(name=self.name, values=values,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class PEG(Factor):
    """PEG = PE / 盈利增长率 - 反向因子
    
    PEG < 1 被认为低估
    """
    name = "peg"
    group = "valuation"
    direction = -1
    
    def calculate(self, df: pd.DataFrame) -> FactorResult:
        # 需要pe_ttm和profit_growth
        latest = df.groupby("symbol").tail(1)
        pe = latest.set_index("symbol")["pe_ttm"]
        growth = latest.set_index("symbol")["profit_growth"]
        
        # 盈利增长率为负的，PEG设为较大值
        peg = pe / growth.abs()
        peg = peg.replace([np.inf, -np.inf], np.nan)
        peg = winsorize(peg, 0.01, 0.99).clip(upper=5)
        
        return FactorResult(name=self.name, values=peg,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class EP(Factor):
    """EP = 1/PE - 盈利收益率 - 正向因子"""
    name = "ep"
    group = "valuation"
    direction = 1
    
    def calculate(self, df: pd.DataFrame) -> FactorResult:
        latest = df.groupby("symbol").tail(1)
        pe = latest.set_index("symbol")["pe_ttm"]
        ep = (1 / pe).replace([np.inf, -np.inf], np.nan)
        ep = winsorize(ep, 0.01, 0.99)
        return FactorResult(name=self.name, values=ep,
                          direction=self.direction, group=self.group)
