"""动量因子"""
import pandas as pd
import numpy as np

from .base import Factor, FactorRegistry, FactorResult, winsorize


@FactorRegistry.register
class Momentum20D(Factor):
    """20日动量 - 正向因子（中期动量）"""
    name = "momentum_20d"
    group = "momentum"
    direction = 1
    
    def calculate(self, df: pd.DataFrame) -> FactorResult:
        """计算20日收益率"""
        def calc_mom(group):
            if len(group) < 20:
                return np.nan
            return group.iloc[-1] / group.iloc[-20] - 1
        
        # 确保按日期排序
        df = df.sort_values(["symbol", "trade_date"])
        mom = df.groupby("symbol")["close"].apply(calc_mom)
        mom = winsorize(mom, 0.01, 0.99).clip(-0.5, 0.5)
        return FactorResult(name=self.name, values=mom,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class Momentum60D(Factor):
    """60日动量 - 正向因子（中长期动量）"""
    name = "momentum_60d"
    group = "momentum"
    direction = 1
    
    def calculate(self, df: pd.DataFrame) -> FactorResult:
        def calc_mom(group):
            if len(group) < 60:
                return np.nan
            return group.iloc[-1] / group.iloc[-60] - 1
        
        df = df.sort_values(["symbol", "trade_date"])
        mom = df.groupby("symbol")["close"].apply(calc_mom)
        mom = winsorize(mom, 0.01, 0.99).clip(-1, 1)
        return FactorResult(name=self.name, values=mom,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class Momentum120D(Factor):
    """120日动量 - 正向因子（长期动量）"""
    name = "momentum_120d"
    group = "momentum"
    direction = 1
    
    def calculate(self, df: pd.DataFrame) -> FactorResult:
        def calc_mom(group):
            if len(group) < 120:
                return np.nan
            return group.iloc[-1] / group.iloc[-120] - 1
        
        df = df.sort_values(["symbol", "trade_date"])
        mom = df.groupby("symbol")["close"].apply(calc_mom)
        mom = winsorize(mom, 0.01, 0.99).clip(-2, 2)
        return FactorResult(name=self.name, values=mom,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class High52WRatio(Factor):
    """52周新高距离 - 正向因子
    
    越接近52周新高越好
    """
    name = "high_52w_ratio"
    group = "momentum"
    direction = 1
    
    def calculate(self, df: pd.DataFrame) -> FactorResult:
        df = df.sort_values(["symbol", "trade_date"])
        
        def calc_ratio(group):
            if len(group) < 60:  # 至少60天数据
                return np.nan
            high_52w = group.tail(252).max() if len(group) >= 252 else group.max()
            current = group.iloc[-1]
            if high_52w > 0:
                return current / high_52w
            return np.nan
        
        ratio = df.groupby("symbol")["close"].apply(calc_ratio)
        ratio = winsorize(ratio, 0.01, 0.99).clip(0, 1)
        return FactorResult(name=self.name, values=ratio,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class MomentumAcceleration(Factor):
    """动量加速度 = 短期动量 - 长期动量
    
    正向因子，捕捉动量加速的股票
    """
    name = "momentum_accel"
    group = "momentum"
    direction = 1
    
    def calculate(self, df: pd.DataFrame) -> FactorResult:
        df = df.sort_values(["symbol", "trade_date"])
        
        def calc_accel(group):
            if len(group) < 60:
                return np.nan
            mom_short = group.iloc[-1] / group.iloc[-20] - 1 if len(group) >= 20 else 0
            mom_long = group.iloc[-1] / group.iloc[-60] - 1
            return mom_short - mom_long
        
        accel = df.groupby("symbol")["close"].apply(calc_accel)
        accel = winsorize(accel, 0.01, 0.99).clip(-0.5, 0.5)
        return FactorResult(name=self.name, values=accel,
                          direction=self.direction, group=self.group)
