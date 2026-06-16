"""动量因子 — 短线择时精简版

20日短线择时场景下，仅保留 Momentum20D 作为核心动量因子。
Momentum60D/120D/52周新高/动量加速度与20日动量高度相关，已删除。
"""
import pandas as pd
import numpy as np

from .base import Factor, FactorRegistry, FactorResult, winsorize


@FactorRegistry.register
class Momentum20D(Factor):
    """20日动量 - 正向因子（核心动量因子）"""
    name = "momentum_20d"
    group = "momentum"
    direction = 1

    def calculate(self, df: pd.DataFrame) -> FactorResult:
        """计算20日收益率"""
        def calc_mom(group):
            if len(group) < 20:
                return np.nan
            return group.iloc[-1] / group.iloc[-20] - 1

        df = df.sort_values(["symbol", "trade_date"])
        mom = df.groupby("symbol")["close"].apply(calc_mom)
        mom = winsorize(mom, 0.01, 0.99).clip(-0.5, 0.5)
        return FactorResult(name=self.name, values=mom,
                          direction=self.direction, group=self.group)
