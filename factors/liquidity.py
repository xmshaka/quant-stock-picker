"""流动性因子

基于真实成交数据计算，数据来源：AKShare stock_zh_a_spot_em / stock_zh_a_daily
"""
import pandas as pd
import numpy as np

from .base import Factor, FactorRegistry, FactorResult, winsorize


@FactorRegistry.register
class Turnover20D(Factor):
    """20日平均换手率 - 反向因子（适度流动性即可，过高可能投机）

    换手率过高往往伴随短线投机，稳定性差。
    """
    name = "turnover_20d"
    group = "liquidity"
    direction = -1  # 反向：换手率过高不好

    def calculate(self, df: pd.DataFrame) -> FactorResult:
        df = df.sort_values(["symbol", "trade_date"])

        def calc_turnover(group):
            if len(group) < 20:
                return np.nan
            return group.tail(20).mean()

        turnover = df.groupby("symbol")["turnover"].apply(calc_turnover)
        turnover = winsorize(turnover, 0.01, 0.99).clip(0, 0.5)
        return FactorResult(name=self.name, values=turnover,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class TurnoverRatio(Factor):
    """流通市值换手率 = 成交额 / 流通市值 - 反向因子

    反映资金在流通盘中的活跃程度。
    """
    name = "amt_per_cap"
    group = "liquidity"
    direction = -1

    def calculate(self, df: pd.DataFrame) -> FactorResult:
        latest = df.groupby("symbol").tail(1).set_index("symbol")
        amount = latest.get("amount", pd.Series(np.nan, index=latest.index))
        float_mv = latest.get("float_mv", pd.Series(1, index=latest.index))

        # amount 单位：元（AKShare返回的成交额通常是元）
        # float_mv 单位：元（测试中构造的是元）
        ratio = amount / float_mv
        ratio = winsorize(ratio, 0.01, 0.99).clip(0, 10)
        return FactorResult(name=self.name, values=ratio,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class VolumeStd20D(Factor):
    """20日成交量波动率 - 反向因子

    成交量波动大说明筹码不稳定。
    """
    name = "volume_std_20d"
    group = "liquidity"
    direction = -1

    def calculate(self, df: pd.DataFrame) -> FactorResult:
        df = df.sort_values(["symbol", "trade_date"])

        def calc_vol(group):
            if len(group) < 20:
                return np.nan
            vol = group.tail(20)
            return vol.std() / vol.mean()  # 变异系数

        vstd = df.groupby("symbol")["volume"].apply(calc_vol)
        vstd = winsorize(vstd, 0.01, 0.99).clip(0, 5)
        return FactorResult(name=self.name, values=vstd,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class LiquidityComposite(Factor):
    """流动性综合因子 = 换手率 + 成交量波动 反向合成

    流动性适中最好：太低不好交易，太高波动大。
    """
    name = "liquidity_composite"
    group = "liquidity"
    direction = -1

    def calculate(self, df: pd.DataFrame) -> FactorResult:
        df = df.sort_values(["symbol", "trade_date"])
        latest = df.groupby("symbol").tail(1).set_index("symbol")

        turnover = latest.get("turnover", pd.Series(np.nan, index=latest.index))

        # 计算20日成交量变异系数
        def calc_vstd(group):
            if len(group) < 20:
                return np.nan
            vol = group.tail(20)
            return vol.std() / vol.mean()

        vstd = df.groupby("symbol")["volume"].apply(calc_vstd)
        vstd = vstd.reindex(latest.index)

        from .base import zscore
        composite = (zscore(turnover.fillna(turnover.median())) +
                    zscore(vstd.fillna(vstd.median()))) / 2

        composite = winsorize(composite, 0.01, 0.99)
        return FactorResult(name=self.name, values=composite,
                          direction=self.direction, group=self.group)
