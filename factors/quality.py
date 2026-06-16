"""质量因子 — 短线择时精简版

20日短线择时场景下，ROE/ROA/毛利率/净利率作为基本面质量筛选，
帮助排除垃圾股。营收增长率和净利润增长率（季频）边际贡献低，已删除。
"""
import pandas as pd
import numpy as np

from .base import Factor, FactorRegistry, FactorResult, winsorize, zscore


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
