"""估值因子 — 短线择时精简版

20日短线择时场景下，P/E TTM/PEG/EP 等估值比率边际贡献极低。
仅保留 PB（市净率），对金融/周期类股票有辅助筛选价值。
"""
import pandas as pd
import numpy as np

from .base import Factor, FactorRegistry, FactorResult, winsorize


@FactorRegistry.register
class PB(Factor):
    """市净率 - 反向因子（辅助筛选，权重极低）"""
    name = "pb"
    group = "valuation"
    direction = -1

    def calculate(self, df: pd.DataFrame) -> FactorResult:
        latest = df.groupby("symbol").tail(1)
        values = latest.set_index("symbol")["pb"]
        values = winsorize(values, 0.01, 0.99)
        return FactorResult(name=self.name, values=values,
                          direction=self.direction, group=self.group)
