"""情绪因子

数据来源：
- 北向资金（沪深港通持股）：AKShare stock_hsgt_stock_em
- 融资融券余额：AKShare stock_margin_detail_em

⚠️ 所有数据必须来自真实API，严禁编造。
"""
import pandas as pd
import numpy as np

from .base import Factor, FactorRegistry, FactorResult, winsorize


@FactorRegistry.register
class NorthHoldPct(Factor):
    """北向资金持股比例 - 正向因子

    北向资金（沪深港通）通常被视为"聪明钱"，持股比例高的股票可能更受外资青睐。
    数据来源：AKShare stock_hsgt_stock_em
    """
    name = "north_hold_pct"
    group = "sentiment"
    direction = 1

    def calculate(self, df: pd.DataFrame) -> FactorResult:
        latest = df.groupby("symbol").tail(1)
        values = latest.set_index("symbol")["north_hold_pct"]
        values = winsorize(values, 0.01, 0.99).clip(0, 1)
        return FactorResult(name=self.name, values=values,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class NorthHoldChange20D(Factor):
    """北向资金20日持股变动 - 正向因子

    北向资金近期加仓的股票，可能预示外资看好。
    需要历史数据：当前持股比例 - 20日前持股比例
    """
    name = "north_hold_change_20d"
    group = "sentiment"
    direction = 1

    def calculate(self, df: pd.DataFrame) -> FactorResult:
        df = df.sort_values(["symbol", "trade_date"])

        def calc_change(group):
            if len(group) < 20:
                return np.nan
            current = group.iloc[-1]
            past = group.iloc[-20]
            return current - past

        change = df.groupby("symbol")["north_hold_pct"].apply(calc_change)
        change = winsorize(change, 0.01, 0.99).clip(-0.5, 0.5)
        return FactorResult(name=self.name, values=change,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class MarginBalanceRatio(Factor):
    """融资余额占流通市值比 - 正向因子（适度杠杆代表市场看好）

    融资余额高说明散户/机构愿意借钱买，情绪乐观。
    数据来源：AKShare stock_margin_detail_em
    """
    name = "margin_balance_ratio"
    group = "sentiment"
    direction = 1

    def calculate(self, df: pd.DataFrame) -> FactorResult:
        latest = df.groupby("symbol").tail(1)
        margin = latest.set_index("symbol")["margin_balance"]
        float_mv = latest.set_index("symbol")["float_mv"]
        # 融资余额占流通市值比（万元/万元）
        ratio = margin / float_mv
        ratio = winsorize(ratio, 0.01, 0.99).clip(0, 0.5)
        return FactorResult(name=self.name, values=ratio,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class MarginBalanceChange20D(Factor):
    """融资余额20日变动率 - 正向因子

    融资余额增加代表杠杆资金流入，情绪升温。
    """
    name = "margin_change_20d"
    group = "sentiment"
    direction = 1

    def calculate(self, df: pd.DataFrame) -> FactorResult:
        df = df.sort_values(["symbol", "trade_date"])

        def calc_change(group):
            if len(group) < 20:
                return np.nan
            current = group.iloc[-1]
            past = group.iloc[-20]
            if past == 0:
                return np.nan
            return (current - past) / past

        change = df.groupby("symbol")["margin_balance"].apply(calc_change)
        change = winsorize(change, 0.01, 0.99).clip(-1, 1)
        return FactorResult(name=self.name, values=change,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class SentimentComposite(Factor):
    """情绪综合因子 = 北向持股变动 + 融资余额变动 等权合成

    综合反映外资和杠杆资金的情绪动向。
    """
    name = "sentiment_composite"
    group = "sentiment"
    direction = 1

    def calculate(self, df: pd.DataFrame) -> FactorResult:
        df = df.sort_values(["symbol", "trade_date"])
        latest = df.groupby("symbol").tail(1).set_index("symbol")

        north = latest.get("north_hold_pct", pd.Series(np.nan, index=latest.index))
        margin = latest.get("margin_balance", pd.Series(np.nan, index=latest.index))
        float_mv = latest.get("float_mv", pd.Series(1, index=latest.index))

        # 北向持股比例 + 融资余额占流通市值比
        margin_ratio = margin / float_mv
        from .base import zscore
        composite = (zscore(north.fillna(north.median())) +
                    zscore(margin_ratio.fillna(margin_ratio.median()))) / 2

        composite = winsorize(composite, 0.01, 0.99)
        return FactorResult(name=self.name, values=composite,
                          direction=self.direction, group=self.group)
