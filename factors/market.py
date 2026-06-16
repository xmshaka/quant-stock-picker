"""市场/规模因子 — 短线择时专用

新增因子（2026-06-16）:
- FloatMarketCap: 流通市值对数，控制大小盘风格暴露
- FreeFloatTurnover: 自由流通市值换手率，反映资金活跃度
- High20DDistance: 20日最高价距离，判断突破/追高风险
- RelativeStrength5D: 5日相对强度（相对全市场中位数）
- VolumeRatio: 量比 = 5日均量 / 20日均量，捕捉放量
- LimitDistance: 涨跌停距离，排除流动性陷阱
"""
import pandas as pd
import numpy as np

from .base import Factor, FactorRegistry, FactorResult, winsorize


@FactorRegistry.register
class FloatMarketCap(Factor):
    """流通市值（对数） - 正向因子（适度偏中小盘）

    ln(流通市值)，去极值后标准化。
    短线场景下中小盘弹性更大，但也不是越小越好（排除垃圾微盘）。
    """
    name = "float_market_cap"
    group = "market"
    direction = -1  # 反向：市值越小弹性越大（但设置下限过滤微盘）

    def calculate(self, df: pd.DataFrame) -> FactorResult:
        latest = df.groupby("symbol").tail(1).set_index("symbol")

        # 优先使用 float_mv（流通市值），否则用 total_mv（总市值）
        if "float_mv" in latest.columns:
            mv = latest["float_mv"]
        elif "total_mv" in latest.columns:
            mv = latest["total_mv"]
        else:
            # 兜底：从 amount/turnover 推算
            amount = latest.get("amount", pd.Series(np.nan, index=latest.index))
            turnover = latest.get("turnover", pd.Series(np.nan, index=latest.index))
            mv = amount / turnover.replace(0, np.nan)

        # 过滤：市值 < 20亿的微盘股（流动性差）
        mv = mv.clip(lower=2e9)
        log_mv = np.log(mv)
        log_mv = winsorize(log_mv, 0.01, 0.99)
        return FactorResult(name=self.name, values=log_mv,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class FreeFloatTurnover(Factor):
    """自由流通市值换手率 - 正向因子

    = 成交额 / 流通市值，反映资金在流通盘中的真实活跃度。
    适度偏高更好（有资金关注），但不能过高（排除对倒出货）。
    """
    name = "free_float_turnover"
    group = "market"
    direction = 1

    def calculate(self, df: pd.DataFrame) -> FactorResult:
        latest = df.groupby("symbol").tail(1).set_index("symbol")
        amount = latest.get("amount", pd.Series(np.nan, index=latest.index))
        float_mv = latest.get("float_mv", pd.Series(np.nan, index=latest.index))

        ratio = amount / float_mv.replace(0, np.nan)
        # 换手率 > 20% 可能异常（对倒/新股），clip 到合理范围
        ratio = winsorize(ratio, 0.01, 0.99).clip(0.001, 0.20)
        return FactorResult(name=self.name, values=ratio,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class High20DDistance(Factor):
    """20日最高价距离 - 正向因子

    = (收盘价 - 20日最低价) / (20日最高价 - 20日最低价)
    越接近20日高点越好（趋势延续），但也不能已经在高点（排除追高）。
    配合 Volatility20D 使用效果更好。
    """
    name = "high_20d_distance"
    group = "market"
    direction = 1

    def calculate(self, df: pd.DataFrame) -> FactorResult:
        df = df.sort_values(["symbol", "trade_date"])

        def calc_dist(group):
            if len(group) < 20:
                return np.nan
            recent = group.tail(20)
            hh = recent.max()
            ll = recent.min()
            if hh == ll:
                return 0.5
            return (recent.iloc[-1] - ll) / (hh - ll)

        dist = df.groupby("symbol")["close"].apply(calc_dist)
        dist = winsorize(dist, 0.01, 0.99).clip(0, 1)
        return FactorResult(name=self.name, values=dist,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class RelativeStrength5D(Factor):
    """5日相对强度 - 正向因子

    = 个股5日涨幅 - 全市场中位数5日涨幅
    正值表示跑赢市场，短线强势股通常有正的相对强度。
    """
    name = "relative_strength_5d"
    group = "market"
    direction = 1

    def calculate(self, df: pd.DataFrame) -> FactorResult:
        df = df.sort_values(["symbol", "trade_date"])

        def calc_ret(group):
            if len(group) < 5:
                return np.nan
            return group.iloc[-1] / group.iloc[-5] - 1

        ret_5d = df.groupby("symbol")["close"].apply(calc_ret)

        # 相对市场整体中位数
        median_ret = ret_5d.median()
        rel_strength = ret_5d - median_ret
        rel_strength = winsorize(rel_strength, 0.01, 0.99).clip(-0.3, 0.3)
        return FactorResult(name=self.name, values=rel_strength,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class VolumeRatio(Factor):
    """量比 = 5日均量 / 20日均量 - 正向因子

    量比 > 1 表示近期放量，有资金关注。
    短线择时核心因子：放量上涨是有效突破的必要条件。
    """
    name = "volume_ratio"
    group = "market"
    direction = 1

    def calculate(self, df: pd.DataFrame) -> FactorResult:
        df = df.sort_values(["symbol", "trade_date"])

        def calc_ratio(group):
            if len(group) < 20:
                return np.nan
            vol_5d = group.tail(5).mean()
            vol_20d = group.tail(20).mean()
            if vol_20d == 0:
                return 1.0
            return vol_5d / vol_20d

        ratio = df.groupby("symbol")["volume"].apply(calc_ratio)
        ratio = winsorize(ratio, 0.01, 0.99).clip(0.3, 5.0)
        return FactorResult(name=self.name, values=ratio,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class LimitDistance(Factor):
    """涨跌停距离 - 正向因子

    = min(涨停距离%, 跌停距离%) / 涨跌停幅度
    距离涨跌停越远越好 → 流动性充足，不会被封板卡住。
    距离涨停太近有买不到风险，距离跌停太近有流动性危机。
    """
    name = "limit_distance"
    group = "market"
    direction = 1

    def calculate(self, df: pd.DataFrame) -> FactorResult:
        latest = df.groupby("symbol").tail(1).set_index("symbol")
        close = latest.get("close", pd.Series(np.nan, index=latest.index))

        # 前一日收盘作为参考
        prev_close = latest.get("pre_close", pd.Series(np.nan, index=latest.index))
        if prev_close.isna().all():
            prev_close = close  # fallback

        # 涨停价 = prev_close * 1.10, 跌停价 = prev_close * 0.90（主板10%）
        limit_up = prev_close * 1.10
        limit_down = prev_close * 0.90

        # 距离 = min(到涨停距离, 到跌停距离) / (涨跌停幅度)
        up_dist = (limit_up - close) / prev_close
        down_dist = (close - limit_down) / prev_close
        dist = np.minimum(up_dist, down_dist) / 0.20  # 归一化到 [0, 1]

        dist = winsorize(dist, 0.01, 0.99).clip(0, 1)
        return FactorResult(name=self.name, values=dist,
                          direction=self.direction, group=self.group)
