"""技术因子

基于真实行情数据计算的技术面指标。
所有公式公开可查，数据仅来源于 AKShare/Tushare 的 OHLCV 数据。
"""
import pandas as pd
import numpy as np

from .base import Factor, FactorRegistry, FactorResult, winsorize


@FactorRegistry.register
class RSI14(Factor):
    """RSI 14日相对强弱指标 - 反向因子

    公式来源: J. Welles Wilder, New Concepts in Technical Trading Systems (1978)
    RSI = 100 - 100 / (1 + RS), RS = 平均上涨幅度 / 平均下跌幅度
    超买(>70)时反向，超卖(<30)时正向，但回测通常显示中等RSI更好，故处理为反向因子
    """
    name = "rsi_14"
    group = "technical"
    direction = -1  # 反向：RSI过高往往预示回调

    def calculate(self, df: pd.DataFrame) -> FactorResult:
        df = df.sort_values(["symbol", "trade_date"])

        def calc_rsi(group):
            if len(group) < 14:
                return np.nan
            close = group.values
            deltas = np.diff(close)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            if avg_loss == 0:
                return 100.0
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

        rsi = df.groupby("symbol")["close"].apply(calc_rsi)
        rsi = winsorize(rsi, 0.01, 0.99).clip(0, 100)
        return FactorResult(name=self.name, values=rsi,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class MACD_Hist(Factor):
    """MACD 柱状线 - 正向因子

    公式来源: Gerald Appel (1979)
    EMA12 = EMA(close, 12), EMA26 = EMA(close, 26)
    DIF = EMA12 - EMA26, DEA = EMA(DIF, 9), HIST = DIF - DEA
    HIST > 0 且扩大表示多头增强
    """
    name = "macd_hist"
    group = "technical"
    direction = 1

    def calculate(self, df: pd.DataFrame) -> FactorResult:
        df = df.sort_values(["symbol", "trade_date"])

        def calc_macd(group):
            if len(group) < 35:  # 26 + 9
                return np.nan
            close = group.values
            ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().iloc[-1]
            ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().iloc[-1]
            dif = ema12 - ema26
            # DEA 需要历史 DIF，这里用简化版：对整段 DIF 算 EMA9
            dif_series = pd.Series(close).ewm(span=12, adjust=False).mean() - \
                        pd.Series(close).ewm(span=26, adjust=False).mean()
            dea = dif_series.ewm(span=9, adjust=False).mean().iloc[-1]
            return dif - dea

        hist = df.groupby("symbol")["close"].apply(calc_macd)
        hist = winsorize(hist, 0.01, 0.99)
        return FactorResult(name=self.name, values=hist,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class BollingerPosition(Factor):
    """布林带位置 - 正向因子

    公式来源: John Bollinger (1980s)
    中轨 = MA20, 上轨 = MA20 + 2*STD20, 下轨 = MA20 - 2*STD20
    位置 = (close - 下轨) / (上轨 - 下轨)，0~1 之间
    靠近下轨(<0.2)可能超卖，靠近上轨(>0.8)可能超买
    取位置值作为正向因子：越高说明越强（趋势延续）
    """
    name = "bband_position"
    group = "technical"
    direction = 1

    def calculate(self, df: pd.DataFrame) -> FactorResult:
        df = df.sort_values(["symbol", "trade_date"])

        def calc_pos(group):
            if len(group) < 20:
                return np.nan
            close = group.tail(20).values
            ma20 = np.mean(close)
            std20 = np.std(close)
            upper = ma20 + 2 * std20
            lower = ma20 - 2 * std20
            if upper == lower:
                return 0.5
            current = close[-1]
            pos = (current - lower) / (upper - lower)
            return pos

        pos = df.groupby("symbol")["close"].apply(calc_pos)
        pos = winsorize(pos, 0.01, 0.99).clip(0, 1)
        return FactorResult(name=self.name, values=pos,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class BollingerWidth(Factor):
    """布林带宽度 - 反向因子（低波动后可能高波动，但回测通常显示低波动更稳）

    公式来源: John Bollinger
    宽度 = (上轨 - 下轨) / 中轨
    宽度收窄预示变盘，此处作为反向因子：越低越好（低波动环境）
    """
    name = "bband_width"
    group = "technical"
    direction = -1  # 反向：带宽越窄越稳定

    def calculate(self, df: pd.DataFrame) -> FactorResult:
        df = df.sort_values(["symbol", "trade_date"])

        def calc_width(group):
            if len(group) < 20:
                return np.nan
            close = group.tail(20).values
            ma20 = np.mean(close)
            std20 = np.std(close)
            if ma20 == 0:
                return np.nan
            width = (4 * std20) / ma20  # (upper - lower) / mid = 4*std / ma
            return width

        width = df.groupby("symbol")["close"].apply(calc_width)
        width = winsorize(width, 0.01, 0.99).clip(0, 1)
        return FactorResult(name=self.name, values=width,
                          direction=self.direction, group=self.group)


@FactorRegistry.register
class Volatility20D(Factor):
    """20日收益率波动率（标准差） - 反向因子

    公式来源: 经典风险度量
    波动率 = std(日收益率, 20) * sqrt(252) 年化
    越低越好
    """
    name = "volatility_20d"
    group = "technical"
    direction = -1

    def calculate(self, df: pd.DataFrame) -> FactorResult:
        df = df.sort_values(["symbol", "trade_date"])

        def calc_vol(group):
            if len(group) < 20:
                return np.nan
            returns = np.diff(group.tail(20).values) / group.tail(20).values[:-1]
            vol = np.std(returns) * np.sqrt(252)
            return vol

        vol = df.groupby("symbol")["close"].apply(calc_vol)
        vol = winsorize(vol, 0.01, 0.99).clip(0, 2)
        return FactorResult(name=self.name, values=vol,
                          direction=self.direction, group=self.group)



