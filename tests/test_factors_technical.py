"""测试技术因子"""
import pytest
import pandas as pd
import numpy as np

from factors.technical import (
    RSI14, MACD_Hist, BollingerPosition, BollingerWidth,
    Volatility20D, 
)


class TestRSI14:
    """测试RSI因子"""

    def test_rsi_calculation(self, sample_bars_30d):
        factor = RSI14()
        result = factor.calculate(sample_bars_30d)

        assert result.name == "rsi_14"
        assert result.group == "technical"
        assert result.direction == -1
        assert len(result.values) == 5

        # RSI应在0~100之间
        assert result.values.min() >= 0
        assert result.values.max() <= 100

    def test_rsi_insufficient_data(self):
        df = pd.DataFrame({
            "symbol": ["A"] * 5,
            "trade_date": pd.date_range("2025-01-01", periods=5),
            "close": [10, 11, 12, 13, 14],
        })
        factor = RSI14()
        result = factor.calculate(df)
        assert np.isnan(result.values["A"])

    def test_rsi_perfect_uptrend(self):
        # 连续14天上涨，RSI应为100
        df = pd.DataFrame({
            "symbol": ["A"] * 20,
            "trade_date": pd.date_range("2025-01-01", periods=20),
            "close": list(range(10, 30)),
        })
        factor = RSI14()
        result = factor.calculate(df)
        assert result.values["A"] == pytest.approx(100.0, abs=1e-6)

    def test_rsi_perfect_downtrend(self):
        # 连续14天下跌，RSI应为0
        df = pd.DataFrame({
            "symbol": ["A"] * 20,
            "trade_date": pd.date_range("2025-01-01", periods=20),
            "close": list(range(30, 10, -1)),
        })
        factor = RSI14()
        result = factor.calculate(df)
        assert result.values["A"] == pytest.approx(0.0, abs=1e-6)


class TestMACD_Hist:
    """测试MACD柱状线因子"""

    def test_macd_calculation(self, sample_bars_30d):
        factor = MACD_Hist()
        result = factor.calculate(sample_bars_30d)

        assert result.name == "macd_hist"
        assert result.group == "technical"
        assert result.direction == 1
        assert len(result.values) == 5

    def test_macd_insufficient_data(self):
        df = pd.DataFrame({
            "symbol": ["A"] * 10,
            "trade_date": pd.date_range("2025-01-01", periods=10),
            "close": [10] * 10,
        })
        factor = MACD_Hist()
        result = factor.calculate(df)
        assert np.isnan(result.values["A"])


class TestBollingerPosition:
    """测试布林带位置因子"""

    def test_position_calculation(self, sample_bars_30d):
        factor = BollingerPosition()
        result = factor.calculate(sample_bars_30d)

        assert result.name == "bband_position"
        assert result.direction == 1
        assert len(result.values) == 5

        # 应在0~1之间
        assert result.values.min() >= 0
        assert result.values.max() <= 1

    def test_position_at_upper_band(self):
        # 收盘价在布林带最上端时应接近1
        close = [10.0] * 19 + [12.0]  # 前19天在10，最后一天跳涨到12
        df = pd.DataFrame({
            "symbol": ["A"] * 20,
            "trade_date": pd.date_range("2025-01-01", periods=20),
            "close": close,
        })
        factor = BollingerPosition()
        result = factor.calculate(df)
        assert result.values["A"] > 0.5

    def test_position_at_lower_band(self):
        # 收盘价在布林带最下端时应接近0
        close = [10.0] * 19 + [8.0]
        df = pd.DataFrame({
            "symbol": ["A"] * 20,
            "trade_date": pd.date_range("2025-01-01", periods=20),
            "close": close,
        })
        factor = BollingerPosition()
        result = factor.calculate(df)
        assert result.values["A"] < 0.5


class TestBollingerWidth:
    """测试布林带宽度因子"""

    def test_width_calculation(self, sample_bars_30d):
        factor = BollingerWidth()
        result = factor.calculate(sample_bars_30d)

        assert result.name == "bband_width"
        assert result.direction == -1
        assert len(result.values) == 5

        # 宽度应为非负数
        assert result.values.min() >= 0


class TestVolatility20D:
    """测试20日波动率因子"""

    def test_volatility_calculation(self, sample_bars_30d):
        factor = Volatility20D()
        result = factor.calculate(sample_bars_30d)

        assert result.name == "volatility_20d"
        assert result.direction == -1
        assert len(result.values) == 5

        # 年化波动率应为正数
        assert result.values.min() >= 0

    def test_volatility_zero(self):
        # 价格不变，波动率为0
        df = pd.DataFrame({
            "symbol": ["A"] * 25,
            "trade_date": pd.date_range("2025-01-01", periods=25),
            "close": [10.0] * 25,
        })
        factor = Volatility20D()
        result = factor.calculate(df)
        assert result.values["A"] == pytest.approx(0.0, abs=1e-6)


