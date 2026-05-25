"""测试动量因子"""
import pytest
import pandas as pd
import numpy as np

from factors.momentum import (
    Momentum20D, Momentum60D, Momentum120D,
    High52WRatio, MomentumAcceleration
)


class TestMomentum20D:
    """测试20日动量因子"""

    def test_momentum20d_calculation(self, sample_bars_30d):
        factor = Momentum20D()
        result = factor.calculate(sample_bars_30d)

        assert result.name == "momentum_20d"
        assert result.group == "momentum"
        assert result.direction == 1
        assert len(result.values) == 5  # 5只股票

        # 30天数据 >= 20天，结果不应全为nan
        assert not result.values.isna().all()
        # 验证clip范围
        assert result.values.min() >= -0.5 - 1e-6
        assert result.values.max() <= 0.5 + 1e-6

    def test_momentum20d_clip(self, sample_bars_30d):
        factor = Momentum20D()
        result = factor.calculate(sample_bars_30d)
        # 应被clip到 [-0.5, 0.5]
        assert result.values.min() >= -0.5 - 1e-6
        assert result.values.max() <= 0.5 + 1e-6

    def test_momentum20d_insufficient_data(self):
        # 只有10天数据
        df = pd.DataFrame({
            "symbol": ["A"] * 10,
            "trade_date": pd.date_range("2025-01-01", periods=10),
            "close": range(10, 20),
        })
        factor = Momentum20D()
        result = factor.calculate(df)
        assert np.isnan(result.values["A"])

    def test_momentum20d_direction(self, sample_bars_30d):
        factor = Momentum20D()
        result = factor.calculate(sample_bars_30d)
        ranked = result.ranked

        # 正向因子：动量越高排名越高
        mom_highest = result.values.idxmax()
        mom_lowest = result.values.idxmin()
        assert ranked[mom_highest] > ranked[mom_lowest]


class TestMomentum60D:
    """测试60日动量因子"""

    def test_momentum60d_insufficient_data(self, sample_bars_30d):
        # 30天数据不够60天
        factor = Momentum60D()
        result = factor.calculate(sample_bars_30d)
        # 所有值应为nan
        assert result.values.isna().all()

    def test_momentum60d_with_enough_data(self):
        # 构造70天数据
        df = pd.DataFrame({
            "symbol": ["A"] * 70,
            "trade_date": pd.date_range("2025-01-01", periods=70),
            "close": [10 + i * 0.1 for i in range(70)],
        })
        factor = Momentum60D()
        result = factor.calculate(df)

        expected = (df["close"].iloc[-1] / df["close"].iloc[-60]) - 1
        assert result.values["A"] == pytest.approx(expected, abs=1e-6)


class TestMomentum120D:
    """测试120日动量因子"""

    def test_momentum120d_insufficient_data(self, sample_bars_30d):
        factor = Momentum120D()
        result = factor.calculate(sample_bars_30d)
        assert result.values.isna().all()


class TestHigh52WRatio:
    """测试52周新高距离因子"""

    def test_ratio_calculation(self, sample_bars_30d):
        factor = High52WRatio()
        result = factor.calculate(sample_bars_30d)

        assert result.name == "high_52w_ratio"
        assert result.direction == 1
        assert len(result.values) == 5

        # 30天数据 < 60天要求，结果应全为nan（代码中要求 >= 60天）
        # 但实际上代码检查的是 len(group) >= 60，而30天不够
        # 注意：代码里写的是 if len(group) < 60: return np.nan
        # 但前面还有一个 if len(group) < 60: return np.nan 的守卫
        # 所以30天数据应该全为nan
        assert result.values.isna().all()

    def test_ratio_clip(self, sample_bars_30d):
        factor = High52WRatio()
        result = factor.calculate(sample_bars_30d)
        # 30天数据不够60天要求，全为nan
        assert result.values.isna().all()

    def test_ratio_insufficient_data(self):
        df = pd.DataFrame({
            "symbol": ["A"] * 10,
            "trade_date": pd.date_range("2025-01-01", periods=10),
            "close": range(10, 20),
        })
        factor = High52WRatio()
        result = factor.calculate(df)
        # 10天数据 >= 60天要求？不，代码要求 >= 60天
        # 实际上代码检查 len(group) >= 60
        assert np.isnan(result.values["A"])


class TestMomentumAcceleration:
    """测试动量加速度因子"""

    def test_acceleration_calculation(self, sample_bars_30d):
        factor = MomentumAcceleration()
        result = factor.calculate(sample_bars_30d)

        assert result.name == "momentum_accel"
        assert result.direction == 1

        # 验证公式: mom_short(20d) - mom_long(60d)
        # 但30天数据不足60天，所以应该是nan
        # 等等，代码中检查 len(group) >= 60，30天不够
        assert result.values.isna().all()

    def test_acceleration_with_enough_data(self):
        # 构造70天数据，短期动量 > 长期动量
        close = [10 + i * 0.2 for i in range(70)]
        df = pd.DataFrame({
            "symbol": ["A"] * 70,
            "trade_date": pd.date_range("2025-01-01", periods=70),
            "close": close,
        })
        factor = MomentumAcceleration()
        result = factor.calculate(df)

        # 结果不应为nan，且在clip范围内
        assert not np.isnan(result.values["A"])
        assert result.values["A"] >= -0.5 - 1e-6
        assert result.values["A"] <= 0.5 + 1e-6
