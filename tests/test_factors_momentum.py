"""测试动量因子"""
import pytest
import pandas as pd
import numpy as np

from factors.momentum import (
    Momentum20D, 
    
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


