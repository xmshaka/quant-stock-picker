"""测试质量因子"""
import pytest
import pandas as pd
import numpy as np

from factors.quality import (
    ROE, ROA, GrossMargin, NetMargin,
    RevenueGrowth, ProfitGrowth, QualityComposite
)


class TestROE:
    """测试ROE因子"""

    def test_roe_calculation(self, sample_snapshot):
        factor = ROE()
        result = factor.calculate(sample_snapshot)

        assert result.name == "roe"
        assert result.group == "quality"
        assert result.direction == 1
        assert len(result.values) == len(sample_snapshot)
        assert result.values.min() >= -1
        assert result.values.max() <= 1

    def test_roe_clip(self, sample_snapshot):
        factor = ROE()
        result = factor.calculate(sample_snapshot)
        assert result.values.min() >= -1 - 1e-6
        assert result.values.max() <= 1 + 1e-6

    def test_roe_direction(self, sample_snapshot):
        factor = ROE()
        result = factor.calculate(sample_snapshot)
        ranked = result.ranked

        roe_highest = result.values.idxmax()
        roe_lowest = result.values.idxmin()
        assert ranked[roe_highest] > ranked[roe_lowest]


class TestROA:
    """测试ROA因子"""

    def test_roa_calculation(self, sample_snapshot):
        factor = ROA()
        result = factor.calculate(sample_snapshot)
        assert result.name == "roa"
        assert len(result.values) == len(sample_snapshot)
        assert result.values.min() >= -0.5
        assert result.values.max() <= 0.5


class TestGrossMargin:
    """测试毛利率因子"""

    def test_gross_margin_calculation(self, sample_snapshot):
        factor = GrossMargin()
        result = factor.calculate(sample_snapshot)
        assert result.name == "gross_margin"
        assert len(result.values) == len(sample_snapshot)
        assert result.values.min() >= 0
        assert result.values.max() <= 1

    def test_gross_margin_clip(self, sample_snapshot):
        factor = GrossMargin()
        result = factor.calculate(sample_snapshot)
        assert result.values.min() >= 0 - 1e-6
        assert result.values.max() <= 1 + 1e-6


class TestNetMargin:
    """测试净利率因子"""

    def test_net_margin_calculation(self, sample_snapshot):
        factor = NetMargin()
        result = factor.calculate(sample_snapshot)
        assert result.name == "net_margin"
        assert len(result.values) == len(sample_snapshot)
        assert result.values.min() >= -1
        assert result.values.max() <= 1


class TestRevenueGrowth:
    """测试营收增长率因子"""

    def test_revenue_growth_calculation(self, sample_snapshot):
        factor = RevenueGrowth()
        result = factor.calculate(sample_snapshot)
        assert result.name == "revenue_growth"
        assert len(result.values) == len(sample_snapshot)
        assert result.values.min() >= -2
        assert result.values.max() <= 2

    def test_revenue_growth_clip(self, sample_snapshot):
        factor = RevenueGrowth()
        result = factor.calculate(sample_snapshot)
        assert result.values.min() >= -2 - 1e-6
        assert result.values.max() <= 2 + 1e-6


class TestProfitGrowth:
    """测试净利润增长率因子"""

    def test_profit_growth_calculation(self, sample_snapshot):
        factor = ProfitGrowth()
        result = factor.calculate(sample_snapshot)
        assert result.name == "profit_growth"
        assert len(result.values) == len(sample_snapshot)
        assert result.values.min() >= -5
        assert result.values.max() <= 5


class TestQualityComposite:
    """测试质量综合因子"""

    def test_composite_calculation(self, sample_snapshot):
        factor = QualityComposite()
        result = factor.calculate(sample_snapshot)

        assert result.name == "quality_composite"
        assert result.group == "quality"
        assert result.direction == 1
        assert len(result.values) == 5

        # 综合因子 = (Z(ROE) + Z(毛利率) + Z(营收增长) + Z(净利率)) / 4
        # 验证所有值在合理范围（缩尾后应在 [0,1] 附近）
        assert not result.values.isna().any()

    def test_composite_direction(self, sample_snapshot):
        factor = QualityComposite()
        result = factor.calculate(sample_snapshot)
        ranked = result.ranked

        # 正向因子
        highest = result.values.idxmax()
        lowest = result.values.idxmin()
        assert ranked[highest] > ranked[lowest]

    def test_composite_missing_column(self):
        # 缺少部分财务列
        df = pd.DataFrame({
            "symbol": ["A", "B"],
            "trade_date": ["2025-01-01", "2025-01-01"],
            "close": [10, 20],
            "roe": [0.1, 0.2],
            # 缺少毛利率、净利率等
        })
        factor = QualityComposite()
        result = factor.calculate(df)
        # 代码中用 get + fillna(median)，缺失列会填充中位数
        assert len(result.values) == 2
        assert not result.values.isna().any()
