"""测试质量因子"""
import pytest
import pandas as pd
import numpy as np

from factors.quality import (
    ROE, ROA, GrossMargin, NetMargin,
    
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


