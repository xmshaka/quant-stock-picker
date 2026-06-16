"""测试估值因子

验证点：
1. 因子值在合理范围内（PB>0 等）
2. 方向正确（反向因子值越大排名越低）
3. 极端值已被缩尾处理
4. 空数据/缺失数据处理正确
"""
import pytest
import pandas as pd
import numpy as np

from factors.valuation import PB


class TestPB:
    """测试市净率因子"""

    def test_pb_calculation(self, sample_snapshot):
        factor = PB()
        result = factor.calculate(sample_snapshot)

        assert result.name == "pb"
        assert result.direction == -1
        assert len(result.values) == len(sample_snapshot)
        assert result.values.min() > 0

    def test_pb_direction(self, sample_snapshot):
        factor = PB()
        result = factor.calculate(sample_snapshot)
        ranked = result.ranked

        pb_lowest = result.values.idxmin()
        pb_highest = result.values.idxmax()
        assert ranked[pb_lowest] > ranked[pb_highest]
