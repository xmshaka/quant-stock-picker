"""测试估值因子

验证点：
1. 因子值在合理范围内（PE_TTM >0, PB>0 等）
2. 方向正确（反向因子值越大排名越低）
3. 极端值已被缩尾处理
4. 空数据/缺失数据处理正确
"""
import pytest
import pandas as pd
import numpy as np

from factors.valuation import PE_TTM, PB, PEG, EP


class TestPE_TTM:
    """测试市盈率因子"""

    def test_pe_calculation(self, sample_snapshot):
        factor = PE_TTM()
        result = factor.calculate(sample_snapshot)

        assert result.name == "pe_ttm"
        assert result.group == "valuation"
        assert result.direction == -1  # 反向因子
        assert len(result.values) == len(sample_snapshot)

        # PE值应为正数，在合理范围内（winsorize可能轻微改变边缘值）
        assert result.values.min() > 0
        assert result.values.max() < 100  # 缩尾后不应有极端值

    def test_pe_direction(self, sample_snapshot):
        factor = PE_TTM()
        result = factor.calculate(sample_snapshot)
        ranked = result.ranked

        # 反向因子：PE越低排名越高
        pe_lowest = result.values.idxmin()
        pe_highest = result.values.idxmax()
        assert ranked[pe_lowest] > ranked[pe_highest]

    def test_pe_winsorize(self, sample_snapshot):
        # 制造极端值
        df = sample_snapshot.copy()
        df.loc[0, "pe_ttm"] = 99999.0
        factor = PE_TTM()
        result = factor.calculate(df)

        # 极端值应被缩尾
        assert result.values.max() < 99999.0

    def test_pe_missing_column(self):
        df = pd.DataFrame({
            "symbol": ["A", "B"],
            "trade_date": ["2025-01-01", "2025-01-01"],
            "close": [10, 20],
        })
        factor = PE_TTM()
        with pytest.raises(KeyError):
            factor.calculate(df)

    def test_pe_empty(self, empty_df):
        factor = PE_TTM()
        # 空DataFrame需要包含必要的列结构
        df = pd.DataFrame(columns=["symbol", "trade_date", "pe_ttm"])
        result = factor.calculate(df)
        assert result.values.empty


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


class TestPEG:
    """测试PEG因子"""

    def test_peg_calculation(self, sample_snapshot):
        factor = PEG()
        result = factor.calculate(sample_snapshot)

        assert result.name == "peg"
        assert result.direction == -1

        # PEG = PE / |盈利增长率|，且被clip到上限5
        pe = sample_snapshot.set_index("symbol")["pe_ttm"]
        growth = sample_snapshot.set_index("symbol")["profit_growth"]
        expected = (pe / growth.abs()).clip(upper=5)

        # 注意：growth为负时，abs后算PEG
        for sym in expected.index:
            assert result.values[sym] == pytest.approx(expected[sym], abs=1e-6)

    def test_peg_zero_growth(self, sample_snapshot):
        # 盈利增长率为0时，PEG应被处理为较大值
        df = sample_snapshot.copy()
        df.loc[df["symbol"] == "000001", "profit_growth"] = 0.0
        factor = PEG()
        result = factor.calculate(df)

        # growth为0会导致除零，代码中用abs()后除，会是inf，然后被replace为nan
        # 实际行为：inf -> nan，然后winsorize会保留nan
        val = result.values["000001"]
        assert np.isnan(val) or val == pytest.approx(5.0, abs=1.1)

    def test_peg_clip_upper(self, sample_snapshot):
        factor = PEG()
        result = factor.calculate(sample_snapshot)
        # 所有PEG应被clip到5以内
        assert result.values.max() <= 5.0 + 1e-6


class TestEP:
    """测试EP（盈利收益率）因子"""

    def test_ep_calculation(self, sample_snapshot):
        factor = EP()
        result = factor.calculate(sample_snapshot)

        assert result.name == "ep"
        assert result.direction == 1  # 正向因子
        assert len(result.values) == len(sample_snapshot)
        assert result.values.min() > 0  # EP = 1/PE，PE为正则EP为正

    def test_ep_direction(self, sample_snapshot):
        factor = EP()
        result = factor.calculate(sample_snapshot)
        ranked = result.ranked

        # 正向因子：EP越高排名越高
        ep_highest = result.values.idxmax()
        ep_lowest = result.values.idxmin()
        assert ranked[ep_highest] > ranked[ep_lowest]
