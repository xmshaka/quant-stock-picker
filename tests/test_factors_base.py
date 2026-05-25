"""测试因子基类与注册机制"""
import pytest
import pandas as pd
import numpy as np

from factors.base import (
    Factor, FactorRegistry, FactorResult,
    winsorize, zscore, rank_ic
)


class DummyFactor(Factor):
    """测试用虚拟因子"""
    name = "dummy_test"
    group = "test"
    direction = 1

    def calculate(self, df: pd.DataFrame) -> FactorResult:
        values = pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"])
        return FactorResult(name=self.name, values=values,
                          direction=self.direction, group=self.group)


class TestFactorResult:
    """测试 FactorResult 数据类"""

    def test_ranked_normalization(self):
        values = pd.Series([10, 20, 30, 40, 50], index=["A", "B", "C", "D", "E"])
        fr = FactorResult(name="test", values=values, direction=1, group="test")
        ranked = fr.ranked

        assert len(ranked) == 5
        assert ranked.min() == pytest.approx(0.0, abs=1e-10)
        assert ranked.max() == pytest.approx(1.0, abs=1e-10)
        assert ranked["A"] < ranked["E"]

    def test_ranked_reverse_direction(self):
        values = pd.Series([10, 20, 30], index=["A", "B", "C"])
        fr = FactorResult(name="test", values=values, direction=-1, group="test")
        ranked = fr.ranked

        # 反向因子：值越大排名越低
        assert ranked["A"] > ranked["C"]

    def test_ranked_empty_series(self):
        values = pd.Series([], dtype=float)
        fr = FactorResult(name="test", values=values, direction=1, group="test")
        ranked = fr.ranked
        assert ranked.empty

    def test_ranked_constant_values(self):
        values = pd.Series([5.0, 5.0, 5.0], index=["A", "B", "C"])
        fr = FactorResult(name="test", values=values, direction=1, group="test")
        ranked = fr.ranked
        assert (ranked == 0.5).all()


class TestFactorRegistry:
    """测试因子注册中心"""

    def setup_method(self):
        # 保存原始注册表
        self._original = dict(FactorRegistry._factors)

    def teardown_method(self):
        # 恢复原始注册表
        FactorRegistry._factors = self._original

    def test_register_factor(self):
        @FactorRegistry.register
        class MyFactor(Factor):
            name = "my_factor"
            group = "test"
            direction = 1

            def calculate(self, df: pd.DataFrame) -> FactorResult:
                return FactorResult(name=self.name, values=pd.Series(),
                                  direction=self.direction, group=self.group)

        assert "my_factor" in FactorRegistry._factors
        assert FactorRegistry._factors["my_factor"] == MyFactor

    def test_get_factor(self):
        FactorRegistry._factors["existing"] = DummyFactor
        cls = FactorRegistry.get("existing")
        assert cls == DummyFactor

    def test_get_nonexistent(self):
        assert FactorRegistry.get("not_exist") is None

    def test_list_factors(self):
        FactorRegistry._factors = {
            "f1": DummyFactor,
            "f2": DummyFactor,
        }
        names = FactorRegistry.list_factors()
        assert set(names) == {"f1", "f2"}

    def test_list_factors_by_group(self):
        class G1Factor(DummyFactor):
            name = "g1"
            group = "group_a"

        class G2Factor(DummyFactor):
            name = "g2"
            group = "group_b"

        FactorRegistry._factors = {
            "g1": G1Factor,
            "g2": G2Factor,
        }
        assert FactorRegistry.list_factors("group_a") == ["g1"]
        assert FactorRegistry.list_factors("group_b") == ["g2"]

    def test_build_all(self):
        FactorRegistry._factors = {"dummy": DummyFactor}
        instances = FactorRegistry.build_all()
        assert len(instances) == 1
        assert isinstance(instances[0], DummyFactor)


class TestUtilityFunctions:
    """测试工具函数"""

    def test_winsorize(self):
        s = pd.Series([1, 2, 3, 4, 100], index=["A", "B", "C", "D", "E"])
        result = winsorize(s, lower=0.01, upper=0.99)
        # 极端值被缩尾
        assert result["E"] <= 100
        assert result.min() >= s.quantile(0.01)
        assert result.max() <= s.quantile(0.99)

    def test_winsorize_no_extremes(self):
        s = pd.Series([1, 2, 3, 4, 5], index=["A", "B", "C", "D", "E"])
        result = winsorize(s, lower=0.01, upper=0.99)
        # 无极端值时 winsorize 不改变顺序，值近似相等（允许 quantile 插值微小差异）
        assert list(result.sort_index()) == pytest.approx([1, 2, 3, 4, 5], abs=0.1)

    def test_zscore(self):
        s = pd.Series([1, 2, 3, 4, 5], index=["A", "B", "C", "D", "E"])
        result = zscore(s)
        assert pytest.approx(result.mean(), abs=1e-10) == 0.0
        assert pytest.approx(result.std(), abs=1e-10) == 1.0

    def test_zscore_zero_std(self):
        s = pd.Series([5.0, 5.0, 5.0], index=["A", "B", "C"])
        result = zscore(s)
        assert (result == 0).all()

    def test_rank_ic_perfect_correlation(self):
        # rank_ic 要求至少10个样本
        idx = [f"A{i}" for i in range(1, 21)]
        factor = pd.Series(range(1, 21), index=idx)
        returns = pd.Series([i * 0.1 for i in range(1, 21)], index=idx)
        ic = rank_ic(factor, returns)
        assert pytest.approx(ic, abs=0.01) == 1.0

    def test_rank_ic_perfect_negative(self):
        idx = [f"A{i}" for i in range(1, 21)]
        factor = pd.Series(range(1, 21), index=idx)
        returns = pd.Series([i * 0.1 for i in range(20, 0, -1)], index=idx)
        ic = rank_ic(factor, returns)
        assert pytest.approx(ic, abs=0.01) == -1.0

    def test_rank_ic_no_overlap(self):
        factor = pd.Series([1, 2, 3], index=["A", "B", "C"])
        returns = pd.Series([0.1, 0.2], index=["D", "E"])
        ic = rank_ic(factor, returns)
        assert np.isnan(ic)

    def test_rank_ic_insufficient_data(self):
        factor = pd.Series([1, 2], index=["A", "B"])
        returns = pd.Series([0.1, 0.2], index=["A", "B"])
        ic = rank_ic(factor, returns)
        assert np.isnan(ic)
