"""测试选股模型"""
import pytest
import pandas as pd
import numpy as np

from models.stock_picker import MultiFactorScorer, SectorRotator


class TestMultiFactorScorer:
    """测试多因子打分器"""

    def test_equal_weights_scoring(self):
        # 3只股票，2个因子
        factor_matrix = pd.DataFrame({
            "pe_ttm": [0.2, 0.5, 0.8],   # 反向因子已反转
            "roe":    [0.9, 0.5, 0.1],   # 正向因子
        }, index=["A", "B", "C"])

        scorer = MultiFactorScorer()  # 等权
        result = scorer.score(factor_matrix)

        assert len(result) == 3
        assert "total_score" in result.columns

        # A: PE低(好)=0.2, ROE高(好)=0.9 → 总分应最高
        # C: PE高(差)=0.8, ROE低(差)=0.1 → 总分应最低
        assert result.loc["A", "total_score"] > result.loc["C", "total_score"]

    def test_custom_weights(self):
        # 注意：ranked值越大越好。
        # PE是反向因子，ranked 0.8 = PE最低 = 最好
        # ROE是正向因子，ranked 0.9 = ROE最高 = 最好
        factor_matrix = pd.DataFrame({
            "pe_ttm": [0.8, 0.5, 0.2],   # A的PE最低（最好），C的PE最高（最差）
            "roe":    [0.9, 0.5, 0.1],   # A的ROE最高（最好），C的ROE最低（最差）
        }, index=["A", "B", "C"])

        scorer = MultiFactorScorer(factor_weights={"pe_ttm": 0.8, "roe": 0.2})
        result = scorer.score(factor_matrix)

        # A在两个因子上都最好，应排第一
        assert result.index[0] == "A"

    def test_empty_matrix(self):
        scorer = MultiFactorScorer()
        result = scorer.score(pd.DataFrame())
        assert result.empty

    def test_sector_whitelist(self):
        factor_matrix = pd.DataFrame({
            "pe_ttm": [0.2, 0.5, 0.8],
        }, index=["A", "B", "C"])
        sector_map = pd.Series({"A": "银行", "B": "科技", "C": "消费"}, name="sector")

        scorer = MultiFactorScorer(sector_whitelist={"银行", "消费"})
        result = scorer.score(factor_matrix, sector_map=sector_map)

        assert "B" not in result.index
        assert set(result.index) == {"A", "C"}

    def test_sector_blacklist(self):
        factor_matrix = pd.DataFrame({
            "pe_ttm": [0.2, 0.5, 0.8],
        }, index=["A", "B", "C"])
        sector_map = pd.Series({"A": "银行", "B": "科技", "C": "消费"}, name="sector")

        scorer = MultiFactorScorer(sector_blacklist={"科技"})
        result = scorer.score(factor_matrix, sector_map=sector_map)

        assert "B" not in result.index

    def test_mv_filter(self):
        factor_matrix = pd.DataFrame({
            "pe_ttm": [0.2, 0.5, 0.8],
        }, index=["A", "B", "C"])
        mv = pd.Series({"A": 1000000, "B": 500000, "C": 2000000}, name="mv")

        scorer = MultiFactorScorer(min_mv=800000)
        result = scorer.score(factor_matrix, mv_series=mv)

        assert "B" not in result.index
        assert set(result.index) == {"A", "C"}

    def test_turnover_filter(self):
        factor_matrix = pd.DataFrame({
            "pe_ttm": [0.2, 0.5, 0.8],
        }, index=["A", "B", "C"])
        turnover = pd.Series({"A": 0.02, "B": 0.005, "C": 0.03}, name="turnover")

        scorer = MultiFactorScorer(min_turnover=0.01)
        result = scorer.score(factor_matrix, turnover_series=turnover)

        assert "B" not in result.index

    def test_hotspot_weight(self):
        factor_matrix = pd.DataFrame({
            "pe_ttm": [0.5, 0.5, 0.5],
        }, index=["A", "B", "C"])
        sector_map = pd.Series({"A": "银行", "B": "科技", "C": "消费"}, name="sector")
        hotspot = {"科技"}

        scorer = MultiFactorScorer(hotspot_weight=0.2)
        result = scorer.score(factor_matrix, sector_map=sector_map, hotspot_sectors=hotspot)

        # B是热点板块，应加分
        assert result.loc["B", "total_score"] > result.loc["A", "total_score"]
        assert result.loc["B", "total_score"] > result.loc["C", "total_score"]

    def test_missing_factor_weight_fallback(self):
        # 权重中指定了不存在的因子
        factor_matrix = pd.DataFrame({
            "pe_ttm": [0.2, 0.5],
        }, index=["A", "B"])

        scorer = MultiFactorScorer(factor_weights={"pe_ttm": 0.5, "nonexistent": 0.5})
        result = scorer.score(factor_matrix)

        # 只使用存在的因子，pe_ttm权重被归一化为1
        assert len(result) == 2
        assert result.loc["A", "total_score"] < result.loc["B", "total_score"]

    def test_get_top_n(self):
        # ranked值越大越好。PE是反向因子，ranked 0.5 = PE最低 = 最好
        factor_matrix = pd.DataFrame({
            "pe_ttm": [0.5, 0.4, 0.3, 0.2, 0.1],  # A的PE最低（最好）
        }, index=["A", "B", "C", "D", "E"])

        scorer = MultiFactorScorer()
        top3 = scorer.get_top_n(factor_matrix, n=3)

        assert len(top3) == 3
        assert top3.index[0] == "A"  # A的PE ranked最高 = PE最低 = 最好


class TestSectorRotator:
    """测试板块轮动筛选器"""

    def test_get_hot_sectors(self):
        sector_df = pd.DataFrame({
            "sector_code": ["BK001", "BK002", "BK003", "BK001", "BK002", "BK003"],
            "sector_name": ["银行", "科技", "消费", "银行", "科技", "消费"],
            "trade_date": ["2025-01-01", "2025-01-01", "2025-01-01",
                          "2025-01-02", "2025-01-02", "2025-01-02"],
            "close": [100, 200, 150, 102, 210, 148],
            "pct_change": [0.02, 0.05, -0.01, 0.02, 0.05, -0.013],
            "turnover": [0.01, 0.03, 0.02, 0.01, 0.03, 0.02],
            "up_count": [5, 15, 8, 5, 15, 7],
            "down_count": [3, 2, 5, 3, 2, 6],
        })
        sector_df["trade_date"] = pd.to_datetime(sector_df["trade_date"])

        rotator = SectorRotator()
        hot = rotator.get_hot_sectors(sector_df, top_n=2)

        assert len(hot) == 2
        # 科技板块涨幅最高且上涨家数最多，应排第一
        assert hot.iloc[0]["sector_name"] == "科技"

    def test_get_hot_sectors_empty(self):
        rotator = SectorRotator()
        hot = rotator.get_hot_sectors(pd.DataFrame(), top_n=5)
        assert hot.empty
