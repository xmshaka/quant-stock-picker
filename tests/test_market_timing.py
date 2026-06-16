"""测试大盘择时模型"""
import pytest
from datetime import date
from market.timing import (
    MarketTimingModel, PositionBracket, POSITION_BRACKETS, create_market_timing
)


class TestPositionBrackets:
    """测试仓位档位"""

    def test_bracket_count(self):
        assert len(POSITION_BRACKETS) == 5

    def test_bracket_ranges_non_overlapping(self):
        for i in range(len(POSITION_BRACKETS) - 1):
            assert POSITION_BRACKETS[i].max_score <= POSITION_BRACKETS[i+1].min_score

    def test_full_coverage(self):
        assert POSITION_BRACKETS[0].min_score == 0
        assert POSITION_BRACKETS[-1].max_score == 100

    def test_position_pct_bounds(self):
        for b in POSITION_BRACKETS:
            assert 0 <= b.position_pct <= 1.0
            assert 0 <= b.per_entry_mult <= 1.0

    def test_bracket_lookup(self):
        """测试评分→仓位档位映射"""
        test_cases = [
            (0, "防御"),
            (10, "防御"),
            (19, "防御"),
            (20, "低仓"),
            (35, "低仓"),
            (39, "低仓"),
            (40, "中等"),
            (55, "中等"),
            (59, "中等"),
            (60, "高仓"),
            (75, "高仓"),
            (79, "高仓"),
            (80, "满仓"),
            (95, "满仓"),
            (99, "满仓"),
        ]
        for score, expected_label in test_cases:
            for bracket in POSITION_BRACKETS:
                if bracket.min_score <= score < bracket.max_score:
                    assert bracket.label == expected_label, f"score={score}: expected {expected_label}, got {bracket.label}"
                    break


class TestMarketTimingModel:
    """测试大盘择时模型核心功能"""

    def test_initialization(self):
        model = MarketTimingModel()
        assert not model._loaded
        assert model.score_on(date(2025, 6, 15)) == 50.0  # 未加载返回中性
        assert model.position_multiplier_on(date(2025, 6, 15)) == 0.78  # 默认高仓位
        assert model.bracket_on(date(2025, 6, 15)).label == "高仓"

    def test_position_multiplier_range(self):
        """验证所有档位倍数在 [0, 1] 范围内"""
        for bracket in POSITION_BRACKETS:
            assert 0 <= bracket.per_entry_mult <= 1.0
            assert 0 <= bracket.position_pct <= 1.0

    def test_detail_default(self):
        model = MarketTimingModel()
        detail = model.detail_on(date(2025, 6, 15))
        assert detail == {'trend': 12.5, 'capital': 12.5, 'leverage': 12.5, 'activity': 12.5}

    def test_empty_dataframe(self):
        model = MarketTimingModel()
        df = model.to_dataframe()
        assert df.empty

    def test_fallback_on_future_date(self):
        """未来日期应回退到最近已知交易日"""
        model = MarketTimingModel()
        # 未加载数据，返回默认值
        score = model.score_on(date(2030, 1, 1))
        assert score == 50.0


class TestMarketTimingWithData:
    """使用真实Tushare数据的集成测试"""

    @pytest.fixture
    def model(self):
        import os
        token = os.getenv("TUSHARE_TOKEN", "")
        if not token:
            pytest.skip("需要 TUSHARE_TOKEN")
        m = MarketTimingModel(tushare_token=token)
        m.fetch_all("20250601", "20250616")
        return m

    def test_fetch_all_returns_scores(self, model):
        assert model._loaded
        assert len(model._scores) > 0

    def test_scores_in_range(self, model):
        for d, score in model._scores.items():
            assert 0 <= score <= 100, f"Score {score} out of range on {d}"

    def test_detail_keys(self, model):
        for d, detail in model._details.items():
            assert set(detail.keys()) == {'trend', 'capital', 'leverage', 'activity'}
            for k, v in detail.items():
                assert 0 <= v <= 25, f"{k}={v} out of range on {d}"

    def test_brackets_assigned(self, model):
        for d, bracket in model._brackets.items():
            assert bracket is not None
            assert bracket.label in {"防御", "低仓", "中等", "高仓", "满仓"}

    def test_dataframe_export(self, model):
        df = model.to_dataframe()
        assert not df.empty
        assert list(df.columns) == [
            'date', 'score', 'trend', 'capital', 'leverage', 'activity',
            'bracket', 'position_pct', 'multiplier'
        ]
        assert len(df) == len(model._scores)
