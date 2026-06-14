"""Dashboard 指标卡样式策略测试。"""

from dashboard.theme import (
    _metric_row_compact,
    _metric_value_classes,
    _metric_value_needs_compact,
)


def test_metric_value_needs_compact_for_units_and_long_values():
    """数字+中文/百分号/货币/日期等都应触发行级紧凑显示。"""
    assert _metric_value_needs_compact("6笔") is True
    assert _metric_value_needs_compact("3轮") is True
    assert _metric_value_needs_compact("0.5000%") is True
    assert _metric_value_needs_compact("¥5,819") is True
    assert _metric_value_needs_compact("2026-06-12") is True
    assert _metric_value_needs_compact("1534s") is True
    assert _metric_value_needs_compact("20") is False


def test_metric_row_uses_one_compact_policy_for_mixed_values():
    """同一行有中文单位/百分号时，所有指标值统一 compact，避免 5819/0.5000%/6笔 字号跳动。"""
    metrics = [
        {"label": "滑点成本", "value": "¥5,819", "color": "red"},
        {"label": "加权滑点率", "value": "0.5000%", "color": "yellow"},
        {"label": "成交笔数", "value": "6笔"},
    ]

    assert _metric_row_compact(metrics) is True
    assert _metric_value_classes(metrics[0], True) == "red compact"
    assert _metric_value_classes(metrics[1], True) == "yellow compact"
    assert _metric_value_classes(metrics[2], True) == "compact"


def test_metric_row_keeps_plain_small_counts_regular():
    """纯短数字行保留常规字号，用于量化选股首页信号数量。"""
    metrics = [
        {"label": "强力买入", "value": "20", "color": "green"},
        {"label": "一般买入", "value": "0", "color": "yellow"},
    ]

    assert _metric_row_compact(metrics) is False
    assert _metric_value_classes(metrics[0], False) == "green"
    assert _metric_value_classes(metrics[1], False) == "yellow"


def test_legacy_value_class_forces_row_compact():
    """兼容旧 value_class=small，但统一转为 compact。"""
    metrics = [
        {"label": "快照日期", "value": "2026-06-12", "value_class": "small"},
        {"label": "股票池", "value": "4538 只"},
    ]

    assert _metric_row_compact(metrics) is True
    assert _metric_value_classes(metrics[0], True) == "compact"
    assert _metric_value_classes(metrics[1], True) == "compact"
