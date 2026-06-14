"""回测记录页筛选逻辑测试。"""

from datetime import datetime

import pandas as pd

from dashboard.history_filters import filter_backtest_runs, unique_non_empty


def _runs():
    return pd.DataFrame([
        {
            "run_id": "20260613_170000_value",
            "scheme_name": "低波价值",
            "pool_mode": "自定义代码",
            "symbols": "002145",
            "created_at": "2026-06-13 17:00:00",
        },
        {
            "run_id": "20260612_220000_momentum",
            "scheme_name": "趋势动量",
            "pool_mode": "全A",
            "symbols": "5156150,600519",
            "created_at": "2026-06-12 22:00:00",
        },
        {
            "run_id": "20260601_100000_value",
            "scheme_name": "低波价值",
            "pool_mode": "观察池",
            "symbols": "300628",
            "created_at": "2026-06-01 10:00:00",
        },
    ])


def test_unique_non_empty_sorted():
    assert unique_non_empty(["低波价值", "", None, "趋势动量", "低波价值"]) == ["低波价值", "趋势动量"]


def test_filter_by_symbol_query():
    out = filter_backtest_runs(_runs(), symbol_query="5156150")
    assert out["run_id"].tolist() == ["20260612_220000_momentum"]


def test_filter_by_run_id_query():
    out = filter_backtest_runs(_runs(), symbol_query="170000")
    assert out["symbols"].tolist() == ["002145"]


def test_filter_by_scheme_and_pool():
    out = filter_backtest_runs(_runs(), scheme_name="低波价值", pool_mode="观察池")
    assert out["symbols"].tolist() == ["300628"]


def test_filter_recent_days():
    out = filter_backtest_runs(_runs(), recent_days=3, now=datetime(2026, 6, 13, 18, 0, 0))
    assert out["run_id"].tolist() == ["20260613_170000_value", "20260612_220000_momentum"]


def test_filter_combined_no_match():
    out = filter_backtest_runs(_runs(), symbol_query="002145", scheme_name="趋势动量")
    assert out.empty
