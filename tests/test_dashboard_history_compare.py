"""回测记录多 run 对比逻辑测试。"""

import pandas as pd

from dashboard.history_compare import (
    build_run_compare_table,
    best_run_summary,
    compare_table_to_csv,
    equity_to_cumulative_return,
    format_compare_table,
    plot_multi_equity_curves,
)


def _runs():
    return pd.DataFrame([
        {
            "run_id": "run_b",
            "scheme_name": "趋势动量",
            "pool_mode": "全A",
            "symbols": "5156150",
            "start_date": "2026-01-01",
            "end_date": "2026-06-13",
            "total_return": 0.12,
            "annual_return": 0.24,
            "max_drawdown": 0.08,
            "win_rate": 0.55,
            "sharpe_ratio": 1.2,
            "trade_count": 5,
            "buy_count": 5,
            "sell_count": 4,
            "created_at": "2026-06-13 17:00:00",
        },
        {
            "run_id": "run_a",
            "scheme_name": "低波价值",
            "pool_mode": "自定义代码",
            "symbols": "002145",
            "start_date": "2026-01-01",
            "end_date": "2026-06-13",
            "total_return": 0.08,
            "annual_return": 0.16,
            "max_drawdown": 0.04,
            "win_rate": 0.60,
            "sharpe_ratio": 1.5,
            "trade_count": 3,
            "buy_count": 3,
            "sell_count": 2,
            "created_at": "2026-06-13 16:00:00",
        },
    ])


def test_build_compare_table_keeps_user_selected_order():
    out = build_run_compare_table(_runs(), ["run_a", "run_b"])
    assert out["run_id"].tolist() == ["run_a", "run_b"]


def test_best_run_summary():
    table = build_run_compare_table(_runs(), ["run_b", "run_a"])
    summary = best_run_summary(table)
    assert summary["count"] == 2
    assert summary["best_return_run_id"] == "run_b"
    assert summary["best_return"] == 0.12
    assert summary["best_drawdown_run_id"] == "run_a"
    assert summary["best_drawdown"] == 0.04
    assert summary["best_sharpe_run_id"] == "run_a"
    assert summary["best_sharpe"] == 1.5


def test_format_compare_table():
    table = build_run_compare_table(_runs(), ["run_a"])
    display = format_compare_table(table)
    assert display.loc[0, "Run ID"] == "run_a"
    assert display.loc[0, "总收益"] == "+8.00%"
    assert display.loc[0, "最大回撤"] == "4.00%"
    assert display.loc[0, "夏普"] == "1.500"


def test_compare_table_to_csv_keeps_raw_numeric_values():
    table = build_run_compare_table(_runs(), ["run_a"])
    data = compare_table_to_csv(table)
    text = data.decode("utf-8-sig")
    assert "run_id,scheme_name" in text
    assert "run_a" in text
    assert "0.08" in text
    assert "+8.00%" not in text


def test_empty_compare_table():
    out = build_run_compare_table(_runs(), [])
    assert out.empty
    assert best_run_summary(out) == {}
    assert format_compare_table(out).empty


def test_equity_to_cumulative_return_absolute_equity():
    equity = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "equity": [1_000_000, 1_050_000]})
    curve = equity_to_cumulative_return(equity)
    assert curve["cum_return_pct"].round(4).tolist() == [0.0, 5.0]


def test_equity_to_cumulative_return_daily_returns():
    equity = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "equity": [0.01, -0.02]})
    curve = equity_to_cumulative_return(equity)
    assert curve["cum_return_pct"].round(4).tolist() == [1.0, -1.02]


def test_plot_multi_equity_curves_has_one_trace_per_run():
    equity = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "equity": [1_000_000, 1_050_000]})
    fig = plot_multi_equity_curves({"run_a": equity, "run_b": equity})
    assert len(fig.data) == 2
    assert [trace.name for trace in fig.data] == ["run_a", "run_b"]
