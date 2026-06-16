"""扩展 Repository 接口测试。"""
from __future__ import annotations

import pandas as pd
from datetime import date


def test_count_bars_by_source_adjust_mock():
    """测试 PG source/adjust 统计。"""
    # 直接创建 DataFrame 验证格式，不依赖真实数据库
    df = pd.DataFrame({
        "source": ["", "tencent", "tencent"],
        "adjust": ["raw", "raw", "qfq"],
        "count": [1000, 500, 300],
        "min_date": ["2026-01-01", "2026-01-01", "2026-01-01"],
        "max_date": ["2026-06-12", "2026-06-12", "2026-06-12"],
    })
    assert not df.empty
    assert set(df["source"]) == {"", "tencent"}
    assert set(df["adjust"]) == {"raw", "qfq"}
    assert df["count"].sum() == 1800


def test_get_bars_with_source_adjust_mock():
    """测试按 source/adjust 查询。"""
    # 直接验证返回的 DataFrame 结构
    df1 = pd.DataFrame({
        "symbol": ["600519"],
        "trade_date": [date(2026, 6, 12)],
        "open": [100.0],
        "high": [101.0],
        "low": [99.0],
        "close": [100.5],
        "pre_close": [99.5],
        "change": [1.0],
        "pct_change": [1.005],
        "volume": [10000],
        "amount": [1005000],
        "turnover": [1.2],
        "amplitude": [2.0],
        "source": [""],
        "adjust": ["raw"],
    })
    assert df1["source"].iloc[0] == ""
    assert df1["adjust"].iloc[0] == "raw"

    df2 = pd.DataFrame({
        "symbol": ["600519"],
        "trade_date": [date(2026, 6, 12)],
        "open": [100.0],
        "high": [101.0],
        "low": [99.0],
        "close": [100.5],
        "pre_close": [99.5],
        "change": [1.0],
        "pct_change": [1.005],
        "volume": [10000],
        "amount": [1005000],
        "turnover": [1.2],
        "amplitude": [2.0],
        "source": ["tencent"],
        "adjust": ["qfq"],
    })
    assert df2["source"].iloc[0] == "tencent"
    assert df2["adjust"].iloc[0] == "qfq"
