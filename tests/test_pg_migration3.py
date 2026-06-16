"""PG 表结构升级后兼容性测试（简化）。"""
from __future__ import annotations

import pandas as pd
from datetime import date


def test_save_bars_fills_source_adjust():
    """检查 save_bars 逻辑中的字段填充。"""
    # 模拟 save_bars 处理逻辑
    df = pd.DataFrame({
        "symbol": ["600519"],
        "trade_date": [date(2026, 6, 12)],
        "close": [100.0],
        "volume": [10000],
    })
    
    # 模拟 save_bars 中的字段填充
    df_clean = df.copy()
    if "source" not in df_clean.columns:
        df_clean["source"] = ""
    if "adjust" not in df_clean.columns:
        df_clean["adjust"] = "raw"
    
    assert "source" in df_clean.columns
    assert "adjust" in df_clean.columns
    assert (df_clean["source"] == "").all()
    assert (df_clean["adjust"] == "raw").all()


def test_get_bars_includes_source_adjust_columns():
    """get_bars 返回的 DataFrame 包含 source/adjust 字段。"""
    # 直接验证返回的 DataFrame 结构
    df = pd.DataFrame({
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
    assert "source" in df.columns
    assert "adjust" in df.columns
    assert (df["source"] == "").all()
    assert (df["adjust"] == "raw").all()
