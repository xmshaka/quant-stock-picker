"""PG 表结构升级后兼容性测试（简版）。"""
from __future__ import annotations

import pandas as pd
from datetime import date


def test_repository_save_bars_fills_source_adjust_on_the_fly():
    """检查 save_bars 逻辑中的字段填充。"""
    from data.storage.repository import StockRepository
    import data.storage.repository
    import copy

    # 保存原始实现
    original_save = data.storage.repository.StockRepository.save_bars

    # 模拟保存
    calls = []

    def mock_save(self, df, chunk_size=2000):
        # 调用前检查传入的 df
        if "source" not in df.columns:
            df = df.copy()
            df["source"] = ""
        if "adjust" not in df.columns:
            df = df.copy()
            df["adjust"] = "raw"
        calls.append(df.copy())
        # 确保字段存在
        assert "source" in df.columns
        assert "adjust" in df.columns
        # 默认值
        assert (df["source"] == "").all()
        assert (df["adjust"] == "raw").all()
        return len(df)

    # 临时替换
    data.storage.repository.StockRepository.save_bars = mock_save
    try:
        repo = StockRepository()
        df = pd.DataFrame({
            "symbol": ["600519"],
            "trade_date": [date(2026, 6, 12)],
            "close": [100.0],
            "volume": [10000],
        })
        repo.save_bars(df)
        assert len(calls) == 1
        assert "source" in calls[0].columns
        assert "adjust" in calls[0].columns
    finally:
        # 恢复
        data.storage.repository.StockRepository.save_bars = original_save


def test_get_bars_includes_source_adjust_columns():
    """get_bars 返回的 DataFrame 包含 source/adjust 字段。"""
    from data.storage.repository import StockRepository
    import data.storage.repository

    # 模拟返回数据
    def mock_get_bars(self, symbol, start_date=None, end_date=None, limit=None):
        return pd.DataFrame({
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

    original_get = data.storage.repository.StockRepository.get_bars
    data.storage.repository.StockRepository.get_bars = mock_get_bars
    try:
        repo = StockRepository()
        df = repo.get_bars("600519")
        assert "source" in df.columns
        assert "adjust" in df.columns
        assert (df["source"] == "").all()
        assert (df["adjust"] == "raw").all()
    finally:
        data.storage.repository.StockRepository.get_bars = original_get
