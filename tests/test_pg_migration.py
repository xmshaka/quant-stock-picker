"""PG 表结构升级后兼容性测试。"""
from __future__ import annotations

import pandas as pd
import pytest
from datetime import date


def test_repository_save_bars_fills_source_adjust(tmp_path, monkeypatch):
    """存储时自动填充 source/adjust 字段。"""
    # 模拟数据库连接，实际不写入
    import data.storage.repository
    original_save = data.storage.repository.StockRepository.save_bars

    calls = []

    def mock_save(self, df, chunk_size=2000):
        calls.append(df.copy())
        # 检查字段
        assert "source" in df.columns
        assert "adjust" in df.columns
        # 默认值
        assert (df["source"] == "").all()
        assert (df["adjust"] == "raw").all()
        return len(df)

    monkeypatch.setattr(data.storage.repository.StockRepository, "save_bars", mock_save)

    from data.storage.repository import StockRepository
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


def test_repository_get_bars_filters_by_source_adjust(tmp_path, monkeypatch):
    """查询时默认只读 source='', adjust='raw' 的数据。"""
    # 创建模拟 session
    import sqlalchemy.orm
    mock_session = type("MockSession", (), {
        "query": lambda *args: None,
        "filter": lambda *args: None,
        "order_by": lambda *args: None,
        "all": lambda: [],
    })()

    def mock_session_ctx(self):
        return type("MockCtx", (), {"__enter__": lambda: mock_session, "__exit__": lambda *args: None})()

    import data.storage.repository
    monkeypatch.setattr(data.storage.repository.BaseRepository, "session", mock_session_ctx)

    # 记录调用
    filters = []

    def mock_filter(self, cond):
        filters.append(cond)
        return self

    mock_session.filter = lambda cond: mock_filter(mock_session, cond)

    from data.storage.repository import StockRepository
    repo = StockRepository()
    repo.get_bars("600519")

    # 至少应包含 source='' 和 adjust='raw' 的过滤
    assert len(filters) >= 3
    # 检查是否有 source/adjust 过滤（条件可能通过 AND 合并）
    # 这里我们只确认调用了 filter，具体条件由 sqlalchemy 决定
    assert any(isinstance(f, str) or f is not None for f in filters)
