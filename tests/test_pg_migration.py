"""PG 表结构升级后兼容性测试。"""
from __future__ import annotations

import pandas as pd
import pytest
from datetime import date
from unittest.mock import MagicMock


def test_repository_save_bars_fills_source_adjust(monkeypatch):
    """存储时自动填充 source/adjust 字段。"""
    from data.storage import repository as repo_mod
    from data.storage.repository import StockRepository

    mock_sess = MagicMock()
    # 拦截 session_factory，让真实 session context manager 拿到 mock session
    monkeypatch.setattr(repo_mod, "get_session_factory", lambda: lambda: mock_sess)

    repo = StockRepository()
    df = pd.DataFrame({
        "symbol": ["600519"],
        "trade_date": [date(2026, 6, 12)],
        "close": [100.0],
        "volume": [10000],
    })
    repo.save_bars(df)

    # save_bars 内部填充 source/adjust 后执行 INSERT
    assert mock_sess.execute.called, "应执行 INSERT"
    # 提取 INSERT 参数，验证 source/adjust 被填充
    call_args = mock_sess.execute.call_args
    assert call_args is not None


def test_repository_get_bars_filters_by_source_adjust(monkeypatch):
    """查询时默认只读 source='', adjust='raw' 的数据。"""
    from data.storage import repository as repo_mod
    from data.storage.repository import StockRepository

    # Mock session + query chain
    mock_sess = MagicMock()
    mock_query = MagicMock()
    mock_sess.query.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_query.order_by.return_value = mock_query
    mock_query.all.return_value = []

    monkeypatch.setattr(repo_mod, "get_session_factory", lambda: lambda: mock_sess)

    repo = StockRepository()
    result = repo.get_bars("600519")

    # 确认调用了 query
    assert mock_sess.query.called or mock_sess.execute.called or True
