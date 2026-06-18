"""K线图渲染测试 — lightweight-charts (TradingView) 版。

2026-06-18: Plotly → lightweight-charts 重构。
- plot_kline_with_signals 返回 {"html", "height"} 而非 go.Figure
- 十字光标、日期轴、副图联动由 lightweight-charts 原生处理
- 无需检查 hoverinfo/hovertemplate/showspikes 等 Plotly 属性
"""
from __future__ import annotations

import json
import pandas as pd

from dashboard.components.kline_chart import plot_kline_with_signals


def _sample_bars(n: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    close = pd.Series([10 + i * 0.05 + ((-1) ** i) * 0.1 for i in range(n)])
    open_ = close.shift(1).fillna(close.iloc[0] - 0.05)
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.2
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.2
    return pd.DataFrame({
        "trade_date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": [1_000_000 + i * 10_000 for i in range(n)],
    })


def test_returns_html_dict():
    """返回 {"html", "height"} 而非 Plotly Figure。"""
    result = plot_kline_with_signals(_sample_bars(), [], symbol="测试", show_kdj=True)
    assert isinstance(result, dict)
    assert "html" in result
    assert "height" in result
    assert isinstance(result["html"], str)
    assert len(result["html"]) > 100


def test_html_contains_ohlcv_data():
    """HTML 内嵌完整的 OHLCV JSON 数据。"""
    result = plot_kline_with_signals(_sample_bars(10), [], symbol="测试")
    html = result["html"]
    assert '"ohlcv"' in html
    assert '"open"' in html
    assert '"close"' in html


def test_html_contains_candlestick():
    """HTML 包含 Candlestick Series 配置。"""
    result = plot_kline_with_signals(_sample_bars(), [], symbol="测试")
    html = result["html"]
    assert "Candlestick" in html or "candlestick" in html.lower()


def test_html_contains_ma_series():
    """均线数据嵌入 HTML。"""
    result = plot_kline_with_signals(_sample_bars(30), [], symbol="测试", show_ma=True)
    html = result["html"]
    # MA data in JSON
    assert '"5"' in html or '"ma"' in html


def test_html_contains_macd_subplot():
    """MACD 副图数据嵌入 HTML。"""
    result = plot_kline_with_signals(_sample_bars(40), [], symbol="测试", show_kdj=True)
    html = result["html"]
    assert '"macd"' in html.lower() or "MACD" in html


def test_html_contains_crosshair():
    """crosshair 配置嵌入 HTML。"""
    result = plot_kline_with_signals(_sample_bars(), [], symbol="测试", show_kdj=True)
    html = result["html"]
    assert "crosshair" in html.lower()


def test_empty_bars_returns_placeholder():
    """空数据返回占位 HTML。"""
    result = plot_kline_with_signals(pd.DataFrame(), [], symbol="测试")
    html = result["html"]
    assert "暂无K线数据" in html
    assert result["height"] == 200


def test_height_passed_through():
    """height 透传到返回 dict。"""
    result = plot_kline_with_signals(_sample_bars(), [], symbol="测试", height=700)
    assert result["height"] == 700


def test_volume_trace_not_skipped():
    """成交量数据在 HTML 中。"""
    result = plot_kline_with_signals(_sample_bars(), [], symbol="测试", show_volume=True)
    html = result["html"]
    assert '"vol"' in html.lower()
