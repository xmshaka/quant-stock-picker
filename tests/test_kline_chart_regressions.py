"""K线图交互与显示回归测试。"""

from __future__ import annotations

import pandas as pd

from dashboard.components.kline_chart import plot_kline_with_signals


def _sample_bars(n: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
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


def test_kline_hover_template_uses_chinese_labels():
    fig = plot_kline_with_signals(_sample_bars(), [], symbol="测试", show_kdj=True)
    hover_templates = "\n".join(str(getattr(trace, "hovertemplate", "")) for trace in fig.data)
    kline_trace = next(trace for trace in fig.data if getattr(trace, "name", "") == "K线")

    assert kline_trace.hoverinfo is None
    assert "开盘" in hover_templates
    assert "最高" in hover_templates
    assert "最低" in hover_templates
    assert "收盘" in hover_templates
    assert "成交量" in hover_templates
    assert "O:" not in hover_templates
    assert "C:" not in hover_templates


def test_kline_unified_tip_does_not_duplicate_main_traces():
    """主图只允许一个 K线 hover 锚点，影线/实体/均线/买卖点不参与 hover。"""
    fig = plot_kline_with_signals(_sample_bars(), [], symbol="测试", show_kdj=True)
    main_hover_templates = [
        trace.hovertemplate
        for trace in fig.data
        if trace.hovertemplate and "开盘" in str(trace.hovertemplate)
    ]
    skipped = [trace.name for trace in fig.data if getattr(trace, "hoverinfo", None) == "skip"]

    assert len(main_hover_templates) == 1
    assert {"阳线", "阴线", "MA5", "MA10", "MA20"}.issubset(set(skipped))


def test_kline_adds_macd_subplot_by_default():
    fig = plot_kline_with_signals(_sample_bars(), [], symbol="测试", show_kdj=True)
    names = {trace.name for trace in fig.data if getattr(trace, "name", None)}

    assert {"MACD柱", "DIF", "DEA"}.issubset(names)
    assert fig.layout.height >= 760


def test_kline_crosshair_and_x_axes_are_synced():
    fig = plot_kline_with_signals(_sample_bars(), [], symbol="测试", show_kdj=True)
    layout = fig.layout.to_plotly_json()
    xaxes = {k: v for k, v in layout.items() if k.startswith("xaxis")}

    assert len(xaxes) >= 4  # 主图 + 成交量 + KDJ + MACD
    assert all(axis.get("matches") == "x" for axis in xaxes.values())
    assert all(axis.get("showspikes") is True for axis in xaxes.values())
    assert all(axis.get("spikemode") == "across+toaxis" for axis in xaxes.values())
    assert fig.layout.hovermode == "x unified"
    assert fig.layout.hoversubplots == "axis"
    assert fig.layout.spikedistance == -1


def test_kdj_yaxis_is_fixed_to_prevent_zoom_clipping():
    fig = plot_kline_with_signals(_sample_bars(), [], symbol="测试", show_kdj=True)

    assert list(fig.layout.yaxis3.range) == [-20, 120]


def test_volume_trace_is_not_gray_or_translucent():
    fig = plot_kline_with_signals(_sample_bars(), [], symbol="测试", show_kdj=True)
    volume_trace = next(trace for trace in fig.data if getattr(trace, "name", "") == "成交量")

    colors = list(volume_trace.marker.color)
    assert colors
    assert all("0.82" in color for color in colors)
    assert volume_trace.opacity == 1.0
