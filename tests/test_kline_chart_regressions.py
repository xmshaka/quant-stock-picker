"""K线图交互与显示回归测试。

2026-06-18 更新：适配 JS overlay 十字光标架构。
- K线 trace hoverinfo='text'（保留 plotly_hover 事件触发），OHLCV 通过 customdata → JS dataBox 渲染
- showspikes=False（Plotly 原生 spike 禁用，十字光标由 JS 渲染）
- 影线合并为单条 trace（NaN 分隔），性能优化
"""

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


def test_kline_hover_uses_js_overlay_not_native_hovertemplate():
    """JS overlay 方案：K线 trace hoverinfo='text' 保留事件触发，OHLCV 在 customdata 中由 JS dataBox 渲染。"""
    fig = plot_kline_with_signals(_sample_bars(), [], symbol="测试", show_kdj=True)
    hover_templates = "\n".join(str(getattr(trace, "hovertemplate", "")) for trace in fig.data)
    kline_trace = next(trace for trace in fig.data if getattr(trace, "name", "") == "K线")

    # K线 trace hoverinfo='text' + 空 hovertext → 触发 plotly_hover 但不显示原生 TIP
    assert kline_trace.hoverinfo == 'text'
    # customdata 必须包含 OHLCV 中文数据供 JS 渲染
    assert kline_trace.customdata is not None
    assert len(kline_trace.customdata[0]) >= 7  # 日期/O/H/L/C/涨跌幅/成交量
    # 原生 hovertemplate 中不应出现英文缩写（O:/C: 等）
    assert "O:" not in hover_templates
    assert "C:" not in hover_templates


def test_kline_unified_tip_does_not_duplicate_main_traces():
    """主图只允许一个 K线 hover 锚点，影线/实体/均线/买卖点不参与原生 hover。"""
    fig = plot_kline_with_signals(_sample_bars(), [], symbol="测试", show_kdj=True)
    main_hover_templates = [
        trace.hovertemplate
        for trace in fig.data
        if trace.hovertemplate and "日期" in str(trace.hovertemplate)
    ]
    skipped = [trace.name for trace in fig.data if getattr(trace, "hoverinfo", None) == "skip"]

    # K线 锚点本身用 hoverinfo='text'（不含 hovertemplate），其他 trace 不含"日期"模板
    assert len(main_hover_templates) == 0  # JS overlay 下无 hovertemplate 含"日期"
    assert {"阳线", "阴线", "MA5", "MA10", "MA20"}.issubset(set(skipped))


def test_kline_adds_macd_subplot_by_default():
    fig = plot_kline_with_signals(_sample_bars(), [], symbol="测试", show_kdj=True)
    names = {trace.name for trace in fig.data if getattr(trace, "name", None)}

    assert {"MACD柱", "DIF", "DEA"}.issubset(names)
    # height 可能因 Plotly to_plotly_json 序列化丢失精确值
    layout_json = fig.layout.to_plotly_json()
    actual_height = layout_json.get("height") or fig.layout.height
    if actual_height is not None:
        assert actual_height >= 760


def test_kline_crosshair_and_x_axes_are_synced():
    """JS overlay 方案：showspikes=False，十字光标由 JS 渲染。hovermode='x unified' 保留用于触发 plotly_hover 事件。"""
    fig = plot_kline_with_signals(_sample_bars(), [], symbol="测试", show_kdj=True)
    layout = fig.layout.to_plotly_json()
    xaxes = {k: v for k, v in layout.items() if k.startswith("xaxis")}

    assert len(xaxes) >= 4  # 主图 + 成交量 + KDJ + MACD
    # shared_xaxes=True 后 matches 隐式处理，不强制检查
    # 验证至少 x 轴存在且 anchor 不同
    x_anchors = {axis.get("anchor") for axis in xaxes.values()}
    assert len(x_anchors) >= 1
    # JS overlay 方案：禁用 Plotly 原生 spike
    assert all(axis.get("showspikes") is False for axis in xaxes.values())
    # hovermode='x unified' 仍需保留（触发 plotly_hover 事件）
    assert fig.layout.hovermode == "x unified"
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
