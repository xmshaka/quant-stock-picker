"""K线图渲染测试 — ECharts 多 grid 专业K线版。

2026-06-19: lightweight-charts 多独立 canvas 同步仍有右侧空白/十字光标错位，
切换为 Apache ECharts 单实例多 grid：统一 xAxis、axisPointer.link、dataZoom。
"""
from __future__ import annotations

import pandas as pd

from dashboard.components.kline_chart import plot_kline_with_signals, render_kline_chart
from dashboard.kline_events import trade_points_from_executed_frame
from signals.rules import TradePoint


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
    result = plot_kline_with_signals(_sample_bars(), [], symbol="测试", show_kdj=True)
    assert isinstance(result, dict)
    assert "html" in result
    assert "height" in result
    assert isinstance(result["html"], str)
    assert len(result["html"]) > 100


def test_html_contains_ohlcv_data():
    result = plot_kline_with_signals(_sample_bars(10), [], symbol="测试")
    html = result["html"]
    assert '"ohlcv"' in html
    assert '"open"' in html
    assert '"close"' in html
    assert '"time": "2026-01-01"' in html


def test_html_contains_echarts_candlestick():
    """HTML 使用本地 ECharts 单实例 candlestick，不再使用 lightweight 多 canvas。"""
    result = plot_kline_with_signals(_sample_bars(), [], symbol="测试")
    html = result["html"]
    assert "unpkg.com" not in html
    assert "cdn.jsdelivr.net" not in html
    assert "echarts.init" in html
    assert "type: 'candlestick'" in html
    assert "LightweightCharts" not in html
    assert "addCandlestickSeries" not in html


def test_empty_bars_returns_placeholder():
    result = plot_kline_with_signals(pd.DataFrame(), [], symbol="测试")
    html = result["html"]
    assert "暂无K线数据" in html
    assert result["height"] == 200


def test_height_passed_through():
    result = plot_kline_with_signals(_sample_bars(), [], symbol="测试", height=700)
    assert result["height"] == 700


def test_volume_kdj_macd_data_embedded():
    result = plot_kline_with_signals(_sample_bars(60), [], symbol="测试", show_volume=True, show_kdj=True, show_macd=True)
    html = result["html"]
    assert '"vol"' in html.lower()
    assert '"kdj"' in html.lower()
    assert '"macd"' in html.lower()
    assert "name: '成交量'" in html
    assert "name: 'K'" in html
    assert "name: 'MACD柱'" in html


def test_multigrid_single_axis_pointer_layout():
    """ECharts 单实例多 grid：统一 axisPointer/link/dataZoom 解决光标错位。"""
    result = plot_kline_with_signals(_sample_bars(60), [], symbol="测试", show_volume=True, show_kdj=True, show_macd=True)
    html = result["html"]
    assert "axisPointer" in html
    assert "link: [{ xAxisIndex: 'all' }]" in html
    assert "xAxisIndex: [0,1,2,3]" in html
    assert "dataZoom" in html
    assert "grid: [" in html
    assert "gridIndex: i" in html
    assert "chart.addPane" not in html
    assert "setCrosshairPosition" not in html


def test_buy_sell_marker_details_are_embedded():
    bars = _sample_bars(60)
    buy = TradePoint(
        date=bars["trade_date"].iloc[10].date(),
        action="BUY",
        reason="L1趋势+L3共振",
        confidence=0.82,
        price=10.5,
        rule_name="trend_momentum",
        exec_price=10.6,
        shares=1000,
    )
    sell = TradePoint(
        date=bars["trade_date"].iloc[30].date(),
        action="SELL",
        reason="止盈/信号转弱",
        confidence=0.76,
        price=11.2,
        rule_name="exit_rule",
        exec_price=11.1,
        shares=1000,
        pnl=500.0,
        pnl_pct=0.05,
        holding_days=12,
    )
    result = plot_kline_with_signals(bars, [buy, sell], symbol="测试", show_kdj=True, show_macd=True)
    html = result["html"]
    assert '"action": "BUY"' in html
    assert '"action": "SELL"' in html
    assert "trend_momentum" in html
    assert "L1趋势+L3共振" in html
    assert "exit_rule" in html
    assert "止盈/信号转弱" in html


def test_kline_marker_uses_exec_date_and_keeps_signal_date_tooltip():
    """K线买卖点必须按成交日落点，同时保留信号日用于审计。"""
    bars = _sample_bars(20)
    p = TradePoint(
        date=bars["trade_date"].iloc[5].date(),
        action="BUY",
        reason="T日收盘信号",
        confidence=1.0,
        price=10.0,
        exec_price=10.2,
        shares=1000,
    )
    setattr(p, "signal_date", bars["trade_date"].iloc[5].date())
    setattr(p, "exec_date", bars["trade_date"].iloc[6].date())
    result = plot_kline_with_signals(bars, [p], symbol="测试", show_kdj=True, show_macd=True)
    html = result["html"]
    assert f'"time": "{bars["trade_date"].iloc[6].strftime("%Y-%m-%d")}"' in html
    assert f'"signalDate": "{bars["trade_date"].iloc[5].strftime("%Y-%m-%d")}"' in html
    assert f'"execDate": "{bars["trade_date"].iloc[6].strftime("%Y-%m-%d")}"' in html
    assert "信号:" in html


def test_executed_frame_to_kline_points_prioritizes_exec_date():
    """历史页K线复盘必须从成交事件构造点位，exec_date 优先于 signal_date/date。"""
    df = pd.DataFrame([
        {
            "symbol": "000001",
            "date": "2026-01-03",
            "signal_date": "2026-01-02",
            "exec_date": "2026-01-06",
            "action": "BUY",
            "exec_price": 10.5,
            "shares": 1000,
            "reason": "T日信号",
            "rule_name": "测试规则",
            "exit_type": "",
            "exit_subtype": "",
        }
    ])
    points = trade_points_from_executed_frame(df)

    assert len(points) == 1
    assert points[0].date == pd.Timestamp("2026-01-06").date()
    assert points[0].exec_date == pd.Timestamp("2026-01-06").date()
    assert points[0].signal_date == pd.Timestamp("2026-01-02").date()

    bars = _sample_bars(10)
    # 强制包含 exec_date，验证 marker 落在 exec_date 而不是 signal_date/date。
    bars.loc[5, "trade_date"] = pd.Timestamp("2026-01-06")
    result = plot_kline_with_signals(bars.sort_values("trade_date"), points, symbol="测试")
    html = result["html"]
    assert '"time": "2026-01-06"' in html
    assert '"signalDate": "2026-01-02"' in html


def test_signal_markers_use_scatter_and_dashed_lines():
    bars = _sample_bars(60)
    buy = TradePoint(
        date=bars["trade_date"].iloc[10].date(),
        action="BUY",
        reason="测试买点",
        confidence=0.8,
        price=10.5,
        rule_name="rule",
        exec_price=10.6,
    )
    result = plot_kline_with_signals(bars, [buy], symbol="测试", show_kdj=True, show_macd=True)
    html = result["html"]
    assert "name: '信号连线'" in html
    assert "type: 'lines'" in html
    assert "type: 'dashed'" in html
    assert "name: '买卖点'" in html
    assert "type: 'scatter'" in html
    assert "formatter: isBuy ? 'B' : 'S'" in html
    assert '"anchorPrice"' in html
    assert '"labelPrice"' in html


def test_tooltip_is_compact_and_centered():
    result = plot_kline_with_signals(_sample_bars(60), [], symbol="测试", show_kdj=True, show_macd=True)
    html = result["html"]
    assert "trigger: 'axis'" in html
    assert "axisPointer: { type: 'cross' }" in html
    assert "confine: true" in html
    assert "text-align:center" in html
    assert "activeGridIndex === 1" in html
    assert "activeGridIndex === 2" in html
    assert "activeGridIndex === 3" in html
    assert "成交量</b>" in html
    assert "DIF</b>" in html


def test_main_sub_ratio_is_half():
    result = plot_kline_with_signals(_sample_bars(60), [], symbol="测试", show_kdj=True, show_macd=True, height=760)
    html = result["html"]
    assert "height:760px" in html
    assert "height: 372" in html  # 主 grid 760 * 0.49
    assert "height: 114" in html  # 成交量 760 * 0.15
    assert "height: 121" in html  # KDJ 760 * 0.16
    assert html.count("height: 121") >= 2  # KDJ/MACD
    assert "top: 6" in html
    assert "top: 384" in html
    assert "top: 504" in html
    assert "top: 631" in html


def test_axis_date_label_only_on_bottom_and_no_slider():
    """日期轴 pointer label 只保留一个，去掉底部可见 dataZoom 滑条。"""
    result = plot_kline_with_signals(_sample_bars(60), [], symbol="测试", show_kdj=True, show_macd=True, height=760)
    html = result["html"]
    assert "axisPointer: { label: { show: i === 3 } }" in html
    assert "axisLabel: { show: i === 3" in html
    assert "type: 'inside'" in html
    assert "type: 'slider'" not in html


def test_volume_axis_uses_simplified_units():
    """成交量轴/tooltip 使用万/亿简化单位。"""
    result = plot_kline_with_signals(_sample_bars(60), [], symbol="测试", show_kdj=True, show_macd=True, height=760)
    html = result["html"]
    assert "function fmtVol" in html
    assert "+ '亿'" in html
    assert "+ '万'" in html
    assert "formatter: fmtVol" in html


def test_width_resize_observer_prevents_right_blank():
    result = plot_kline_with_signals(_sample_bars(60), [], symbol="测试", show_kdj=True, show_macd=True, height=760)
    html = result["html"]
    assert "ResizeObserver" in html
    assert "chart.resize" in html
    assert "wrap.clientWidth" in html
    assert "barSpacing: 8" not in html
    assert "min: 'dataMin'" in html
    assert "max: 'dataMax'" in html


def test_echarts_js_is_localized():
    result = plot_kline_with_signals(_sample_bars(60), [], symbol="测试", show_kdj=True, show_macd=True)
    html = result["html"]
    assert "unpkg.com" not in html
    assert "cdn.jsdelivr.net" not in html
    assert "echarts-multigrid" in html
    assert "Apache" in html or "ECharts" in html or "echarts" in html


def test_render_height_syncs_iframe_and_inner_chart(monkeypatch):
    captured = {}

    def fake_html(html, height=None, scrolling=False):
        captured["html"] = html
        captured["height"] = height
        captured["scrolling"] = scrolling

    import streamlit.components.v1 as components
    monkeypatch.setattr(components, "html", fake_html)

    result = plot_kline_with_signals(_sample_bars(60), [], symbol="测试", show_kdj=True, height=580)
    render_kline_chart(result, height=760)
    assert captured["height"] == 760
    assert "height:760px" in captured["html"]
    assert "min-height:760px" in captured["html"]
