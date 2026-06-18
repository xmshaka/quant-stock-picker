"""K线 + 买卖点可视化 — 严格匹配参考图风格

参考图特征：
- 米白 Financial Professional 背景
- 阳线：红色实心  阴线：绿色实心（A股惯例）
- 上下影线同色，细线
- MA5: 黄色  MA10: 蓝色  MA20: 紫色
- 成交量：阳红阴绿柱
- KDJ 副图：K 黄 / D 蓝 / J 紫
- 买入 ▲ 黄色三角  卖出 ▼ 紫色三角
- 当前价格：右侧红/绿色标签
- 十字光标线：灰色虚线
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional

from signals.rules import TradePoint

# ── 配色 ──
KC = {
    "bg":           "#ede7e0",
    "plot_bg":      "#e6ded3",
    "grid":         "#d2c7b8",
    "axis":         "#5f5648",
    "text":         "#2f2a22",
    "text_dim":     "#8a7f6d",
    "up":           "#ef5350",   # 阳线红
    "down":         "#26a69a",   # 阴线绿
    "ma5":          "#f0b90b",
    "ma10":         "#1e88e5",
    "ma20":         "#ab47bc",
    "vol_up":       "rgba(239,83,80,0.82)",
    "vol_down":     "rgba(38,166,154,0.82)",
    "buy":          "#f0b90b",
    "sell":         "#ab47bc",
    "buy_label_bg":  "rgba(240,185,11,0.18)",
    "sell_label_bg": "rgba(171,71,188,0.18)",
    "rsi":          "#29b6f6",
    "kdj_k":        "#f0b90b",
    "kdj_d":        "#1e88e5",
    "kdj_j":        "#ab47bc",
    "macd":         "#5f5648",
    "macd_signal":  "#1e88e5",
    "macd_hist_up": "rgba(239,83,80,0.72)",
    "macd_hist_down": "rgba(38,166,154,0.72)",
    "crosshair":    "#8a7f6d",
    "wick":         "#8a7f6d",
    "tooltip_bg":   "#f6f3ed",
    "tooltip_text": "#2f2a22",
    "tooltip_border": "#c2b39f",
}


def _style_axis(fig, row: int = None, col: int = None):
    """统一坐标轴样式（十字光标由 JS 渲染，禁用 Plotly 原生 spike）"""
    kw = dict(row=row, col=col) if row else {}
    fig.update_xaxes(
        gridcolor=KC['grid'], showgrid=True, griddash='dot',
        zeroline=False, showline=False,
        tickfont=dict(size=9, color=KC['axis']),
        showspikes=False,
        **kw,
    )
    fig.update_yaxes(
        gridcolor=KC['grid'], showgrid=True, griddash='dot',
        zeroline=False, showline=False, side='right',
        tickfont=dict(size=9, color=KC['axis']),
        tickformat='.2f',
        showspikes=False,
        **kw,
    )


def _format_cn_datetime(series: pd.Series) -> pd.Series:
    """中文习惯日期格式：YYYY-MM-DD。"""
    return pd.to_datetime(series).dt.strftime('%Y-%m-%d')


def _volume_unit_text(volume: pd.Series) -> pd.Series:
    """成交量中文化，按 A 股习惯显示“万手”。输入 volume 为股数。"""
    return (volume.astype(float) / 10000.0).map(lambda x: f"{x:,.2f} 万手")


def _price_hover_customdata(bars: pd.DataFrame, volume: pd.Series) -> np.ndarray:
    """K线 tooltip 统一中文字段，避免 O/H/L/C 英文缩写。"""
    prev_close = bars['close'].astype(float).shift(1)
    pct = (bars['close'].astype(float) / prev_close - 1.0) * 100
    pct = pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return np.column_stack([
        _format_cn_datetime(bars['trade_date']),
        bars['open'].astype(float),
        bars['high'].astype(float),
        bars['low'].astype(float),
        bars['close'].astype(float),
        pct,
        _volume_unit_text(volume),
    ])


def _set_no_hover(*traces):
    """批量关闭 trace hover，避免 unified TIP 重复。"""
    for trace in traces:
        trace.hoverinfo = 'skip'
        trace.hovertemplate = None
    return traces[0] if len(traces) == 1 else traces


def _hover_participate(trace, n: int):
    """让 trace 参与 hover 事件（出现在 evt.points 中），但 tooltip 不可见。"""
    trace.hoverinfo = 'text'
    trace.hovertext = [''] * n
    trace.hovertemplate = None


def plot_kline_with_signals(
    bars: pd.DataFrame,
    trade_points: List[TradePoint],
    symbol: str = "",
    show_ma: bool = True,
    show_volume: bool = True,
    show_rsi: bool = False,
    show_kdj: bool = False,
    show_macd: bool = True,
    height: int = 580,
) -> go.Figure:
    """绘制专业K线图（蜡烛图 + 影线 + 均线 + 成交量 + KDJ/RSI）"""
    bars = bars.copy().sort_values('trade_date').reset_index(drop=True)
    bars['trade_date'] = pd.to_datetime(bars['trade_date'])
    o = bars['open'].astype(float)
    h = bars['high'].astype(float)
    l = bars['low'].astype(float)
    c = bars['close'].astype(float)
    v = bars['volume'].astype(float) if 'volume' in bars.columns else pd.Series(0, index=bars.index)
    hover_cd = _price_hover_customdata(bars, v)

    n = len(bars)
    if n == 0:
        return go.Figure()

    # 价格范围提前计算：买卖标记需要与当日 high/low 保持固定留白
    price_min = float(l.min())
    price_max = float(h.max())
    price_range = max(price_max - price_min, max(abs(price_max), 1.0) * 0.02)
    # B/S 标记必须脱离K线实体和上下影线，避免遮挡柱线。
    # 用全图价格区间 + 单根K线中位振幅取较大值，窄幅震荡时也能留足空间。
    median_bar_range = float((h - l).replace(0, np.nan).median())
    if not np.isfinite(median_bar_range) or median_bar_range <= 0:
        median_bar_range = price_range * 0.04
    marker_gap = max(price_range * 0.105, median_bar_range * 1.35)
    label_gap = max(price_range * 0.155, median_bar_range * 1.85)
    # 固定轨道：买点统一在全图最低价下方，卖点统一在全图最高价上方。
    # 不再按单日 high/low 摆放，避免图标遮挡相邻K柱。
    buy_lane_y = price_min - marker_gap
    buy_price_lane_y = buy_lane_y - label_gap * 0.55
    sell_lane_y = price_max + marker_gap
    sell_price_lane_y = sell_lane_y + label_gap * 0.55
    date_key = bars['trade_date'].dt.date
    low_by_date = dict(zip(date_key, l.astype(float)))
    high_by_date = dict(zip(date_key, h.astype(float)))

    # ── 子图布局 ──
    rows = 1
    row_heights = [0.58]
    if show_volume:
        rows += 1; row_heights.append(0.12)
    indicator_row = None
    if show_rsi or show_kdj:
        rows += 1; row_heights.append(0.18)
        indicator_row = rows
    macd_row = None
    if show_macd:
        rows += 1; row_heights.append(0.16)
        macd_row = rows

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.055,
        row_heights=row_heights,
    )

    # ── 动态柱宽：按数据密度计算，确保柱线可辨识 ──
    n_bars = len(bars)
    date_range_ms = (bars['trade_date'].iloc[-1] - bars['trade_date'].iloc[0]).total_seconds() * 1000 if n_bars > 1 else 86_400_000
    bar_width_ms = max(43_200_000, min(86_400_000, date_range_ms / max(n_bars, 1) * 0.75))

    # K线 hover 锚点：lines+markers 双重覆盖，消灭周末 gap 盲区。
    # alpha=0.005 肉眼不可见但非零，避免 Plotly 优化跳过 hover 命中。
    # JS 通过日期匹配（而非 pointIndex）定位数据，确保 gap 区域命中最近交易日。
    # 原生 TIP 通过透明 hoverlabel + hovermode='x' 设为不可见。
    fig.add_trace(go.Scatter(
        x=bars['trade_date'],
        y=c,
        mode='lines+markers',
        line=dict(color='rgba(0,0,0,0.005)', width=3),
        marker=dict(size=30, color='rgba(0,0,0,0.005)', line=dict(width=0)),
        name='K线',
        showlegend=False,
        customdata=hover_cd,
        hoverinfo='text',
        hovertext=[''] * n,
    ), row=1, col=1)

    # 影线：合并为单条 trace（NaN 分隔），替代 n 条独立 trace 避免渲染性能问题
    x_wick = []
    y_wick = []
    for i in range(n):
        x_wick.extend([bars['trade_date'].iloc[i], bars['trade_date'].iloc[i], None])
        y_wick.extend([l.iloc[i], h.iloc[i], None])
    fig.add_trace(go.Scatter(
        x=x_wick, y=y_wick,
        mode='lines',
        line=dict(color=KC['wick'], width=1),
        showlegend=False,
        hoverinfo='skip',
        connectgaps=False,
    ), row=1, col=1)

    # 实体：阳线
    up_mask = c >= o
    if up_mask.any():
        up_idx = bars.index[up_mask]
        fig.add_trace(go.Bar(
            x=bars.loc[up_idx, 'trade_date'],
            y=(c[up_idx] - o[up_idx]).abs().clip(lower=0.001),
            base=o[up_idx],
            marker_color=KC['up'],
            marker_line_color=KC['up'],
            marker_line_width=0,
            width=bar_width_ms,
            name='阳线',
            showlegend=False,
            hoverinfo='skip',
        ), row=1, col=1)

    # 实体：阴线
    down_mask = ~up_mask
    if down_mask.any():
        dn_idx = bars.index[down_mask]
        fig.add_trace(go.Bar(
            x=bars.loc[dn_idx, 'trade_date'],
            y=(o[dn_idx] - c[dn_idx]).abs().clip(lower=0.001),
            base=c[dn_idx],
            marker_color=KC['down'],
            marker_line_color=KC['down'],
            marker_line_width=0,
            width=bar_width_ms,
            name='阴线',
            showlegend=False,
            hoverinfo='skip',
        ), row=1, col=1)

    # ── 均线 ──
    if show_ma:
        for period, color, name in [(5, KC['ma5'], 'MA5'), (10, KC['ma10'], 'MA10'), (20, KC['ma20'], 'MA20')]:
            if n >= period:
                ma = c.rolling(period).mean()
                fig.add_trace(go.Scatter(
                    x=bars['trade_date'], y=ma,
                    mode='lines', name=name,
                    line=dict(width=1, color=color),
                    hoverinfo='skip',
                ), row=1, col=1)

    # ── 买卖点标记 ──
    buy_pts = [p for p in trade_points if p.action == "BUY"]
    sell_pts = [p for p in trade_points if p.action == "SELL"]

    buy_marker_y = []
    sell_marker_y = []

    if buy_pts:
        buy_dates = pd.to_datetime([p.date for p in buy_pts])
        buy_prices = [float(getattr(p, 'exec_price', 0) or p.price) for p in buy_pts]
        buy_marker_y = [
            buy_lane_y
            for p, price in zip(buy_pts, buy_prices)
        ]
        # 只保留透明散点做 hover/legend；可见 B 图标强制用 annotation 绘制，避免 Plotly text 被吞
        buy_hover = go.Scatter(
            x=buy_dates,
            y=buy_marker_y,
            mode='markers',
            marker=dict(symbol='circle', size=24, color='rgba(240,185,11,0.01)',
                        line=dict(width=0, color='rgba(0,0,0,0)')),
            name='买入B',
            cliponaxis=False,
            customdata=[[p.reason, price] for p, price in zip(buy_pts, buy_prices)],
        )
        _hover_participate(buy_hover, len(buy_pts))
        fig.add_trace(buy_hover, row=1, col=1)
        # 可见图标与价格使用 Scatter 文本强制绘制，比 annotation 更不容易被 Streamlit 复用/吞掉
        fig.add_trace(go.Scatter(
            x=buy_dates, y=buy_marker_y,
            mode='markers+text',
            marker=dict(symbol='square', size=21, color=KC['buy'], line=dict(width=1, color='#111')),
            text=['B' for _ in buy_pts],
            textposition='middle center',
            textfont=dict(size=13, color='#111', family='monospace'),
            showlegend=False,
            hoverinfo='skip',
            cliponaxis=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=buy_dates,
            y=[buy_price_lane_y for _ in buy_marker_y],
            mode='text',
            text=[f"{price:.2f}" for price in buy_prices],
            textposition='bottom center',
            textfont=dict(size=11, color=KC['buy'], family='monospace'),
            showlegend=False,
            hoverinfo='skip',
            cliponaxis=False,
        ), row=1, col=1)
        # 参考图样式：用虚线连接买点标志与对应K线最低点，明确是哪根柱
        for p, x, y_marker in zip(buy_pts, buy_dates, buy_marker_y):
            k_date = pd.Timestamp(p.date).date()
            y_kline = low_by_date.get(k_date)
            if y_kline is not None:
                fig.add_shape(
                    type='line',
                    x0=x, x1=x,
                    y0=y_marker + marker_gap * 0.10,
                    y1=y_kline,
                    line=dict(color=KC['buy'], width=1.0, dash='dot'),
                    opacity=0.75,
                    row=1, col=1,
                )

    if sell_pts:
        sell_dates = pd.to_datetime([p.date for p in sell_pts])
        sell_prices = [float(getattr(p, 'exec_price', 0) or p.price) for p in sell_pts]
        sell_marker_y = [
            sell_lane_y
            for p, price in zip(sell_pts, sell_prices)
        ]
        sell_hover = go.Scatter(
            x=sell_dates,
            y=sell_marker_y,
            mode='markers',
            marker=dict(symbol='circle', size=24, color='rgba(171,71,188,0.01)',
                        line=dict(width=0, color='rgba(0,0,0,0)')),
            name='卖出S',
            cliponaxis=False,
            customdata=[[p.reason, price] for p, price in zip(sell_pts, sell_prices)],
        )
        _hover_participate(sell_hover, len(sell_pts))
        fig.add_trace(sell_hover, row=1, col=1)
        fig.add_trace(go.Scatter(
            x=sell_dates,
            y=[sell_price_lane_y for _ in sell_marker_y],
            mode='text',
            text=[f"{price:.2f}" for price in sell_prices],
            textposition='top center',
            textfont=dict(size=11, color=KC['sell'], family='monospace'),
            showlegend=False,
            hoverinfo='skip',
            cliponaxis=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=sell_dates, y=sell_marker_y,
            mode='markers+text',
            marker=dict(symbol='square', size=21, color=KC['sell'], line=dict(width=1, color='#111')),
            text=['S' for _ in sell_pts],
            textposition='middle center',
            textfont=dict(size=13, color='#fff', family='monospace'),
            showlegend=False,
            hoverinfo='skip',
            cliponaxis=False,
        ), row=1, col=1)
        # 参考图样式：用虚线连接卖点标志与对应K线最高点
        for p, x, y_marker in zip(sell_pts, sell_dates, sell_marker_y):
            k_date = pd.Timestamp(p.date).date()
            y_kline = high_by_date.get(k_date)
            if y_kline is not None:
                fig.add_shape(
                    type='line',
                    x0=x, x1=x,
                    y0=y_kline,
                    y1=y_marker - marker_gap * 0.10,
                    line=dict(color=KC['sell'], width=1.0, dash='dot'),
                    opacity=0.75,
                    row=1, col=1,
                )

    # ── 成交量 ──
    current_row = 2
    if show_volume:
        vol_colors = [KC['vol_up'] if ci >= oi else KC['vol_down']
                      for oi, ci in zip(o, c)]
        vol_trace = go.Bar(
            x=bars['trade_date'], y=v,
            marker_color=vol_colors,
            marker_line_color=vol_colors,
            marker_line_width=0,
            opacity=1.0,
            width=bar_width_ms,
            name='成交量',
            showlegend=False,
            customdata=np.column_stack([_format_cn_datetime(bars['trade_date']), _volume_unit_text(v)]),
        )
        _hover_participate(vol_trace, n)
        fig.add_trace(vol_trace, row=current_row, col=1)
        fig.update_yaxes(tickformat='.2s', row=current_row, col=1)
        _style_axis(fig, row=current_row)
        current_row += 1

    # ── RSI 副图 ──
    if show_rsi and indicator_row:
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, min_periods=14).mean()
        avg_loss = loss.ewm(com=13, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)

        fig.add_trace(go.Scatter(
            x=bars['trade_date'], y=rsi,
            mode='lines', name='RSI14',
            line=dict(width=1.2, color=KC['rsi']),
            customdata=np.column_stack([_format_cn_datetime(bars['trade_date']), rsi.fillna(0)]),
            hoverinfo='skip',
        ), row=indicator_row, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color=KC['up'], line_width=0.5,
                      row=indicator_row, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color=KC['down'], line_width=0.5,
                      row=indicator_row, col=1)
        fig.update_yaxes(range=[0, 100], dtick=25, row=indicator_row, col=1)
        _style_axis(fig, row=indicator_row)

    # ── KDJ 副图 ──
    if show_kdj and indicator_row:
        period = 9
        lowest_l = l.rolling(period).min()
        highest_h = h.rolling(period).max()
        rsv = (c - lowest_l) / (highest_h - lowest_l).replace(0, np.nan) * 100
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d

        for vals, color, name in [(k, KC['kdj_k'], 'K'), (d, KC['kdj_d'], 'D'), (j, KC['kdj_j'], 'J')]:
            t = go.Scatter(
                x=bars['trade_date'], y=vals,
                mode='lines', name=name,
                line=dict(width=1.2, color=color),
                customdata=np.column_stack([_format_cn_datetime(bars['trade_date']), vals.fillna(0)]),
            )
            _hover_participate(t, n)
            fig.add_trace(t, row=indicator_row, col=1)
        fig.add_hline(y=80, line_dash="dot", line_color=KC['up'], line_width=0.5,
                      row=indicator_row, col=1)
        fig.add_hline(y=20, line_dash="dot", line_color=KC['down'], line_width=0.5,
                      row=indicator_row, col=1)
        # 固定 KDJ 显示区间，避免 x 轴 zoom 后 y 轴 autorange 把 J 线截断。
        fig.update_yaxes(range=[-20, 120], dtick=20, row=indicator_row, col=1)
        _style_axis(fig, row=indicator_row)

    # ── MACD 副图 ──
    if show_macd and macd_row:
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        hist = (dif - dea) * 2
        hist_colors = [KC['macd_hist_up'] if val >= 0 else KC['macd_hist_down'] for val in hist.fillna(0)]

        macd_cd = np.column_stack([
            _format_cn_datetime(bars['trade_date']),
            dif.fillna(0),
            dea.fillna(0),
            hist.fillna(0),
        ])
        macd_bar = go.Bar(
            x=bars['trade_date'], y=hist,
            marker_color=hist_colors,
            marker_line_width=0,
            width=bar_width_ms,
            name='MACD柱',
            customdata=macd_cd,
        )
        _hover_participate(macd_bar, n)
        fig.add_trace(macd_bar, row=macd_row, col=1)
        dif_t = go.Scatter(
            x=bars['trade_date'], y=dif,
            mode='lines', name='DIF',
            line=dict(width=1.1, color=KC['macd']),
            customdata=macd_cd,
        )
        _hover_participate(dif_t, n)
        fig.add_trace(dif_t, row=macd_row, col=1)
        dea_t = go.Scatter(
            x=bars['trade_date'], y=dea,
            mode='lines', name='DEA',
            line=dict(width=1.1, color=KC['macd_signal']),
            customdata=macd_cd,
        )
        _hover_participate(dea_t, n)
        fig.add_trace(dea_t, row=macd_row, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color=KC['grid'], line_width=0.6,
                      row=macd_row, col=1)
        fig.update_yaxes(tickformat='.3f', row=macd_row, col=1)
        _style_axis(fig, row=macd_row)

    # ── 统一样式 ──
    fig.update_layout(
        height=max(height, 760 if show_macd and show_volume and (show_kdj or show_rsi) else height),
        template="plotly_white",
        paper_bgcolor=KC['bg'],
        plot_bgcolor=KC['plot_bg'],
        font=dict(family="monospace", color=KC['text'], size=11),
        title=dict(text=symbol, font=dict(size=13, color=KC['text']), x=0.01, y=0.98),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.01,
            font=dict(size=9, family="monospace"),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=55, r=15, t=30, b=25),
        hovermode="x unified",
        hoverdistance=-1,
        spikedistance=-1,
        hoverlabel=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)',
            font=dict(size=1, color='rgba(0,0,0,0)'),
        ),
        bargap=0.15,
    )
    _style_axis(fig, row=1)
    # 显式 date 轴：go.Bar 在 shared_xaxes 下默认走 categorical（index 对齐），
    # 会导致 K 线柱体与均线/日期错位。强制 type='date' 修复。
    fig.update_xaxes(type='date', showspikes=False)
    fig.update_xaxes(showticklabels=True, row=rows, col=1)
    for axis_name in [name for name in fig.layout if str(name).startswith('xaxis')]:
        axis = fig.layout[axis_name]
        axis.rangeslider = dict(visible=False)

    # Y 轴范围：基于 high/low 加 padding，确保 K 线不压扁
    if n > 0:
        # 留出买卖标签空间，避免标志/价格被边缘遮挡
        buy_label_y = [buy_price_lane_y for _ in buy_marker_y]
        sell_label_y = [sell_price_lane_y for _ in sell_marker_y]
        y_min = min([price_min] + buy_marker_y + buy_label_y) if buy_marker_y else price_min
        y_max = max([price_max] + sell_marker_y + sell_label_y) if sell_marker_y else price_max
        padding = max(price_range * 0.22, label_gap * 1.2)
        fig.update_yaxes(
            range=[y_min - padding, y_max + padding],
            row=1, col=1,
        )

    # ── 当前价格标签（最右侧） ──
    if n > 0:
        last_price = float(c.iloc[-1])
        last_color = KC['up'] if last_price >= float(o.iloc[-1]) else KC['down']
        fig.add_annotation(
            x=bars['trade_date'].iloc[-1], y=last_price,
            text=f" {last_price:.2f} ",
            showarrow=False,
            font=dict(size=11, color='#fff', family="monospace"),
            bgcolor=last_color,
            bordercolor=last_color,
            borderpad=3,
            xanchor='left',
            row=1, col=1,
        )

    return fig


def plot_equity_curve(
    equity_curve: dict,
    benchmark: Optional[dict] = None,
    title: str = "权益曲线",
) -> go.Figure:
    """绘制权益曲线"""
    fig = go.Figure()

    if equity_curve and len(equity_curve) > 1:
        dates = list(equity_curve.keys())
        values = list(equity_curve.values())

        # 兼容两种格式：绝对权益值（大数）或日收益率（小数）
        if values and abs(values[0]) > 100:
            # 绝对权益值，转为收益率百分比
            initial = values[0]
            cum_returns = [(v / initial - 1) * 100 for v in values]
        else:
            # 日收益率，逐日连乘
            cum = 1.0
            cum_returns = []
            for v in values:
                cum *= (1 + v)
                cum_returns.append((cum - 1) * 100)

        fig.add_trace(go.Scatter(
            x=pd.to_datetime(dates), y=cum_returns,
            mode='lines', name='策略',
            line=dict(width=2, color=KC['ma10']),
            fill='tozeroy',
            fillcolor='rgba(30,58,138,0.08)',
        ))

    if benchmark and len(benchmark) > 1:
        b_dates = list(benchmark.keys())
        b_values = list(benchmark.values())
        cum = 1.0
        b_cum = []
        for v in b_values:
            cum *= (1 + v)
            b_cum.append((cum - 1) * 100)
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(b_dates), y=b_cum,
            mode='lines', name='基准',
            line=dict(width=1, color=KC['text_dim'], dash='dash'),
        ))

    fig.update_layout(
        title=title,
        height=350,
        template="plotly_white",
        paper_bgcolor=KC['bg'],
        plot_bgcolor=KC['plot_bg'],
        font=dict(family="monospace", color=KC['text'], size=11),
        yaxis_title="累计收益 (%)",
        margin=dict(l=55, r=15, t=30, b=25),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.01,
            font=dict(size=9, family="monospace"),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=KC['tooltip_bg'],
            bordercolor=KC['tooltip_border'],
            font=dict(size=10, family="monospace", color=KC['tooltip_text']),
        ),
    )
    fig.update_yaxes(tickformat='.1f')
    _style_axis(fig)
    fig.add_hline(y=0, line_dash="dash", line_color=KC['grid'])

    return fig


# ══════════════════════════════════════════════════════════════════
# JS 十字光标跨子图同步渲染
# ══════════════════════════════════════════════════════════════════

CROSSHAIR_JS = r"""
<script>
(function() {
    var pollCount = 0;
    function initCrosshair() {
        pollCount++;
        var gd = document.querySelector('.js-plotly-plot');
        if (!gd) {
            if (pollCount < 50) setTimeout(initCrosshair, 200);
            return;
        }
        try {
            if (!gd._fullLayout || !gd._fullLayout.xaxis) {
                if (pollCount < 50) setTimeout(initCrosshair, 200);
                return;
            }
        } catch(e) {
            if (pollCount < 50) setTimeout(initCrosshair, 200);
            return;
        }

        // ── 十字光标竖线 ──
        var vLine = document.createElement('div');
        vLine.style.cssText = 'position:absolute;width:1px;background:rgba(138,127,109,0.55);pointer-events:none;display:none;z-index:999;top:0;';
        gd.appendChild(vLine);

        // ── 十字光标横线（K线子图） ──
        var hLine = document.createElement('div');
        hLine.style.cssText = 'position:absolute;height:1px;background:rgba(138,127,109,0.55);pointer-events:none;display:none;z-index:999;left:0;';
        gd.appendChild(hLine);

        // ── 顶部 OHLCV 数据框 ──
        var dataBox = document.createElement('div');
        dataBox.style.cssText =
            'position:absolute;top:0;left:50%;transform:translateX(-50%);' +
            'background:rgba(237,231,224,0.96);border:1px solid #c2b39f;border-radius:4px;' +
            'padding:4px 10px;font-family:monospace;font-size:11px;color:#2f2a22;' +
            'pointer-events:none;display:none;z-index:1000;white-space:nowrap;' +
            'box-shadow:0 1px 3px rgba(0,0,0,0.08);';
        gd.appendChild(dataBox);

        // ── 成交量子图数据框 ──
        var volBox = document.createElement('div');
        volBox.style.cssText =
            'position:absolute;left:50%;transform:translate(-50%,-50%);' +
            'background:rgba(237,231,224,0.90);border:1px solid #c2b39f;border-radius:3px;' +
            'padding:2px 7px;font-family:monospace;font-size:10px;color:#5f5648;' +
            'pointer-events:none;display:none;z-index:1000;white-space:nowrap;';
        gd.appendChild(volBox);

        // ── KDJ 子图数据框 ──
        var kdjBox = document.createElement('div');
        kdjBox.style.cssText =
            'position:absolute;left:50%;transform:translate(-50%,-50%);' +
            'background:rgba(237,231,224,0.90);border:1px solid #c2b39f;border-radius:3px;' +
            'padding:2px 7px;font-family:monospace;font-size:10px;color:#5f5648;' +
            'pointer-events:none;display:none;z-index:1000;white-space:nowrap;';
        gd.appendChild(kdjBox);

        // ── MACD 子图数据框 ──
        var macdBox = document.createElement('div');
        macdBox.style.cssText =
            'position:absolute;left:50%;transform:translate(-50%,-50%);' +
            'background:rgba(237,231,224,0.90);border:1px solid #c2b39f;border-radius:3px;' +
            'padding:2px 7px;font-family:monospace;font-size:10px;color:#5f5648;' +
            'pointer-events:none;display:none;z-index:1000;white-space:nowrap;';
        gd.appendChild(macdBox);

        // ── X 轴日期标签 ──
        var xLabel = document.createElement('div');
        xLabel.style.cssText =
            'position:absolute;bottom:0;transform:translateX(-50%);' +
            'background:rgba(138,127,109,0.90);color:#fff;font-family:monospace;font-size:9px;' +
            'padding:2px 6px;border-radius:2px;pointer-events:none;display:none;z-index:1000;white-space:nowrap;';
        gd.appendChild(xLabel);

        // ── Y 轴价格标签 ──
        var yLabel = document.createElement('div');
        yLabel.style.cssText =
            'position:absolute;right:2px;transform:translateY(-50%);' +
            'background:rgba(138,127,109,0.90);color:#fff;font-family:monospace;font-size:9px;' +
            'padding:2px 6px;border-radius:2px;pointer-events:none;display:none;z-index:1000;white-space:nowrap;';
        gd.appendChild(yLabel);

        function fmtNum(v, dec) {
            if (v == null || isNaN(v)) return '--';
            return parseFloat(v).toFixed(dec || 2);
        }

        gd.on('plotly_hover', function(evt) {
            if (!evt || !evt.points || !evt.points.length) return;
            var xval = evt.points[0].x;
            var fullLayout = gd._fullLayout;
            var xaxis = fullLayout.xaxis;
            if (!xaxis || typeof xaxis.d2p !== 'function') return;

            var margin = fullLayout.margin || {t:30, b:25, l:55, r:15};
            var xPixel = xaxis.d2p(xval);
            var xOffset = (typeof xaxis._offset === 'number') ? xaxis._offset : 0;
            var containerH = gd.offsetHeight || gd.clientHeight || 0;
            var containerW = gd.offsetWidth || gd.clientWidth || 0;

            // 竖线
            vLine.style.left = (xOffset + xPixel) + 'px';
            vLine.style.top = margin.t + 'px';
            vLine.style.height = (containerH - margin.t - margin.b) + 'px';
            vLine.style.display = 'block';

            // X 轴日期标签
            var xDate = new Date(xval);
            var xDateStr = xDate.getFullYear() + '-' +
                ('0' + (xDate.getMonth() + 1)).slice(-2) + '-' +
                ('0' + xDate.getDate()).slice(-2);
            xLabel.textContent = xDateStr;
            xLabel.style.left = (xOffset + xPixel) + 'px';
            xLabel.style.bottom = '0px';
            xLabel.style.display = 'block';

            // 横线 + Y轴标签（K线子图）
            var yaxis = fullLayout.yaxis;
            var yVal = null;
            for (var i = 0; i < evt.points.length; i++) {
                if (evt.points[i].data.name === 'K线') {
                    yVal = evt.points[i].y; break;
                }
            }
            if (yVal == null && evt.points[0].y !== undefined) yVal = evt.points[0].y;
            if (yVal != null && yaxis && typeof yaxis.d2p === 'function') {
                var yPixel = yaxis.d2p(yVal);
                var yOffset = (typeof yaxis._offset === 'number') ? yaxis._offset : 0;
                hLine.style.top = (yOffset + yPixel) + 'px';
                hLine.style.left = margin.l + 'px';
                hLine.style.width = (containerW - margin.l - margin.r) + 'px';
                hLine.style.display = 'block';
                yLabel.textContent = fmtNum(yVal);
                yLabel.style.top = (yOffset + yPixel) + 'px';
                yLabel.style.display = 'block';
            }

            // ── 顶部 OHLCV 数据框 ──
            var cd = null, pointIdx = -1;
            for (var i = 0; i < evt.points.length; i++) {
                var pt = evt.points[i];
                if (pt.data && pt.data.name === 'K线' && pt.data.customdata && pt.data.customdata.length > pt.pointIndex) {
                    pointIdx = pt.pointIndex; var row = pt.data.customdata[pointIdx];
                    if (row && row.length >= 6) { cd = row; break; }
                }
            }
            if (cd) {
                var chgColor = parseFloat(cd[5]) >= 0 ? '#ef5350' : '#26a69a';
                var chgSign = parseFloat(cd[5]) >= 0 ? '+' : '';
                dataBox.innerHTML =
                    '<span style="color:#5f5648;">' + cd[0] + '</span>' +
                    '&nbsp; O <b>' + fmtNum(cd[1]) + '</b>' +
                    '&nbsp; H <b style="color:#ef5350;">' + fmtNum(cd[2]) + '</b>' +
                    '&nbsp; L <b style="color:#26a69a;">' + fmtNum(cd[3]) + '</b>' +
                    '&nbsp; C <b>' + fmtNum(cd[4]) + '</b>' +
                    '&nbsp; <b style="color:' + chgColor + ';">' + chgSign + fmtNum(cd[5]) + '%</b>' +
                    '&nbsp; <span style="color:#8a7f6d;font-size:10px;">' + (cd[6] || '') + '</span>';
                dataBox.style.display = 'block';
            }

            // ── 子图数据框：从 gd.data 数组取值（trace hoverinfo='skip' 不会出现在 evt.points 中）──
            var volY = null, kY = null, dY = null, jY = null;
            var difY = null, deaY = null, macdY = null;
            var buyInfo = null, sellInfo = null;
            if (pointIdx >= 0) {
                for (var i = 0; i < gd.data.length; i++) {
                    var d = gd.data[i], nm = d.name || '';
                    if (nm === '成交量' && d.y && pointIdx < d.y.length) volY = d.y[pointIdx];
                    if (nm === 'K' && d.y && pointIdx < d.y.length) kY = d.y[pointIdx];
                    if (nm === 'D' && d.y && pointIdx < d.y.length) dY = d.y[pointIdx];
                    if (nm === 'J' && d.y && pointIdx < d.y.length) jY = d.y[pointIdx];
                    if (nm === 'DIF' && d.y && pointIdx < d.y.length) difY = d.y[pointIdx];
                    if (nm === 'DEA' && d.y && pointIdx < d.y.length) deaY = d.y[pointIdx];
                    if (nm === 'MACD柱' && d.y && pointIdx < d.y.length) macdY = d.y[pointIdx];
                    if (nm === '买入B' && d.customdata && pointIdx < d.customdata.length) {
                        var cd2 = d.customdata[pointIdx];
                        if (cd2 && cd2.length >= 2) buyInfo = {reason: cd2[0], price: cd2[1]};
                    }
                    if (nm === '卖出S' && d.customdata && pointIdx < d.customdata.length) {
                        var cd3 = d.customdata[pointIdx];
                        if (cd3 && cd3.length >= 2) sellInfo = {reason: cd3[0], price: cd3[1]};
                    }
                }
            }

            // ── 顶部数据框追加买卖点信息 ──
            if (buyInfo) {
                dataBox.innerHTML += '<br><span style="color:#d4a017;">▶ 买入 @ ' + fmtNum(buyInfo.price) + '</span>' +
                    ' <span style="color:#8a7f6d;font-size:10px;">' + (buyInfo.reason || '') + '</span>';
            }
            if (sellInfo) {
                dataBox.innerHTML += '<br><span style="color:#ab47bc;">▶ 卖出 @ ' + fmtNum(sellInfo.price) + '</span>' +
                    ' <span style="color:#8a7f6d;font-size:10px;">' + (sellInfo.reason || '') + '</span>';
            }

            // 成交量框（以子图边界为中心，translateY(-50%) 避免遮挡）
            if (fullLayout.yaxis2 && typeof fullLayout.yaxis2._offset === 'number') {
                volBox.style.top = fullLayout.yaxis2._offset + 'px';
                if (volY != null) {
                    volBox.innerHTML = '<b>VOL</b> ' + fmtNum(volY, 0) + ' 手';
                    volBox.style.display = 'block';
                }
            }

            // KDJ 框（以子图边界为中心）
            if (fullLayout.yaxis3 && typeof fullLayout.yaxis3._offset === 'number') {
                kdjBox.style.top = fullLayout.yaxis3._offset + 'px';
                if (kY != null || dY != null || jY != null) {
                    kdjBox.innerHTML =
                        '<span style="color:#f0b90b;">K ' + fmtNum(kY, 2) + '</span>' +
                        '&nbsp; <span style="color:#1e88e5;">D ' + fmtNum(dY, 2) + '</span>' +
                        '&nbsp; <span style="color:#ab47bc;">J ' + fmtNum(jY, 2) + '</span>';
                    kdjBox.style.display = 'block';
                }
            }

            // MACD 框（以子图边界为中心）
            if (fullLayout.yaxis4 && typeof fullLayout.yaxis4._offset === 'number') {
                macdBox.style.top = fullLayout.yaxis4._offset + 'px';
                if (difY != null || deaY != null || macdY != null) {
                    var histColor = (macdY != null && parseFloat(macdY) >= 0) ? '#ef5350' : '#26a69a';
                    macdBox.innerHTML =
                        '<span style="color:#5f5648;">DIF ' + fmtNum(difY, 4) + '</span>' +
                        '&nbsp; <span style="color:#1e88e5;">DEA ' + fmtNum(deaY, 4) + '</span>' +
                        '&nbsp; <span style="color:' + histColor + ';">MACD ' + fmtNum(macdY, 4) + '</span>';
                    macdBox.style.display = 'block';
                }
            }
        });

        gd.on('plotly_unhover', function() {
            vLine.style.display = 'none';
            hLine.style.display = 'none';
            dataBox.style.display = 'none';
            volBox.style.display = 'none';
            kdjBox.style.display = 'none';
            macdBox.style.display = 'none';
            xLabel.style.display = 'none';
            yLabel.style.display = 'none';
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() { setTimeout(initCrosshair, 300); });
    } else {
        setTimeout(initCrosshair, 100);
    }
})();
</script>
"""


# ── 十字光标 JS（通过 st.markdown 直接注入主页面 DOM）──
_CROSSHAIR_INJECT = r"""
<style>
.js-plotly-plot .hoverlayer { display: none !important; }
</style>
<script>
(function() {
    var attempts = 0;
    function init() {
        attempts++;
        // 查找所有尚未注入十字光标的 Plotly 图表
        var allCharts = document.querySelectorAll('.js-plotly-plot');
        var found = false;
        for (var c = 0; c < allCharts.length; c++) {
            var gd = allCharts[c];
            if (gd.__crosshairInited) continue;
            try { if (!gd._fullLayout || !gd._fullLayout.xaxis) continue; }
            catch(e) { continue; }
            gd.__crosshairInited = true;
            found = true;
            _initChart(gd);
        }
        if (attempts < 100 && !found) setTimeout(init, 200);
    }

    function _initChart(gd) {

        var D = document;
        var margin = gd._fullLayout.margin || {t:30, b:25, l:55, r:15};

        function fmtNum(v, dec) { if (v == null || isNaN(v)) return '--'; return parseFloat(v).toFixed(dec || 2); }

        function makeBox() {
            var b = D.createElement('div');
            b.style.cssText = 'position:absolute;left:50%;transform:translate(-50%,-50%);background:rgba(237,231,224,0.90);border:1px solid #c2b39f;border-radius:3px;padding:2px 7px;font-family:monospace;font-size:10px;color:#5f5648;pointer-events:none;display:none;z-index:1000;white-space:nowrap;';
            gd.appendChild(b); return b;
        }

        var vLine = D.createElement('div');
        vLine.style.cssText = 'position:absolute;width:1px;background:rgba(138,127,109,0.55);pointer-events:none;display:none;z-index:999;top:0;';
        gd.appendChild(vLine);
        var hLine = D.createElement('div');
        hLine.style.cssText = 'position:absolute;height:1px;background:rgba(138,127,109,0.55);pointer-events:none;display:none;z-index:999;left:0;';
        gd.appendChild(hLine);

        var dataBox = D.createElement('div');
        dataBox.style.cssText = 'position:absolute;top:0;left:50%;transform:translateX(-50%);background:rgba(237,231,224,0.96);border:1px solid #c2b39f;border-radius:4px;padding:4px 10px;font-family:monospace;font-size:11px;color:#2f2a22;pointer-events:none;display:none;z-index:1000;white-space:nowrap;box-shadow:0 1px 3px rgba(0,0,0,0.08);';
        gd.appendChild(dataBox);
        var volBox = makeBox(), kdjBox = makeBox(), macdBox = makeBox();

        var xLabel = D.createElement('div');
        xLabel.style.cssText = 'position:absolute;bottom:0;transform:translateX(-50%);background:rgba(138,127,109,0.90);color:#fff;font-family:monospace;font-size:9px;padding:2px 6px;border-radius:2px;pointer-events:none;display:none;z-index:1000;white-space:nowrap;';
        gd.appendChild(xLabel);
        var yLabel = D.createElement('div');
        yLabel.style.cssText = 'position:absolute;right:2px;transform:translateY(-50%);background:rgba(138,127,109,0.90);color:#fff;font-family:monospace;font-size:9px;padding:2px 6px;border-radius:2px;pointer-events:none;display:none;z-index:1000;white-space:nowrap;';
        gd.appendChild(yLabel);

        function onHover(evt) {
            if (!evt || !evt.points || !evt.points.length) return;
            var fl = gd._fullLayout;
            var xaxis = fl.xaxis;
            if (!xaxis || typeof xaxis.d2p !== 'function') return;
            margin = fl.margin || margin;

            var cH = gd.offsetHeight || gd.clientHeight || 0;
            var cW = gd.offsetWidth || gd.clientWidth || 0;

            // ── 日期匹配：在 K线 trace 的 x 数组中找最近的交易日索引 ──
            // 不依赖 Plotly 的 pointIndex（mode='lines' 时不可靠）
            var klineTrace = null, klineIdx = -1;
            for (var i = 0; i < gd.data.length; i++) {
                if (gd.data[i].name === 'K线' && gd.data[i].x) { klineTrace = gd.data[i]; break; }
            }
            if (klineTrace && klineTrace.x.length > 0) {
                var hoverX = evt.points[0].x;
                var bestDist = Infinity;
                for (var j = 0; j < klineTrace.x.length; j++) {
                    var dist = Math.abs(new Date(klineTrace.x[j]) - new Date(hoverX));
                    if (dist < bestDist) { bestDist = dist; klineIdx = j; }
                }
            }

            // ── 锁定到最近数据点的 x 值，确保竖线/日期/数据三位一体对齐 ──
            var xval;
            if (klineIdx >= 0 && klineTrace) {
                xval = klineTrace.x[klineIdx];
            } else {
                xval = evt.points[0].x;
            }

            var xPixel = xaxis.d2p(xval);
            var xOff = (typeof xaxis._offset === 'number') ? xaxis._offset : 0;

            vLine.style.left = (xOff + xPixel) + 'px';
            vLine.style.top = margin.t + 'px';
            vLine.style.height = (cH - margin.t - margin.b) + 'px';
            vLine.style.display = 'block';

            // X 轴日期标签
            var xDate = new Date(xval);
            xLabel.textContent = xDate.getFullYear() + '-' + ('0'+(xDate.getMonth()+1)).slice(-2) + '-' + ('0'+xDate.getDate()).slice(-2);
            xLabel.style.left = (xOff + xPixel) + 'px';
            xLabel.style.display = 'block';

            // 横线 + Y轴标签（K线子图）
            var yaxis = fl.yaxis, yVal = null;
            if (klineIdx >= 0 && klineTrace && klineTrace.y && klineIdx < klineTrace.y.length) {
                yVal = klineTrace.y[klineIdx];
            }
            if (yVal == null && evt.points[0].y !== undefined) yVal = evt.points[0].y;
            if (yVal != null && yaxis && typeof yaxis.d2p === 'function') {
                var yPx = yaxis.d2p(yVal), yOff = (typeof yaxis._offset === 'number') ? yaxis._offset : 0;
                hLine.style.top = (yOff + yPx) + 'px';
                hLine.style.left = margin.l + 'px';
                hLine.style.width = (cW - margin.l - margin.r) + 'px';
                hLine.style.display = 'block';
                yLabel.textContent = fmtNum(yVal);
                yLabel.style.top = (yOff + yPx) + 'px';
                yLabel.style.display = 'block';
            }

            // ── 用匹配到的索引从 gd.data 取值（date-safe）──
            var cd = null, volY = null, kY = null, dY = null, jY = null;
            var difY = null, deaY = null, macdY = null;
            var buyInfo = null, sellInfo = null;
            if (klineIdx >= 0) {
                if (klineTrace.customdata && klineIdx < klineTrace.customdata.length) {
                    var row = klineTrace.customdata[klineIdx];
                    if (row && row.length >= 6) cd = row;
                }
                for (var i = 0; i < gd.data.length; i++) {
                    var d = gd.data[i], nm = d.name || '', yarr = d.y, xarr = d.x;
                    if (nm === '成交量' && yarr && klineIdx < yarr.length) volY = yarr[klineIdx];
                    if (nm === 'K' && yarr && klineIdx < yarr.length) kY = yarr[klineIdx];
                    if (nm === 'D' && yarr && klineIdx < yarr.length) dY = yarr[klineIdx];
                    if (nm === 'J' && yarr && klineIdx < yarr.length) jY = yarr[klineIdx];
                    if (nm === 'DIF' && yarr && klineIdx < yarr.length) difY = yarr[klineIdx];
                    if (nm === 'DEA' && yarr && klineIdx < yarr.length) deaY = yarr[klineIdx];
                    if (nm === 'MACD柱' && yarr && klineIdx < yarr.length) macdY = yarr[klineIdx];
                    // 买卖点：匹配日期
                    if ((nm === '买入B' || nm === '卖出S') && d.customdata && xarr) {
                        for (var k = 0; k < xarr.length; k++) {
                            var xd = new Date(xarr[k]); var nd = new Date(klineTrace.x[klineIdx]);
                            if (Math.abs(xd - nd) < 3600000) {  // 同一天（1小时容差）
                                var cd2 = d.customdata[k];
                                if (cd2 && cd2.length >= 2) {
                                    if (nm === '买入B') buyInfo = {reason: cd2[0], price: cd2[1]};
                                    else sellInfo = {reason: cd2[0], price: cd2[1]};
                                }
                                break;
                            }
                        }
                    }
                }
            }
            if (cd) {
                var chgC = parseFloat(cd[5]) >= 0 ? '#ef5350' : '#26a69a';
                var chgS = parseFloat(cd[5]) >= 0 ? '+' : '';
                dataBox.innerHTML = '<span style="color:#5f5648;">' + cd[0] + '</span>&nbsp; O <b>' + fmtNum(cd[1]) + '</b>&nbsp; H <b style="color:#ef5350;">' + fmtNum(cd[2]) + '</b>&nbsp; L <b style="color:#26a69a;">' + fmtNum(cd[3]) + '</b>&nbsp; C <b>' + fmtNum(cd[4]) + '</b>&nbsp; <b style="color:' + chgC + ';">' + chgS + fmtNum(cd[5]) + '%</b>&nbsp; <span style="color:#8a7f6d;font-size:10px;">' + (cd[6] || '') + '</span>';
                if (buyInfo) dataBox.innerHTML += '<br><span style="color:#d4a017;">▶ 买入 @ ' + fmtNum(buyInfo.price) + '</span> <span style="color:#8a7f6d;font-size:10px;">' + (buyInfo.reason || '') + '</span>';
                if (sellInfo) dataBox.innerHTML += '<br><span style="color:#ab47bc;">▶ 卖出 @ ' + fmtNum(sellInfo.price) + '</span> <span style="color:#8a7f6d;font-size:10px;">' + (sellInfo.reason || '') + '</span>';
                dataBox.style.display = 'block';
            }

            // 副图数据框以子图边界为中心（translateY(-50%)），不遮挡线条和坐标轴
            if (fl.yaxis2 && typeof fl.yaxis2._offset === 'number') { volBox.style.top = fl.yaxis2._offset + 'px'; if (volY != null) { volBox.innerHTML = '<b>VOL</b> ' + fmtNum(volY, 0) + ' 手'; volBox.style.display = 'block'; } }
            if (fl.yaxis3 && typeof fl.yaxis3._offset === 'number') { kdjBox.style.top = fl.yaxis3._offset + 'px'; if (kY != null || dY != null || jY != null) { kdjBox.innerHTML = '<span style="color:#f0b90b;">K ' + fmtNum(kY,2) + '</span>&nbsp; <span style="color:#1e88e5;">D ' + fmtNum(dY,2) + '</span>&nbsp; <span style="color:#ab47bc;">J ' + fmtNum(jY,2) + '</span>'; kdjBox.style.display = 'block'; } }
            if (fl.yaxis4 && typeof fl.yaxis4._offset === 'number') { macdBox.style.top = fl.yaxis4._offset + 'px'; if (difY != null || deaY != null || macdY != null) { var hc = (macdY != null && parseFloat(macdY) >= 0) ? '#ef5350' : '#26a69a'; macdBox.innerHTML = '<span style="color:#5f5648;">DIF ' + fmtNum(difY,4) + '</span>&nbsp; <span style="color:#1e88e5;">DEA ' + fmtNum(deaY,4) + '</span>&nbsp; <span style="color:' + hc + ';">MACD ' + fmtNum(macdY,4) + '</span>'; macdBox.style.display = 'block'; } }
        }

        function onUnhover() {
            vLine.style.display = 'none'; hLine.style.display = 'none';
            dataBox.style.display = 'none'; volBox.style.display = 'none';
            kdjBox.style.display = 'none'; macdBox.style.display = 'none';
            xLabel.style.display = 'none'; yLabel.style.display = 'none';
        }

        gd.on('plotly_hover', onHover);
        gd.on('plotly_unhover', onUnhover);
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', function() { setTimeout(init, 300); });
    else setTimeout(init, 100);
})();
</script>
"""


def render_kline_chart(fig: go.Figure, key: str = "", height: int = 760):
    """
    渲染 K 线图（带 JS 十字光标跨子图同步）。

    使用 components.html + CDN Plotly.js 方案（浏览器首次加载后 CDN 缓存，后续极快）。
    K线影线已合并为单条 trace（NaN 分隔），trace 数从 48→19。
    """
    import streamlit.components.v1 as components

    fig_html = fig.to_html(
        include_plotlyjs='cdn',
        full_html=True,
        config={
            'responsive': True,
            'displayModeBar': True,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'displaylogo': False,
        }
    )

    # 在 </body> 前注入 JS
    fig_html = fig_html.replace('</body>', _CROSSHAIR_INJECT + '\n</body>')

    components.html(fig_html, height=height, scrolling=False)
