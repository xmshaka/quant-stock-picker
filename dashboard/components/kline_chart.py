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
    "crosshair":    "#c2b39f",
    "tooltip_bg":   "#f6f3ed",
    "tooltip_text": "#2f2a22",
    "tooltip_border": "#c2b39f",
}


def _style_axis(fig, row: int = None, col: int = None):
    """统一坐标轴样式（含十字光标线）"""
    kw = dict(row=row, col=col) if row else {}
    fig.update_xaxes(
        gridcolor=KC['grid'], showgrid=True, griddash='dot',
        zeroline=False, showline=False,
        tickfont=dict(size=9, color=KC['axis']),
        showspikes=True, spikemode='across+toaxis', spikethickness=1,
        spikecolor=KC['crosshair'], spikesnap='cursor',
        **kw,
    )
    fig.update_yaxes(
        gridcolor=KC['grid'], showgrid=True, griddash='dot',
        zeroline=False, showline=False, side='right',
        tickfont=dict(size=9, color=KC['axis']),
        tickformat='.2f',
        showspikes=True, spikemode='across+toaxis', spikethickness=1,
        spikecolor=KC['crosshair'], spikesnap='cursor',
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

    # ── K线蜡烛（用 Scatter 画影线 + Bar 画实体） ──
    bar_width_ms = 51_840_000  # 0.6 天

    # 只保留一个主图 TIP 锚点参与 hover；实体/影线/均线关闭 hover。
    # 否则 Plotly x unified 会把同一天的所有 trace 都塞进 TIP，造成大块重复空白。
    fig.add_trace(go.Scatter(
        x=bars['trade_date'],
        y=c,
        mode='markers',
        marker=dict(size=16, color='rgba(0,0,0,0)', line=dict(width=0)),
        name='K线',
        showlegend=False,
        customdata=hover_cd,
        hovertemplate=(
            '日期：%{customdata[0]}<br>'
            '开盘：%{customdata[1]:.2f}<br>'
            '最高：%{customdata[2]:.2f}<br>'
            '最低：%{customdata[3]:.2f}<br>'
            '收盘：%{customdata[4]:.2f}<br>'
            '涨跌幅：%{customdata[5]:+.2f}%<br>'
            '成交量：%{customdata[6]}'
            '<extra></extra>'
        ),
    ), row=1, col=1)

    # 影线：上下影线用一条竖线（从 low 到 high）
    for i in range(n):
        color = KC['up'] if c.iloc[i] >= o.iloc[i] else KC['down']
        fig.add_trace(go.Scatter(
            x=[bars['trade_date'].iloc[i], bars['trade_date'].iloc[i]],
            y=[l.iloc[i], h.iloc[i]],
            mode='lines',
            line=dict(color=color, width=1),
            showlegend=False,
            hoverinfo='skip',
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
        fig.add_trace(go.Scatter(
            x=buy_dates,
            y=buy_marker_y,
            mode='markers',
            marker=dict(symbol='circle', size=24, color='rgba(240,185,11,0.01)',
                        line=dict(width=0, color='rgba(0,0,0,0)')),
            name='买入B',
            cliponaxis=False,
            customdata=[[p.reason, price] for p, price in zip(buy_pts, buy_prices)],
            hoverinfo='skip',
        ), row=1, col=1)
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
        fig.add_trace(go.Scatter(
            x=sell_dates,
            y=sell_marker_y,
            mode='markers',
            marker=dict(symbol='circle', size=24, color='rgba(171,71,188,0.01)',
                        line=dict(width=0, color='rgba(0,0,0,0)')),
            name='卖出S',
            cliponaxis=False,
            customdata=[[p.reason, price] for p, price in zip(sell_pts, sell_prices)],
            hoverinfo='skip',
        ), row=1, col=1)
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
        fig.add_trace(go.Bar(
            x=bars['trade_date'], y=v,
            marker_color=vol_colors,
            marker_line_color=vol_colors,
            marker_line_width=0,
            opacity=1.0,
            width=bar_width_ms,
            name='成交量',
            showlegend=False,
            customdata=np.column_stack([_format_cn_datetime(bars['trade_date']), _volume_unit_text(v)]),
            hovertemplate='日期：%{customdata[0]}<br>成交量：%{customdata[1]}<extra></extra>',
        ), row=current_row, col=1)
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
            hovertemplate='RSI14：%{customdata[1]:.2f}<extra></extra>',
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
            fig.add_trace(go.Scatter(
                x=bars['trade_date'], y=vals,
                mode='lines', name=name,
                line=dict(width=1.2, color=color),
                customdata=np.column_stack([_format_cn_datetime(bars['trade_date']), vals.fillna(0)]),
                hovertemplate=f'{name}：%{{customdata[1]:.2f}}<extra></extra>',
            ), row=indicator_row, col=1)
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
        fig.add_trace(go.Bar(
            x=bars['trade_date'], y=hist,
            marker_color=hist_colors,
            marker_line_width=0,
            width=bar_width_ms,
            name='MACD柱',
            customdata=macd_cd,
            hovertemplate='MACD柱：%{customdata[3]:.4f}<extra></extra>',
        ), row=macd_row, col=1)
        fig.add_trace(go.Scatter(
            x=bars['trade_date'], y=dif,
            mode='lines', name='DIF',
            line=dict(width=1.1, color=KC['macd']),
            customdata=macd_cd,
            hovertemplate='DIF：%{customdata[1]:.4f}<extra></extra>',
        ), row=macd_row, col=1)
        fig.add_trace(go.Scatter(
            x=bars['trade_date'], y=dea,
            mode='lines', name='DEA',
            line=dict(width=1.1, color=KC['macd_signal']),
            customdata=macd_cd,
            hovertemplate='DEA：%{customdata[2]:.4f}<extra></extra>',
        ), row=macd_row, col=1)
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
        hoversubplots="axis",
        hoverdistance=50,
        spikedistance=-1,
        hoverlabel=dict(
            bgcolor=KC['tooltip_bg'],
            bordercolor=KC['tooltip_border'],
            font=dict(size=10, family="monospace", color=KC['tooltip_text']),
        ),
        bargap=0.15,
    )
    _style_axis(fig, row=1)
    fig.update_xaxes(matches='x', showspikes=True, spikemode='across+toaxis')
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

        var line = document.createElement('div');
        line.className = 'custom-crosshair-v';
        line.style.cssText = 'position:absolute;width:1px;background:rgba(136,136,136,0.5);pointer-events:none;display:none;z-index:999;top:0;';
        gd.appendChild(line);

        gd.on('plotly_hover', function(evt) {
            if (!evt || !evt.points || !evt.points.length) return;
            var xval = evt.points[0].x;
            var fullLayout = gd._fullLayout;
            var xaxis = fullLayout.xaxis;
            if (!xaxis || typeof xaxis.d2p !== 'function') return;

            var xPixel = xaxis.d2p(xval);
            var xOffset = (typeof xaxis._offset === 'number') ? xaxis._offset : 0;
            var margin = fullLayout.margin || {t:0, b:0};
            var containerH = gd.offsetHeight || gd.clientHeight || 0;

            line.style.left = (xOffset + xPixel) + 'px';
            line.style.top = margin.t + 'px';
            line.style.height = (containerH - margin.t - margin.b) + 'px';
            line.style.display = 'block';
        });

        gd.on('plotly_unhover', function() {
            line.style.display = 'none';
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


def render_kline_chart(fig: go.Figure, key: str = "", height: int = 760):
    """
    渲染 K 线图（带 JS 十字光标跨子图同步）。

    替代 st.plotly_chart()，注入 JavaScript 监听 plotly_hover 事件，
    在所有子图（K线/成交量/KDJ/MACD）上同步绘制十字光标竖线。

    原理：
    - Plotly 内置 spikemode='across' 仅作用于当前 hover 子图
    - JS 方案监听 plotly_hover，获取 x 值后在全图高度绘制竖线
    - 使用 xaxis.d2p() 将数据坐标转为像素坐标，精确定位
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
    fig_html = fig_html.replace('</body>', CROSSHAIR_JS + '\n</body>')

    components.html(fig_html, height=height, scrolling=False)
