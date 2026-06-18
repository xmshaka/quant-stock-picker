"""K线 + 买卖点可视化 — lightweight-charts (TradingView) 版。

为什么放弃 Plotly 自建十字光标：
1. go.Bar 和 go.Scatter 日期轴定位算法不同 → K柱与均线系统性错位
2. hovermode='x unified' + 离散 markers 领地冲突 → gap边缘K柱无法选中
3. mode='lines' 触发插值 → pointIndex 指向插值点而非原始数据
4. 自建JS十字光标需绕过Plotly hover引擎 → 复杂度指数增长

lightweight-charts 原生解决以上所有问题：
- 日期轴：原生金融时间轴，周末 gap 自动留白
- 十字光标：内置 crosshair，所有 pane 自动同步
- 点击/悬停：基于 canvas 坐标反算，100% 精准
- 渲染性能：WebGL canvas，远优于 Plotly SVG
"""
from __future__ import annotations

import json
import pandas as pd
import numpy as np
from typing import List, Optional

from signals.rules import TradePoint

# ══════════════════════════════════════════════════════════════════
# 配色（保持原来白配色一致）
# ══════════════════════════════════════════════════════════════════

KC = {
    "bg": "#ede7e0",
    "grid": "#d2c7b8",
    "text": "#5f5648",
    "up": "#ef5350",
    "down": "#26a69a",
    "ma5": "#f0b90b",
    "ma10": "#1e88e5",
    "ma20": "#ab47bc",
    "buy": "#f0b90b",
    "sell": "#ab47bc",
    "kdj_k": "#f0b90b",
    "kdj_d": "#1e88e5",
    "kdj_j": "#ab47bc",
    "macd": "#5f5648",
    "macd_signal": "#1e88e5",
    "macd_hist_up": "#ef5350",
    "macd_hist_down": "#26a69a",
    "crosshair": "rgba(138,127,109,0.55)",
}


def _ts(dt) -> int:
    """日期 → Unix 秒。"""
    return int(pd.Timestamp(dt).timestamp())


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
):
    """生成 lightweight-charts K线图 HTML。

    返回 dict，包含 html 和 height，供 render_kline_chart 使用。
    """
    # 空 DataFrame 无 trade_date 列时直接返回占位
    if 'trade_date' not in bars.columns or len(bars) == 0:
        return {"html": "<div style='color:#5f5648;text-align:center;padding:40px;'>暂无K线数据</div>", "height": 200}

    bars = bars.copy().sort_values('trade_date').reset_index(drop=True)
    bars['trade_date'] = pd.to_datetime(bars['trade_date'])
    n = len(bars)

    o = bars['open'].astype(float)
    h = bars['high'].astype(float)
    l = bars['low'].astype(float)
    c = bars['close'].astype(float)
    v = bars['volume'].astype(float) if 'volume' in bars.columns else pd.Series(0, index=bars.index)

    # ── OHLCV JSON ──
    ohlcv_data = []
    for i in range(n):
        ohlcv_data.append({
            "time": _ts(bars['trade_date'].iloc[i]),
            "open": float(o.iloc[i]),
            "high": float(h.iloc[i]),
            "low": float(l.iloc[i]),
            "close": float(c.iloc[i]),
            "volume": float(v.iloc[i]),
        })

    # ── 均线 ──
    ma_series = {}
    if show_ma:
        for period in [5, 10, 20]:
            if n >= period:
                ma = c.rolling(period).mean()
                pts = []
                for i in range(n):
                    val = ma.iloc[i]
                    if pd.notna(val):
                        pts.append({"time": _ts(bars['trade_date'].iloc[i]), "value": float(val)})
                if pts:
                    ma_series[str(period)] = pts

    # ── 成交量 ──
    vol_data = []
    for i in range(n):
        vc = KC["up"] if c.iloc[i] >= o.iloc[i] else KC["down"]
        vol_data.append({
            "time": _ts(bars['trade_date'].iloc[i]),
            "value": float(v.iloc[i]),
            "color": vc,
        })

    # ── KDJ ──
    kdj_data = {}
    if show_kdj and n >= 9:
        period = 9
        lowest_l = l.rolling(period).min()
        highest_h = h.rolling(period).max()
        rsv = (c - lowest_l) / (highest_h - lowest_l).replace(0, np.nan) * 100
        k_vals = rsv.ewm(com=2, adjust=False).mean()
        d_vals = k_vals.ewm(com=2, adjust=False).mean()
        j_vals = 3 * k_vals - 2 * d_vals
        kdj_data["k"] = [{"time": _ts(bars['trade_date'].iloc[i]), "value": float(k_vals.iloc[i])}
                          for i in range(n) if pd.notna(k_vals.iloc[i])]
        kdj_data["d"] = [{"time": _ts(bars['trade_date'].iloc[i]), "value": float(d_vals.iloc[i])}
                          for i in range(n) if pd.notna(d_vals.iloc[i])]
        kdj_data["j"] = [{"time": _ts(bars['trade_date'].iloc[i]), "value": float(j_vals.iloc[i])}
                          for i in range(n) if pd.notna(j_vals.iloc[i])]

    # ── MACD ──
    macd_data = {}
    if show_macd and n >= 26:
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        hist = (dif - dea) * 2
        macd_data["dif"] = [{"time": _ts(bars['trade_date'].iloc[i]), "value": float(dif.iloc[i])}
                            for i in range(n) if pd.notna(dif.iloc[i])]
        macd_data["dea"] = [{"time": _ts(bars['trade_date'].iloc[i]), "value": float(dea.iloc[i])}
                            for i in range(n) if pd.notna(dea.iloc[i])]
        macd_data["hist"] = []
        for i in range(n):
            hv = hist.iloc[i]
            if pd.notna(hv):
                hc = KC["macd_hist_up"] if hv >= 0 else KC["macd_hist_down"]
                macd_data["hist"].append({"time": _ts(bars['trade_date'].iloc[i]), "value": float(hv), "color": hc})

    # ── 买卖点 ──
    markers = []
    for p in trade_points:
        ts = _ts(p.date)
        price = float(getattr(p, 'exec_price', 0) or p.price)
        if p.action == "BUY":
            markers.append({
                "time": ts, "position": "belowBar",
                "color": KC["buy"], "shape": "arrowUp",
                "text": f"B {price:.2f}", "size": 2,
            })
        elif p.action == "SELL":
            markers.append({
                "time": ts, "position": "aboveBar",
                "color": KC["sell"], "shape": "arrowDown",
                "text": f"S {price:.2f}", "size": 2,
            })

    html = _build_lwc_html(symbol, ohlcv_data, ma_series, vol_data,
                           kdj_data, macd_data, markers, height)
    return {"html": html, "height": height}


# ══════════════════════════════════════════════════════════════════
# HTML 模板：用 __KEY__ 占位符替代 %s，避免 JS % 符号冲突
# ══════════════════════════════════════════════════════════════════

_LWC_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:__BG__;overflow:hidden;">
<div id="chart" style="width:100%;height:__HEIGHT__px;"></div>
<div id="tooltip" style="position:absolute;top:4px;left:50%;transform:translateX(-50%);background:rgba(237,231,224,0.96);border:1px solid #c2b39f;border-radius:4px;padding:4px 10px;font-family:monospace;font-size:11px;color:#2f2a22;pointer-events:none;display:none;z-index:1000;white-space:nowrap;box-shadow:0 1px 3px rgba(0,0,0,0.08);"></div>

<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<script>
(function() {
  var D = __DATA__;
  var container = document.getElementById('chart');
  var chart = LightweightCharts.createChart(container, {
    layout: { background: { color: '__BG__' }, textColor: '__TEXT__' },
    grid: { vertLines: { color: '__GRID__', style: 3 }, horzLines: { color: '__GRID__', style: 3 } },
    crosshair: {
      mode: 0,
      vertLine: { color: '__CROSSHAIR__', width: 1, style: 0, labelBackgroundColor: 'rgba(138,127,109,0.9)' },
      horzLine: { color: '__CROSSHAIR__', width: 1, style: 0, labelBackgroundColor: 'rgba(138,127,109,0.9)' },
    },
    rightPriceScale: { borderColor: '__GRID__' },
    timeScale: { borderColor: '__GRID__', timeVisible: true, secondsVisible: false },
  });

  // 主图 pane
  var klineSeries = chart.addCandlestickSeries({
    upColor: '__UP__', downColor: '__DOWN__',
    borderUpColor: '__UP__', borderDownColor: '__DOWN__',
    wickUpColor: '__UP__', wickDownColor: '__DOWN__',
  });
  klineSeries.setData(D.ohlcv);
  if (D.markers.length) klineSeries.setMarkers(D.markers);

  // 均线
  var maColors = D.maColors || {};
  Object.keys(D.ma).forEach(function(p) {
    if (p === 'maColors') return;
    var s = chart.addLineSeries({ color: maColors[p] || '#888', lineWidth: 1 });
    s.setData(D.ma[p]);
  });

  // 成交量 pane
  var volPane = chart.addPane({ height: __VOL_H__ });
  var volSeries = volPane.addHistogramSeries({ priceFormat: { type: 'volume' } });
  volSeries.setData(D.vol);

  // KDJ pane
  if (D.kdj.k && D.kdj.k.length) {
    var kdjPane = chart.addPane({ height: __KDJ_H__ });
    kdjPane.addLineSeries({ color: '__KDJ_K__', lineWidth: 1 }).setData(D.kdj.k);
    kdjPane.addLineSeries({ color: '__KDJ_D__', lineWidth: 1 }).setData(D.kdj.d);
    kdjPane.addLineSeries({ color: '__KDJ_J__', lineWidth: 1 }).setData(D.kdj.j);
    kdjPane.addLineSeries({ color: '__GRID__', lineWidth: 1, lineStyle: 2 }).setData(
      [{ time: D.ohlcv[0].time, value: 80 }, { time: D.ohlcv[D.ohlcv.length-1].time, value: 80 }]
    );
    kdjPane.addLineSeries({ color: '__GRID__', lineWidth: 1, lineStyle: 2 }).setData(
      [{ time: D.ohlcv[0].time, value: 20 }, { time: D.ohlcv[D.ohlcv.length-1].time, value: 20 }]
    );
  }

  // MACD pane
  if (D.macd.hist && D.macd.hist.length) {
    var macdPane = chart.addPane({ height: __MACD_H__ });
    macdPane.addHistogramSeries().setData(D.macd.hist);
    macdPane.addLineSeries({ color: '__MACD__', lineWidth: 1 }).setData(D.macd.dif);
    macdPane.addLineSeries({ color: '__MACD_SIGNAL__', lineWidth: 1 }).setData(D.macd.dea);
  }

  // 十字光标 tooltip
  function fmtNum(v, dec) {
    if (v == null || isNaN(v)) return '--';
    return parseFloat(v).toFixed(dec || 2);
  }
  function fmtVol(v) {
    if (v == null || isNaN(v)) return '--';
    return fmtNum(v / 10000, 0) + ' 万手';
  }

  var tooltip = document.getElementById('tooltip');
  chart.subscribeCrosshairMove(function(param) {
    if (!param.time || !param.point) { tooltip.style.display = 'none'; return; }
    var ts = param.time, idx = -1;
    for (var i = 0; i < D.ohlcv.length; i++) {
      if (D.ohlcv[i].time === ts) { idx = i; break; }
    }
    if (idx < 0) { tooltip.style.display = 'none'; return; }
    var d = D.ohlcv[idx];
    var prevClose = idx > 0 ? D.ohlcv[idx - 1].close : d.open;
    var chgPct = prevClose > 0 ? ((d.close - prevClose) / prevClose * 100) : 0;
    var chgColor = chgPct >= 0 ? '__UP__' : '__DOWN__';
    var chgSign = chgPct >= 0 ? '+' : '';
    var dt = new Date(d.time * 1000);
    var dateLabel = dt.getFullYear() + '-' + ('0'+(dt.getMonth()+1)).slice(-2) + '-' + ('0'+dt.getDate()).slice(-2);

    var html = '<span style="color:#5f5648;">' + dateLabel + '</span>'
      + '&nbsp; O <b>' + fmtNum(d.open) + '</b>'
      + '&nbsp; H <b style="color:__UP__;">' + fmtNum(d.high) + '</b>'
      + '&nbsp; L <b style="color:__DOWN__;">' + fmtNum(d.low) + '</b>'
      + '&nbsp; C <b>' + fmtNum(d.close) + '</b>'
      + '&nbsp; <b style="color:' + chgColor + ';">' + chgSign + fmtNum(chgPct) + '%</b>'
      + '&nbsp; <span style="color:#8a7f6d;font-size:10px;">' + fmtVol(d.volume) + '</span>';

    for (var j = 0; j < D.markers.length; j++) {
      if (D.markers[j].time === ts) {
        var m = D.markers[j];
        if (m.shape === 'arrowUp')
          html += '<br><span style="color:#d4a017;">&#9654; 买入 @ ' + fmtNum(parseFloat(m.text.replace('B ',''))) + '</span>';
        else
          html += '<br><span style="color:#ab47bc;">&#9654; 卖出 @ ' + fmtNum(parseFloat(m.text.replace('S ',''))) + '</span>';
      }
    }
    tooltip.innerHTML = html;
    tooltip.style.display = 'block';
  });

  // resize
  var rt;
  window.addEventListener('resize', function() {
    clearTimeout(rt);
    rt = setTimeout(function() {
      chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
    }, 100);
  });
})();
</script>
</body>
</html>"""


def _build_lwc_html(
    symbol: str,
    ohlcv_data: list,
    ma_series: dict,
    vol_data: list,
    kdj_data: dict,
    macd_data: dict,
    markers: list,
    height: int,
) -> str:
    """构建 lightweight-charts HTML。"""
    # 预计算 JS 用到的颜色映射
    ma_colors = {"5": KC["ma5"], "10": KC["ma10"], "20": KC["ma20"]}

    data_json = json.dumps({
        "ohlcv": ohlcv_data,
        "ma": ma_series,
        "vol": vol_data,
        "kdj": kdj_data,
        "macd": macd_data,
        "markers": markers,
        "symbol": symbol,
        "maColors": ma_colors,
    }, ensure_ascii=False)

    html = _LWC_HTML_TEMPLATE
    html = html.replace("__BG__", KC["bg"])
    html = html.replace("__TEXT__", KC["text"])
    html = html.replace("__GRID__", KC["grid"])
    html = html.replace("__CROSSHAIR__", KC["crosshair"])
    html = html.replace("__UP__", KC["up"])
    html = html.replace("__DOWN__", KC["down"])
    html = html.replace("__KDJ_K__", KC["kdj_k"])
    html = html.replace("__KDJ_D__", KC["kdj_d"])
    html = html.replace("__KDJ_J__", KC["kdj_j"])
    html = html.replace("__MACD__", KC["macd"])
    html = html.replace("__MACD_SIGNAL__", KC["macd_signal"])
    html = html.replace("__HEIGHT__", str(height))
    html = html.replace("__VOL_H__", str(int(height * 0.12)))
    html = html.replace("__KDJ_H__", str(int(height * 0.14)))
    html = html.replace("__MACD_H__", str(int(height * 0.16)))
    html = html.replace("__DATA__", data_json)
    return html


def render_kline_chart(result, key: str = "", height: int = 760):
    """渲染 lightweight-charts K 线图。

    result: plot_kline_with_signals 返回的 dict，包含 html 和 height。
    """
    import streamlit.components.v1 as components

    html = result.get("html", "")
    actual_height = result.get("height", height)
    components.html(html, height=actual_height, scrolling=False)


# ══════════════════════════════════════════════════════════════════
# 权益曲线保留 Plotly（静态图无需十字光标交互）
# ══════════════════════════════════════════════════════════════════

import plotly.graph_objects as go


def plot_equity_curve(
    equity_curve: dict,
    benchmark: Optional[dict] = None,
    title: str = "权益曲线",
) -> go.Figure:
    """绘制权益曲线（保留 Plotly——静态图无需十字光标）。"""
    fig = go.Figure()

    if equity_curve and len(equity_curve) > 1:
        dates = list(equity_curve.keys())
        values = list(equity_curve.values())
        if values and abs(values[0]) > 100:
            initial = values[0]
            cum_returns = [(v / initial - 1) * 100 for v in values]
        else:
            cum = 1.0
            cum_returns = []
            for v in values:
                cum *= (1 + v)
                cum_returns.append((cum - 1) * 100)

        fig.add_trace(go.Scatter(
            x=pd.to_datetime(dates), y=cum_returns,
            mode='lines', name='策略',
            line=dict(width=2, color=KC["ma10"]),
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
            line=dict(width=1, color=KC["text"], dash='dash'),
        ))

    fig.update_layout(
        title=title,
        height=350,
        template="plotly_white",
        paper_bgcolor=KC["bg"],
        plot_bgcolor=KC["bg"],
        font=dict(family="monospace", color=KC["text"], size=11),
        yaxis_title="累计收益 (%)",
        margin=dict(l=55, r=15, t=30, b=25),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.01,
            font=dict(size=9, family="monospace"), bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#f6f3ed", bordercolor="#c2b39f",
            font=dict(size=10, family="monospace", color="#2f2a22"),
        ),
    )
    fig.update_yaxes(tickformat='.1f')

    fig.update_xaxes(
        gridcolor=KC["grid"], showgrid=True, griddash='dot',
        zeroline=False, showline=False,
        tickfont=dict(size=9, color=KC["text"]),
    )
    fig.update_yaxes(
        gridcolor=KC["grid"], showgrid=True, griddash='dot',
        zeroline=False, showline=False, side='right',
        tickfont=dict(size=9, color=KC["text"]),
    )
    fig.add_hline(y=0, line_dash="dash", line_color=KC["grid"])

    return fig
