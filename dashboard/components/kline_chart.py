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
import time
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Optional

from signals.rules import TradePoint

_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"
_LWC_JS_PATH = _ASSET_DIR / "lightweight-charts-5.2.0.standalone.production.js"
_ECHARTS_JS_PATH = _ASSET_DIR / "echarts-5.5.1.min.js"
_KLINE_VIEW_VERSION = "echarts-multigrid-polish-v20260619-1910"


def _load_lwc_js() -> str:
    """读取本地 lightweight-charts，避免 CDN/CSP/外网波动导致 K线空白。"""
    if not _LWC_JS_PATH.exists():
        raise FileNotFoundError(f"lightweight-charts 本地资源缺失: {_LWC_JS_PATH}")
    return _LWC_JS_PATH.read_text(encoding="utf-8")


def _load_echarts_js() -> str:
    """读取本地 ECharts，避免 CDN/CSP/外网波动导致 K线空白。"""
    if not _ECHARTS_JS_PATH.exists():
        raise FileNotFoundError(f"ECharts 本地资源缺失: {_ECHARTS_JS_PATH}")
    return _ECHARTS_JS_PATH.read_text(encoding="utf-8")

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


def _ts(dt) -> str:
    """日期 → YYYY-MM-DD。日线图统一用业务日期，避免坐标 tooltip 显示时分秒。"""
    return pd.Timestamp(dt).strftime("%Y-%m-%d")


def _safe_ts(dt) -> str:
    """可空日期 → YYYY-MM-DD；用于 signal_date 等审计字段。"""
    if dt is None:
        return ""
    if isinstance(dt, str) and dt.strip() == "":
        return ""
    try:
        if pd.isna(dt):
            return ""
    except (TypeError, ValueError):
        pass
    return _ts(dt)


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
    bar_by_ts = {
        _ts(bars['trade_date'].iloc[i]): {"high": float(h.iloc[i]), "low": float(l.iloc[i])}
        for i in range(n)
    }
    price_span = max(float(h.max() - l.min()), float(c.mean()) * 0.05, 0.01)
    marker_gap = price_span * 0.14
    for p in trade_points:
        exec_dt = getattr(p, 'exec_date', None) or getattr(p, 'date', None)
        signal_dt = getattr(p, 'signal_date', None) or ''
        ts = _ts(exec_dt)
        price = float(getattr(p, 'exec_price', 0) or p.price)
        action = str(getattr(p, 'action', '')).upper()
        bar = bar_by_ts.get(ts, {})
        anchor_high = float(bar.get("high", price))
        anchor_low = float(bar.get("low", price))
        label_price = anchor_low - marker_gap if action == "BUY" else anchor_high + marker_gap
        anchor_price = anchor_low if action == "BUY" else anchor_high
        marker_base = {
            "time": ts,
            "signalDate": _safe_ts(signal_dt),
            "execDate": ts,
            "price": price,
            "anchorPrice": anchor_price,
            "labelPrice": label_price,
            "reason": str(getattr(p, 'reason', '') or ''),
            "rule": str(getattr(p, 'rule_name', '') or ''),
            "confidence": float(getattr(p, 'confidence', 0) or 0),
            "shares": int(getattr(p, 'shares', 0) or 0),
            "pnl": float(getattr(p, 'pnl', 0) or 0),
            "pnlPct": float(getattr(p, 'pnl_pct', 0) or 0),
            "holdingDays": int(getattr(p, 'holding_days', 0) or 0),
        }
        if action == "BUY":
            markers.append({
                **marker_base, "position": "belowBar", "action": "BUY",
                "color": KC["buy"], "shape": "arrowUp",
                "text": "", "size": 1,
            })
        elif action == "SELL":
            markers.append({
                **marker_base, "position": "aboveBar", "action": "SELL",
                "color": KC["sell"], "shape": "arrowDown",
                "text": "", "size": 1,
            })

    html = _build_lwc_html(symbol, ohlcv_data, ma_series, vol_data,
                           kdj_data, macd_data, markers, height)
    return {"html": html, "height": height}


# ══════════════════════════════════════════════════════════════════
# HTML 模板：用 __KEY__ 占位符替代 %s，避免 JS % 符号冲突
# ══════════════════════════════════════════════════════════════════

_LWC_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>html,body{width:100%;height:100%;}</style></head>
<body style="margin:0;padding:0;background:__BG__;overflow:hidden;min-height:__HEIGHT__px;">
<div id="chartWrap" style="position:relative;width:100%;height:__HEIGHT__px;background:__BG__;">
  <div id="klineVersion" style="position:absolute;left:6px;bottom:2px;z-index:1001;font-size:9px;color:rgba(95,86,72,0.45);font-family:monospace;pointer-events:none;">__VIEW_VERSION__</div>
  <div id="echartsKline" style="width:100%;height:__HEIGHT__px;"></div>
</div>
<div id="lwc-error" style="display:none;color:#ef5350;background:#fff3f3;border:1px solid #ef5350;margin:12px;padding:12px;font-family:monospace;font-size:13px;white-space:pre-wrap;"></div>
<script>
__ECHARTS_JS__
</script>
<script>
(function() {
  var errEl = document.getElementById('lwc-error');
  function fail(e) {
    errEl.style.display = 'block';
    errEl.textContent = 'K线渲染失败: ' + (e && e.message ? e.message : e);
    if (window.console && console.error) console.error(e);
  }
  try {
    if (typeof echarts === 'undefined') { fail('ECharts is undefined'); return; }
    var D = __DATA__;
    var wrap = document.getElementById('chartWrap');
    var el = document.getElementById('echartsKline');
    var chart = echarts.init(el, null, { renderer: 'canvas' });
    var categories = D.ohlcv.map(function(d) { return d.time; });
    var kData = D.ohlcv.map(function(d) { return [d.open, d.close, d.low, d.high]; });
    var volumes = D.vol.map(function(d) { return { value: d.value, itemStyle: { color: d.color } }; });
    function alignValues(arr) {
      var mp = {};
      (arr || []).forEach(function(p) { mp[p.time] = p.value; });
      return categories.map(function(t) { return Object.prototype.hasOwnProperty.call(mp, t) ? mp[t] : null; });
    }
    function fmtNum(v, dec) {
      if (v == null || isNaN(v)) return '--';
      return parseFloat(v).toFixed(dec == null ? 2 : dec);
    }
    function fmtVol(v) {
      if (v == null || isNaN(v)) return '--';
      if (Math.abs(v) >= 100000000) return fmtNum(v / 100000000, 2) + '亿';
      if (Math.abs(v) >= 10000) return fmtNum(v / 10000, 1) + '万';
      return fmtNum(v, 0);
    }
    function esc(s) {
      return String(s || '').replace(/[&<>"']/g, function(c) {
        return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
      });
    }
    var markerByTime = {};
    (D.markers || []).forEach(function(m) {
      if (!markerByTime[m.time]) markerByTime[m.time] = [];
      markerByTime[m.time].push(m);
    });
    var signalScatter = (D.markers || []).map(function(m) {
      var isBuy = m.action === 'BUY';
      return {
        value: [m.time, m.labelPrice],
        action: m.action,
        price: m.price,
        itemStyle: { color: isBuy ? '__BUY__' : '__SELL__' },
        label: {
          show: true,
          formatter: isBuy ? 'B' : 'S',
          color: isBuy ? '__BUY__' : '__SELL__',
          backgroundColor: '__BG__',
          borderColor: isBuy ? '__BUY__' : '__SELL__',
          borderWidth: 1,
          borderRadius: 3,
          padding: [2, 5],
          fontWeight: 800,
          fontSize: 11,
        },
        symbolSize: 1
      };
    });
    var signalLines = (D.markers || []).map(function(m) {
      var isBuy = m.action === 'BUY';
      return {
        coords: [[m.time, m.anchorPrice], [m.time, m.labelPrice]],
        lineStyle: { color: isBuy ? '__BUY__' : '__SELL__', type: 'dashed', width: 1, opacity: 0.9 }
      };
    });
    var activeGridIndex = 0;
    var gridRanges = [
      { idx: 0, top: __MAIN_TOP__, bottom: __MAIN_TOP__ + __MAIN_H__ },
      { idx: 1, top: __VOL_TOP__, bottom: __VOL_TOP__ + __VOL_H__ },
      { idx: 2, top: __KDJ_TOP__, bottom: __KDJ_TOP__ + __KDJ_H__ },
      { idx: 3, top: __MACD_TOP__, bottom: __MACD_TOP__ + __MACD_H__ }
    ];
    function updateActiveGridByY(y) {
      for (var gi = 0; gi < gridRanges.length; gi++) {
        if (y >= gridRanges[gi].top && y <= gridRanges[gi].bottom) {
          activeGridIndex = gridRanges[gi].idx;
          return;
        }
      }
    }
    function valuesAt(idx) {
      return {
        k: D.kdj.k ? alignValues(D.kdj.k)[idx] : null,
        d: D.kdj.d ? alignValues(D.kdj.d)[idx] : null,
        j: D.kdj.j ? alignValues(D.kdj.j)[idx] : null,
        dif: D.macd.dif ? alignValues(D.macd.dif)[idx] : null,
        dea: D.macd.dea ? alignValues(D.macd.dea)[idx] : null,
        hist: D.macd.hist ? alignValues(D.macd.hist)[idx] : null
      };
    }
    var option = {
      animation: false,
      backgroundColor: '__BG__',
      color: ['__KDJ_K__','__KDJ_D__','__KDJ_J__','__MACD__','__MACD_SIGNAL__'],
      axisPointer: {
        link: [{ xAxisIndex: 'all' }],
        label: { backgroundColor: 'rgba(138,127,109,0.9)' }
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'cross' },
        confine: true,
        backgroundColor: 'rgba(237,231,224,0.92)',
        borderColor: '#c2b39f',
        textStyle: { color: '#2f2a22', fontSize: 10, fontFamily: 'monospace' },
        extraCssText: 'text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.06);',
        formatter: function(params) {
          var axis = params && params.length ? params[0].axisValue : '';
          var idx = categories.indexOf(axis);
          if (idx < 0) return axis;
          var d = D.ohlcv[idx];
          var vals = valuesAt(idx);
          if (activeGridIndex === 1) {
            return '<span style="color:#5f5648;">' + axis + '</span><br>'
              + '<b>成交量</b>&nbsp;' + fmtVol(d.volume);
          }
          if (activeGridIndex === 2) {
            return '<span style="color:#5f5648;">' + axis + '</span><br>'
              + '<b style="color:__KDJ_K__;">K</b> ' + fmtNum(vals.k)
              + '&nbsp; <b style="color:__KDJ_D__;">D</b> ' + fmtNum(vals.d)
              + '&nbsp; <b style="color:__KDJ_J__;">J</b> ' + fmtNum(vals.j);
          }
          if (activeGridIndex === 3) {
            return '<span style="color:#5f5648;">' + axis + '</span><br>'
              + '<b style="color:__MACD__;">DIF</b> ' + fmtNum(vals.dif, 3)
              + '&nbsp; <b style="color:__MACD_SIGNAL__;">DEA</b> ' + fmtNum(vals.dea, 3)
              + '&nbsp; <b>MACD</b> ' + fmtNum(vals.hist, 3);
          }
          var prevClose = idx > 0 ? D.ohlcv[idx - 1].close : d.open;
          var chgPct = prevClose > 0 ? ((d.close - prevClose) / prevClose * 100) : 0;
          var chgColor = chgPct >= 0 ? '__UP__' : '__DOWN__';
          var html = '<span style="color:#5f5648;">' + axis + '</span>'
            + '&nbsp; O <b>' + fmtNum(d.open) + '</b>'
            + '&nbsp; H <b style="color:__UP__;">' + fmtNum(d.high) + '</b>'
            + '&nbsp; L <b style="color:__DOWN__;">' + fmtNum(d.low) + '</b>'
            + '&nbsp; C <b>' + fmtNum(d.close) + '</b>'
            + '&nbsp; <b style="color:' + chgColor + ';">' + (chgPct >= 0 ? '+' : '') + fmtNum(chgPct) + '%</b>'
            + '&nbsp; <span style="color:#8a7f6d;font-size:10px;">' + fmtVol(d.volume) + '</span>';
          (markerByTime[axis] || []).forEach(function(m) {
            var isBuy = m.action === 'BUY';
            var color = isBuy ? '#d4a017' : '#ab47bc';
            html += '<br><span style="color:' + color + ';font-weight:bold;">&#9654; '
              + (isBuy ? '买入' : '卖出') + ' @ ' + fmtNum(m.price) + '</span>';
            if (m.shares) html += '&nbsp; <span style="color:#5f5648;">' + m.shares + '股</span>';
            if (m.signalDate && m.signalDate !== m.execDate) html += '&nbsp; <span style="color:#8a7f6d;">信号:' + esc(m.signalDate) + '</span>';
            if (m.pnl || m.pnlPct) html += '&nbsp; <span style="color:' + ((m.pnl || 0) >= 0 ? '__UP__' : '__DOWN__') + ';">' + fmtNum((m.pnlPct || 0) * 100, 2) + '%</span>';
            var shortReason = esc(m.reason || m.rule || '');
            if (shortReason.length > 32) shortReason = shortReason.slice(0, 32) + '...';
            if (shortReason) html += '<br><span style="color:#8a7f6d;">' + shortReason + '</span>';
          });
          return html;
        }
      },
      grid: [
        { left: 48, right: 58, top: __MAIN_TOP__, height: __MAIN_H__, containLabel: false },
        { left: 48, right: 58, top: __VOL_TOP__, height: __VOL_H__, containLabel: false },
        { left: 48, right: 58, top: __KDJ_TOP__, height: __KDJ_H__, containLabel: false },
        { left: 48, right: 58, top: __MACD_TOP__, height: __MACD_H__, containLabel: false }
      ],
      xAxis: [0,1,2,3].map(function(i) {
        return {
          type: 'category',
          data: categories,
          gridIndex: i,
          boundaryGap: true,
          axisLine: { lineStyle: { color: '__GRID__' } },
          axisPointer: { label: { show: i === 3 } },
          axisTick: { show: false },
          axisLabel: { show: i === 3, color: '__TEXT__', fontSize: 10 },
          splitLine: { show: false },
          min: 'dataMin',
          max: 'dataMax'
        };
      }),
      yAxis: [
        { scale: true, gridIndex: 0, position: 'right', axisLine: { lineStyle: { color: '__GRID__' } }, axisLabel: { color: '__TEXT__', fontSize: 10 }, splitLine: { lineStyle: { color: '__GRID__', type: 'dashed' } } },
        { scale: true, gridIndex: 1, position: 'right', axisLine: { lineStyle: { color: '__GRID__' } }, axisLabel: { color: '__TEXT__', fontSize: 9, formatter: fmtVol }, splitLine: { lineStyle: { color: '__GRID__', type: 'dashed' } } },
        { scale: true, gridIndex: 2, position: 'right', min: 0, max: 120, axisLine: { lineStyle: { color: '__GRID__' } }, axisLabel: { color: '__TEXT__', fontSize: 9 }, splitLine: { lineStyle: { color: '__GRID__', type: 'dashed' } } },
        { scale: true, gridIndex: 3, position: 'right', axisLine: { lineStyle: { color: '__GRID__' } }, axisLabel: { color: '__TEXT__', fontSize: 9 }, splitLine: { lineStyle: { color: '__GRID__', type: 'dashed' } } }
      ],
      dataZoom: [
        { type: 'inside', xAxisIndex: [0,1,2,3], start: 0, end: 100, zoomOnMouseWheel: true, moveOnMouseMove: true }
      ],
      series: [
        { name: 'K线', type: 'candlestick', xAxisIndex: 0, yAxisIndex: 0, data: kData,
          itemStyle: { color: '__UP__', color0: '__DOWN__', borderColor: '__UP__', borderColor0: '__DOWN__' } },
        { name: 'MA5', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: alignValues(D.ma['5']), smooth: false, showSymbol: false, lineStyle: { width: 1, color: D.maColors['5'] }, emphasis: { disabled: true } },
        { name: 'MA10', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: alignValues(D.ma['10']), smooth: false, showSymbol: false, lineStyle: { width: 1, color: D.maColors['10'] }, emphasis: { disabled: true } },
        { name: 'MA20', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: alignValues(D.ma['20']), smooth: false, showSymbol: false, lineStyle: { width: 1, color: D.maColors['20'] }, emphasis: { disabled: true } },
        { name: '成交量', type: 'bar', xAxisIndex: 1, yAxisIndex: 1, data: volumes, barWidth: '60%' },
        { name: 'K', type: 'line', xAxisIndex: 2, yAxisIndex: 2, data: alignValues(D.kdj.k), showSymbol: false, lineStyle: { width: 1, color: '__KDJ_K__' } },
        { name: 'D', type: 'line', xAxisIndex: 2, yAxisIndex: 2, data: alignValues(D.kdj.d), showSymbol: false, lineStyle: { width: 1, color: '__KDJ_D__' } },
        { name: 'J', type: 'line', xAxisIndex: 2, yAxisIndex: 2, data: alignValues(D.kdj.j), showSymbol: false, lineStyle: { width: 1, color: '__KDJ_J__' } },
        { name: 'MACD柱', type: 'bar', xAxisIndex: 3, yAxisIndex: 3, data: alignValues(D.macd.hist).map(function(v, i) { return { value: v, itemStyle: { color: v >= 0 ? '#ef5350' : '#26a69a' } }; }), barWidth: '60%' },
        { name: 'DIF', type: 'line', xAxisIndex: 3, yAxisIndex: 3, data: alignValues(D.macd.dif), showSymbol: false, lineStyle: { width: 1, color: '__MACD__' } },
        { name: 'DEA', type: 'line', xAxisIndex: 3, yAxisIndex: 3, data: alignValues(D.macd.dea), showSymbol: false, lineStyle: { width: 1, color: '__MACD_SIGNAL__' } },
        { name: '信号连线', type: 'lines', coordinateSystem: 'cartesian2d', xAxisIndex: 0, yAxisIndex: 0, data: signalLines, polyline: false, symbol: ['none','none'], silent: true, z: 10 },
        { name: '买卖点', type: 'scatter', xAxisIndex: 0, yAxisIndex: 0, data: signalScatter, symbol: 'circle', z: 11, tooltip: { show: false } }
      ]
    };
    chart.setOption(option, true);
    chart.getZr().on('mousemove', function(e) { updateActiveGridByY(e.offsetY); });
    function resizeChart() {
      chart.resize({ width: wrap.clientWidth || document.body.clientWidth || window.innerWidth, height: __HEIGHT__ });
    }
    window.addEventListener('resize', resizeChart);
    if (typeof ResizeObserver !== 'undefined') {
      var ro = new ResizeObserver(function() { resizeChart(); });
      ro.observe(wrap);
      ro.observe(el);
    }
    setTimeout(resizeChart, 0);
    setTimeout(resizeChart, 120);
    setTimeout(resizeChart, 360);
  } catch (e) { fail(e); }
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
    echarts_js = _load_echarts_js()

    html = _LWC_HTML_TEMPLATE
    html = html.replace("__BG__", KC["bg"])
    html = html.replace("__TEXT__", KC["text"])
    html = html.replace("__GRID__", KC["grid"])
    html = html.replace("__CROSSHAIR__", KC["crosshair"])
    html = html.replace("__UP__", KC["up"])
    html = html.replace("__DOWN__", KC["down"])
    html = html.replace("__BUY__", KC["buy"])
    html = html.replace("__SELL__", KC["sell"])
    html = html.replace("__KDJ_K__", KC["kdj_k"])
    html = html.replace("__KDJ_D__", KC["kdj_d"])
    html = html.replace("__KDJ_J__", KC["kdj_j"])
    html = html.replace("__MACD__", KC["macd"])
    html = html.replace("__MACD_SIGNAL__", KC["macd_signal"])
    html = f"<!-- kline-view-version:{_KLINE_VIEW_VERSION};nonce:{int(time.time() * 1000)} -->\n" + html
    html = html.replace("__HEIGHT__", str(height))
    gap = 6
    main_top = 6
    bottom_pad = 8
    main_h = int(height * 0.49)
    vol_h = int(height * 0.15)
    kdj_h = int(height * 0.16)
    macd_h = max(80, height - main_top - bottom_pad - main_h - vol_h - kdj_h - gap * 3)
    vol_top = main_top + main_h + gap
    kdj_top = vol_top + vol_h + gap
    macd_top = kdj_top + kdj_h + gap
    html = html.replace("__MAIN_TOP__", str(main_top))
    html = html.replace("__MAIN_H__", str(main_h))
    html = html.replace("__VIEW_VERSION__", _KLINE_VIEW_VERSION)
    # 默认布局：主图约占 1/2，副图合计约占 1/2；grid 间留 6px 防止 KDJ/MACD 裁切。
    html = html.replace("__VOL_H__", str(vol_h))
    html = html.replace("__KDJ_H__", str(kdj_h))
    html = html.replace("__MACD_H__", str(macd_h))
    html = html.replace("__VOL_TOP__", str(vol_top))
    html = html.replace("__KDJ_TOP__", str(kdj_top))
    html = html.replace("__MACD_TOP__", str(macd_top))
    html = html.replace("__DATA__", data_json)
    html = html.replace("__ECHARTS_JS__", echarts_js)
    return html


def render_kline_chart(result, key: str = "", height: int = 760):
    """渲染 lightweight-charts K 线图。

    result: plot_kline_with_signals 返回的 dict，包含 html 和 height。
    """
    import streamlit.components.v1 as components

    html = result.get("html", "")
    result_height = int(result.get("height", height) or height)
    actual_height = max(result_height, int(height or result_height))
    # 调用方常传 height=760；lightweight 图表高度写在 HTML 内部，需同步替换，
    # 否则 iframe 足够高但 canvas 仍按默认 580px 布局，副图会被压缩。
    if actual_height != result_height:
        html = html.replace(f"height:{result_height}px", f"height:{actual_height}px")
        html = html.replace(f"min-height:{result_height}px", f"min-height:{actual_height}px")
    # 每次渲染追加 nonce，避免 Streamlit/frontend 复用旧 iframe HTML。
    html = html + f"\n<!-- render-key:{key};render-nonce:{int(time.time() * 1000)} -->"
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
