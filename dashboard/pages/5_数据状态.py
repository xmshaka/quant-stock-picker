"""数据状态页面"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import streamlit as st
import pandas as pd
from datetime import datetime

from theme import inject_theme, metric_row, section_header, badge, empty_state, C

st.set_page_config(page_title="数据状态", page_icon="🩺", layout="wide")
inject_theme()

section_header("数据状态")

# ========== 快照状态 ==========
from data.daily_factors import latest_snapshot_date, load_snapshot_meta

snap_date = latest_snapshot_date()
if snap_date:
    meta = load_snapshot_meta(snap_date)
    if meta:
        metric_row([
            {"label": "快照日期", "value": snap_date},
            {"label": "股票池", "value": f"{meta.get('universe_size', 0)} 只"},
            {"label": "因子行数", "value": f"{meta.get('factor_rows', 0):,}"},
            {"label": "价格行数", "value": f"{meta.get('price_rows', 0):,}"},
            {"label": "耗时", "value": f"{meta.get('elapsed_seconds', 0):.0f}s"},
        ], cols=5)

        fnames = meta.get('factor_names', [])
        if fnames:
            section_header("因子列表", f"({len(fnames)})")
            tags = " ".join([badge(f, "hold") for f in fnames])
            st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:4px;">{tags}</div>', unsafe_allow_html=True)
else:
    empty_state("🩺", "暂无快照数据")

# ========== 扫描历史 ==========
section_header("扫描历史")
from data.scan_status import load_scan_reports

try:
    reports = load_scan_reports()
    if not reports.empty:
        for _, row in reports.head(10).iterrows():
            ts = str(row.get("ts", ""))[:16]
            total = row.get("total_symbols", 0)
            updated = row.get("updated_count", 0)
            failed = row.get("failed_count", 0)
            elapsed = row.get("elapsed_seconds", 0)

            status_b = badge("成功", "buy") if failed == 0 else badge(f"失败{failed}", "sell")
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;padding:7px 0;border-bottom:1px solid {C['border']};">
                <span style="font-size:0.78rem;width:140px;color:{C['text']};">{ts}</span>
                {status_b}
                <span style="font-size:0.72rem;color:{C['text2']};">池 {total} · 更新 {updated} · 耗时 {elapsed:.0f}s</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        empty_state("📋", "暂无扫描记录")
except Exception as e:
    st.warning(f"读取扫描记录失败: {e}")

# ========== 调度配置 ==========
section_header("调度配置")
from config.settings import settings

config_items = [
    ("每日扫描", "启用" if settings.daily_scan_enabled else "禁用"),
    ("扫描时间", f"{settings.daily_scan_hour:02d}:{settings.daily_scan_minute:02d}"),
    ("因子预计算", "启用" if settings.daily_factor_enabled else "禁用"),
    ("因子时间", f"{settings.daily_factor_hour:02d}:{settings.daily_factor_minute:02d}"),
    ("数据源", settings.data_source_order),
    ("股票池", settings.universe_source),
]
for label, value in config_items:
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid {C['border']};">
        <span style="font-size:0.78rem;color:{C['text2']};">{label}</span>
        <span style="font-size:0.78rem;font-weight:600;color:{C['text']};">{value}</span>
    </div>
    """, unsafe_allow_html=True)
