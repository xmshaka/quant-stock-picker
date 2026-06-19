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
try:
    from data.daily_factors import latest_snapshot_date, load_snapshot_meta, latest_data_source_meta, snapshot_coverage_report
    HAS_DAILY_FACTORS = True
except ImportError as e:
    print(f"Warning: Failed to import from data.daily_factors: {e}")
    HAS_DAILY_FACTORS = False
    # 提供默认实现
    def latest_snapshot_date():
        return None
    def load_snapshot_meta(date_str):
        return None
    def latest_data_source_meta():
        return {"snapshot_date": "", "snapshot_source": "none", "primary_source": "", "quote_source": "", "daily_basic_source": "", "daily_basic_date": "", "computed_at": ""}
    def snapshot_coverage_report(date_str=None):
        return {}

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

        coverage = snapshot_coverage_report(snap_date)
        if coverage:
            section_header("快照覆盖度")
            metric_row([
                {"label": "全局最新交易日", "value": coverage.get("global_latest_date") or "—"},
                {"label": "最新覆盖", "value": f"{coverage.get('fresh_symbols', 0):,} 只"},
                {"label": "滞后个股", "value": f"{coverage.get('stale_symbols', 0):,} 只"},
                {"label": "覆盖率", "value": f"{coverage.get('coverage_pct', 0):.2f}%"},
            ], cols=4)
            dist = coverage.get("date_distribution", {})
            if dist:
                dist_text = " · ".join([f"{d}: {n}只" for d, n in dist.items()])
                st.caption(f"各股票自身最新交易日分布：{dist_text}")
                if coverage.get("stale_symbols", 0):
                    st.warning("存在个股快照落后于全局最新交易日；观察页会回退展示该股票自身最新日。请重新跑增量扫描 + 全池因子预计算补齐后再用于正式信号。")
else:
    empty_state("🩺", "暂无快照数据")

# ========== 数据来源追踪 ==========
section_header("数据来源")
if HAS_DAILY_FACTORS:
    source_meta = latest_data_source_meta()
else:
    source_meta = {"snapshot_date": "", "snapshot_source": "none", "primary_source": "", "quote_source": "", "daily_basic_source": "", "daily_basic_date": "", "computed_at": ""}
source_items = [
    ("快照来源", source_meta.get("snapshot_source", "none")),
    ("股票池主源", source_meta.get("primary_source", "")),
    ("行情补充", source_meta.get("quote_source", "")),
    ("估值缓存", source_meta.get("daily_basic_source", "")),
    ("估值日期", source_meta.get("daily_basic_date", "")),
]
for label, value in source_items:
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid {C['border']};">
        <span style="font-size:0.78rem;color:{C['text2']};">{label}</span>
        <span style="font-size:0.78rem;font-weight:600;color:{C['text']};">{value or '—'}</span>
    </div>
    """, unsafe_allow_html=True)

# ========== Baostock 可用性 ==========
try:
    from data.fetchers.baostock_fetcher import BaostockFetcher
    bs_status = BaostockFetcher.status()
    section_header("Baostock 可用性")
    ok_badge = badge("可用", "buy") if bs_status.get("last_ok") else badge("待验证", "hold")
    if not bs_status.get("installed"):
        ok_badge = badge("未安装", "sell")
    bs_items = [
        ("安装状态", "已安装" if bs_status.get("installed") else "未安装"),
        ("版本", bs_status.get("version", "unknown")),
        ("最近状态", ok_badge),
        ("最近操作", bs_status.get("last_operation", "—")),
        ("最近行数", bs_status.get("last_rows", 0)),
        ("检查时间", bs_status.get("last_checked_at", "—")),
        ("日K缓存", f"{bs_status.get('cached_daily_files', 0)} 个"),
    ]
    for label, value in bs_items:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid {C['border']};">
            <span style="font-size:0.78rem;color:{C['text2']};">{label}</span>
            <span style="font-size:0.78rem;font-weight:600;color:{C['text']};">{value}</span>
        </div>
        """, unsafe_allow_html=True)
except Exception as e:
    st.warning(f"读取 Baostock 状态失败: {e}")

# ========== 缓存健康 ==========
try:
    from data.cache_health import cache_health_summary
    try:
        from data.storage.repository import stock_repo
        HAS_STOCK_REPO = True
    except ImportError as e:
        print(f"Warning: Failed to import stock_repo: {e}")
        HAS_STOCK_REPO = False
        # 提供默认实现
        class MockStockRepo:
            def count_bars_by_source_adjust(self):
                return pd.DataFrame()
    cache_status = cache_health_summary()
    kline_cache = cache_status.get("l2_kline", {})
    section_header("缓存健康")

    l3_badge = badge("启用", "sell") if kline_cache.get("l3_enabled") else badge("禁用", "buy")
    old_count = int(kline_cache.get("old_files", 0) or 0)
    old_badge = badge(f"旧缓存 {old_count}", "sell") if old_count else badge("无旧缓存", "buy")
    cache_items = [
        ("L2日K新缓存", f"{kline_cache.get('new_files', 0)} 个"),
        ("L2快照缓存", f"{cache_status.get('snapshot_files', 0)} 个"),
        ("L3 PG K线", l3_badge),
        ("旧版K线缓存", old_badge),
    ]
    for label, value in cache_items:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid {C['border']};">
            <span style="font-size:0.78rem;color:{C['text2']};">{label}</span>
            <span style="font-size:0.78rem;font-weight:600;color:{C['text']};">{value}</span>
        </div>
        """, unsafe_allow_html=True)

    dist = kline_cache.get("distribution", [])
    if dist:
        st.markdown(f"<div style='font-size:0.78rem;color:{C['text2']};margin-top:8px;'>L2日K分布</div>", unsafe_allow_html=True)
        tags = " ".join([badge(f"{d['source']}/{d['adjust']} {d['files']}", "hold") for d in dist])
        st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:4px;">{tags}</div>', unsafe_allow_html=True)

    # PG source/adjust 统计
    if HAS_STOCK_REPO:
        pg_stats = stock_repo().count_bars_by_source_adjust()
    else:
        pg_stats = pd.DataFrame()
    
    if not pg_stats.empty:
        st.markdown(f"<div style='font-size:0.78rem;color:{C['text2']};margin-top:12px;'>PG K线分布</div>", unsafe_allow_html=True)
        pg_tags = []
        for _, row in pg_stats.iterrows():
            src = row["source"] if row["source"] else "（空）"
            pg_tags.append(badge(f"{src}/{row['adjust']} {row['count']}", "hold"))
        st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:4px;">{", ".join(pg_tags)}</div>', unsafe_allow_html=True)

    if old_count:
        examples = ", ".join(kline_cache.get("old_examples", []))
        st.warning(f"检测到旧版 K线缓存，当前新链路不会读取；可在确认无回滚需求后清理。示例：{examples}")
except Exception as e:
    st.warning(f"读取缓存健康失败: {e}")

# ========== 扫描历史 ==========
section_header("扫描历史")
from data.scan_status import load_scan_reports

try:
    reports = load_scan_reports()
    if not reports.empty:
        for idx, (_, row) in enumerate(reports.head(10).iterrows()):
            ts = str(row.get("ts", ""))[:16]
            total = row.get("total_symbols", 0)
            skipped = row.get("skipped_up_to_date", 0)
            updated = row.get("updated_count", 0)
            new_rows = row.get("new_rows", 0)
            failed = row.get("failed_count", 0)
            elapsed = row.get("elapsed_seconds", 0)
            row_bg = C['surface2'] if idx % 2 == 0 else C['surface']
            tone_color = C['green'] if failed == 0 else C['red']

            status_b = badge("成功", "buy") if failed == 0 else badge(f"失败{failed}", "sell")
            st.markdown(f"""
            <div style="background:{row_bg};border:1px solid {C['border']};border-left:4px solid {tone_color};border-radius:0.375rem;margin-bottom:7px;padding:9px 12px;display:flex;align-items:center;gap:14px;">
                <span style="font-size:0.78rem;font-weight:600;min-width:140px;color:{C['text']};">{ts}</span>
                <span style="min-width:58px;">{status_b}</span>
                <span style="font-size:0.72rem;color:{C['text2']};">股票池 {total} · 已最新 {skipped} · 拉取更新 {updated} · 新增行 {new_rows} · 耗时 {elapsed:.0f}s</span>
            </div>
            """, unsafe_allow_html=True)
        st.caption("说明：扫描历史的“拉取更新”来自增量扫描日志，表示本次实际请求并写入过的股票数；它不等于每日因子快照中覆盖到全局最新交易日的股票数。")
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
