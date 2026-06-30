"""数据状态页面"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import streamlit as st
import pandas as pd
from datetime import datetime

from theme import inject_theme, metric_row, section_header, badge, empty_state, C
from data.repair import repair_status, run_scan_repair, run_factor_repair

st.set_page_config(page_title="数据状态", page_icon="🩺", layout="wide")
inject_theme()


def _trigger_scan():
    r = run_scan_repair()
    if r["success"]:
        st.toast(f"🧮 增量扫描已后台启动", icon="🧮")
        st.info(f"✅ {r['message']}")
    else:
        st.error(r["message"])


def _trigger_factor():
    r = run_factor_repair()
    if r["success"]:
        st.toast(f"🧮 因子预计算已后台启动", icon="🧮")
        st.info(f"✅ {r['message']}")
    else:
        st.error(r["message"])

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
    # 空数据状态：提供补全入口
    rs = repair_status()
    st.warning(f"💡 {rs['message']}")
    cols = st.columns(2)
    with cols[0]:
        if rs["need_scan"]:
            st.button("🔄 补拉数据（增量扫描）", key="btn_empty_scan", use_container_width=True,
                      help="拉取全A股K线数据", on_click=_trigger_scan)
    with cols[1]:
        if rs["need_factor"]:
            st.button("🧮 补快照（因子预计算）", key="btn_empty_factor", use_container_width=True,
                      help="生成全池因子快照", on_click=_trigger_factor)

# ========== 因子数据完整性 ==========
if snap_date:
    section_header("因子数据完整性", f"({snap_date})")
    try:
        import pandas as pd
        from pathlib import Path as P
        factor_path = P("data/daily_factors") / f"factors_{snap_date}.parquet"
        if factor_path.exists():
            fdf = pd.read_parquet(factor_path)
            # 分类因子
            CATEGORIES = {
                "技术指标": ["rsi14", "boll_position", "volatility_20d", "ma5", "ma20"],
                "动量反转": ["momentum_5d", "momentum_20d", "reversal", "high_20d_distance"],
                "量能换手": ["volume_ratio", "relative_turnover_5d", "relative_turnover_20d",
                          "turnover_percentile_60d", "amount_percentile_60d"],
                "估值规模": ["pb", "float_market_cap", "north_hold_change"],
                "资金流": ["main_net_mf_amount", "main_net_mf_pct_amount", "main_net_mf_rank",
                        "large_elg_net_mf_amount", "large_elg_net_mf_pct_amount", "large_elg_net_mf_rank",
                        "large_net_mf_amount", "elg_net_mf_amount"],
            }
            total_rows = len(fdf)
            # FIX: 使用最新日空值率判定分类健康度，避免历史早期字段缺失拉高全量均值导致误报
            latest_trade_date = fdf["trade_date"].max() if "trade_date" in fdf.columns else None
            latest_df = fdf[fdf["trade_date"] == latest_trade_date] if latest_trade_date is not None else fdf
            for cat_name, cols in CATEGORIES.items():
                present = [c for c in cols if c in fdf.columns]
                if not present:
                    continue
                cat_tags = []
                cat_ok = True
                for c in present:
                    # 展示用最新日空值率（全量平均会掩盖当日真实状态）
                    day_null_pct = round(latest_df[c].isna().mean() * 100, 1) if not latest_df.empty else 0.0
                    all_null_pct = round(fdf[c].isna().mean() * 100, 1)
                    null_pct = day_null_pct
                    if null_pct <= 5:
                        cat_tags.append(badge(f"{c} {null_pct}%", "buy"))
                    elif null_pct <= 20:
                        cat_tags.append(badge(f"{c} {null_pct}%", "hold"))
                        cat_ok = False
                    else:
                        cat_tags.append(badge(f"{c} {null_pct}%", "sell"))
                        cat_ok = False
                cat_badge = badge("✓ 正常", "buy") if cat_ok else badge("⚠ 缺失", "sell")
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0;">'
                    f'<span style="min-width:72px;font-size:0.78rem;color:{C["text2"]};">{cat_name}</span>'
                    f'{cat_badge}'
                    f'<span style="display:flex;flex-wrap:wrap;gap:4px;">{"".join(cat_tags)}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            # ── 最新日空值专项检查（关键：全历史平均会掩盖当日 100% NaN）──
            # 分类检查：阻断级（技术/动量/估值 >20%→报错）vs 警告级（资金流/换手 >20%→仅提示 T+1 延迟）
            BLOCKING_FIELDS = set([
                "volume_ratio", "rsi14", "boll_position", "volatility_20d",
                "momentum_5d", "momentum_20d", "pb", "float_market_cap",
                "ma5", "ma20", "reversal", "high_20d_distance",
            ])
            WARNING_FIELDS = set([
                "main_net_mf_amount", "main_net_mf_pct_amount", "main_net_mf_rank",
                "large_elg_net_mf_amount", "large_elg_net_mf_pct_amount", "large_elg_net_mf_rank",
                "large_net_mf_amount", "elg_net_mf_amount",
                "relative_turnover_5d", "relative_turnover_20d",
                "turnover_percentile_60d", "amount_percentile_60d",
                "north_hold_change",
            ])
            if latest_trade_date is not None:
                blocking_alerts = []
                warning_alerts = []
                for cat_name, cols in CATEGORIES.items():
                    present = [c for c in cols if c in fdf.columns]
                    for c in present:
                        day_null_pct = round(latest_df[c].isna().mean() * 100, 1)
                        if day_null_pct > 20:
                            if c in BLOCKING_FIELDS:
                                blocking_alerts.append((cat_name, c, day_null_pct))
                            elif c in WARNING_FIELDS:
                                warning_alerts.append((cat_name, c, day_null_pct))
                if blocking_alerts:
                    alert_lines = [f"{cat}/{col} {pct}%" for cat, col, pct in blocking_alerts]
                    st.error(f"🚫 最新日 {str(latest_trade_date)[:10]} 阻断级字段空值率 >20%：{'、'.join(alert_lines)}。数据未就绪，信号/回测结果不可信。")
                elif warning_alerts:
                    warn_lines = [f"{col} {pct}%" for _, col, pct in warning_alerts]
                    st.warning(f"⚠️ 最新日 {str(latest_trade_date)[:10]} 资金流/换手字段空值率 >20%（{', '.join(warn_lines)}），属 T+1 延迟正常现象。信号已自动回退纯技术模式，19:30 后快照补全。")
                else:
                    st.success(f"✅ 最新日 {str(latest_trade_date)[:10]} 所有因子字段空值率 ≤20%，数据就绪。")

            # ── 手动补全入口（基于 repair_status 条件判断）──
            rs = repair_status()
            st.divider()
            if rs["status"] == "ok":
                st.success(f"✅ {rs['message']}")
                # ok 时仅在用户明确需要时提供重生成入口
                cols = st.columns(2)
                with cols[0]:
                    if rs.get("need_scan"):
                        st.button("🔄 补拉数据", key="btn_scan_force", use_container_width=True,
                                  help="强制拉取最新K线数据", on_click=_trigger_scan)
                with cols[1]:
                    if rs.get("need_factor"):
                        st.button("🧮 补快照", key="btn_factor_force", use_container_width=True,
                                  help="强制重新生成因子快照", on_click=_trigger_factor)
            elif rs["status"] in ("stale", "gap", "missing"):
                st.warning(f"⚠️ {rs['message']}")
                cols = st.columns(2)
                with cols[0]:
                    if rs["need_scan"]:
                        st.button("🔄 补拉数据（增量扫描）", key="btn_scan_repair", use_container_width=True,
                                  help="拉取最新K线数据", on_click=_trigger_scan)
                    else:
                        st.caption("✅ 数据拉取已是最新")
                with cols[1]:
                    if rs["need_factor"]:
                        st.button("🧮 补快照（因子预计算）", key="btn_factor_repair", use_container_width=True,
                                  help="重新生成因子快照", on_click=_trigger_factor)
                    else:
                        st.caption("✅ 因子快照正常")

            # 数据源说明
            meta = load_snapshot_meta(snap_date)
            if meta:
                mf_src = meta.get("moneyflow_source", "")
                mf_date = meta.get("moneyflow_date", "")
                if mf_src or mf_date:
                    st.caption(f"资金流数据源: {mf_src or '未知'} · 资金流日期: {mf_date or '未知'} · 总行数: {total_rows:,}")
        else:
            st.caption(f"快照文件不存在: {factor_path}")
    except Exception as e:
        st.warning(f"读取因子完整性失败: {e}")

# ========== 因子计算执行日志 ==========
section_header("因子计算执行日志")
try:
    import json
    from pathlib import Path as P
    meta_files = sorted(P("data/daily_factors").glob("meta_*.json"), reverse=True)
    if meta_files:
        rows = []
        for mp in meta_files[:20]:
            m = json.loads(mp.read_text())
            rows.append({
                "日期": mp.stem.replace("meta_", ""),
                "计算时间": str(m.get("computed_at", "?"))[:16],
                "耗时": f"{m.get('elapsed_seconds', 0):.0f}s",
                "因子行数": f"{m.get('factor_rows', 0):,}",
                "股票池": f"{m.get('universe_size', 0)} 只",
                "资金流日期": m.get("moneyflow_date", "?"),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                         column_config={"计算时间": st.column_config.TextColumn(width="small")})
        else:
            st.caption("无因子计算记录")
except Exception as e:
    st.warning(f"读取因子计算日志失败: {e}")

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
