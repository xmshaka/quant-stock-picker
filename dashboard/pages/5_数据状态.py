"""数据更新状态页"""
import sys
from datetime import datetime

sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import streamlit as st

from data.scan_status import (
    latest_scan_report,
    parse_report_time,
    read_scan_reports,
    reports_to_frame,
    resolve_alert_path,
    scan_health,
)
from data.scheduler import run_daily_job
from config.settings import settings


st.set_page_config(page_title="数据状态", page_icon="🩺", layout="centered")

st.markdown("""
<style>
    .block-container { padding-top: 3.5rem !important; padding-left: 0.8rem; padding-right: 0.8rem; }
    h1 { font-size: 1.3rem !important; margin-top: 0.5rem !important; }
    h2 { font-size: 1.1rem !important; margin-top: 0.6rem !important; }
    .stMetric { background: #f8f9fa; border-radius: 6px; padding: 6px 4px; }
    .stMetric label { font-size: 0.7rem !important; }
    .stMetric div[data-testid="stMetricValue"] { font-size: 1rem !important; }
</style>
""", unsafe_allow_html=True)

st.title("🩺 数据状态")

col1, col2 = st.columns([3, 1])
with col1:
    st.caption(f"状态文件: `{resolve_alert_path()}`")
with col2:
    if st.button("🔄 刷新", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


@st.cache_data(ttl=60)
def _load_reports():
    reports = read_scan_reports(limit=20)
    return latest_scan_report(), reports


latest, reports = _load_reports()
status, msg = scan_health(latest)

if status == "OK":
    st.success(f"✅ {msg}")
elif status == "WARN":
    st.warning(f"⚠️ {msg}")
elif status == "ERROR":
    st.error(f"❌ {msg}")
else:
    st.info(f"ℹ️ {msg}")

if latest:
    ts = parse_report_time(latest)
    ts_text = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else str(latest.get("ts", "-"))
    age_text = "-"
    if ts:
        age_min = max((datetime.now() - ts).total_seconds() / 60, 0)
        age_text = f"{age_min:.0f} 分钟前" if age_min < 1440 else f"{age_min/1440:.1f} 天前"

    st.subheader("最近一次扫描")
    c1, c2, c3 = st.columns(3)
    c1.metric("扫描时间", ts_text[-8:] if ts_text != "-" else "-")
    c2.metric("距离现在", age_text)
    c3.metric("级别", latest.get("level", "-"))

    c4, c5, c6 = st.columns(3)
    c4.metric("股票池", int(latest.get("total_symbols", 0) or 0))
    c5.metric("成功更新", int(latest.get("updated_count", 0) or 0))
    c6.metric("失败", int(latest.get("failed_count", 0) or 0))

    c7, c8, c9 = st.columns(3)
    c7.metric("已最新", int(latest.get("skipped_up_to_date", 0) or 0))
    c8.metric("新增行数", int(latest.get("new_rows", 0) or 0))
    c9.metric("耗时", f"{float(latest.get('elapsed_seconds', 0) or 0):.1f}s")

    failures = latest.get("failures") or []
    if failures:
        with st.expander(f"失败明细 ({len(failures)})", expanded=False):
            st.dataframe(failures, use_container_width=True, hide_index=True)

    with st.expander("原始摘要", expanded=False):
        st.code(latest.get("summary", ""), language="text")
else:
    st.warning("还没有扫描报告。可以先执行：`python -m data.scheduler --once`")

st.subheader("手动扫描")
with st.expander("小范围手动扫描", expanded=False):
    st.caption("用于临时补数/验证数据源。为避免误触发全量扫描，这里最多允许 10 只股票。")
    manual_symbols = st.text_input("股票代码", value="600519,000001", help="逗号分隔，例如 600519,000001")
    m1, m2 = st.columns(2)
    with m1:
        lookback_days = st.number_input("回看天数", min_value=1, max_value=365, value=10, step=1)
    with m2:
        max_workers = st.number_input("并发数", min_value=1, max_value=4, value=2, step=1)

    symbols = [s.strip() for s in manual_symbols.split(",") if s.strip()]
    if len(symbols) > 10:
        st.error("一次最多扫描 10 只，请减少代码数量。")
    elif st.button("▶️ 立即扫描", type="primary", use_container_width=True, disabled=not symbols):
        with st.spinner(f"正在扫描 {len(symbols)} 只股票..."):
            try:
                report = run_daily_job(
                    lookback_days=int(lookback_days),
                    max_workers=int(max_workers),
                    symbols=symbols,
                )
                st.success("扫描完成，报告已写入状态文件。")
                st.code(report.summary(), language="text")
                st.cache_data.clear()
            except Exception as e:
                st.error(f"扫描失败: {e}")

st.subheader("历史扫描")
hist_df = reports_to_frame(reports)
if hist_df.empty:
    st.info("暂无历史记录")
else:
    st.dataframe(hist_df.sort_values("时间", ascending=False), use_container_width=True, hide_index=True)

st.subheader("调度配置")
conf_col1, conf_col2 = st.columns(2)
conf_col1.metric("每日扫描", "启用" if settings.daily_scan_enabled else "关闭")
conf_col2.metric("时间", f"{settings.daily_scan_hour:02d}:{settings.daily_scan_minute:02d}")
st.caption(f"降级顺序: `{settings.data_source_order}` · 失败阈值: `{settings.daily_scan_failure_threshold}`")
