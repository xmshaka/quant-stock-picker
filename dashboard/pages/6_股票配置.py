"""股票池与扫描配置页"""
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import streamlit as st
from config.settings import settings
from data.universe import Universe
from theme import inject_theme, section_header, badge, C

st.set_page_config(page_title="股票配置", page_icon="⚙", layout="centered")
inject_theme()

# ========== env 写入 ==========
ENV_KEYS = {
    "universe_source": "UNIVERSE_SOURCE", "max_stocks": "MAX_STOCKS",
    "exclude_st": "EXCLUDE_ST", "exclude_delisting": "EXCLUDE_DELISTING",
    "exclude_suspended": "EXCLUDE_SUSPENDED", "exclude_new_stock_days": "EXCLUDE_NEW_STOCK_DAYS",
    "exclude_bj": "EXCLUDE_BJ", "min_float_mv_yi": "MIN_FLOAT_MV_YI",
    "max_float_mv_yi": "MAX_FLOAT_MV_YI", "min_avg_turnover": "MIN_AVG_TURNOVER",
    "data_source_order": "DATA_SOURCE_ORDER", "daily_scan_enabled": "DAILY_SCAN_ENABLED",
    "daily_scan_hour": "DAILY_SCAN_HOUR", "daily_scan_minute": "DAILY_SCAN_MINUTE",
    "daily_scan_lookback_days": "DAILY_SCAN_LOOKBACK_DAYS",
    "daily_scan_failure_threshold": "DAILY_SCAN_FAILURE_THRESHOLD",
}

def _fmt(v):
    return "true" if v is True else "false" if v is False else str(v)

def update_env_file(values: dict):
    env_path = settings.project_root / ".env"
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    unset = {ENV_KEYS[k] for k, v in values.items() if k in ENV_KEYS and v is None}
    wanted = {ENV_KEYS[k]: _fmt(v) for k, v in values.items() if k in ENV_KEYS and ENV_KEYS[k] not in unset}
    seen, out = set(), []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or "=" not in line:
            out.append(line); continue
        key = line.split("=", 1)[0].strip()
        if key in wanted:
            out.append(f"{key}={wanted[key]}"); seen.add(key)
        else:
            out.append(line)
    out = [l for l in out if l.split("=", 1)[0].strip() not in unset]
    missing = [k for k in wanted if k not in seen]
    if missing:
        if out and out[-1].strip(): out.append("")
        out.append("# Dashboard stock selection config")
        for k in missing: out.append(f"{k}={wanted[k]}")
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")

@st.cache_data(ttl=300, show_spinner=False)
def preview_universe(overrides):
    return Universe().load(use_cache=True, **overrides)

# ========== UI ==========
section_header("股票选择配置")
st.caption("配置股票池过滤与每日扫描参数。保存后需重启服务生效。")

with st.expander("当前生效配置", expanded=False):
    st.json({
        "universe_source": settings.universe_source, "max_stocks": settings.max_stocks,
        "exclude_st": settings.exclude_st, "exclude_delisting": settings.exclude_delisting,
        "exclude_suspended": settings.exclude_suspended, "exclude_bj": settings.exclude_bj,
        "daily_scan_time": f"{settings.daily_scan_hour:02d}:{settings.daily_scan_minute:02d}",
        "data_source_order": settings.data_source_order,
    })

section_header("股票池范围")
source_options = ["all_a", "hs300", "zz500", "zz1000"]
universe_source = st.selectbox("来源", source_options,
    index=source_options.index(settings.universe_source) if settings.universe_source in source_options else 0)
max_stocks = st.number_input("最大数量", 1, 8000, int(settings.max_stocks), step=100)

section_header("过滤条件")
c1, c2 = st.columns(2)
with c1:
    exclude_st = st.checkbox("排除 ST", value=bool(settings.exclude_st))
    exclude_delisting = st.checkbox("排除退市", value=bool(settings.exclude_delisting))
    exclude_suspended = st.checkbox("排除停牌", value=bool(settings.exclude_suspended))
with c2:
    exclude_bj = st.checkbox("排除北交所", value=bool(settings.exclude_bj))
    exclude_new = st.number_input("上市不足N天排除", 0, 1000, int(settings.exclude_new_stock_days), step=10)

c3, c4 = st.columns(2)
with c3:
    min_mv = st.number_input("最小流通市值(亿)", 0.0, value=float(settings.min_float_mv_yi), step=5.0)
with c4:
    max_mv_raw = st.number_input("最大流通市值(亿, 0=不限)", 0.0, value=float(settings.max_float_mv_yi or 0), step=50.0)
min_turn = st.number_input("最小日均换手率(%)", 0.0, value=float(settings.min_avg_turnover), step=0.1)
max_mv = None if max_mv_raw <= 0 else max_mv_raw

overrides = {
    "universe_source": universe_source, "max_stocks": int(max_stocks),
    "exclude_st": bool(exclude_st), "exclude_delisting": bool(exclude_delisting),
    "exclude_suspended": bool(exclude_suspended), "exclude_new_stock_days": int(exclude_new),
    "exclude_bj": bool(exclude_bj), "min_float_mv_yi": float(min_mv),
    "max_float_mv_yi": max_mv, "min_avg_turnover": float(min_turn),
}

section_header("预览")
if st.button("🔍 预览股票池", width="stretch"):
    with st.spinner("构建中..."):
        try:
            df = preview_universe(overrides)
            st.success(f"筛选后: {len(df)} 只")
            cols = [c for c in ["symbol", "name", "close", "pct_change", "float_mv", "turnover_rate"] if c in df.columns]
            st.dataframe(df[cols].head(50) if cols else df.head(50), width="stretch", hide_index=True)
        except Exception as e:
            st.error(f"预览失败: {e}")

section_header("每日扫描")
s1, s2 = st.columns(2)
with s1:
    scan_enabled = st.checkbox("启用", value=bool(settings.daily_scan_enabled))
    scan_hour = st.number_input("小时", 0, 23, int(settings.daily_scan_hour))
with s2:
    scan_minute = st.number_input("分钟", 0, 59, int(settings.daily_scan_minute))
    scan_lookback = st.number_input("回看天数", 1, 1000, int(settings.daily_scan_lookback_days), step=10)

source_order = st.text_input("数据源降级", value=settings.data_source_order)
scan_threshold = st.number_input("失败告警阈值", 1, 1000, int(settings.daily_scan_failure_threshold))

scan_vals = {
    "data_source_order": source_order.strip(), "daily_scan_enabled": bool(scan_enabled),
    "daily_scan_hour": int(scan_hour), "daily_scan_minute": int(scan_minute),
    "daily_scan_lookback_days": int(scan_lookback), "daily_scan_failure_threshold": int(scan_threshold),
}

st.divider()
if st.button("💾 保存到 .env", type="primary", width="stretch"):
    try:
        update_env_file({**overrides, **scan_vals})
        st.success("已保存")
        st.warning("⚠️ 需重启看板和调度服务后生效。")
        st.code("systemctl restart quant-stock-picker.service quant-scheduler.service", language="bash")
    except Exception as e:
        st.error(f"保存失败: {e}")

st.info("💡 临时验证用「预览」，持久化才点「保存」。")
