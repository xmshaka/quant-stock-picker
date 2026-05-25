"""股票池与扫描配置页"""
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import streamlit as st

from config.settings import settings
from data.universe import Universe


st.set_page_config(page_title="股票配置", page_icon="⚙️", layout="centered")

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


ENV_KEYS = {
    "universe_source": "UNIVERSE_SOURCE",
    "max_stocks": "MAX_STOCKS",
    "exclude_st": "EXCLUDE_ST",
    "exclude_delisting": "EXCLUDE_DELISTING",
    "exclude_suspended": "EXCLUDE_SUSPENDED",
    "exclude_new_stock_days": "EXCLUDE_NEW_STOCK_DAYS",
    "exclude_bj": "EXCLUDE_BJ",
    "min_float_mv_yi": "MIN_FLOAT_MV_YI",
    "max_float_mv_yi": "MAX_FLOAT_MV_YI",
    "min_avg_turnover": "MIN_AVG_TURNOVER",
    "data_source_order": "DATA_SOURCE_ORDER",
    "daily_scan_enabled": "DAILY_SCAN_ENABLED",
    "daily_scan_hour": "DAILY_SCAN_HOUR",
    "daily_scan_minute": "DAILY_SCAN_MINUTE",
    "daily_scan_lookback_days": "DAILY_SCAN_LOOKBACK_DAYS",
    "daily_scan_failure_threshold": "DAILY_SCAN_FAILURE_THRESHOLD",
}


def _format_env_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def update_env_file(values: dict[str, Any]) -> None:
    """Upsert selected config keys into project .env without touching secrets."""
    env_path = settings.project_root / ".env"
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    wanted = {ENV_KEYS[k]: _format_env_value(v) for k, v in values.items() if k in ENV_KEYS}

    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            out.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in wanted:
            out.append(f"{key}={wanted[key]}")
            seen.add(key)
        else:
            out.append(line)

    missing = [k for k in wanted if k not in seen]
    if missing:
        if out and out[-1].strip():
            out.append("")
        out.append("# Dashboard stock selection config")
        for key in missing:
            out.append(f"{key}={wanted[key]}")

    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")


@st.cache_data(ttl=300)
def preview_universe(overrides: dict[str, Any]):
    df = Universe().load(use_cache=True, **overrides)
    return df


st.title("⚙️ 股票选择配置")
st.caption("这里配置的是股票池过滤与每日扫描参数。保存后需要重启看板/调度服务才会对后台任务完全生效。")

with st.expander("当前生效配置", expanded=False):
    st.json({
        "universe_source": settings.universe_source,
        "max_stocks": settings.max_stocks,
        "exclude_st": settings.exclude_st,
        "exclude_delisting": settings.exclude_delisting,
        "exclude_suspended": settings.exclude_suspended,
        "exclude_new_stock_days": settings.exclude_new_stock_days,
        "exclude_bj": settings.exclude_bj,
        "min_float_mv_yi": settings.min_float_mv_yi,
        "max_float_mv_yi": settings.max_float_mv_yi,
        "min_avg_turnover": settings.min_avg_turnover,
        "data_source_order": settings.data_source_order,
        "daily_scan_enabled": settings.daily_scan_enabled,
        "daily_scan_time": f"{settings.daily_scan_hour:02d}:{settings.daily_scan_minute:02d}",
        "daily_scan_lookback_days": settings.daily_scan_lookback_days,
        "daily_scan_failure_threshold": settings.daily_scan_failure_threshold,
    })

st.subheader("股票池范围")
source_options = ["all_a", "hs300", "zz500", "zz1000"]
universe_source = st.selectbox(
    "股票池来源",
    source_options,
    index=source_options.index(settings.universe_source) if settings.universe_source in source_options else 0,
    help="all_a=全A；hs300/zz500/zz1000 为指数成分池",
)
max_stocks = st.number_input("最大股票数", min_value=1, max_value=8000, value=int(settings.max_stocks), step=100)

st.subheader("过滤条件")
c1, c2 = st.columns(2)
with c1:
    exclude_st = st.checkbox("排除 ST / *ST", value=bool(settings.exclude_st))
    exclude_delisting = st.checkbox("排除退市整理", value=bool(settings.exclude_delisting))
    exclude_suspended = st.checkbox("排除停牌/无成交", value=bool(settings.exclude_suspended))
with c2:
    exclude_bj = st.checkbox("排除北交所", value=bool(settings.exclude_bj))
    exclude_new_stock_days = st.number_input("排除上市不足 N 天", min_value=0, max_value=1000, value=int(settings.exclude_new_stock_days), step=10)

c3, c4 = st.columns(2)
with c3:
    min_float_mv_yi = st.number_input("最小流通市值(亿元)", min_value=0.0, value=float(settings.min_float_mv_yi), step=5.0)
with c4:
    max_mv_default = float(settings.max_float_mv_yi or 0.0)
    max_float_mv_yi_raw = st.number_input("最大流通市值(亿元，0=不限)", min_value=0.0, value=max_mv_default, step=50.0)
min_avg_turnover = st.number_input("最小日均换手率(%)", min_value=0.0, value=float(settings.min_avg_turnover), step=0.1)
max_float_mv_yi = None if max_float_mv_yi_raw <= 0 else max_float_mv_yi_raw

stock_overrides = {
    "universe_source": universe_source,
    "max_stocks": int(max_stocks),
    "exclude_st": bool(exclude_st),
    "exclude_delisting": bool(exclude_delisting),
    "exclude_suspended": bool(exclude_suspended),
    "exclude_new_stock_days": int(exclude_new_stock_days),
    "exclude_bj": bool(exclude_bj),
    "min_float_mv_yi": float(min_float_mv_yi),
    "max_float_mv_yi": max_float_mv_yi,
    "min_avg_turnover": float(min_avg_turnover),
}

st.subheader("预览")
if st.button("🔍 预览股票池", use_container_width=True):
    with st.spinner("正在构建股票池..."):
        try:
            df = preview_universe(stock_overrides)
            st.success(f"筛选后股票数：{len(df)}")
            cols = [c for c in ["symbol", "name", "close", "pct_change", "float_mv", "turnover_rate", "list_date"] if c in df.columns]
            if cols:
                st.dataframe(df[cols].head(50), use_container_width=True, hide_index=True)
            else:
                st.dataframe(df.head(50), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"预览失败：{e}")

st.subheader("每日扫描")
s1, s2 = st.columns(2)
with s1:
    daily_scan_enabled = st.checkbox("启用每日扫描", value=bool(settings.daily_scan_enabled))
    daily_scan_hour = st.number_input("扫描小时", min_value=0, max_value=23, value=int(settings.daily_scan_hour), step=1)
with s2:
    daily_scan_minute = st.number_input("扫描分钟", min_value=0, max_value=59, value=int(settings.daily_scan_minute), step=1)
    daily_scan_lookback_days = st.number_input("扫描回看天数", min_value=1, max_value=1000, value=int(settings.daily_scan_lookback_days), step=10)

data_source_order = st.text_input("数据源降级顺序", value=settings.data_source_order, help="逗号分隔，例如 tencent,tushare,akshare")
daily_scan_failure_threshold = st.number_input("失败告警阈值", min_value=1, max_value=1000, value=int(settings.daily_scan_failure_threshold), step=1)

scan_values = {
    "data_source_order": data_source_order.strip(),
    "daily_scan_enabled": bool(daily_scan_enabled),
    "daily_scan_hour": int(daily_scan_hour),
    "daily_scan_minute": int(daily_scan_minute),
    "daily_scan_lookback_days": int(daily_scan_lookback_days),
    "daily_scan_failure_threshold": int(daily_scan_failure_threshold),
}

st.divider()
if st.button("💾 保存到 .env", type="primary", use_container_width=True):
    try:
        update_env_file({**stock_overrides, **scan_values})
        st.success("已保存到 .env")
        st.warning("⚠️ 配置修改后不会立刻影响已运行进程，需要重启看板和调度服务后才会完全生效。")
        st.code("systemctl restart quant-stock-picker.service quant-scheduler.service", language="bash")
        st.caption("重启后：看板会读取新的股票选择配置；每日扫描调度器会按新的股票池/时间/数据源顺序运行。")
    except Exception as e:
        st.error(f"保存失败：{e}")

st.info("提示：如果只是临时验证筛选结果，点“预览股票池”即可；只有点“保存到 .env”才会修改持久配置。保存后记得重启服务。")
