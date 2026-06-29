"""参数网格结果页面。"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import streamlit as st

from dashboard.param_grid_view import (
    format_grid_results_for_display,
    list_grid_audit_runs,
    load_grid_audit_run,
    summarize_grid_run,
)
from theme import inject_theme, metric_row, section_header, empty_state

st.set_page_config(page_title="参数网格", page_icon="🧪", layout="wide")
inject_theme()

section_header("参数网格结果")
st.warning("⚠️ **CLI-Only**：参数网格优化仅通过 CLI 运行 `scripts/run_param_grid.py --audit-root data/grid_results`，本页面仅用于**只读浏览**历史结果，不可在线编辑或发起新实验。", icon="🧪")
st.caption("读取 data/grid_results/*：grid_results/config/summary，用于参数稳定性审计。排序口径：低回撤优先，不按收益单独优化。")

runs = list_grid_audit_runs()
if runs.empty:
    empty_state("🧪", "暂无参数网格实验，请先运行 scripts/run_param_grid.py --audit-root data/grid_results")
    st.stop()

# ── 实验列表 ──
section_header("实验列表")
display = runs.copy()
for col in ["best_return", "best_drawdown"]:
    if col in display.columns:
        display[col] = display[col].map(lambda x: f"{float(x):.4%}")
if "initial_capital" in display.columns:
    display["initial_capital"] = display["initial_capital"].map(lambda x: f"{float(x):,.0f}")
show_cols = [
    "run_id", "strategy_id", "created_at", "max_runs", "lookback_days", "top_n",
    "initial_capital", "row_count", "eligible_count", "best_return", "best_drawdown",
]
st.dataframe(display[[c for c in show_cols if c in display.columns]], width="stretch", hide_index=True)

options = runs["run_id"].tolist()
selected = st.selectbox("选择实验", options, index=0)
run_path = runs.loc[runs["run_id"] == selected, "path"].iloc[0]
loaded = load_grid_audit_run(run_path)
config = loaded["config"]
results = loaded["results"]

section_header("实验摘要", selected)
summary_items = summarize_grid_run(config, results)
metric_row([
    {"label": k, "value": v} for k, v in summary_items.items()
], cols=5)

with st.expander("审计配置 config.json", expanded=False):
    st.json(config)

# ── 结果表 ──
section_header("结果表", "低回撤优先")
if results.empty:
    empty_state("📉", "该实验没有结果行")
else:
    st.dataframe(format_grid_results_for_display(results), width="stretch", hide_index=True)

# ── 摘要 Markdown ──
with st.expander("summary.md", expanded=True):
    if loaded["summary"]:
        st.markdown(loaded["summary"])
    else:
        empty_state("📝", "该实验暂无 summary.md")
