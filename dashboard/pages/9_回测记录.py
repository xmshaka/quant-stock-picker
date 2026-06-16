"""历史回测记录页面 — 读取 data/backtest_runs 落盘结果。"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import pandas as pd
import streamlit as st

from backtest.records import (
    LIQUIDITY_BUCKET_LABELS,
    delete_backtest_run,
    list_backtest_runs,
    load_backtest_run,
    summarize_liquidity_slippage,
    validate_trade_schema,
)
from backtest.scheme_backtest import _fetch_ohlcv
from dashboard.components.kline_chart import plot_equity_curve, plot_kline_with_signals, render_kline_chart
from dashboard.history_compare import (
    build_run_compare_table,
    best_run_summary,
    compare_table_to_csv,
    format_compare_table,
    plot_multi_equity_curves,
)
from dashboard.history_filters import filter_backtest_runs, unique_non_empty
from dashboard.history_state import ensure_history_state, reset_history_state, sync_history_state, valid_default, valid_default_list
from signals.rules import TradePoint
from theme import inject_theme, metric_row, section_header, empty_state, C

st.set_page_config(page_title="回测记录", page_icon="🧾", layout="wide")
inject_theme()


def _record_table_height(row_count: int) -> int:
    """历史列表高度自适应：少量记录全展开，过多记录保留内部滚动条。"""
    header_height = 38
    row_height = 35
    padding = 8
    min_height = 180
    max_height = 620
    height = header_height + max(int(row_count), 1) * row_height + padding
    return min(max(height, min_height), max_height)

section_header("历史回测记录")
st.caption("读取 data/backtest_runs/*：metrics/config/trades/equity/signals，用于复盘和审计。")
ensure_history_state(st.session_state)


runs = list_backtest_runs()
if runs.empty:
    empty_state("🧾", "暂无回测记录，请先在策略回测页持久化一次结果")
    st.stop()

# ── 筛选 ──
with st.expander("筛选回测记录", expanded=True):
    reset_col, hint_col = st.columns([1, 4])
    with reset_col:
        if st.button("重置筛选", width="stretch"):
            reset_history_state(st.session_state)
            st.rerun()
    with hint_col:
        st.caption("筛选条件会在刷新/切页后保留；如列表为空或记录变少，可一键重置。")
    f1, f2, f3, f4 = st.columns([2, 1.4, 1.2, 1.2])
    with f1:
        symbol_query = st.text_input(
            "股票代码 / Run ID",
            value=st.session_state.history_symbol_query,
            placeholder="如 002145 / 5156150 / 20260613",
        )
    with f2:
        scheme_options = ["全部"] + unique_non_empty(runs.get("scheme_name", []))
        scheme_default = valid_default(st.session_state.history_scheme_filter, scheme_options, "全部")
        scheme_filter = st.selectbox("策略方案", scheme_options, index=scheme_options.index(scheme_default))
    with f3:
        pool_options = ["全部"] + unique_non_empty(runs.get("pool_mode", []))
        pool_default = valid_default(st.session_state.history_pool_filter, pool_options, "全部")
        pool_filter = st.selectbox("股票池", pool_options, index=pool_options.index(pool_default))
    with f4:
        recent_options = [0, 1, 3, 7, 30]
        recent_default = valid_default(st.session_state.history_recent_days, recent_options, 0)
        recent_days = st.selectbox("保存时间", recent_options, format_func=lambda x: "全部" if x == 0 else f"最近{x}天", index=recent_options.index(recent_default))

sync_history_state(
    st.session_state,
    history_symbol_query=symbol_query,
    history_scheme_filter=scheme_filter,
    history_pool_filter=pool_filter,
    history_recent_days=recent_days,
)

filtered_runs = filter_backtest_runs(
    runs,
    symbol_query=symbol_query,
    scheme_name=scheme_filter,
    pool_mode=pool_filter,
    recent_days=recent_days,
)
if filtered_runs.empty:
    empty_state("🔎", "没有匹配的回测记录，请调整筛选条件")
    st.stop()

# ── 列表 ──
display = filtered_runs.copy()
for col in ("total_return", "annual_return", "max_drawdown", "win_rate"):
    display[col] = display[col].map(lambda x: f"{float(x):+.2%}" if col in ("total_return", "annual_return") else f"{float(x):.2%}")
display["sharpe_ratio"] = display["sharpe_ratio"].map(lambda x: f"{float(x):.3f}")
display["交易"] = display.apply(lambda r: f"买{int(r['buy_count'])} 卖{int(r['sell_count'])}（{int(r['trade_count'])}轮）", axis=1)
display["一致性"] = display["consistency_ok"].map(lambda x: "PASS" if x else "FAIL")

# 确保数据源字段存在（兼容旧记录）
for col in ["data_source", "data_adjust", "data_version"]:
    if col not in display.columns:
        display[col] = ""

# 数据源格式化
def format_data_source(row):
    source = row["data_source"] if pd.notna(row["data_source"]) else ""
    adjust = row["data_adjust"] if pd.notna(row["data_adjust"]) else "raw"
    version = row["data_version"] if pd.notna(row["data_version"]) else ""
    if not source:
        return "未知"
    base = f"{source}/{adjust}"
    # 如果版本信息是默认格式，不显示
    default_version = f"source={source}, adjust={adjust}"
    if version and version != default_version and not version.startswith(default_version + ","):
        # 简化显示
        if len(version) > 20:
            version = version[:17] + "..."
        return f"{base} ({version})"
    return base

display["数据源"] = display.apply(format_data_source, axis=1)

show_cols = {
    "run_id": "Run ID",
    "scheme_name": "策略",
    "pool_mode": "股票池",
    "symbols": "代码",
    "start_date": "开始",
    "end_date": "结束",
    "total_return": "总收益",
    "annual_return": "年化",
    "max_drawdown": "最大回撤",
    "win_rate": "胜率",
    "sharpe_ratio": "夏普",
    "交易": "交易",
    "一致性": "一致性",
    "数据源": "数据源",
    "created_at": "保存时间",
}
st.dataframe(
    display[list(show_cols.keys())].rename(columns=show_cols),
    width="stretch",
    hide_index=True,
    height=_record_table_height(len(display)),
)
st.caption(f"筛选后 {len(display)} / 全部 {len(runs)} 条回测记录；列表高度随记录数自适应，超过可视上限后可在表格内滚动。")

# ── 多 run 横向对比 ──
with st.expander("多记录横向对比", expanded=False):
    default_compare = filtered_runs["run_id"].head(min(3, len(filtered_runs))).tolist()
    persisted_compare = valid_default_list(st.session_state.history_compare_run_ids, filtered_runs["run_id"].tolist())
    compare_run_ids = st.multiselect(
        "选择要对比的回测记录",
        filtered_runs["run_id"].tolist(),
        default=persisted_compare or default_compare,
        help="基于当前筛选结果选择多个 Run，横向比较收益、回撤、夏普、交易次数。",
    )
    sync_history_state(st.session_state, history_compare_run_ids=compare_run_ids)
    compare_table = build_run_compare_table(filtered_runs, compare_run_ids)
    if compare_table.empty:
        empty_state("📊", "请选择至少一条回测记录进行对比")
    else:
        compare_summary = best_run_summary(compare_table)
        metric_row([
            {"label": "对比数量", "value": f"{compare_summary.get('count', 0)}条"},
            {"label": "最高收益", "value": f"{float(compare_summary.get('best_return', 0)):+.2%}", "color": "green" if float(compare_summary.get('best_return', 0)) > 0 else "red"},
            {"label": "最低回撤", "value": f"{float(compare_summary.get('best_drawdown', 0)):.2%}"},
            {"label": "最高夏普", "value": f"{float(compare_summary.get('best_sharpe', 0)):.3f}"},
        ], cols=4)
        st.caption(
            f"最高收益: {compare_summary.get('best_return_run_id', '-')}；"
            f"最低回撤: {compare_summary.get('best_drawdown_run_id', '-')}；"
            f"最高夏普: {compare_summary.get('best_sharpe_run_id', '-')}"
        )
        
        # 数据源统计
        if "data_source" in compare_table.columns and "data_adjust" in compare_table.columns:
            # 确保字段存在并填充默认值
            compare_table["data_source"] = compare_table.get("data_source", "")
            compare_table["data_adjust"] = compare_table.get("data_adjust", "raw")
            
            source_stats = compare_table.groupby(["data_source", "data_adjust"]).size().reset_index(name="count")
            if not source_stats.empty:
                source_text = "，".join([f"{row['data_source']}/{row['data_adjust']}×{row['count']}" 
                                       for _, row in source_stats.iterrows()])
                st.caption(f"数据源分布: {source_text}")
        
        st.download_button(
            "导出对比CSV",
            data=compare_table_to_csv(compare_table),
            file_name="backtest_run_compare.csv",
            mime="text/csv",
            width="stretch",
        )
        st.dataframe(format_compare_table(compare_table), width="stretch", hide_index=True)
        equity_map = {}
        for cmp_run_id in compare_table["run_id"].tolist():
            try:
                cmp_run = load_backtest_run(cmp_run_id)
                if not cmp_run["equity"].empty:
                    equity_map[cmp_run_id] = cmp_run["equity"]
            except Exception:
                continue
        if equity_map:
            st.plotly_chart(plot_multi_equity_curves(equity_map), width="stretch", key="history_compare_equity")
        else:
            empty_state("📈", "选中记录暂无权益曲线数据")

c1, c2, c3 = st.columns([3, 1, 1])
with c1:
    run_id = st.selectbox("选择回测记录", filtered_runs["run_id"].tolist(), index=0)
with c2:
    if st.button("刷新列表", width="stretch"):
        st.rerun()
with c3:
    delete_confirm = st.checkbox("确认删除", key=f"delete_confirm_{run_id}")
    if st.button("删除记录", type="secondary", width="stretch", disabled=not delete_confirm):
        try:
            dst = delete_backtest_run(run_id, trash=True)
            st.success(f"已移入回收站: {dst}")
            st.rerun()
        except Exception as e:
            st.error(f"删除失败: {e}")

run = load_backtest_run(run_id)
metrics = run["metrics"]
config = run["config"]
trades = run["trades"]
equity = run["equity"]
signals_executed = run["signals_executed"]
signals_raw = run["signals_raw"]

section_header("记录详情", f"{run_id} · {config.get('scheme_name', '')}")
metric_row([
    {"label": "总收益", "value": f"{float(metrics.get('total_return', 0)):+.2%}", "color": "green" if float(metrics.get('total_return', 0)) > 0 else "red"},
    {"label": "年化", "value": f"{float(metrics.get('annual_return', 0)):+.2%}", "color": "green" if float(metrics.get('annual_return', 0)) > 0 else "red"},
    {"label": "最大回撤", "value": f"{float(metrics.get('max_drawdown', 0)):.2%}", "color": "red" if float(metrics.get('max_drawdown', 0)) > 0.15 else "yellow"},
    {"label": "胜率", "value": f"{float(metrics.get('win_rate', 0)):.0%}"},
    {"label": "夏普", "value": f"{float(metrics.get('sharpe_ratio', 0)):.3f}"},
    {"label": "交易", "value": f"{int(metrics.get('trade_count', 0))}轮"},
], cols=6)

schema_check = validate_trade_schema(trades)
consistency = metrics.get("consistency", {}) if isinstance(metrics.get("consistency", {}), dict) else {}
if schema_check["ok"] and consistency.get("ok", False):
    st.success("✅ trades schema 与成交/K线一致性均通过", icon="🧾")
else:
    st.warning(f"⚠️ 审计检查异常: schema={schema_check}, consistency={consistency}")

tab_summary, tab_trades, tab_liquidity, tab_equity, tab_kline, tab_raw = st.tabs(["概览", "交易流水", "滑点审计", "权益曲线", "K线复盘", "配置/报告"])

with tab_summary:
    c1, c2, c3 = st.columns(3)
    c1.write("**配置**")
    c1.json(config)
    c2.write("**一致性**")
    c2.json(consistency)
    c3.write("**文件**")
    c3.code(str(run["path"]), language=None)

with tab_trades:
    if trades.empty:
        empty_state("🧾", "trades 为空")
    else:
        display_trades = trades.copy()
        if "liquidity_bucket" in display_trades.columns:
            display_trades["流动性分层"] = display_trades["liquidity_bucket"].fillna("").astype(str).map(lambda x: LIQUIDITY_BUCKET_LABELS.get(x, x))
        for col in ["exec_price", "amount", "commission", "stamp_duty", "transfer_fee", "slippage", "slippage_rate", "turnover_amount", "avg_cost", "pnl"]:
            if col in display_trades.columns:
                display_trades[col] = pd.to_numeric(display_trades[col], errors="coerce").round(4)
        preferred_cols = [
            "symbol", "date", "action", "event_type", "exec_price", "shares", "amount",
            "slippage_rate", "slippage", "流动性分层", "turnover_amount",
            "commission", "stamp_duty", "transfer_fee", "avg_cost", "pnl", "pnl_pct", "reason",
        ]
        preferred_cols = [c for c in preferred_cols if c in display_trades.columns]
        rest_cols = [c for c in display_trades.columns if c not in preferred_cols]
        display_trades = display_trades[preferred_cols + rest_cols]
        st.dataframe(display_trades, width="stretch", hide_index=True)
        st.caption(f"共 {len(trades)} 笔成交；字段数 {len(trades.columns)}")

with tab_liquidity:
    summary = summarize_liquidity_slippage(trades)
    if not summary.get("ok"):
        empty_state("🧾", "暂无滑点审计数据")
    else:
        if summary.get("is_legacy_audit"):
            missing_cols = ", ".join(summary.get("missing_audit_columns", []))
            st.warning(
                f"旧口径记录：缺少新版流动性审计字段 {missing_cols}。"
                "当前仅按已有成交数据做兼容展示；如需准确分层滑点，请重新运行回测生成新版记录。",
                icon="⚠️",
            )
        metric_row([
            {"label": "成交笔数", "value": f"{summary['rows']}笔"},
            {"label": "成交额", "value": f"¥{summary['total_amount']:,.0f}"},
            {"label": "滑点成本", "value": f"¥{summary['total_slippage']:,.0f}", "color": "red" if summary['total_slippage'] > 0 else ""},
            {"label": "加权滑点率", "value": f"{summary['weighted_slippage_rate']:.4%}", "color": "red" if summary['weighted_slippage_rate'] > 0.005 else "yellow"},
        ], cols=4)
        buckets = summary["buckets"].copy()
        for col in ["成交额", "滑点成本", "平均市场成交额"]:
            if col in buckets.columns:
                buckets[col] = pd.to_numeric(buckets[col], errors="coerce").round(2)
        if "加权滑点率" in buckets.columns:
            buckets["加权滑点率"] = pd.to_numeric(buckets["加权滑点率"], errors="coerce").map(lambda x: f"{x:.4%}")
        st.dataframe(buckets, width="stretch", hide_index=True)
        st.caption("分层规则：>5亿=0.2%，1亿–5亿=0.5%，<1亿=1.0%；成交额缺失回退默认滑点。")

with tab_equity:
    if equity.empty:
        empty_state("📈", "equity 为空")
    else:
        eq_dict = dict(zip(equity["date"].astype(str), pd.to_numeric(equity["equity"], errors="coerce")))
        fig = plot_equity_curve(eq_dict, title=f"{run_id} 权益曲线")
        st.plotly_chart(fig, width="stretch", key=f"history_eq_{run_id}")
        st.dataframe(equity, width="stretch", hide_index=True)

with tab_kline:
    if signals_executed.empty:
        empty_state("📉", "signals_executed 为空，无法复盘K线")
    else:
        symbols = sorted(signals_executed["symbol"].dropna().unique().tolist())
        sym = st.selectbox("选择股票", symbols, key=f"history_symbol_{run_id}")
        sym_sigs = signals_executed[signals_executed["symbol"] == sym].copy()
        points = []
        for _, row in sym_sigs.iterrows():
            try:
                points.append(TradePoint(
                    date=pd.Timestamp(row.get("date") or row.get("exec_date")).date(),
                    action=str(row.get("action", "")),
                    reason=str(row.get("reason", "")),
                    confidence=float(row.get("confidence", 1.0) or 1.0),
                    price=float(row.get("exec_price", row.get("price", 0.0)) or 0.0),
                    rule_name=str(row.get("rule_name", "历史成交")),
                    exec_price=float(row.get("exec_price", row.get("price", 0.0)) or 0.0),
                    shares=int(row.get("shares", 0) or 0),
                    cash_after=float(row.get("cash_after", 0.0) or 0.0),
                    position_shares=int(row.get("position_after", row.get("position_shares", 0)) or 0),
                    avg_cost=float(row.get("avg_cost", 0.0) or 0.0),
                    pnl=float(row.get("pnl", 0.0) or 0.0),
                    pnl_pct=float(row.get("pnl_pct", 0.0) or 0.0),
                    holding_days=int(row.get("holding_days", 0) or 0),
                ))
            except Exception:
                continue
        lookback_days = int(config.get("lookback_days", 80) or 80)
        bars = _fetch_ohlcv([sym], max(lookback_days, 80))
        if bars.empty:
            empty_state("📉", f"{sym} K线数据为空")
        else:
            bars["trade_date"] = pd.to_datetime(bars["trade_date"])
            fig = plot_kline_with_signals(bars[bars["symbol"] == sym], points, symbol=sym, show_ma=True, show_volume=True, show_kdj=True)
            render_kline_chart(fig, key=f"history_kline_{run_id}_{sym}", height=760)
            st.dataframe(sym_sigs, width="stretch", hide_index=True)

with tab_raw:
    st.write("**报告**")
    st.markdown(run.get("report", ""))
    with st.expander("signals_executed"):
        st.dataframe(signals_executed, width="stretch", hide_index=True)
    with st.expander("signals_raw"):
        st.dataframe(signals_raw, width="stretch", hide_index=True)
