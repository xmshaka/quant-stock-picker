"""策略回测页面"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import streamlit as st
import pandas as pd

from data_loader import load_data, FACTOR_NAME_MAP, NAME_MAP
from strategy.registry import SchemeRegistry
from strategy.schemes import StrategyScheme
from backtest.scheme_backtest import SchemeBacktester, run_multi_scheme_backtest
from backtest.records import BacktestRunConfig, load_backtest_run, summarize_liquidity_slippage
from dashboard.backtest_state import backtest_context_signature, clear_stale_compare
from dashboard.components.kline_chart import plot_kline_with_signals, plot_equity_curve
from signals.rules import TradePoint
from theme import inject_theme, metric_row, section_header, badge, empty_state, C

st.set_page_config(page_title="策略回测", page_icon="📈", layout="wide")
inject_theme()

# ========== 数据 ==========
@st.cache_data(ttl=300, show_spinner=False)
def get_data(n_stocks=200):
    return load_data(data_source="real", n_stocks=n_stocks, n_days=252)

registry = SchemeRegistry()
schemes = registry.list_all()
scheme_names = {s.name: s for s in schemes}


def _restore_widget_state(widget_key: str, durable_key: str, default):
    """跨页面保留参数：Streamlit 会清理未渲染 widget key，另存 durable key。"""
    if durable_key not in st.session_state:
        st.session_state[durable_key] = default
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state[durable_key]


def _save_widget_state(widget_key: str, durable_key: str):
    st.session_state[durable_key] = st.session_state.get(widget_key)


_restore_widget_state("bt_scheme_name", "bt_pref_scheme_name", list(scheme_names.keys())[0])
_restore_widget_state("bt_top_n", "bt_pref_top_n", 10)
_restore_widget_state("bt_lookback", "bt_pref_lookback", 60)
_restore_widget_state("bt_capital_wan", "bt_pref_capital_wan", 100)
_restore_widget_state("bt_pool_mode", "bt_pref_pool_mode", "全A")
_restore_widget_state("bt_custom_codes", "bt_pref_custom_codes", "")

section_header("策略回测")
st.caption("选择方案 → 选股池 → 回测区间 → 查看绩效和K线买卖点")

# ========== 参数 ==========
c1, c2, c3, c4 = st.columns(4)
with c1:
    selected_name = st.selectbox(
        "策略方案", list(scheme_names.keys()),
        index=list(scheme_names.keys()).index(st.session_state.bt_scheme_name) if st.session_state.bt_scheme_name in scheme_names else 0,
        key="bt_scheme_name",
        on_change=_save_widget_state,
        args=("bt_scheme_name", "bt_pref_scheme_name"),
    )
    selected_scheme = scheme_names[selected_name]
with c2:
    top_n = st.slider("选股数量", 3, 30, key="bt_top_n", on_change=_save_widget_state, args=("bt_top_n", "bt_pref_top_n"))
with c3:
    lookback = st.slider("回测天数", 20, 120, key="bt_lookback", on_change=_save_widget_state, args=("bt_lookback", "bt_pref_lookback"))
with c4:
    capital = st.number_input("初始资金(万)", 10, 1000, step=10, key="bt_capital_wan", on_change=_save_widget_state, args=("bt_capital_wan", "bt_pref_capital_wan")) * 10000

# 股票池选择
pool_mode = st.radio(
    "股票池", ["全A", "自定义代码", "观察池", "持仓池"],
    horizontal=True, key="bt_pool_mode",
    on_change=_save_widget_state, args=("bt_pool_mode", "bt_pref_pool_mode"),
)
custom_symbols = None
if pool_mode == "自定义代码":
    code_input = st.text_input(
        "股票代码（逗号分隔）", placeholder="600519,000001,002594", key="bt_custom_codes",
        on_change=_save_widget_state, args=("bt_custom_codes", "bt_pref_custom_codes"),
    )
    if code_input:
        custom_symbols = [s.strip() for s in code_input.split(",") if s.strip()]
elif pool_mode == "观察池":
    from signals.portfolio import PortfolioManager
    pm = PortfolioManager("main")
    custom_symbols = [item.symbol for item in pm.watch_list]
    if not custom_symbols:
        st.warning("观察池为空")
elif pool_mode == "持仓池":
    from signals.portfolio import PortfolioManager
    pm = PortfolioManager("main")
    custom_symbols = [item.symbol for item in pm.hold_list]
    if not custom_symbols:
        st.warning("持仓池为空")

current_context_signature = backtest_context_signature(pool_mode, custom_symbols, lookback, top_n, capital)
clear_stale_compare(st.session_state, current_context_signature)

# ========== 执行回测 ==========
if st.button("▶ 运行回测", type="primary", width="stretch"):
    factor_df, price_df, factor_names = get_data(n_stocks=300)

    progress_bar = st.progress(0)
    status_text = st.empty()

    def _update_progress(step, total, msg=""):
        progress_bar.progress(min(step / total, 1.0))
        if msg:
            status_text.caption(msg)

    backtester = SchemeBacktester()
    result = backtester.run(
        scheme=selected_scheme,
        factor_df=factor_df,
        price_df=price_df,
        factor_names=factor_names,
        symbols=custom_symbols,
        lookback_days=lookback,
        top_n=top_n,
        initial_capital=capital,
        progress_callback=_update_progress,
    )

    progress_bar.progress(1.0)
    status_text.caption("✅ 回测完成")
    st.session_state.bt_result = result
    st.session_state.bt_result_signature = current_context_signature
    # FIX: 股票池/参数已发生新回测时，旧“方案对比”结果不再可信，必须清空。
    st.session_state.pop("bt_compare", None)
    st.session_state.pop("bt_compare_signature", None)
    st.session_state.bt_run_config = BacktestRunConfig(
        run_id=result.run_id,
        scheme_id=selected_scheme.scheme_id,
        scheme_name=selected_scheme.name,
        start_date=result.start_date,
        end_date=result.end_date,
        lookback_days=lookback,
        top_n=top_n,
        initial_capital=capital,
        pool_mode=pool_mode,
        symbols=custom_symbols or [],
        cost={
            "commission": 0.00025,
            "stamp_duty": 0.001,
            "transfer_fee": 0.00001,
            "slippage": 0.002,
        },
        risk={"single_position_cap": 0.20, "total_position_cap": 0.90},
    )
    try:
        run_dir = result.persist(config=st.session_state.bt_run_config)
        st.session_state.bt_last_saved_dir = str(run_dir)
        status_text.caption(f"✅ 回测完成并已自动保存: {run_dir.name}")
    except Exception as e:
        st.session_state.bt_last_saved_dir = ""
        st.warning(f"回测完成，但自动保存失败: {e}")

# ========== 展示结果 ==========
if "bt_result" in st.session_state:
    result = st.session_state.bt_result

    # 绩效指标
    section_header("回测绩效", f"{result.scheme_name} · {result.run_id}")

    # P0: 一致性校验提示
    from backtest.records import validate_backtest_consistency
    consistency = validate_backtest_consistency(result)
    if consistency.get('ok'):
        st.success(f"✅ 事件源一致: 买{consistency['buy_points']}点=卖{consistency['sell_points']}点=明细{consistency['trade_detail_rows']}行", icon="📊")
    else:
        st.warning(f"⚠️ 事件源不一致: result.buy={result.buy_count} vs K线买={consistency['buy_points']} vs 明细={consistency['trade_detail_rows']}")

    metric_row([
        {"label": "总收益", "value": result.fmt('total_return'), "color": "green" if result.total_return > 0 else "red"},
        {"label": "年化收益", "value": result.fmt('annual_return'), "color": "green" if result.annual_return > 0 else "red"},
        {"label": "夏普比率", "value": result.fmt('sharpe_ratio'), "color": "green" if result.sharpe_ratio > 0.5 else "red" if result.sharpe_ratio < 0 else ""},
        {"label": "最大回撤", "value": result.fmt('max_drawdown'), "color": "red" if result.max_drawdown > 0.15 else "yellow" if result.max_drawdown > 0.08 else "green"},
        {"label": "胜率", "value": result.fmt('win_rate'), "color": "green" if result.win_rate > 0.5 else "red" if result.trade_count > 0 else ""},
        {"label": "交易", "value": f"{result.trade_count}轮"},
    ], cols=6)

    # 持久化按钮
    if st.session_state.get("bt_last_saved_dir"):
        st.caption(f"最近自动保存: {st.session_state.bt_last_saved_dir}")
    if st.button("💾 持久化回测结果到 parquet", width="stretch", type="secondary"):
        try:
            config = st.session_state.get("bt_run_config")
            if config is not None:
                config.run_id = result.run_id
            run_dir = result.persist(config=config)
            st.success(f"已保存至 {run_dir}")
        except Exception as e:
            st.error(f"持久化失败: {e}")

    # 权益曲线
    if result.equity_curve:
        section_header("权益曲线")
        fig_eq = plot_equity_curve(result.equity_curve, title=f"{result.scheme_name} 权益曲线")
        st.plotly_chart(fig_eq, width="stretch")

    # P1: 流动性分层滑点审计
    if result.trade_details:
        section_header("流动性滑点审计")
        audit_trades = pd.DataFrame(result.trade_details)
        audit_source = "内存回测结果"
        saved_dir = st.session_state.get("bt_last_saved_dir", "")
        if saved_dir:
            try:
                saved_run = load_backtest_run(result.run_id)
                if not saved_run.get("trades", pd.DataFrame()).empty:
                    audit_trades = saved_run["trades"]
                    audit_source = "已保存 trades.parquet"
            except Exception:
                audit_source = "内存回测结果（保存文件读取失败）"
        liq_summary = summarize_liquidity_slippage(audit_trades)
        if liq_summary.get("ok"):
            metric_row([
                {"label": "滑点成本", "value": f"¥{liq_summary['total_slippage']:,.0f}", "color": "red" if liq_summary['total_slippage'] > 0 else ""},
                {"label": "加权滑点率", "value": f"{liq_summary['weighted_slippage_rate']:.4%}", "color": "red" if liq_summary['weighted_slippage_rate'] > 0.005 else "yellow"},
                {"label": "成交额", "value": f"¥{liq_summary['total_amount']:,.0f}"},
                {"label": "成交笔数", "value": f"{liq_summary['rows']}笔"},
            ], cols=4)
            st.caption(f"滑点审计数据源：{audit_source}")
            liq_buckets = liq_summary["buckets"].copy()
            for col in ["成交额", "滑点成本", "平均市场成交额"]:
                if col in liq_buckets.columns:
                    liq_buckets[col] = pd.to_numeric(liq_buckets[col], errors="coerce").round(2)
            if "加权滑点率" in liq_buckets.columns:
                liq_buckets["加权滑点率"] = pd.to_numeric(liq_buckets["加权滑点率"], errors="coerce").map(lambda x: f"{x:.4%}")
            st.dataframe(liq_buckets, width="stretch", hide_index=True)

    # P0: K线默认展示 signals_executed，提供原始信号叠加选项
    show_points = result.signals_executed if result.signals_executed else result.stock_signals
    overlay_raw = False
    if result.signals_raw and show_points:
        overlay_raw = st.checkbox("叠加原始规则信号（仅参考，不参与统计）", value=False, key="overlay_raw")

    # K线买卖点
    if show_points:
        section_header("个股K线买卖点（实际成交）", f"({len(show_points)} 只)")

        # 拉取完整 OHLCV 供 K 线图使用
        from backtest.scheme_backtest import _fetch_ohlcv
        ohlcv_df = _fetch_ohlcv(list(show_points.keys()), lookback)
        have_ohlcv = not ohlcv_df.empty
        if have_ohlcv:
            ohlcv_df['trade_date'] = pd.to_datetime(ohlcv_df['trade_date'])
        # 回退：用 price_df（仅 close）
        factor_df, price_df, factor_names = get_data(n_stocks=300)

        for sym, points in sorted(show_points.items()):
            # P0: 若勾选原始规则信号叠加层
            raw_points = result.signals_raw.get(sym, []) if overlay_raw else []
            sym_name = NAME_MAP.get(sym, "")
            label = f"{sym} {sym_name}" if sym_name else sym
            buy_count = len([p for p in points if p.action == "BUY"])
            sell_count = len([p for p in points if p.action == "SELL"])

            with st.expander(f"**{label}** — 买{buy_count} 卖{sell_count}", expanded=False):
                if have_ohlcv:
                    sym_bars = ohlcv_df[ohlcv_df['symbol'] == sym].copy()
                else:
                    sym_bars = price_df[price_df['symbol'] == sym].copy()
                if not sym_bars.empty:
                    # 确保有 OHLCV 列（只补 close 的近似值，volume 不能用 close 代替）
                    for col in ('open', 'high', 'low'):
                        if col not in sym_bars.columns:
                            sym_bars[col] = sym_bars['close']
                    if 'volume' not in sym_bars.columns:
                        # 无成交量数据时隐藏成交量子图
                        show_vol = False
                    else:
                        show_vol = True
                    all_points = points + raw_points
                    fig = plot_kline_with_signals(
                        sym_bars, all_points, symbol=label,
                        show_ma=True, show_volume=show_vol, show_rsi=False, show_kdj=True,
                    )
                    st.plotly_chart(fig, width="stretch", key=f"kline_bs_v7_lane_line_{sym}")

                    # P0: 买卖点明细 — 默认显示执行信号，可选叠加强调raw
                    if all_points:
                        pts_data = []
                        for p in all_points:
                            row_meta = ""
                            if p in raw_points and p not in points:
                                row_meta = badge("原始", "regime")
                            elif p in points:
                                row_meta = badge("执行", "hold")
                            row = {
                                "日期": str(p.date),
                                "来源": row_meta,
                                "方向": badge("买", "buy") if p.action == "BUY" else badge("卖", "sell"),
                                "置信度": f"{p.confidence:.0%}",
                                "价格": f"{p.price:.2f}",
                                "原因": p.reason,
                                "规则": p.rule_name,
                            }
                            if p.action == 'BUY':
                                ps = getattr(p, 'position_shares', 0)
                                ca = getattr(p, 'cash_after', 0)
                                sl = getattr(p, 'stop_loss', 0)
                                tp = getattr(p, 'take_profit', 0)
                                row["持股"] = f"{ps:,}" if ps else ""
                                row["余额"] = f"¥{ca:,.0f}" if ca else ""
                                row["止损"] = f"{sl:.2f}" if sl else ""
                                row["止盈"] = f"{tp:.2f}" if tp else ""
                            elif p.action == 'SELL':
                                ca = getattr(p, 'cash_after', 0)
                                pnl = getattr(p, 'pnl', 0)
                                pnl_pct = getattr(p, 'pnl_pct', 0)
                                hd = getattr(p, 'holding_days', 0)
                                row["持股"] = "0"
                                row["余额"] = f"¥{ca:,.0f}" if ca else ""
                                if pnl:
                                    row["盈亏"] = f"¥{pnl:+,.0f}"
                                if pnl_pct:
                                    row["盈亏%"] = f"{pnl_pct:+.2%}"
                                if hd:
                                    row["持仓天数"] = str(hd)
                            pts_data.append(row)

                        if pts_data:
                            st.dataframe(pd.DataFrame(pts_data), width="stretch", hide_index=True,
                                         column_config={"来源": st.column_config.TextColumn(), "方向": st.column_config.TextColumn()})
                else:
                    st.warning(f"{sym} 无K线数据")

    # 方案对比（批量回测）
    st.divider()
    if st.button("📊 对比全部方案", width="stretch"):
        factor_df, price_df, factor_names = get_data(n_stocks=300)
        all_schemes = registry.list_all()
        compare_progress = st.progress(0)
        compare_status = st.empty()
        all_results = []
        for i, scheme in enumerate(all_schemes):
            compare_status.caption(f"回测 [{scheme.name}] ({i+1}/{len(all_schemes)})...")
            backtester = SchemeBacktester()
            result = backtester.run(
                scheme=scheme, factor_df=factor_df, price_df=price_df,
                factor_names=factor_names, symbols=custom_symbols,
                lookback_days=lookback, top_n=top_n, initial_capital=capital,
            )
            all_results.append(result)
            compare_progress.progress((i + 1) / len(all_schemes))
        compare_status.caption("✅ 对比完成")
        all_results.sort(key=lambda r: r.total_return, reverse=True)
        st.session_state.bt_compare = all_results
        st.session_state.bt_compare_signature = current_context_signature

    if "bt_compare" in st.session_state and st.session_state.get("bt_compare_signature") == current_context_signature:
        section_header("方案对比")
        compare_symbols = current_context_signature["symbols"]
        symbol_label = ",".join(compare_symbols) if compare_symbols else current_context_signature["pool_mode"]
        st.caption(
            f"对比上下文：股票池={symbol_label}，回测天数={current_context_signature['lookback_days']}，"
            f"选股数={current_context_signature['top_n']}，资金=¥{current_context_signature['initial_capital']:,.0f}"
        )
        compare_data = []
        for r in st.session_state.bt_compare:
            compare_data.append({
                "方案": r.scheme_name,
                "总收益": f"{r.total_return:+.2%}",
                "年化": f"{r.annual_return:+.2%}",
                "夏普": f"{r.sharpe_ratio:.3f}",
                "最大回撤": f"{r.max_drawdown:.2%}",
                "胜率": f"{r.win_rate:.0%}",
                "交易": r.trade_summary,
            })
        st.dataframe(pd.DataFrame(compare_data), width="stretch", hide_index=True)

else:
    empty_state("📈", "点击「运行回测」开始")
