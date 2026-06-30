"""策略回测页面"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

from copy import deepcopy
from dataclasses import fields
import streamlit as st
import pandas as pd
import logging

logger = logging.getLogger(__name__)

from data_loader import load_data, FACTOR_NAME_MAP, NAME_MAP
from strategy.registry import SchemeRegistry
from strategy.schemes import StrategyScheme, ExitConfig
from backtest.scheme_backtest import SchemeBacktester, run_multi_scheme_backtest
from backtest.records import BacktestRunConfig, load_backtest_run, summarize_liquidity_slippage, scheme_audit_snapshot
from dashboard.backtest_state import backtest_context_signature, clear_stale_compare
from dashboard.components.kline_chart import plot_kline_with_signals, plot_equity_curve, render_kline_chart
from signals.rules import TradePoint
from theme import inject_theme, metric_row, section_header, badge, empty_state, C

st.set_page_config(page_title="策略回测", page_icon="📈", layout="wide")
inject_theme()

# ========== 数据 ==========
@st.cache_data(ttl=300, show_spinner=False)
def get_data(n_stocks=200):
    return load_data(data_source="real", prefer_snapshot=True, n_stocks=n_stocks, n_days=252)

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


default_scheme_name = next((name for name, scheme in scheme_names.items() if scheme.scheme_id == "balanced"), list(scheme_names.keys())[0])
_restore_widget_state("bt_scheme_name", "bt_pref_scheme_name", default_scheme_name)
_restore_widget_state("bt_top_n", "bt_pref_top_n", 10)


def _sync_to_scan() -> None:
    """将当前回测参数持久化到文件，量化选股页可一键加载。"""
    import json, os
    sid = selected_scheme_runtime.scheme_id

    # 从 widget 读取当前所有回测参数
    fw = {}
    for k, v in st.session_state.items():
        if k.startswith("bt_fw_"):
            fw[k[6:]] = float(v)
    data = {
        sid: {
            "factor_weights": fw,
            "resonance_config": {
                "min_confirmations": int(st.session_state.get("bt_l3_min", 3)),
            },
            "stop_loss_atr_mult": float(st.session_state.get("bt_sl_atr", 2.0)),
            "take_profit_atr_mult": float(st.session_state.get("bt_tp_atr", 3.0)),
            "trailing_atr_mult": float(st.session_state.get("bt_trail_atr", 2.0)),
            "atr_period": int(st.session_state.get("bt_atr_p", 14)),
            "position_pct_per_entry": float(st.session_state.get("bt_pos_pct", 0.30)),
            "max_add_times": int(st.session_state.get("bt_max_add", 2)),
            "max_single_pct": float(st.session_state.get("bt_max_single", 30)) / 100.0,
            "min_entry_condition_count": int(st.session_state.get("bt_entry_cc", 3)),
            "enable_market_timing": bool(st.session_state.get("bt_market_timing", True)),
        },
        "_meta": {
            "synced_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "synced_from": "策略回测页",
        },
    }

    path = "data/synced_backtest_params.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"同步 {sid} 参数到 {path}")


def _load_from_scan() -> None:
    """从量化选股页同步的参数文件中加载，应用到当前回测 widget。"""
    import json, os
    path = "data/synced_scan_params.json"
    if not os.path.exists(path):
        st.toast("⚠️ 暂无选股页同步数据，请先在量化选股页点击「同步到回测」", icon="⚠️")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sid = selected_scheme_runtime.scheme_id
    if sid not in data:
        st.toast(f"⚠️ 同步文件中无 {sid} 策略参数", icon="⚠️")
        return
    params = data[sid]
    # 因子权重
    for f, w in params.get("factor_weights", {}).items():
        st.session_state[f"bt_fw_{f}"] = float(w)
    # L3 共振
    rc = params.get("resonance_config", {})
    if "min_confirmations" in rc:
        st.session_state["bt_l3_min"] = int(rc["min_confirmations"])
    # ATR
    if "stop_loss_atr_mult" in params:
        st.session_state["bt_sl_atr"] = float(params["stop_loss_atr_mult"])
    if "take_profit_atr_mult" in params:
        st.session_state["bt_tp_atr"] = float(params["take_profit_atr_mult"])
    if "trailing_atr_mult" in params:
        st.session_state["bt_trail_atr"] = float(params["trailing_atr_mult"])
    if "atr_period" in params:
        st.session_state["bt_atr_p"] = int(params["atr_period"])
    # 仓位
    if "position_pct_per_entry" in params:
        st.session_state["bt_pos_pct"] = float(params["position_pct_per_entry"])
    if "max_add_times" in params:
        st.session_state["bt_max_add"] = int(params["max_add_times"])
    if "max_single_pct" in params:
        st.session_state["bt_max_single"] = int(float(params["max_single_pct"]) * 100)
    if "min_entry_condition_count" in params:
        st.session_state["bt_entry_cc"] = int(params["min_entry_condition_count"])
    if "enable_market_timing" in params:
        st.session_state["bt_market_timing"] = bool(params["enable_market_timing"])
    meta = data.get("_meta", {})
    st.toast(f"✅ 已从选股页同步参数（{meta.get('synced_at', '未知时间')}）", icon="📥")
_restore_widget_state("bt_lookback", "bt_pref_lookback", 60)
_restore_widget_state("bt_capital_wan", "bt_pref_capital_wan", 100)
_restore_widget_state("bt_pool_mode", "bt_pref_pool_mode", "全A")
_restore_widget_state("bt_custom_codes", "bt_pref_custom_codes", "")


def _persist_all_params():
    """将当前所有 bt_* widget 值写入 bt_pref_* 持久键，并标记已确认。"""
    for k in list(st.session_state.keys()):
        if k.startswith("bt_") and not k.startswith("bt_pref_") and not k.startswith("bt_result") and not k.startswith("bt_compare") and not k.startswith("bt_run_") and not k.startswith("bt_last_") and not k.startswith("bt_market_"):
            pref_key = k.replace("bt_", "bt_pref_", 1)
            st.session_state[pref_key] = st.session_state[k]
    st.session_state["bt_params_confirmed"] = True


def _params_dirty():
    """参数是否未确认。首载或任何 widget 值与已确认值不同时为 True。"""
    if not st.session_state.get("bt_params_confirmed"):
        return True
    # 抽样比对：关键 widget 值 vs 对应 pref
    check_keys = ["bt_scheme_name", "bt_top_n", "bt_lookback", "bt_capital_wan", "bt_pool_mode",
                  "bt_enable_market_defense_exit", "bt_exit_max_holding_days_v3", "bt_sl_atr", "bt_pos_pct",
                  "bt_entry_cc", "bt_market_timing"]
    for k in check_keys:
        pref_key = k.replace("bt_", "bt_pref_", 1)
        if k in st.session_state and pref_key in st.session_state:
            if st.session_state[k] != st.session_state[pref_key]:
                return True
    return False


def _scheme_with_exit_overrides(base: StrategyScheme, exit_cfg: ExitConfig) -> StrategyScheme:
    """复制策略并应用页面上的退出规则覆盖，避免修改全局内置策略。"""
    cloned = StrategyScheme.from_dict(base.to_dict()) if hasattr(base, "to_dict") else deepcopy(base)
    cloned.exit_config = exit_cfg
    return cloned


def _make_exit_config(**kwargs) -> ExitConfig:
    """构造退出配置；过滤旧运行态未加载的新字段，避免 Streamlit 热重载半新半旧时报错。"""
    valid = {f.name for f in fields(ExitConfig)}
    return ExitConfig(**{k: v for k, v in kwargs.items() if k in valid})

section_header("策略回测")
st.caption("选择方案 → 选股池 → 回测区间 → 查看绩效和K线买卖点")

# ========== 参数 ==========
c1, c2, c3, c4 = st.columns(4)
with c1:
    selected_name = st.selectbox(
        "策略方案", list(scheme_names.keys()),
        index=list(scheme_names.keys()).index(st.session_state.bt_scheme_name) if st.session_state.bt_scheme_name in scheme_names else 0,
        key="bt_scheme_name",
    )
    selected_scheme = scheme_names[selected_name]
with c2:
    top_n = st.slider("选股数量", 3, 30, key="bt_top_n")
with c3:
    lookback = st.slider("回测天数", 20, 120, key="bt_lookback")
with c4:
    capital = st.number_input("初始资金(万)", 10, 1000, step=10, key="bt_capital_wan") * 10000

# ── 创建运行时副本，避免修改全局内置策略 ──
selected_scheme_runtime = StrategyScheme.from_dict(selected_scheme.to_dict()) if hasattr(selected_scheme, "to_dict") else deepcopy(selected_scheme)

# ── 恢复默认 / 同步参数 ──
b1, b2, b3 = st.columns(3)
with b1:
    if st.button("🔄 恢复默认参数", key="bt_restore_defaults", help="将所有参数（因子权重、L3条件、ATR、退出规则等）重置为当前策略的内置默认值"):
        to_del = [k for k in list(st.session_state.keys())
                  if k.startswith("bt_") and not k.startswith("bt_pref_") and not k.startswith("bt_result")
                  and not k.startswith("bt_compare") and not k.startswith("bt_run_") and not k.startswith("bt_last_")
                  and not k.startswith("bt_market_")]
        for k in to_del:
            del st.session_state[k]
        st.session_state.pop("_bt_exit_scheme_id", None)
        st.session_state.pop("_bt_exit_defaults_version", None)
        st.session_state["bt_params_confirmed"] = False
        st.rerun()
with b2:
    if st.button("🔄 同步到选股", key="bt_sync_to_scan", help="将当前回测参数持久化，量化选股页可一键加载"):
        _sync_to_scan()
with b3:
    if st.button("📥 从选股同步", key="bt_load_from_scan", help="从量化选股页加载已同步的参数"):
        _load_from_scan()
        st.rerun()

# ── 因子权重（可展开编辑）──
with st.expander("📊 因子权重", expanded=False):
    st.caption("各因子在截面打分中的权重。正数=多头偏好，负数=空头偏好。")
    st.info("⚠️ 本页因子权重修改仅影响**回测**（择时策略打分），**不影响量化选股页面**（选股页面使用策略内置默认权重）。", icon="🔒")
    fw = selected_scheme_runtime.factor_weights
    sorted_factors = sorted(fw.items(), key=lambda x: abs(x[1]), reverse=True)
    edited_weights = {}
    wcols = st.columns(3)
    for idx, (f, w) in enumerate(sorted_factors):
        with wcols[idx % 3]:
            cn = FACTOR_NAME_MAP.get(f, f)
            edited_weights[f] = st.slider(
                cn, -1.0, 1.0, float(w), 0.05,
                key=f"bt_fw_{f}",
                help=f"{f}: 当前 {w:+.2f}"
            )
    selected_scheme_runtime.factor_weights = edited_weights

# ── L3 共振条件（可编辑）──
rc = getattr(selected_scheme_runtime, "resonance_config", None)
if rc is not None:
    with st.expander("🎯 L3 共振条件", expanded=False):
        st.caption("三层过滤第三层：多个条件同时满足才触发信号。调整最低确认数和条件列表。")
        rc.min_confirmations = st.slider(
            "最低确认数（至少N个买入条件满足才触发BUY）",
            1, max(1, len(rc.buy_conditions)), int(rc.min_confirmations),
            key="bt_l3_min_conf"
        )
        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown("**买入条件**")
            buy_labels = {
                "large_elg_net_mf_positive": "超大单净流入 > 5万",
                "main_net_mf_positive": "主力净流入 > 1万",
                "large_elg_net_mf_rank_high": "超大单流入排名 > 70%",
                "main_net_mf_negative_improving": "主力流出改善",
                "large_elg_net_mf_negative_improving": "超大单流出改善",
                "large_elg_net_mf_positive_strong": "超大单>10万(突破)",
                "main_net_mf_positive_strong": "主力>5万(突破)",
                "main_net_mf_not_negative": "主力不净流出",
                "mf_rank_elite": "资金排名前20%",
                "relative_turnover_5d_high": "相对换手活跃 > 1.0x",
                "amount_percentile_60d_high": "成交额分位 > 60%",
                "relative_turnover_5d_low": "相对换手缩量 < 0.9x",
                "turnover_percentile_60d_low": "换手率分位 < 40%",
                "relative_turnover_5d_not_low": "相对换手 > 0.8x",
                "volume_expand": "温和放量",
                "ma5_above_ma20": "MA5高于MA20",
                "momentum_5d_strong": "5日动量强劲 > 2.5%",
                "momentum_20d_strong": "20日动量强劲 > 4%",
                "rsi_not_extreme": "RSI不过热 < 70",
                "rsi_oversold": "RSI超卖 < 45",
                "boll_lower": "布林下轨 < 0.35",
                "pullback_range": "回调幅度 5%-15%",
                "not_break_20d_low": "不破20日低点",
                "volume_calm": "成交量平稳 < 1.0x",
                "near_support": "接近均线支撑",
                "volume_surge": "量比>2x(突破)",
                "break_platform": "突破平台上沿",
                "narrow_range": "平台振幅<8%",
                "boll_upper_break": "突破布林上轨 > 0.7",
                "bullish_body": "实体阳线(突破)",
                "buildup_signal": "前日蓄势信号",
                "sustained_breakout": "连续站稳突破位",
                "boll_expanding": "布林中上轨",
                "momentum_5d_positive": "5日动量为正",
            }
            for cond in list(rc.buy_conditions):
                label = buy_labels.get(cond, cond)
                enabled = st.checkbox(label, value=True, key=f"bt_buy_{cond}")
                if not enabled:
                    rc.buy_conditions.remove(cond)
        with bc2:
            st.markdown("**卖出条件**")
            sell_labels = {
                "main_net_mf_negative": "主力净流出",
                "large_elg_net_mf_negative": "超大单净流出",
                "main_net_mf_negative_worsening": "主力净流出恶化",
                "large_elg_net_mf_negative_worsening": "超大单净流出恶化",
                "relative_turnover_5d_low": "相对换手率低（缩量走弱）",
                "relative_turnover_5d_high": "相对换手率高（放量下跌）",
                "ma5_below_ma20": "MA5低于MA20",
                "macd_bearish": "MACD转弱",
                "volume_price_down": "放量下跌",
                "rsi_overbought": "RSI超买 > 70",
                "boll_upper": "布林上轨 > 0.7",
            }
            for cond in list(rc.sell_conditions):
                label = sell_labels.get(cond, cond)
                enabled = st.checkbox(label, value=True, key=f"bt_sell_{cond}")
                if not enabled:
                    rc.sell_conditions.remove(cond)

# ── 开仓执行契约 ──
with st.expander("🛡️ 开仓执行契约", expanded=False):
    st.caption("信号必须满足≥N个L3条件才允许开仓。0=关闭契约（所有信号都执行）。")
    min_entry_cc = st.slider(
        "min_entry_condition_count（最小L3条件数）",
        0, 12, int(getattr(selected_scheme_runtime, "min_entry_condition_count", 3) or 0),
        key="bt_entry_cc",
        help="信号 condition_count < 此值 → 跳过不执行。该参数直接决定信号是否执行。"
    )
    selected_scheme_runtime.min_entry_condition_count = min_entry_cc

# ── 大盘择时开关 ──
with st.expander("🌍 大盘择时", expanded=False):
    st.caption("启用后根据市场评分动态调整仓位比例。评分来自北向资金/融资/量能/趋势四指标。")
    enable_market_timing = st.checkbox(
        "启用大盘择时仓位调制",
        value=bool(getattr(selected_scheme_runtime, "enable_market_timing", True)),
        key="bt_market_timing"
    )
    selected_scheme_runtime.enable_market_timing = enable_market_timing

# ── ATR 止盈止损 ──
with st.expander("💰 ATR 止盈止损", expanded=False):
    st.caption("基于ATR的动态止盈止损。止损=买入价-N×ATR，止盈=买入价+N×ATR，跟踪止盈=最高价-N×ATR。")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        sl_atr = st.number_input("止损ATR倍数", 0.5, 5.0, float(getattr(selected_scheme_runtime, "stop_loss_atr_mult", 2.0)), 0.5, key="bt_sl_atr")
        selected_scheme_runtime.stop_loss_atr_mult = sl_atr
    with a2:
        tp_atr = st.number_input("止盈ATR倍数", 0.5, 10.0, float(getattr(selected_scheme_runtime, "take_profit_atr_mult", 3.0)), 0.5, key="bt_tp_atr")
        selected_scheme_runtime.take_profit_atr_mult = tp_atr
    with a3:
        trail_atr = st.number_input("跟踪止盈ATR倍数", 0.5, 5.0, float(getattr(selected_scheme_runtime, "trailing_atr_mult", 2.0)), 0.5, key="bt_trail_atr")
        selected_scheme_runtime.trailing_atr_mult = trail_atr
    with a4:
        atr_p = st.number_input("ATR计算周期", 5, 30, int(getattr(selected_scheme_runtime, "atr_period", 14) or 14), 1, key="bt_atr_p")
        selected_scheme_runtime.atr_period = atr_p

# ── 仓位管理 ──
with st.expander("📐 仓位管理", expanded=False):
    st.caption("每次建仓/加仓占可用资金比例，加仓次数，单票最大仓位。")
    pm1, pm2, pm3 = st.columns(3)
    with pm1:
        pos_pct = st.slider("每次建仓比例", 0.05, 1.0, float(getattr(selected_scheme_runtime, "position_pct_per_entry", 0.30)), 0.05, key="bt_pos_pct")
        selected_scheme_runtime.position_pct_per_entry = pos_pct
    with pm2:
        max_add = st.slider("最大加仓次数", 0, 5, int(getattr(selected_scheme_runtime, "max_add_times", 2) or 0), 1, key="bt_max_add")
        selected_scheme_runtime.max_add_times = max_add
    with pm3:
        max_single = st.slider("单票最大仓位%", 5, 30, int((getattr(selected_scheme_runtime, "max_single_pct", 0.30) or 0.30) * 100), 5, key="bt_max_single")
        selected_scheme_runtime.max_single_pct = max_single / 100

# ── 策略失败退出规则说明 ──
strategy_failure_rules = {
    "trend_momentum": "动量失效退出：开盘价跌破 MA20 → 强制卖出",
    "pullback": "回调破位退出：开盘价跌破 20日最低价 → 强制卖出",
    "breakout": "突破失败退出：开盘价跌破 建仓时平台上沿 → 强制卖出",
    "balanced": "无专属策略失败退出（走共振层卖出条件）",
}
with st.expander("🚨 策略失败退出", expanded=False):
    st.caption("开仓后 N 天内跌破关键位则强制退出。与L3共振卖出条件互补。")
    st.info(strategy_failure_rules.get(selected_scheme_runtime.scheme_id, "无"))

# ── 短线退出规则（保持现有控件）──
base_exit_cfg = getattr(selected_scheme_runtime, "exit_config", ExitConfig()) or ExitConfig()
exit_defaults_version = "20260620_exit_defaults_v3"
if (
    st.session_state.get("_bt_exit_scheme_id") != selected_scheme_runtime.scheme_id
    or st.session_state.get("_bt_exit_defaults_version") != exit_defaults_version
):
    st.session_state["bt_enable_market_defense_exit"] = bool(getattr(base_exit_cfg, "enable_market_defense_exit", True))
    st.session_state["bt_enable_strategy_failure_exit"] = bool(getattr(base_exit_cfg, "enable_strategy_failure_exit", True))
    st.session_state["bt_enable_trailing_exit"] = bool(getattr(base_exit_cfg, "enable_trailing_exit", True))
    st.session_state["bt_enable_time_stop"] = bool(getattr(base_exit_cfg, "enable_time_stop", True))
    st.session_state["bt_enable_max_holding_exit"] = bool(getattr(base_exit_cfg, "enable_max_holding_exit", True))
    st.session_state["bt_exit_max_holding_days_v3"] = int(getattr(base_exit_cfg, "max_holding_days", 20) or 20)
    st.session_state["bt_exit_time_stop_days_v3"] = int(getattr(base_exit_cfg, "time_stop_days", 7) or 7)
    st.session_state["bt_exit_time_stop_min_profit_pct_v3"] = float(getattr(base_exit_cfg, "time_stop_min_profit_pct", 0.0) or 0.0) * 100
    st.session_state["bt_exit_market_defense_score_v3"] = float(getattr(base_exit_cfg, "market_defense_score", 20.0) or 20.0)
    st.session_state["bt_exit_failure_window_days_v3"] = int(getattr(base_exit_cfg, "failure_window_days", 3) or 3)
    st.session_state["bt_trailing_activation_pct_v3"] = float(getattr(base_exit_cfg, "trailing_activation_pct", 0.05) or 0.0) * 100
    st.session_state["bt_trailing_activation_atr_mult_v3"] = float(getattr(base_exit_cfg, "trailing_activation_atr_mult", 1.0) or 1.0)
    st.session_state["_bt_exit_scheme_id"] = selected_scheme_runtime.scheme_id
    st.session_state["_bt_exit_defaults_version"] = exit_defaults_version
with st.expander("⏱️ 短线退出规则", expanded=False):
    st.caption("五种退出机制的开关和阈值。调整后立即生效。")
    e1, e2, e3, e4, e5 = st.columns(5)
    with e1:
        enable_market_defense_exit = st.checkbox("大盘防御减仓", value=bool(getattr(base_exit_cfg, "enable_market_defense_exit", True)), key="bt_enable_market_defense_exit")
    with e2:
        enable_strategy_failure_exit = st.checkbox("策略失败退出", value=bool(getattr(base_exit_cfg, "enable_strategy_failure_exit", True)), key="bt_enable_strategy_failure_exit")
    with e3:
        enable_trailing_exit = st.checkbox("跟踪止盈/回撤", value=bool(getattr(base_exit_cfg, "enable_trailing_exit", True)), key="bt_enable_trailing_exit")
    with e4:
        enable_time_stop = st.checkbox("时间止损", value=bool(getattr(base_exit_cfg, "enable_time_stop", True)), key="bt_enable_time_stop")
    with e5:
        enable_max_holding_exit = st.checkbox("最长持仓退出", value=bool(getattr(base_exit_cfg, "enable_max_holding_exit", True)), key="bt_enable_max_holding_exit")

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        max_holding_days = st.number_input("最长持仓天数", min_value=1, max_value=60, value=int(getattr(base_exit_cfg, "max_holding_days", 20) or 20), step=1, key="bt_exit_max_holding_days_v3")
    with p2:
        time_stop_days = st.number_input("时间止损天数", min_value=1, max_value=60, value=int(getattr(base_exit_cfg, "time_stop_days", 7) or 7), step=1, key="bt_exit_time_stop_days_v3")
    with p3:
        time_stop_min_profit_pct = st.number_input("时间止损最低收益%", min_value=-20.0, max_value=50.0, value=float(getattr(base_exit_cfg, "time_stop_min_profit_pct", 0.0) or 0.0) * 100, step=0.5, key="bt_exit_time_stop_min_profit_pct_v3") / 100
    with p4:
        market_defense_score = st.number_input("大盘防御分数", min_value=0.0, max_value=100.0, value=float(getattr(base_exit_cfg, "market_defense_score", 20.0) or 20.0), step=1.0, key="bt_exit_market_defense_score_v3")

    f1, f2, f3 = st.columns(3)
    with f1:
        failure_window_days = st.slider("策略失败观察窗口(交易日)", min_value=0, max_value=20, value=int(getattr(base_exit_cfg, "failure_window_days", 3) or 3), key="bt_exit_failure_window_days_v3")
    with f2:
        trailing_activation_pct = st.number_input("跟踪止盈激活浮盈%", min_value=0.0, max_value=50.0, value=float(getattr(base_exit_cfg, "trailing_activation_pct", 0.05) or 0.0) * 100, step=0.5, key="bt_trailing_activation_pct_v3") / 100
    with f3:
        trailing_activation_atr_mult = st.number_input("跟踪止盈激活ATR倍数", min_value=0.0, max_value=10.0, value=float(getattr(base_exit_cfg, "trailing_activation_atr_mult", 1.0) or 1.0), step=0.1, key="bt_trailing_activation_atr_mult_v3")

selected_scheme_runtime.exit_config = _make_exit_config(
    enable_market_defense_exit=enable_market_defense_exit,
    enable_strategy_failure_exit=enable_strategy_failure_exit,
    enable_trailing_exit=enable_trailing_exit,
    enable_time_stop=enable_time_stop,
    enable_max_holding_exit=enable_max_holding_exit,
    max_holding_days=int(max_holding_days),
    time_stop_days=int(time_stop_days),
    time_stop_min_profit_pct=float(time_stop_min_profit_pct),
    failure_window_days=int(failure_window_days),
    market_defense_score=float(market_defense_score),
    trailing_activation_pct=float(trailing_activation_pct),
    trailing_activation_atr_mult=float(trailing_activation_atr_mult),
)

# 股票池选择
pool_mode = st.radio(
    "股票池", ["全A", "自定义代码", "观察池", "持仓池"],
    horizontal=True, key="bt_pool_mode",
)
custom_symbols = None
if pool_mode == "自定义代码":
    code_input = st.text_input(
        "股票代码（逗号分隔）", placeholder="600519,000001,002594", key="bt_custom_codes",
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
# FIX: 保持签名为 dict。此前把 dict 转为 tuple 后，方案对比区仍按
# current_context_signature["symbols"] 读取，触发
# TypeError: tuple indices must be integers or slices, not str。
current_context_signature["exit_config"] = tuple(sorted(selected_scheme_runtime.exit_config.to_dict().items()))
clear_stale_compare(st.session_state, current_context_signature)

# ========== 确认/恢复参数 ==========
confirm_col, restore_col, status_col = st.columns([1, 1, 3])
with confirm_col:
    if st.button("✅ 确认参数", type="primary", width="stretch", key="bt_confirm_params",
                 help="将当前所有参数写入持久存储，后续会话自动恢复"):
        _persist_all_params()
        st.rerun()
with restore_col:
    if st.button("🔄 恢复默认", width="stretch", key="bt_restore_all",
                 help="清空所有参数修改，恢复为当前策略的内置默认值"):
        to_del = [k for k in list(st.session_state.keys())
                  if k.startswith("bt_") and not k.startswith("bt_pref_") and not k.startswith("bt_result")
                  and not k.startswith("bt_compare") and not k.startswith("bt_run_") and not k.startswith("bt_last_")
                  and not k.startswith("bt_market_")]
        for k in to_del:
            del st.session_state[k]
        st.session_state.pop("_bt_exit_scheme_id", None)
        st.session_state.pop("_bt_exit_defaults_version", None)
        st.session_state["bt_params_confirmed"] = False
        st.rerun()
with status_col:
    if _params_dirty():
        st.warning("⚠️ 参数已修改，点击「确认参数」保存，否则刷新后丢失")
    else:
        st.success("✅ 参数已确认")

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
        scheme=selected_scheme_runtime,
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
    # ── 大盘择时数据 ──
    try:
        from market.timing import MarketTimingModel
        mt = MarketTimingModel()
        s = result.start_date.strftime('%Y%m%d') if hasattr(result.start_date, 'strftime') else str(result.start_date).replace('-', '')
        e = result.end_date.strftime('%Y%m%d') if hasattr(result.end_date, 'strftime') else str(result.end_date).replace('-', '')
        mt.fetch_all(s, e)
        st.session_state.bt_market_scores = mt.to_dataframe()
    except Exception:
        st.session_state.bt_market_scores = None
    # FIX: 股票池/参数已发生新回测时，旧“方案对比”结果不再可信，必须清空。
    st.session_state.pop("bt_compare", None)
    st.session_state.pop("bt_compare_signature", None)
    scheme_snapshot = scheme_audit_snapshot(selected_scheme_runtime)
    st.session_state.bt_run_config = BacktestRunConfig(
        run_id=result.run_id,
        scheme_id=selected_scheme_runtime.scheme_id,
        scheme_name=selected_scheme_runtime.name,
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
        scheme_config=scheme_snapshot["scheme_config"],
        resonance_config=scheme_snapshot["resonance_config"],
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

    # 大盘择时评分
    market_scores_df = st.session_state.get("bt_market_scores")
    if market_scores_df is not None and not market_scores_df.empty:
        with st.expander("📊 大盘择时评分", expanded=False):
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            scores_df = st.session_state["bt_market_scores"]
            fig_mt = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.6, 0.4],
                vertical_spacing=0.08,
                subplot_titles=("市场评分", "仓位档位"),
            )
            fig_mt.add_trace(go.Scatter(
                x=scores_df['date'], y=scores_df['score'],
                mode='lines', name='市场评分',
                line=dict(color='#FF6B35', width=2),
            ), row=1, col=1)
            fig_mt.add_hline(y=80, line_dash="dash", line_color="green", row=1, col=1)
            fig_mt.add_hline(y=60, line_dash="dash", line_color="gray", row=1, col=1)
            fig_mt.add_hline(y=40, line_dash="dash", line_color="gray", row=1, col=1)
            fig_mt.add_hline(y=20, line_dash="dash", line_color="red", row=1, col=1)
            fig_mt.add_trace(go.Bar(
                x=scores_df['date'], y=scores_df['position_pct'],
                name='仓位', marker_color='#2196F3',
            ), row=2, col=1)
            fig_mt.update_yaxes(title_text="评分 (0-100)", row=1, col=1, range=[0, 100])
            fig_mt.update_yaxes(title_text="仓位%", row=2, col=1, range=[0, 1], tickformat=".0%")
            fig_mt.update_layout(height=400, hovermode="x unified", showlegend=False)
            st.plotly_chart(fig_mt, width="stretch")

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

    # P3: K线复盘事件源必须是实际成交事件；严禁用原始 signal 当作买卖点。
    show_points = result.signals_executed
    overlay_raw = False
    if not show_points and result.signals_raw:
        st.warning("K线复盘暂无实际成交事件 signals_executed；原始信号不会作为买卖点展示，避免把未成交信号误认为交易。", icon="⚠️")
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
                    render_kline_chart(fig, key=f"kline_bs_v7_lane_line_{sym}", height=760)

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
                                "成交日": str(getattr(p, 'exec_date', None) or p.date),
                                "信号日": str(getattr(p, 'signal_date', '') or ''),
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
                                ts = getattr(p, 'trailing_stop', 0)
                                row["持股"] = f"{ps:,}" if ps else ""
                                row["余额"] = f"¥{ca:,.0f}" if ca else ""
                                row["止损"] = f"{sl:.2f}" if sl else ""
                                row["跟止"] = f"{ts:.2f}" if ts else ""
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
        all_saved_paths = []
        for i, scheme in enumerate(all_schemes):
            compare_status.caption(f"回测 [{scheme.name}] ({i+1}/{len(all_schemes)})...")
            backtester = SchemeBacktester()
            result = backtester.run(
                scheme=scheme, factor_df=factor_df, price_df=price_df,
                factor_names=factor_names, symbols=custom_symbols,
                lookback_days=lookback, top_n=top_n, initial_capital=capital,
            )
            
            # 新增：保存回测记录到标准回测目录
            try:
                # 使用与单个回测相同的配置，确保一致性
                scheme_snapshot = scheme_audit_snapshot(scheme)
                config = BacktestRunConfig(
                    run_id=result.run_id,
                    scheme_id=scheme.scheme_id,
                    scheme_name=scheme.name,
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
                    scheme_config=scheme_snapshot["scheme_config"],
                    resonance_config=scheme_snapshot["resonance_config"],
                )
                saved_path = result.persist(config=config)
                all_saved_paths.append(saved_path)
                compare_status.caption(f"回测 [{scheme.name}] ({i+1}/{len(all_schemes)})... 已保存到: {saved_path.name}")
                logger.info(f"[Compare] 已保存回测记录: {saved_path}")
            except Exception as e:
                logger.warning(f"[Compare] 保存回测记录失败: {e}")
                compare_status.caption(f"回测 [{scheme.name}] ({i+1}/{len(all_schemes)})... 保存失败: {e}")
            
            all_results.append(result)
            compare_progress.progress((i + 1) / len(all_schemes))
        
        # 记录保存结果统计
        if all_saved_paths:
            saved_count = len(all_saved_paths)
            compare_status.caption(f"✅ 对比完成，已保存 {saved_count}/{len(all_schemes)} 个回测记录")
        else:
            compare_status.caption(f"✅ 对比完成，但未保存任何回测记录")
        compare_status.caption("✅ 对比完成")
        all_results.sort(key=lambda r: r.total_return, reverse=True)
        st.session_state.bt_compare = all_results
        st.session_state.bt_compare_signature = current_context_signature
        
        # 自动保存对比结果到文件
        try:
            import json
            from datetime import datetime
            from pathlib import Path
            
            # 创建保存目录
            compare_dir = Path("data/compare_results")
            compare_dir.mkdir(exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbols_str = "_" + "_".join(current_context_signature["symbols"]) if current_context_signature["symbols"] else "_全池"
            filename = f"compare_{timestamp}{symbols_str}.json"
            filepath = compare_dir / filename
            
            # 准备保存数据
            save_data = {
                "timestamp": datetime.now().isoformat(),
                "context": current_context_signature,
                "results": []
            }
            
            for result in all_results:
                result_data = {
                    "scheme_id": result.scheme_id,
                    "scheme_name": result.scheme_name,
                    "total_return": result.total_return,
                    "annual_return": result.annual_return,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "trade_count": result.trade_count,
                    "buy_count": result.buy_count,
                    "sell_count": result.sell_count,
                    "final_value": result.final_value,
                    "start_date": result.start_date,
                    "end_date": result.end_date,
                    "run_id": result.run_id if hasattr(result, 'run_id') else "",
                    "has_signals": result.trade_count > 0
                }
                save_data["results"].append(result_data)
            
            # 保存到文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            # 显示保存成功消息
            compare_status.caption(f"✅ 对比完成，已保存到: {filepath}")
            st.success(f"对比结果已保存到文件: {filepath}")
            
        except Exception as e:
            st.warning(f"保存对比结果时出错: {e}")
            compare_status.caption(f"✅ 对比完成（保存失败: {e}）")

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
        
        # 显示保存的文件位置
        if "bt_compare_save_path" in st.session_state:
            st.caption(f"📁 对比结果已保存到: {st.session_state.bt_compare_save_path}")

else:
    empty_state("📈", "点击「运行回测」开始")
