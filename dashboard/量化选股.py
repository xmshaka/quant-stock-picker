"""Streamlit 看板 — 量化选股系统

性能优化：
- hotspot 通过 session_state 缓存，不阻塞首屏
- SignalEngine 只在需要时执行，不跟随每次 rerun
- 导航用 st.session_state + radio 替代多按钮触发 rerun
"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import inspect
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from data_loader import load_data, FACTOR_NAME_MAP, NAME_MAP
from signals.engine import SignalEngine, SignalFormatter
from signals.scanner import scan_signals
from signals.portfolio import PortfolioManager, PoolItem
from strategy.schemes import BUILTIN_SCHEMES, StrategyScheme, ExitConfig, ResonanceConfig
from market.timing import MarketTimingModel
from theme import (
    inject_theme, metric_row, section_header, badge, badge_html,
    empty_state, signal_card, topbar, C,
)


def _scan_signals_compat(*, factor_df, price_df, factor_names, scheme_id, market_score, top_n, include_sell_symbols, include_sell_context, scheme_overrides=None):
    """兼容 Streamlit 旧进程缓存的 signals.scanner.scan_signals 签名。

    线上 Streamlit 进程可能在模块更新前已启动，rerun 时会复用 sys.modules 中的旧函数，
    旧函数不接受 include_sell_context / scheme_overrides。这里先避免页面直接 TypeError；进程重启后会走新版签名。
    """
    kwargs = {
        "scheme_id": scheme_id,
        "market_score": market_score,
        "top_n": top_n,
        "include_sell_symbols": list(include_sell_symbols),
    }
    sig_params = inspect.signature(scan_signals).parameters
    try:
        if "include_sell_context" in sig_params:
            kwargs["include_sell_context"] = include_sell_context
        if "scheme_overrides" in sig_params:
            kwargs["scheme_overrides"] = scheme_overrides
    except (TypeError, ValueError):
        kwargs["include_sell_context"] = include_sell_context
        kwargs["scheme_overrides"] = scheme_overrides
    return scan_signals(factor_df, price_df, factor_names, **kwargs)

# ========== 页面配置 ==========
st.set_page_config(page_title="量化选股", page_icon="🎯", layout="centered", initial_sidebar_state="collapsed")
inject_theme()

# ========== 初始化 ==========
pm = PortfolioManager("main")

# ── 数据加载（带缓存）──
@st.cache_data(ttl=300, show_spinner=False)
def get_data(n_stocks=100, include_symbols=None):
    kw = {"n_stocks": n_stocks, "n_days": 120, "prefer_snapshot": True}
    if include_symbols:
        kw["include_symbols"] = include_symbols
    return load_data(data_source="real", **kw)

# ── 热点（异步懒加载，不阻塞首屏）──
def _load_hotspot():
    from hotspot.bridge import get_bridge
    bridge = get_bridge()
    return {
        "bridge": bridge,
        "hot_sectors": bridge.get_hot_sectors_list(top_n=5),
        "hot_stocks": bridge.get_hot_stocks_list(top_n=5),
    }

if "hotspot" not in st.session_state:
    st.session_state.hotspot = None
    # 用 spinner 但不阻塞 —— 首屏先渲染，热点后补
    with st.spinner("加载热点数据..."):
        try:
            st.session_state.hotspot = _load_hotspot()
        except Exception:
            st.session_state.hotspot = {"bridge": None, "hot_sectors": pd.DataFrame(), "hot_stocks": pd.DataFrame()}

hotspot = st.session_state.hotspot

# ── 信号缓存（避免每次 rerun 重算）──
@st.cache_data(ttl=120, show_spinner=False)
def compute_signals(data_key, factor_names_key, scheme_id, market_score, portfolio_symbols_key, portfolio_context_key, overrides_hash_key):
    """按新短线方案扫描信号；用 hashable key 触发缓存，实际数据从 session_state 取。"""
    overrides = st.session_state.get("_signal_scan_overrides")
    buy, sell = _scan_signals_compat(
        factor_df=st.session_state._factor_df,
        price_df=st.session_state._price_df,
        factor_names=st.session_state._factor_names,
        scheme_id=scheme_id,
        market_score=market_score,
        top_n=20,
        include_sell_symbols=list(portfolio_symbols_key),
        include_sell_context=st.session_state.get("_portfolio_signal_context", {}),
        scheme_overrides=overrides,
    )
    return buy, sell

# ── 加载主数据 ──
_portfolio_syms = sorted({item.symbol for item in (pm.watch_list + pm.hold_list)})
st.session_state._portfolio_signal_context = {
    item.symbol: {
        "add_date": item.add_date,
        "add_reason": item.add_reason,
        "note": item.note,
        "signal_strength": item.signal_strength,
        "signal_score": item.signal_score,
    }
    for item in (pm.watch_list + pm.hold_list)
}
_n_stocks = max(100, len(_portfolio_syms) + 100)

with st.spinner("加载行情数据..."):
    factor_df, price_df, factor_names = get_data(n_stocks=_n_stocks, include_symbols=_portfolio_syms)

# 存入 session_state 供缓存函数引用
st.session_state._factor_df = factor_df
st.session_state._price_df = price_df
st.session_state._factor_names = factor_names

latest_date = pd.to_datetime(factor_df['trade_date'].max()).strftime('%Y-%m-%d')

def _date_key(value) -> str:
    """统一日期比较口径，避免 date/Timestamp/字符串混用导致详情为空。"""
    if pd.isna(value):
        return ""
    return pd.to_datetime(value).strftime('%Y-%m-%d')


def _build_data_key(factor_df: pd.DataFrame, price_df: pd.DataFrame, latest_date: str):
    """生成稳定缓存key；禁止用 id(df)，否则切页 rerun 会反复重扫。"""
    factor_symbols = factor_df['symbol'].astype(str) if 'symbol' in factor_df.columns else pd.Series(dtype=str)
    price_symbols = price_df['symbol'].astype(str) if 'symbol' in price_df.columns else pd.Series(dtype=str)
    return (
        latest_date,
        int(len(factor_df)),
        int(len(price_df)),
        int(factor_symbols.nunique()),
        int(price_symbols.nunique()),
        tuple(sorted(factor_symbols.unique())[:10]),
        tuple(sorted(price_symbols.unique())[:10]),
    )


def _signal_resonance_text(signal) -> str:
    """信号卡共振展示，必须使用 scanner 传回的策略配置总数，禁止硬编码 /6。"""
    confirmations = getattr(signal, 'layer3_confirmations', '-')
    total = getattr(signal, 'layer3_total', 0) or 6
    minimum = getattr(signal, 'layer3_min_confirmations', 0) or '-'
    return f"共振 {confirmations}/{total}，最低{minimum}"


def _signal_entry_audit_text(signal) -> str:
    """结构化买点审计摘要。

    moneyflow/相对换手当前只做审计，不作为硬过滤；缺失字段必须直接展示，避免
    用户误以为资金流/成交额分位已经参与交易决策。
    """
    fields = []
    if getattr(signal, "entry_model", ""):
        fields.append(f"买点模型: {getattr(signal, 'entry_model')}")
    if getattr(signal, "fund_flow_context", ""):
        fields.append(f"资金流: {getattr(signal, 'fund_flow_context')}")
    if getattr(signal, "factor_evidence", ""):
        fields.append(f"因子证据: {getattr(signal, 'factor_evidence')}")
    if getattr(signal, "market_context", ""):
        fields.append(f"市场: {getattr(signal, 'market_context')}")
    if getattr(signal, "missing_fields", ""):
        fields.append(f"缺失字段: {getattr(signal, 'missing_fields')}")
    if getattr(signal, "veto_checks", ""):
        fields.append(f"否决/审计: {getattr(signal, 'veto_checks')}")
    return " ｜ ".join(fields)


def _strategy_resonance_summary(scheme_id: str):
    """当前扫描策略的共振配置摘要。balanced 展示三个子策略。"""
    target_ids = ["trend_momentum", "pullback", "breakout"] if scheme_id == "balanced" else [scheme_id]
    rows = []
    for sid in target_ids:
        scheme = BUILTIN_SCHEMES.get(sid)
        if scheme is None:
            continue
        cfg = getattr(scheme, "resonance_config", None)
        if cfg is None:
            continue
        rows.append({
            "策略": scheme.name,
            "最低确认": cfg.min_confirmations,
            "买入条件数": len(cfg.buy_conditions or []),
            "卖出条件数": len(cfg.sell_conditions or []),
            "买入条件": "、".join(cfg.buy_conditions or []),
        })
    return rows

def _strategy_factor_weights_summary(scheme_id: str):
    """当前扫描策略的因子权重摘要。balanced 展示三个子策略。
    
    这些权重来自 BUILTIN_SCHEMES（择股内置默认），与回测页面可编辑的因子权重互不影响。
    """
    target_ids = ["trend_momentum", "pullback", "breakout"] if scheme_id == "balanced" else [scheme_id]
    all_factors = set()
    rows = []
    for sid in target_ids:
        scheme = BUILTIN_SCHEMES.get(sid)
        if scheme is None:
            continue
        fw = getattr(scheme, "factor_weights", {})
        all_factors.update(fw.keys())
        # 按绝对值排序
        sorted_fw = sorted(fw.items(), key=lambda x: abs(x[1]), reverse=True)
        for f, w in sorted_fw:
            cn = FACTOR_NAME_MAP.get(f, f)
            rows.append({"策略": scheme.name, "因子": cn, "权重": w})
    return rows


def _build_scan_overrides(target_ids: list) -> None:
    """从 session_state widget 值构造 scheme_overrides dict，存入 _signal_scan_overrides。"""
    overrides = {}
    from strategy.schemes import StrategyScheme, ExitConfig, ResonanceConfig
    from copy import deepcopy
    for sid in target_ids:
        default_scheme = BUILTIN_SCHEMES.get(sid)
        if default_scheme is None:
            continue
        scheme = StrategyScheme.from_dict(default_scheme.to_dict()) if hasattr(default_scheme, "to_dict") else deepcopy(default_scheme)
        # 因子权重
        fw = {}
        for f in getattr(default_scheme, "factor_weights", {}):
            fw[f] = float(st.session_state.get(f"scan_fw_{sid}_{f}", getattr(default_scheme, "factor_weights", {}).get(f, 0)))
        scheme.factor_weights = fw
        # L3 共振
        rc = getattr(scheme, "resonance_config", ResonanceConfig())
        if rc is not None:
            rc.min_confirmations = int(st.session_state.get(f"scan_l3_min_{sid}", getattr(rc, "min_confirmations", 3)))
            # 过滤启用条件
            buy_enabled = [c for c in (getattr(rc, "buy_conditions", []) or []) if st.session_state.get(f"scan_buy_{sid}_{c}", True)]
            sell_enabled = [c for c in (getattr(rc, "sell_conditions", []) or []) if st.session_state.get(f"scan_sell_{sid}_{c}", True)]
            rc.buy_conditions = buy_enabled
            rc.sell_conditions = sell_enabled
            scheme.resonance_config = rc
        # ATR
        scheme.stop_loss_atr_mult = float(st.session_state.get(f"scan_sl_atr_{sid}", 2.0))
        scheme.take_profit_atr_mult = float(st.session_state.get(f"scan_tp_atr_{sid}", 3.0))
        scheme.trailing_atr_mult = float(st.session_state.get(f"scan_trail_atr_{sid}", 2.0))
        scheme.atr_period = int(st.session_state.get(f"scan_atr_p_{sid}", 14))
        # 仓位
        scheme.position_pct_per_entry = float(st.session_state.get(f"scan_pos_pct_{sid}", 0.30))
        scheme.max_add_times = int(st.session_state.get(f"scan_max_add_{sid}", 2))
        scheme.max_single_pct = float(st.session_state.get(f"scan_max_single_{sid}", 30)) / 100.0
        # 开仓契约
        scheme.min_entry_condition_count = int(st.session_state.get(f"scan_entry_cc_{sid}", 3))
        scheme.enable_market_timing = bool(st.session_state.get(f"scan_market_timing_{sid}", True))
        # 退出规则
        from dataclasses import fields
        ec_fields = {f.name for f in fields(ExitConfig)}
        ec_kwargs = {}
        field_map = {
            "enable_market_defense_exit": f"scan_exit_market_defense_{sid}",
            "enable_strategy_failure_exit": f"scan_exit_strategy_failure_{sid}",
            "enable_trailing_exit": f"scan_exit_trailing_{sid}",
            "enable_time_stop": f"scan_exit_time_stop_{sid}",
            "enable_max_holding_exit": f"scan_exit_max_holding_{sid}",
            "max_holding_days": f"scan_exit_max_days_{sid}",
            "time_stop_days": f"scan_exit_time_stop_days_{sid}",
            "time_stop_min_profit_pct": f"scan_exit_time_stop_profit_{sid}",
            "market_defense_score": f"scan_exit_defense_score_{sid}",
            "failure_window_days": f"scan_exit_failure_window_{sid}",
            "trailing_activation_pct": f"scan_trail_act_pct_{sid}",
            "trailing_activation_atr_mult": f"scan_trail_act_atr_{sid}",
        }
        for field_name, ss_key in field_map.items():
            if field_name in ec_fields and ss_key in st.session_state:
                ec_kwargs[field_name] = st.session_state[ss_key]
        scheme.exit_config = ExitConfig(**{k: v for k, v in ec_kwargs.items() if k in ec_fields})
        overrides[sid] = scheme
    st.session_state["_signal_scan_overrides"] = overrides


def _compute_overrides_hash() -> str:
    """计算当前覆写参数的 hash，用于缓存失效。"""
    import hashlib
    import json
    overrides = st.session_state.get("_signal_scan_overrides", {})
    data = {}
    for sid, scheme in sorted(overrides.items()):
        data[sid] = scheme.to_dict() if hasattr(scheme, "to_dict") else str(scheme)
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ── 大盘择时（session_state 缓存，避免 pickle tushare 对象）──
def _load_market_timing() -> dict:
    """拉取近 90 日大盘择时评分。

    注意：不使用 @st.cache_data 装饰器，因为 tushare DataApi 内部
    使用了 functools.partial 作为 __getattr__ 返回值，pickle 反序列化
    时会触发不兼容的 requests.post(timeout=partial(...)) 调用。
    """
    from datetime import date as dt_date, timedelta
    try:
        model = MarketTimingModel()
        end = dt_date.today().strftime('%Y%m%d')
        start = (dt_date.today() - timedelta(days=90)).strftime('%Y%m%d')
        model.fetch_all(start, end)
        df = model.to_dataframe()
        latest_score = model.score_on(dt_date.today()) if not df.empty else 50.0
        return {"df": df, "latest_score": latest_score, "loaded": True}
    except Exception as e:
        return {"df": None, "latest_score": 50.0, "loaded": False, "error": str(e)}

def _get_market_score() -> float:
    """从 session_state 获取大盘评分，10 分钟内只拉取一次。"""
    import time as _time
    now = _time.time()
    last = st.session_state.get("_mt_last_fetch", 0)
    if now - last > 600 or "_mt_cache" not in st.session_state:
        st.session_state._mt_cache = _load_market_timing()
        st.session_state._mt_last_fetch = now
    return float(st.session_state._mt_cache.get("latest_score", 50.0))


# ── 新方案默认状态 ──
if "signal_scheme_id" not in st.session_state:
    st.session_state.signal_scheme_id = "balanced"

# ========== 顶部 ==========
left_info = f"🎯 <strong>量化选股</strong>"
hot_html = ""
if hotspot and not hotspot["hot_sectors"].empty:
    sectors = hotspot["hot_sectors"].head(3)
    tags = " ".join([f'{badge(s["industry"], "hot")}' for _, s in sectors.iterrows()])
    hot_html = f"热点 {tags}"
topbar(latest_date, left_html=left_info, right_html=hot_html)

# ========== 导航 ==========
PAGES = ["🎯 信号", f"👁 观察({len(pm.watch_list)})", f"💼 持仓({len(pm.hold_list)})", "🔥 热点", "🩺 状态", "⚙ 配置"]
if "nav_page" not in st.session_state:
    st.session_state.nav_page = 0

nav = st.columns(len(PAGES))
for i, label in enumerate(PAGES):
    with nav[i]:
        if st.button(label, key=f"nav_{i}", width="stretch",
                     type="primary" if st.session_state.nav_page == i else "secondary"):
            st.session_state.nav_page = i
            st.rerun()

page_idx = st.session_state.nav_page

# ========== 工具函数 ==========
def fmt_name(symbol: str) -> str:
    name = NAME_MAP.get(symbol, "")
    return f"{symbol} {name}" if name else symbol

def show_factor_detail(symbol, factor_df, price_df, latest_date):
    """个股因子详情面板。"""
    from signals.engine import SignalEngine

    fdf = factor_df.copy()
    fdf['_date_key'] = fdf['trade_date'].map(_date_key)
    latest_key = _date_key(latest_date)
    symbol_data = fdf[fdf['symbol'].astype(str) == str(symbol)].copy()
    day_data = fdf[fdf['_date_key'] == latest_key].copy()

    # 优先展示全局最新交易日；若该股票当天无快照，则回退到该股票自身最新日并明确提示。
    detail = symbol_data[symbol_data['_date_key'] == latest_key]
    if detail.empty and not symbol_data.empty:
        symbol_latest = symbol_data['_date_key'].max()
        detail = symbol_data[symbol_data['_date_key'] == symbol_latest]
        st.warning(f"该股票缺少全局最新日 {latest_key} 快照，当前展示该股票自身最新日 {symbol_latest}。")
        day_data = fdf[fdf['_date_key'] == symbol_latest].copy()
    if detail.empty:
        st.warning("无该股票最新数据")
        return

    row = detail.iloc[0]
    factors = {k: v for k, v in row.items() if k not in ['symbol', 'trade_date', '_date_key'] and pd.notna(v)}

    engine = SignalEngine()
    regime = engine._detect_regime(symbol, price_df, day_data)

    st.markdown(f"**{fmt_name(symbol)}** {badge(regime, 'regime')}", unsafe_allow_html=True)
    st.caption(f"因子日期：{row['_date_key']}；全局最新交易日：{latest_key}")

    FACTOR_DIRS = {
        'rsi14': -1, 'macd_hist': 1, 'boll_position': 1, 'volatility_20d': -1, 'max_dd_60d': -1,
        'north_hold_change': 1, 'turnover_ratio': 1, 'volume_ratio': 1,
        'pe_ttm': -1, 'pb': -1, 'ep': 1,
        'roe': 1, 'gross_margin': 1, 'revenue_growth': 1, 'profit_growth': 1,
        'momentum_5d': -1, 'momentum_20d': -1, 'momentum_60d': 0, 'liquidity': 1, 'reversal': 1,
    }
    DIR_LABEL = {1: "↑ 越高越好", -1: "↓ 越低越好", 0: "— 中性"}

    groups = {
        '技术': ['rsi14', 'macd_hist', 'boll_position', 'volatility_20d', 'max_dd_60d'],
        '情绪': ['north_hold_change', 'turnover_ratio', 'volume_ratio'],
        '估值': ['pe_ttm', 'pb', 'ep'],
        '质量': ['roe', 'gross_margin', 'revenue_growth', 'profit_growth'],
        '动量': ['momentum_5d', 'momentum_20d', 'momentum_60d', 'liquidity', 'reversal'],
    }

    for group_name, gfactors in groups.items():
        rows = []
        for f in gfactors:
            if f not in factors or f not in day_data.columns:
                continue
            vals = pd.to_numeric(day_data[f], errors='coerce').dropna()
            raw = pd.to_numeric(pd.Series([factors.get(f)]), errors='coerce').iloc[0]
            if vals.empty or pd.isna(raw):
                continue
            p25, p75 = vals.quantile(0.25), vals.quantile(0.75)
            pct = float((vals <= raw).mean() * 100)
            d = FACTOR_DIRS.get(f, 1)
            z = (raw - vals.mean()) / vals.std() if vals.std() > 0 else 0
            ds = z * d if d != 0 else np.nan

            if ds != ds:  # NaN
                tag = badge("中性", "neutral")
            elif ds >= 1.0:
                tag = badge("加分", "buy")
            elif ds >= 0.3:
                tag = badge("小加分", "buy")
            elif ds <= -1.0:
                tag = badge("扣分", "sell")
            elif ds <= -0.3:
                tag = badge("小扣分", "sell")
            else:
                tag = badge("平均", "neutral")

            rows.append({
                '因子': FACTOR_NAME_MAP.get(f, f),
                '值': round(raw, 3),
                '位置': f"P{pct:.0f}",
                '区间': f"{p25:.2f} ~ {p75:.2f}",
                '方向': DIR_LABEL.get(d, ""),
                '判断': tag,
            })
        if rows:
            with st.expander(f"**{group_name}** ({len(rows)})", expanded=False):
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                             column_config={"判断": st.column_config.TextColumn()})

# ========== 页面 0: 信号 ==========
if page_idx == 0:
    # ── 新方案扫描控制 ──
    with st.expander("⚙ 择股扫描参数", expanded=True):
        st.info(
            "📌 **择股 vs 择时数据隔离**：本页扫描参数覆写 `BUILTIN_SCHEMES` 内置默认值，"
            "与 `策略回测` 页可编辑参数完全独立。修改回测页参数不影响本页信号扫描结果。",
            icon="🔒"
        )
        # 大盘择时懒加载（首次渲染时拉取，不会在模块级触发 import 错误）
        _market_score = _get_market_score()  # 自动拉取 + 10 分钟缓存
        _mt = st.session_state.get("_mt_cache", {"df": None, "loaded": False})

        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            scheme_options = {
                "balanced": "全部策略/组合器",
                "trend_momentum": "强势追涨",
                "pullback": "回调低吸",
                "breakout": "横盘突破",
            }
            selected_scheme = st.selectbox(
                "扫描策略",
                list(scheme_options.keys()),
                format_func=lambda k: scheme_options[k],
                index=list(scheme_options.keys()).index(st.session_state.signal_scheme_id),
            )
        with c2:
            if _mt["loaded"]:
                score_color = "green" if _market_score >= 60 else "orange" if _market_score >= 40 else "red"
                st.metric("大盘评分（实时）", f"{_market_score:.0f}", delta=None)
                st.caption("来源: Tushare MarketTimingModel")
            else:
                st.metric("大盘评分", "50", delta="⚠️ 离线")
                st.caption(f"不可用: {_mt.get('error', '未知')}")
        with c3:
            bracket_label = (
                "满仓" if _market_score >= 80 else "高仓" if _market_score >= 60
                else "中等" if _market_score >= 40 else "低仓" if _market_score >= 20 else "防御"
            )
            st.metric("仓位档位", bracket_label)
        # 大盘择时评分明细
        if _mt["loaded"] and _mt["df"] is not None and not _mt["df"].empty:
            with st.expander("📊 大盘择时评分明细", expanded=False):
                latest_row = _mt["df"].iloc[-1]
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("趋势强度", f"{latest_row.get('trend', 0):.0f}/25")
                d2.metric("北向资金", f"{latest_row.get('capital', 0):.0f}/25")
                d3.metric("融资热度", f"{latest_row.get('leverage', 0):.0f}/25")
                d4.metric("市场活跃度", f"{latest_row.get('activity', 0):.0f}/25")
                recent = _mt["df"]["score"].tail(5).tolist()
                st.caption(f"近 5 日评分: {recent}")
        st.caption("当前扫描路径：大盘评分 → 股票池过滤 → 三策略独立候选 → L1趋势过滤 → L2形态匹配 → L3共振确认。信号日为T日收盘，建议成交日为T+1。")

        # ── 参数持久化初始化 ──
        target_ids = ["trend_momentum", "pullback", "breakout"] if selected_scheme == "balanced" else [selected_scheme]
        for sid in target_ids:
            default_scheme = BUILTIN_SCHEMES.get(sid)
            if default_scheme is None:
                continue
            # 恢复默认值（切换策略或首次加载时）
            if f"_scan_{sid}_loaded" not in st.session_state or st.session_state.get("_scan_loaded_scheme") != selected_scheme:
                st.session_state[f"_scan_{sid}_loaded"] = True
                for f, w in getattr(default_scheme, "factor_weights", {}).items():
                    st.session_state[f"scan_fw_{sid}_{f}"] = float(w)
                rc = getattr(default_scheme, "resonance_config", ResonanceConfig())
                st.session_state[f"scan_l3_min_{sid}"] = int(getattr(rc, "min_confirmations", 3))
                for cond in (getattr(rc, "buy_conditions", []) or []):
                    st.session_state[f"scan_buy_{sid}_{cond}"] = True
                for cond in (getattr(rc, "sell_conditions", []) or []):
                    st.session_state[f"scan_sell_{sid}_{cond}"] = True
                ec = getattr(default_scheme, "exit_config", ExitConfig())
                st.session_state[f"scan_sl_atr_{sid}"] = float(getattr(default_scheme, "stop_loss_atr_mult", 2.0))
                st.session_state[f"scan_tp_atr_{sid}"] = float(getattr(default_scheme, "take_profit_atr_mult", 3.0))
                st.session_state[f"scan_trail_atr_{sid}"] = float(getattr(default_scheme, "trailing_atr_mult", 2.0))
                st.session_state[f"scan_atr_p_{sid}"] = int(getattr(default_scheme, "atr_period", 14))
                st.session_state[f"scan_pos_pct_{sid}"] = float(getattr(default_scheme, "position_pct_per_entry", 0.30))
                st.session_state[f"scan_max_add_{sid}"] = int(getattr(default_scheme, "max_add_times", 2))
                st.session_state[f"scan_max_single_{sid}"] = int(getattr(default_scheme, "max_single_pct", 0.30) * 100)
                st.session_state[f"scan_entry_cc_{sid}"] = int(getattr(default_scheme, "min_entry_condition_count", 3))
                st.session_state[f"scan_market_timing_{sid}"] = bool(getattr(default_scheme, "enable_market_timing", True))
                # 退出规则
                st.session_state[f"scan_exit_market_defense_{sid}"] = bool(getattr(ec, "enable_market_defense_exit", True))
                st.session_state[f"scan_exit_strategy_failure_{sid}"] = bool(getattr(ec, "enable_strategy_failure_exit", True))
                st.session_state[f"scan_exit_trailing_{sid}"] = bool(getattr(ec, "enable_trailing_exit", True))
                st.session_state[f"scan_exit_time_stop_{sid}"] = bool(getattr(ec, "enable_time_stop", True))
                st.session_state[f"scan_exit_max_holding_{sid}"] = bool(getattr(ec, "enable_max_holding_exit", True))
                st.session_state[f"scan_exit_max_days_{sid}"] = int(getattr(ec, "max_holding_days", 20))
                st.session_state[f"scan_exit_time_stop_days_{sid}"] = int(getattr(ec, "time_stop_days", 7))
                st.session_state[f"scan_exit_time_stop_profit_{sid}"] = float(getattr(ec, "time_stop_min_profit_pct", 0.0))
                st.session_state[f"scan_exit_defense_score_{sid}"] = float(getattr(ec, "market_defense_score", 20.0))
                st.session_state[f"scan_exit_failure_window_{sid}"] = int(getattr(ec, "failure_window_days", 3))
                st.session_state[f"scan_trail_act_pct_{sid}"] = float(getattr(ec, "trailing_activation_pct", 0.05))
                st.session_state[f"scan_trail_act_atr_{sid}"] = float(getattr(ec, "trailing_activation_atr_mult", 1.0))
        st.session_state["_scan_loaded_scheme"] = selected_scheme

        # ── 检测参数变更 ──
        if selected_scheme != st.session_state.signal_scheme_id:
            st.session_state.signal_scheme_id = selected_scheme
            compute_signals.clear()
            st.rerun()

        # ── 各策略参数编辑器 ──
        st.divider()
        st.caption(f"📝 **编辑参数**（{len(target_ids)} 个策略，修改后需点「确认参数」→「重新扫描」生效）")

        for sid in target_ids:
            default_scheme = BUILTIN_SCHEMES.get(sid)
            if default_scheme is None:
                continue
            with st.expander(f"🎯 {default_scheme.name} ({sid})", expanded=(len(target_ids) == 1)):
                # ── 因子权重 ──
                with st.expander("📊 因子权重", expanded=False):
                    st.caption("各因子在截面打分中的权重。正数=多头偏好，负数=空头偏好。")
                    fw = getattr(default_scheme, "factor_weights", {})
                    sorted_factors = sorted(fw.items(), key=lambda x: abs(x[1]), reverse=True)
                    wcols = st.columns(3)
                    for idx, (f, w) in enumerate(sorted_factors):
                        with wcols[idx % 3]:
                            cn = FACTOR_NAME_MAP.get(f, f)
                            st.slider(
                                cn, -1.0, 1.0, float(st.session_state.get(f"scan_fw_{sid}_{f}", w)), 0.05,
                                key=f"scan_fw_{sid}_{f}",
                                help=f"{f}: 默认 {w:+.2f}"
                            )

                # ── L3 共振 ──
                rc = getattr(default_scheme, "resonance_config", ResonanceConfig())
                if rc is not None and hasattr(rc, "buy_conditions"):
                    with st.expander("🎯 L3 共振条件", expanded=False):
                        st.caption("三层过滤第三层：多个条件同时满足才触发信号。")
                        st.slider(
                            "最低确认数",
                            1, max(1, len(rc.buy_conditions or [])),
                            int(st.session_state.get(f"scan_l3_min_{sid}", getattr(rc, "min_confirmations", 3))),
                            key=f"scan_l3_min_{sid}"
                        )
                        buy_labels = {
                            "large_elg_net_mf_positive": "超大单净流入 > 5万",
                            "main_net_mf_positive": "主力净流入 > 1万",
                            "large_elg_net_mf_rank_high": "超大单流入排名 > 70%",
                            "main_net_mf_negative_improving": "主力流出改善",
                            "large_elg_net_mf_negative_improving": "超大单流出改善",
                            "large_elg_net_mf_positive_strong": "超大单>10万(突破)",
                            "main_net_mf_positive_strong": "主力>5万(突破)",
                            "main_net_mf_not_negative": "主力不净流出",
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
                            "momentum_5d_positive": "5日动量为正",
                        }
                        bc1, bc2 = st.columns(2)
                        with bc1:
                            st.markdown("**买入条件**")
                            for cond in list(rc.buy_conditions or []):
                                label = buy_labels.get(cond, cond)
                                default_val = True
                                if f"scan_buy_{sid}_{cond}" not in st.session_state:
                                    st.session_state[f"scan_buy_{sid}_{cond}"] = default_val
                                st.checkbox(label, key=f"scan_buy_{sid}_{cond}")
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
                            for cond in list(rc.sell_conditions or []):
                                label = sell_labels.get(cond, cond)
                                if f"scan_sell_{sid}_{cond}" not in st.session_state:
                                    st.session_state[f"scan_sell_{sid}_{cond}"] = True
                                st.checkbox(label, key=f"scan_sell_{sid}_{cond}")

                # ── 开仓契约 ──
                with st.expander("🛡️ 开仓执行契约", expanded=False):
                    st.slider(
                        "min_entry_condition_count",
                        0, 12,
                        int(st.session_state.get(f"scan_entry_cc_{sid}", getattr(default_scheme, "min_entry_condition_count", 3))),
                        key=f"scan_entry_cc_{sid}",
                        help="信号 condition_count < 此值 → 跳过不执行"
                    )
                    st.checkbox(
                        "启用大盘择时仓位调制",
                        value=bool(st.session_state.get(f"scan_market_timing_{sid}", getattr(default_scheme, "enable_market_timing", True))),
                        key=f"scan_market_timing_{sid}"
                    )

                # ── ATR ──
                with st.expander("💰 ATR 止盈止损", expanded=False):
                    a1, a2, a3, a4 = st.columns(4)
                    with a1:
                        st.number_input("止损ATR倍数", 0.5, 5.0, float(st.session_state.get(f"scan_sl_atr_{sid}", 2.0)), 0.5, key=f"scan_sl_atr_{sid}")
                    with a2:
                        st.number_input("止盈ATR倍数", 0.5, 10.0, float(st.session_state.get(f"scan_tp_atr_{sid}", 3.0)), 0.5, key=f"scan_tp_atr_{sid}")
                    with a3:
                        st.number_input("跟踪止盈ATR倍数", 0.5, 5.0, float(st.session_state.get(f"scan_trail_atr_{sid}", 2.0)), 0.5, key=f"scan_trail_atr_{sid}")
                    with a4:
                        st.number_input("ATR计算周期", 5, 30, int(st.session_state.get(f"scan_atr_p_{sid}", 14)), 1, key=f"scan_atr_p_{sid}")

                # ── 仓位 ──
                with st.expander("📐 仓位管理", expanded=False):
                    pm1, pm2, pm3 = st.columns(3)
                    with pm1:
                        st.slider("建仓比例", 0.05, 1.0, float(st.session_state.get(f"scan_pos_pct_{sid}", 0.30)), 0.05, key=f"scan_pos_pct_{sid}")
                    with pm2:
                        st.slider("最大加仓次数", 0, 5, int(st.session_state.get(f"scan_max_add_{sid}", 2)), 1, key=f"scan_max_add_{sid}")
                    with pm3:
                        st.slider("单票最大仓位%", 5, 30, int(st.session_state.get(f"scan_max_single_{sid}", 30)), 5, key=f"scan_max_single_{sid}")

                # ── 退出规则 ──
                with st.expander("⏱️ 短线退出规则", expanded=False):
                    e1, e2, e3, e4, e5 = st.columns(5)
                    with e1:
                        st.checkbox("大盘防御减仓", key=f"scan_exit_market_defense_{sid}")
                    with e2:
                        st.checkbox("策略失败退出", key=f"scan_exit_strategy_failure_{sid}")
                    with e3:
                        st.checkbox("跟踪止盈/回撤", key=f"scan_exit_trailing_{sid}")
                    with e4:
                        st.checkbox("时间止损", key=f"scan_exit_time_stop_{sid}")
                    with e5:
                        st.checkbox("最长持仓退出", key=f"scan_exit_max_holding_{sid}")
                    x1, x2, x3, x4 = st.columns(4)
                    with x1:
                        st.number_input("最长持仓天数", 1, 60, int(st.session_state.get(f"scan_exit_max_days_{sid}", 20)), 1, key=f"scan_exit_max_days_{sid}")
                    with x2:
                        st.number_input("时间止损天数", 1, 60, int(st.session_state.get(f"scan_exit_time_stop_days_{sid}", 7)), 1, key=f"scan_exit_time_stop_days_{sid}")
                    with x3:
                        st.number_input("时间止损最低收益%", -20.0, 50.0, float(st.session_state.get(f"scan_exit_time_stop_profit_{sid}", 0.0)) * 100, 0.5, key=f"scan_exit_time_stop_profit_{sid}_pct") / 100.0
                    with x4:
                        st.number_input("大盘防御分数", 0.0, 100.0, float(st.session_state.get(f"scan_exit_defense_score_{sid}", 20.0)), 1.0, key=f"scan_exit_defense_score_{sid}")
                    y1, y2, y3 = st.columns(3)
                    with y1:
                        st.slider("策略失败观察窗口(日)", 0, 20, int(st.session_state.get(f"scan_exit_failure_window_{sid}", 3)), key=f"scan_exit_failure_window_{sid}")
                    with y2:
                        st.number_input("跟踪止盈激活浮盈%", 0.0, 50.0, float(st.session_state.get(f"scan_trail_act_pct_{sid}", 0.05)) * 100, 0.5, key=f"scan_trail_act_pct_{sid}_pct") / 100.0
                    with y3:
                        st.number_input("跟踪止盈激活ATR倍数", 0.0, 10.0, float(st.session_state.get(f"scan_trail_act_atr_{sid}", 1.0)), 0.1, key=f"scan_trail_act_atr_{sid}")

        # ── 确认/恢复/重新扫描按钮 ──
        btn1, btn2, btn3 = st.columns(3)
        with btn1:
            if st.button("✅ 确认参数", type="primary", width="stretch", key="scan_confirm_params",
                         help="将当前参数写入持久存储，后续会话自动恢复"):
                _build_scan_overrides(target_ids)
                st.session_state["_signal_overrides_hash"] = _compute_overrides_hash()
                st.session_state["_scan_params_confirmed"] = True
                st.rerun()
        with btn2:
            if st.button("🔄 恢复默认", width="stretch", key="scan_restore_defaults",
                         help="清空所有参数修改，恢复为内置默认值"):
                for sid in target_ids:
                    for k in list(st.session_state.keys()):
                        if k.startswith(f"scan_fw_{sid}_") or k.startswith(f"scan_l3_min_{sid}") or \
                           k.startswith(f"scan_buy_{sid}_") or k.startswith(f"scan_sell_{sid}_") or \
                           k.startswith(f"scan_sl_atr_{sid}") or k.startswith(f"scan_tp_atr_{sid}") or \
                           k.startswith(f"scan_trail_atr_{sid}") or k.startswith(f"scan_atr_p_{sid}") or \
                           k.startswith(f"scan_pos_pct_{sid}") or k.startswith(f"scan_max_add_{sid}") or \
                           k.startswith(f"scan_max_single_{sid}") or k.startswith(f"scan_entry_cc_{sid}") or \
                           k.startswith(f"scan_market_timing_{sid}") or k.startswith(f"scan_exit_") or \
                           k.startswith(f"scan_trail_act_") or k == f"_scan_{sid}_loaded":
                            del st.session_state[k]
                st.session_state["_scan_loaded_scheme"] = None
                st.session_state["_signal_overrides_hash"] = "default"
                st.session_state["_scan_params_confirmed"] = False
                st.session_state["_signal_scan_overrides"] = None
                compute_signals.clear()
                st.rerun()
        with btn3:
            if st.button("🔍 重新扫描", width="stretch", key="scan_rescan",
                         help="用当前参数重新执行信号扫描"):
                _build_scan_overrides(target_ids)
                st.session_state["_signal_overrides_hash"] = _compute_overrides_hash()
                compute_signals.clear()
                st.rerun()

        # 状态提示
        if not st.session_state.get("_scan_params_confirmed"):
            st.warning("⚠️ 参数未确认，关闭页面后丢失。点击「确认参数」保存。")
        else:
            st.success("✅ 参数已确认")

    # ── 计算信号 ──
    portfolio_symbols = tuple(sorted({item.symbol for item in (pm.watch_list + pm.hold_list)}))
    portfolio_context_key = tuple(sorted(
        (
            sym,
            str(ctx.get("add_date", "")),
            str(ctx.get("add_reason", "")),
            str(ctx.get("note", "")),
        )
        for sym, ctx in st.session_state.get("_portfolio_signal_context", {}).items()
    ))
    data_key = _build_data_key(factor_df, price_df, latest_date)
    signal_cache_key = (
        data_key,
        tuple(factor_names),
        st.session_state.signal_scheme_id,
        _market_score,
        portfolio_symbols,
        portfolio_context_key,
        st.session_state.get("_signal_overrides_hash", "default"),
    )
    if st.session_state.get("_latest_signal_cache_key") == signal_cache_key:
        buy_signals, sell_signals = st.session_state.get("_latest_signal_result", ([], []))
    else:
        buy_signals, sell_signals = compute_signals(
            data_key,
            tuple(factor_names),
            st.session_state.signal_scheme_id,
            _market_score,
            portfolio_symbols,
            portfolio_context_key,
            st.session_state.get("_signal_overrides_hash", "default"),
        )
        st.session_state._latest_signal_cache_key = signal_cache_key
        st.session_state._latest_signal_result = (buy_signals, sell_signals)
        st.session_state._latest_signal_map = {s.symbol: s for s in (buy_signals + sell_signals)}

    # ── 顶部指标 ──
    metric_row([
        {"label": "大盘评分", "value": f"{_market_score:.0f}", "color": "green" if _market_score >= 60 else "orange" if _market_score >= 40 else "red"},
        {"label": "买入候选", "value": str(len(buy_signals)), "color": "green"},
        {"label": "风险退出", "value": str(len(sell_signals)), "color": "red" if sell_signals else ""},
        {"label": "扫描策略", "value": scheme_options.get(st.session_state.signal_scheme_id, st.session_state.signal_scheme_id)},
    ])

    # ── 手动加入 ──
    with st.expander("＋ 手动添加股票", expanded=False):
        c1, c2, c3 = st.columns([3, 2, 1])
        with c1:
            manual_symbol = st.text_input("代码", placeholder="000001", label_visibility="collapsed")
        with c2:
            manual_pool = st.selectbox("池", ["观察池", "持仓池"], label_visibility="collapsed")
        with c3:
            if st.button("添加", type="primary", width="stretch"):
                if manual_symbol:
                    sym = manual_symbol.strip().upper()
                    today = str(datetime.now().date())
                    item = PoolItem(symbol=sym, add_date=today, add_reason="手动", signal_strength=0, signal_score=0)
                    if manual_pool == "观察池":
                        pm.add_to_watch(item)
                    else:
                        pm.add_to_hold(item)
                    st.rerun()

    # ── 买入信号 ──
    section_header("买入信号", f"({len(buy_signals)})")
    if buy_signals:
        bridge = hotspot.get("bridge") if hotspot else None
        for s in sorted(buy_signals, key=lambda x: x.strength, reverse=True)[:20]:
            is_hot = bridge.is_hot_stock(s.symbol) if bridge else False
            in_watch = pm.is_in_watch(s.symbol)
            in_hold = pm.is_in_hold(s.symbol)

            regime_b = badge(s.regime[:4], "regime") if s.regime != '震荡整理' else ""
            hot_b = badge("热点", "hot") if is_hot else ""
            risk_b = " ".join([badge(t, "risk") for t in s.risk_tags]) if s.risk_tags else ""
            badges_html = f"{regime_b} {hot_b} {risk_b}"

            col_info, col_action = st.columns([4, 1])
            with col_info:
                signal_card(
                    name=f"<strong>{fmt_name(s.symbol)}</strong>",
                    meta=(
                        f"{s.strategy_name} · 总分 {s.score:.2f} · {_signal_resonance_text(s)} · "
                        f"信号日 {getattr(s, 'signal_date', latest_date)} · T+1 {getattr(s, 'suggested_exec_date', '-') or '-'} · "
                        f"建议仓位 {getattr(s, 'suggested_position_pct', 0) * 100:.1f}%"
                    ),
                    badges_html=badges_html,
                )
                if getattr(s, 'entry_reason', ''):
                    st.caption(s.entry_reason)
                audit_text = _signal_entry_audit_text(s)
                if audit_text:
                    st.caption(audit_text)
            with col_action:
                if in_hold:
                    st.markdown(f'<div style="text-align:center;padding-top:12px;">{badge("持仓", "hold")}</div>', unsafe_allow_html=True)
                elif in_watch:
                    if st.button("买入", key=f"buy_{s.symbol}", width="stretch"):
                        pm.move_to_hold(s.symbol)
                        st.rerun()
                else:
                    if st.button("+观察", key=f"watch_{s.symbol}", width="stretch"):
                        pm.add_to_watch(PoolItem(
                            symbol=s.symbol, add_date=str(latest_date),
                            add_reason=s.strategy_name, signal_strength=s.strength, signal_score=s.score,
                        ))
                        st.rerun()
    else:
        empty_state("📭", "暂无买入信号")

    # ── 卖出信号 ──
    section_header("卖出信号", f"({len(sell_signals)})")
    if sell_signals:
        for s in sorted(sell_signals, key=lambda x: x.strength, reverse=True)[:20]:
            in_hold = pm.is_in_hold(s.symbol)
            regime_b = badge(s.regime[:4], "regime") if s.regime != '震荡整理' else ""
            risk_b = " ".join([badge(t, "risk") for t in s.risk_tags]) if s.risk_tags else ""

            col_info, col_action = st.columns([4, 1])
            with col_info:
                signal_card(
                    name=f"<strong>{fmt_name(s.symbol)}</strong>",
                    meta=(
                        f"{s.strategy_name} · 风险分 {s.score:.2f} · "
                        f"信号日 {getattr(s, 'signal_date', latest_date)} · T+1 {getattr(s, 'suggested_exec_date', '-') or '-'}"
                    ),
                    badges_html=f"{regime_b} {risk_b}",
                )
                if getattr(s, 'entry_reason', ''):
                    st.caption(s.entry_reason)
            with col_action:
                if in_hold:
                    if st.button("卖出", key=f"sell_{s.symbol}", type="primary", width="stretch"):
                        pm.remove_from_hold(s.symbol)
                        st.rerun()
                else:
                    st.markdown(f'<div style="text-align:center;padding-top:12px;color:{C["text2"]};font-size:0.72rem;">未持仓</div>', unsafe_allow_html=True)
    else:
        empty_state("📭", "暂无卖出信号")

# ========== 页面 1: 观察池 ==========
elif page_idx == 1:
    section_header("观察池", f"({len(pm.watch_list)})")
    if pm.watch_list:
        bridge = hotspot.get("bridge") if hotspot else None
        for item in pm.watch_list:
            hot = bridge.get_stock_hot_summary(item.symbol) if bridge else {"has_hot": False}
            hot_b = badge("热点", "hot") if hot.get("has_hot") else ""

            col_info, col_btns = st.columns([3, 2])
            with col_info:
                live_sig = st.session_state.get("_latest_signal_map", {}).get(item.symbol)
                live_meta = (
                    f"最新扫描：{live_sig.strategy_name} · 总分 {live_sig.score:.2f} · {_signal_resonance_text(live_sig)} · {live_sig.signal_date}"
                    if live_sig else
                    f"入池记录：{item.add_reason} · 强度 {item.signal_strength} · {item.add_date}"
                )
                st.markdown(f'<div style="font-size:0.88rem;font-weight:600;">{fmt_name(item.symbol)} {hot_b}</div>', unsafe_allow_html=True)
                st.caption(live_meta)
                if live_sig and getattr(live_sig, 'entry_reason', ''):
                    st.caption(live_sig.entry_reason)
                if live_sig:
                    audit_text = _signal_entry_audit_text(live_sig)
                    if audit_text:
                        st.caption(audit_text)
            with col_btns:
                b1, b2, b3 = st.columns(3)
                with b1:
                    if st.button("📊", key=f"wd_{item.symbol}", help="详情", width="stretch"):
                        st.session_state.detail_symbol = item.symbol
                with b2:
                    if st.button("买入", key=f"wb_{item.symbol}", width="stretch"):
                        pm.move_to_hold(item.symbol)
                        st.rerun()
                with b3:
                    if st.button("✕", key=f"wr_{item.symbol}", width="stretch"):
                        pm.remove_from_watch(item.symbol)
                        st.rerun()

            if st.session_state.get('detail_symbol') == item.symbol:
                show_factor_detail(item.symbol, factor_df, price_df, latest_date)

            st.divider()
    else:
        empty_state("👁", "观察池为空 — 在信号页添加股票")

# ========== 页面 2: 持仓池 ==========
elif page_idx == 2:
    section_header("持仓池", f"({len(pm.hold_list)})")
    if pm.hold_list:
        bridge = hotspot.get("bridge") if hotspot else None

        # 热点提醒
        if bridge:
            alerts = bridge.get_portfolio_hot_alert([i.symbol for i in pm.hold_list])
            if not alerts.empty:
                with st.container():
                    st.markdown(f'<div style="background:{C["orange_bg"]};padding:8px 12px;border-radius:8px;margin-bottom:10px;font-size:0.78rem;">'
                                f'⚠️ <strong>持仓热点提醒</strong></div>', unsafe_allow_html=True)
                    for _, row in alerts.head(3).iterrows():
                        e = badge("利好", "buy") if row["sentiment"] > 0.2 else badge("利空", "sell") if row["sentiment"] < -0.2 else badge("中性", "neutral")
                        st.markdown(f'<div style="font-size:0.72rem;padding-left:8px;">{e} <strong>{fmt_name(row["symbol"])}</strong>: {row["latest_title"]}</div>', unsafe_allow_html=True)

        for item in pm.hold_list:
            hot = bridge.get_stock_hot_summary(item.symbol) if bridge else {"has_hot": False}
            hot_b = badge("热点", "hot") if hot.get("has_hot") else ""

            col_info, col_btns = st.columns([3, 2])
            with col_info:
                live_sig = st.session_state.get("_latest_signal_map", {}).get(item.symbol)
                live_meta = (
                    f"最新扫描：{live_sig.strategy_name} · 风险分 {live_sig.score:.2f} · {live_sig.signal_date}"
                    if live_sig else
                    f"入池记录：{item.add_reason} · 强度 {item.signal_strength} · {item.add_date}"
                )
                st.markdown(f'<div style="font-size:0.88rem;font-weight:600;">{fmt_name(item.symbol)} {hot_b}</div>', unsafe_allow_html=True)
                sent_label = ""
                if hot.get("has_hot"):
                    avg_s = hot.get("avg_sentiment", 0)
                    if avg_s > 0.2:
                        sent_label = badge("利好", "buy")
                    elif avg_s < -0.2:
                        sent_label = badge("利空", "sell")
                st.caption(f"{live_meta} {sent_label}")
                if live_sig and getattr(live_sig, 'entry_reason', ''):
                    st.caption(live_sig.entry_reason)
                if live_sig:
                    audit_text = _signal_entry_audit_text(live_sig)
                    if audit_text:
                        st.caption(audit_text)
            with col_btns:
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("📊", key=f"hd_{item.symbol}", help="详情", width="stretch"):
                        st.session_state.detail_symbol = item.symbol
                with b2:
                    if st.button("卖出", key=f"hs_{item.symbol}", type="primary", width="stretch"):
                        pm.remove_from_hold(item.symbol)
                        st.rerun()

            if st.session_state.get('detail_symbol') == item.symbol:
                show_factor_detail(item.symbol, factor_df, price_df, latest_date)

            st.divider()
    else:
        empty_state("💼", "持仓池为空 — 在观察池点击买入")

# ========== 页面 3: 热点 ==========
elif page_idx == 3:
    section_header("热点追踪")
    if hotspot and not hotspot["hot_sectors"].empty:
        section_header("热门行业")
        for _, row in hotspot["hot_sectors"].head(10).iterrows():
            heat = row.get("heat_score", 0)
            bar_w = min(100, heat * 20) if heat else 0
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:6px 0;">
                <span style="font-size:0.82rem;font-weight:600;width:120px;">{row.get('industry', '')}</span>
                <div class="qsp-progress" style="flex:1;"><div class="fill" style="width:{bar_w}%;background:{C['accent']};"></div></div>
                <span style="font-size:0.72rem;color:{C['text2']};width:60px;text-align:right;">热度 {heat:.1f}</span>
            </div>
            """, unsafe_allow_html=True)

    if hotspot and not hotspot["hot_stocks"].empty:
        section_header("热门个股")
        for _, row in hotspot["hot_stocks"].head(10).iterrows():
            sent = row.get("avg_sentiment", 0)
            sent_b = badge("利好", "buy") if sent > 0.2 else badge("利空", "sell") if sent < -0.2 else badge("中性", "neutral")
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:6px 0;">
                <span style="font-size:0.82rem;font-weight:600;width:120px;">{row.get('stock', '')}</span>
                {sent_b}
                <span style="font-size:0.72rem;color:{C['text2']};">热度 {row.get('heat_score', 0):.1f} · 提及 {row.get('mention_count', 0)}</span>
            </div>
            """, unsafe_allow_html=True)

    if not hotspot or (hotspot["hot_sectors"].empty and hotspot["hot_stocks"].empty):
        empty_state("🔥", "暂无热点数据")

# ========== 页面 4: 状态 ==========
elif page_idx == 4:
    section_header("系统状态")

    from data.daily_factors import latest_snapshot_date, load_snapshot_meta
    snap_date = latest_snapshot_date()
    if snap_date:
        meta = load_snapshot_meta(snap_date)
        if meta:
            metric_row([
                {"label": "快照日期", "value": snap_date},
                {"label": "股票池", "value": f"{meta.get('universe_size', 0)} 只"},
                {"label": "因子行数", "value": f"{meta.get('factor_rows', 0):,}"},
                {"label": "耗时", "value": f"{meta.get('elapsed_seconds', 0):.0f}s"},
            ])

    from data.scan_status import load_scan_reports
    try:
        reports = load_scan_reports()
        if not reports.empty:
            section_header("最近扫描")
            latest = reports.iloc[0]
            metric_row([
                {"label": "扫描时间", "value": str(latest.get("ts", ""))[:16]},
                {"label": "股票池", "value": f"{latest.get('total_symbols', 0)}"},
                {"label": "成功更新", "value": f"{latest.get('updated_count', 0)}", "color": "green"},
                {"label": "失败", "value": f"{latest.get('failed_count', 0)}", "color": "red" if latest.get('failed_count', 0) > 0 else ""},
            ])
    except Exception:
        pass

# ========== 页面 5: 配置 ==========
elif page_idx == 5:
    section_header("系统配置")
    st.caption("跳转到独立配置页面进行股票池、扫描参数等设置。")
    if st.button("打开配置页面 →", type="primary", width="stretch"):
        st.switch_page("pages/6_股票配置.py")

st.divider()
st.caption("量化选股系统 · 因子驱动 · 数据实时")
