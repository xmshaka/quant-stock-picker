"""回测记录管理：run_id、事件源一致性、parquet 落盘。

P0原则：
- K线默认展示 signals_executed
- 绩效统计和交易明细来自同一执行事件源
- raw_signals 仅作为可选叠加层，不参与默认买卖次数统计
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd


BACKTEST_RUN_ROOT = Path(__file__).resolve().parents[1] / "data" / "backtest_runs"


STANDARD_TRADE_COLUMNS = [
    "run_id", "symbol", "date", "signal_date", "exec_date",
    "action", "event_type", "source", "reason", "rule_name",
    "signal_price", "price", "exec_price", "shares", "amount",
    "commission", "stamp_duty", "transfer_fee", "slippage", "slippage_rate", "commission_total",
    "liquidity_bucket", "turnover_amount",
    "avg_cost", "cost", "pnl", "pnl_pct", "holding_days",
    "cash_after", "position_after", "position_shares",
    "stop_loss", "take_profit", "exit_type", "exit_subtype",
    "trigger_price", "projected_pnl",
    "confidence", "confidence_bucket", "confidence_action", "confidence_weight", "confidence_note",
    "entry_model", "main_trigger", "confirmations", "factor_evidence", "market_context",
    "fund_flow_context", "technical_confirmations", "veto_checks", "risk_tags", "missing_fields",
]

SKIPPED_SIGNAL_COLUMNS = [
    "run_id", "symbol", "signal_date", "exec_date", "action", "skip_stage", "skip_reason",
    "reason", "rule_name", "signal_price", "confidence", "confidence_bucket", "confidence_action",
    "confidence_weight", "confidence_note", "entry_model", "main_trigger", "confirmations",
    "factor_evidence", "market_context", "fund_flow_context", "technical_confirmations",
    "veto_checks", "risk_tags", "missing_fields",
]

LIQUIDITY_BUCKET_LABELS = {
    "large_cap_gt_5e": "大盘蓝筹(>5亿)",
    "mid_cap_1e_5e": "中盘(1-5亿)",
    "small_cap_lt_1e": "小盘(<1亿)",
    "unknown_default": "未知(默认)",
    "fixed": "固定滑点",
    "": "未记录",
}

EXIT_TYPE_LABELS = {
    "stop_loss": "止损",
    "take_profit": "止盈",
    "signal_exit": "信号退出",
    "strategy_failure": "策略失败退出",
    "time_exit": "时间退出",
    "market_exit": "大盘风控退出",
    "final_liquidation": "末日清仓",
    "other_exit": "其他退出",
    "": "未记录",
}

EXIT_SUBTYPE_LABELS = {
    "atr_hard_stop": "ATR硬止损",
    "atr_trailing_profit": "ATR跟踪止盈",
    "atr_trailing_profit_failed": "ATR跟踪回撤止损",
    "atr_fixed_profit": "ATR固定止盈",
    "atr_profit_failed": "ATR止盈失败回撤",
    "rule_signal": "规则信号卖出",
    "end_of_backtest": "回测末日清仓",
    "generic_stop_loss": "通用止损",
    "generic_take_profit": "通用止盈",
    "generic_signal": "通用信号退出",
    "time_stop": "时间止损",
    "max_holding_days": "最长持仓退出",
    "trend_momentum_failed": "动量失效退出",
    "pullback_breakdown": "回调破位退出",
    "breakout_failed": "突破失败退出",
    "market_defense": "大盘防御减仓",
    "manual_or_unknown": "手动/未知",
    "": "未记录",
}

RUN_LIST_COLUMNS = [
    "run_id", "scheme_name", "scheme_id", "pool_mode", "symbols", "start_date", "end_date",
    "total_return", "annual_return", "max_drawdown", "sharpe_ratio", "win_rate",
    "trade_count", "buy_count", "sell_count", "final_value",
    "consistency_ok", "created_at", "path",
]


@dataclass
class BacktestRunConfig:
    """单次回测配置快照。"""

    run_id: str
    scheme_id: str
    scheme_name: str
    start_date: str
    end_date: str
    lookback_days: int
    top_n: int
    initial_capital: float
    pool_mode: str = ""
    symbols: List[str] = field(default_factory=list)
    cost: Dict[str, Any] = field(default_factory=dict)
    risk: Dict[str, Any] = field(default_factory=dict)
    scheme_config: Dict[str, Any] = field(default_factory=dict)
    resonance_config: Dict[str, Any] = field(default_factory=dict)
    market_regime_filter: bool = False
    git_commit: str = "unknown"
    data_version: str = "unknown"


def scheme_audit_snapshot(scheme: Any) -> Dict[str, Any]:
    """生成回测配置中可审计的策略快照。

    只保存可 JSON 序列化的策略配置，不保存运行态对象；旧记录不会被改写。
    """
    if scheme is None:
        return {"scheme_config": {}, "resonance_config": {}}
    if hasattr(scheme, "to_dict"):
        scheme_config = scheme.to_dict()
    else:
        scheme_config = dict(scheme) if isinstance(scheme, Mapping) else {}
    resonance_config = scheme_config.get("resonance_config", {}) if isinstance(scheme_config, dict) else {}
    return {
        "scheme_config": scheme_config if isinstance(scheme_config, dict) else {},
        "resonance_config": resonance_config if isinstance(resonance_config, dict) else {},
    }


def make_run_id(scheme_id: str, end_date: Any, suffix: str = "") -> str:
    """生成稳定可读 run_id。

    run_id 表示“本次运行保存时间”，不能使用回测结束交易日。
    否则 6/13 复盘 6/12 收盘数据会生成 20260612_xxx，导致新记录排序/识别混乱。
    回测区间仍保存在 config.start_date / config.end_date 中。
    """
    now = datetime.now()
    dt = now.strftime("%Y%m%d")
    ts = now.strftime("%H%M%S")
    clean_scheme = str(scheme_id or "scheme").replace("/", "_").replace(" ", "_")
    clean_suffix = f"_{suffix}" if suffix else ""
    return f"{dt}_{ts}_{clean_scheme}{clean_suffix}"


def trade_points_to_frame(stock_points: Mapping[str, Iterable[Any]], source: str) -> pd.DataFrame:
    """TradePoint 映射转 DataFrame。"""
    rows: List[Dict[str, Any]] = []
    for symbol, points in (stock_points or {}).items():
        for p in points or []:
            rows.append({
                "symbol": symbol,
                "date": _date_str(getattr(p, "date", None)),
                "action": getattr(p, "action", ""),
                "signal_date": _date_str(getattr(p, "signal_date", None)),
                "exec_date": _date_str(getattr(p, "date", None)),
                "signal_price": float(getattr(p, "signal_price", 0.0) or 0.0),
                "exec_price": float(getattr(p, "exec_price", 0.0) or getattr(p, "price", 0.0) or 0.0),
                "price": float(getattr(p, "price", 0.0) or 0.0),
                "shares": int(getattr(p, "shares", 0) or 0),
                "cash_after": float(getattr(p, "cash_after", 0.0) or 0.0),
                "position_after": int(getattr(p, "position_shares", 0) or 0),
                "avg_cost": float(getattr(p, "avg_cost", 0.0) or 0.0),
                "stop_loss": float(getattr(p, "stop_loss", 0.0) or 0.0),
                "take_profit": float(getattr(p, "take_profit", 0.0) or 0.0),
                "pnl": float(getattr(p, "pnl", 0.0) or 0.0),
                "pnl_pct": float(getattr(p, "pnl_pct", 0.0) or 0.0),
                "holding_days": int(getattr(p, "holding_days", 0) or 0),
                "exit_type": getattr(p, "exit_type", ""),
                "exit_subtype": getattr(p, "exit_subtype", ""),
                "trigger_price": float(getattr(p, "trigger_price", 0.0) or 0.0),
                "projected_pnl": float(getattr(p, "projected_pnl", 0.0) or 0.0),
                "reason": getattr(p, "reason", ""),
                "rule_name": getattr(p, "rule_name", ""),
                "confidence": float(getattr(p, "confidence", 0.0) or 0.0),
                "confidence_bucket": getattr(p, "confidence_bucket", ""),
                "confidence_action": getattr(p, "confidence_action", ""),
                "confidence_weight": float(getattr(p, "confidence_weight", 0.0) or 0.0),
                "confidence_note": getattr(p, "confidence_note", ""),
                "entry_model": getattr(p, "entry_model", ""),
                "main_trigger": getattr(p, "main_trigger", ""),
                "confirmations": getattr(p, "confirmations", ""),
                "factor_evidence": getattr(p, "factor_evidence", ""),
                "market_context": getattr(p, "market_context", ""),
                "fund_flow_context": getattr(p, "fund_flow_context", ""),
                "technical_confirmations": getattr(p, "technical_confirmations", ""),
                "veto_checks": getattr(p, "veto_checks", ""),
                "risk_tags": getattr(p, "risk_tags", ""),
                "missing_fields": getattr(p, "missing_fields", ""),
                "source": source,
            })
    return pd.DataFrame(rows)


def trade_details_to_frame(
    trade_details: Iterable[Mapping[str, Any]],
    *,
    run_id: str = "",
    source: str = "executed",
) -> pd.DataFrame:
    """成交明细标准化为 trades.parquet schema。

    P0：trades.parquet 必须以真实成交明细为准，而不是从 TradePoint 反推，
    否则会丢失佣金、印花税、过户费、滑点、pnl 等审计字段。
    """
    rows: List[Dict[str, Any]] = []
    for raw in trade_details or []:
        row = dict(raw)
        action = str(row.get("action", "") or "").upper()
        exec_price = _float(row.get("exec_price", row.get("price", 0.0)))
        shares = int(row.get("shares", 0) or 0)
        amount = _float(row.get("amount", exec_price * shares))
        slippage = _float(row.get("slippage", 0.0))
        commission = _float(row.get("commission", 0.0))
        stamp_duty = _float(row.get("stamp_duty", 0.0))
        transfer_fee = _float(row.get("transfer_fee", 0.0))
        row.setdefault("run_id", run_id)
        row.setdefault("source", source)
        row["action"] = action
        row.setdefault("event_type", action)
        row.setdefault("date", row.get("exec_date", ""))
        row.setdefault("exec_date", row.get("date", ""))
        row.setdefault("signal_date", "")
        row.setdefault("signal_price", 0.0)
        row.setdefault("price", exec_price)
        row["exec_price"] = exec_price
        row["shares"] = shares
        row["amount"] = amount
        row["commission"] = commission
        row["stamp_duty"] = stamp_duty
        row["transfer_fee"] = transfer_fee
        row["slippage"] = slippage
        row.setdefault("slippage_rate", slippage / amount if amount else 0.0)
        row.setdefault("commission_total", commission + stamp_duty + transfer_fee)
        row.setdefault("liquidity_bucket", "")
        row.setdefault("turnover_amount", 0.0)
        row.setdefault("avg_cost", row.get("cost", 0.0))
        row.setdefault("cost", row.get("avg_cost", 0.0))
        row.setdefault("pnl", 0.0)
        row.setdefault("pnl_pct", 0.0)
        row.setdefault("holding_days", 0)
        row.setdefault("cash_after", 0.0)
        row.setdefault("position_after", row.get("position_shares", 0))
        row.setdefault("position_shares", row.get("position_after", 0))
        row.setdefault("stop_loss", 0.0)
        row.setdefault("take_profit", 0.0)
        row.setdefault("exit_type", "")
        row.setdefault("exit_subtype", "")
        row.setdefault("trigger_price", row.get("price", 0.0) if action == "SELL" else 0.0)
        row.setdefault("projected_pnl", row.get("pnl", 0.0) if action == "SELL" else 0.0)
        row.setdefault("confidence", 0.0)
        row.setdefault("confidence_bucket", "")
        row.setdefault("confidence_action", "")
        row.setdefault("confidence_weight", 0.0)
        row.setdefault("confidence_note", "")
        row.setdefault("entry_model", "")
        row.setdefault("main_trigger", "")
        row.setdefault("confirmations", "")
        row.setdefault("factor_evidence", "")
        row.setdefault("market_context", "")
        row.setdefault("fund_flow_context", "")
        row.setdefault("technical_confirmations", "")
        row.setdefault("veto_checks", "")
        row.setdefault("risk_tags", "")
        row.setdefault("missing_fields", "")
        row.setdefault("reason", "")
        row.setdefault("rule_name", "")
        row["date"] = _date_str(row.get("date"))
        row["exec_date"] = _date_str(row.get("exec_date"))
        row["signal_date"] = _date_str(row.get("signal_date"))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=STANDARD_TRADE_COLUMNS)
    # 确保标准列在前，额外审计字段保留在后。
    for col in STANDARD_TRADE_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col in {"run_id", "symbol", "date", "signal_date", "exec_date", "action", "event_type", "source", "reason", "rule_name", "liquidity_bucket", "exit_type", "exit_subtype", "confidence_bucket", "confidence_action", "confidence_note", "entry_model", "main_trigger", "confirmations", "factor_evidence", "market_context", "fund_flow_context", "technical_confirmations", "veto_checks", "risk_tags", "missing_fields"} else 0
    extra_cols = [c for c in df.columns if c not in STANDARD_TRADE_COLUMNS]
    return df[STANDARD_TRADE_COLUMNS + extra_cols]


def skipped_signals_to_frame(
    skipped_signals: Iterable[Mapping[str, Any]],
    *,
    run_id: str = "",
) -> pd.DataFrame:
    """跳过信号审计表标准化。

    用于解释 signals_raw 有信号但 signals_executed/trades 没成交的原因，例如
    low-confidence observe-only、资金不足、加仓契约拒绝等。当前第一阶段主要记录
    observe-only，后续可扩展其它 skip_stage。
    """
    rows: List[Dict[str, Any]] = []
    for raw in skipped_signals or []:
        row = dict(raw)
        row.setdefault("run_id", run_id)
        row.setdefault("symbol", "")
        row.setdefault("signal_date", "")
        row.setdefault("exec_date", "")
        row.setdefault("action", "")
        row.setdefault("skip_stage", "")
        row.setdefault("skip_reason", "")
        row.setdefault("reason", "")
        row.setdefault("rule_name", "")
        row.setdefault("signal_price", 0.0)
        row.setdefault("confidence", 0.0)
        row.setdefault("confidence_bucket", "")
        row.setdefault("confidence_action", "")
        row.setdefault("confidence_weight", 0.0)
        row.setdefault("confidence_note", "")
        row.setdefault("entry_model", "")
        row.setdefault("main_trigger", "")
        row.setdefault("confirmations", "")
        row.setdefault("factor_evidence", "")
        row.setdefault("market_context", "")
        row.setdefault("fund_flow_context", "")
        row.setdefault("technical_confirmations", "")
        row.setdefault("veto_checks", "")
        row.setdefault("risk_tags", "")
        row.setdefault("missing_fields", "")
        row["signal_date"] = _date_str(row.get("signal_date"))
        row["exec_date"] = _date_str(row.get("exec_date"))
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=SKIPPED_SIGNAL_COLUMNS)
    for col in SKIPPED_SIGNAL_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col not in {"signal_price", "confidence", "confidence_weight"} else 0.0
    extra_cols = [c for c in df.columns if c not in SKIPPED_SIGNAL_COLUMNS]
    return df[SKIPPED_SIGNAL_COLUMNS + extra_cols]


def validate_trade_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """检查 trades.parquet 标准字段是否齐全。"""
    cols = set(df.columns if df is not None else [])
    missing = [c for c in STANDARD_TRADE_COLUMNS if c not in cols]
    return {"ok": not missing, "missing": missing, "columns": list(df.columns if df is not None else [])}


def summarize_liquidity_slippage(trades: pd.DataFrame) -> Dict[str, Any]:
    """汇总成交流水中的流动性分层滑点审计信息。"""
    if trades is None or trades.empty:
        return {
            "ok": False,
            "is_legacy_audit": False,
            "missing_audit_columns": [],
            "rows": 0,
            "total_amount": 0.0,
            "total_slippage": 0.0,
            "weighted_slippage_rate": 0.0,
            "buckets": pd.DataFrame(columns=["liquidity_bucket", "流动性分层", "成交笔数", "成交额", "滑点成本", "加权滑点率"]),
        }

    df = trades.copy()
    required_audit_columns = ["slippage_rate", "liquidity_bucket", "turnover_amount"]
    missing_audit_columns = [c for c in required_audit_columns if c not in df.columns]
    is_legacy_audit = bool(missing_audit_columns)
    for col in ("amount", "slippage", "slippage_rate", "turnover_amount", "exec_price", "shares"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
    # FIX:UI审计兜底。部分内存态回测结果可能 amount=0，但 exec_price/shares 或
    # slippage/slippage_rate 可还原真实策略成交额；否则会出现“滑点成本非0、成交额为0、加权滑点率为0”。
    zero_amount = df["amount"] <= 0
    inferred_amount = df["slippage"] / df["slippage_rate"].replace(0, np.nan)
    inferred_amount = inferred_amount.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df.loc[zero_amount & (inferred_amount > 0), "amount"] = inferred_amount[zero_amount & (inferred_amount > 0)]
    zero_amount = df["amount"] <= 0
    exec_amount = df["exec_price"] * df["shares"].abs()
    df.loc[zero_amount & (exec_amount > 0), "amount"] = exec_amount[zero_amount & (exec_amount > 0)]
    if "liquidity_bucket" not in df.columns:
        df["liquidity_bucket"] = ""
    df["liquidity_bucket"] = df["liquidity_bucket"].fillna("").astype(str)

    total_amount = float(df["amount"].sum())
    total_slippage = float(df["slippage"].sum())
    weighted_rate = total_slippage / total_amount if total_amount > 0 else 0.0

    grouped = (
        df.groupby("liquidity_bucket", dropna=False)
        .agg(
            成交笔数=("symbol", "count"),
            成交额=("amount", "sum"),
            滑点成本=("slippage", "sum"),
            平均市场成交额=("turnover_amount", "mean"),
        )
        .reset_index()
    )
    grouped["流动性分层"] = grouped["liquidity_bucket"].map(lambda x: LIQUIDITY_BUCKET_LABELS.get(str(x), str(x)))
    grouped["加权滑点率"] = grouped.apply(lambda r: float(r["滑点成本"]) / float(r["成交额"]) if float(r["成交额"] or 0) > 0 else 0.0, axis=1)
    grouped = grouped[["liquidity_bucket", "流动性分层", "成交笔数", "成交额", "滑点成本", "加权滑点率", "平均市场成交额"]]

    return {
        "ok": True,
        "is_legacy_audit": is_legacy_audit,
        "missing_audit_columns": missing_audit_columns,
        "rows": int(len(df)),
        "total_amount": total_amount,
        "total_slippage": total_slippage,
        "weighted_slippage_rate": weighted_rate,
        "buckets": grouped.sort_values("成交额", ascending=False).reset_index(drop=True),
    }


def summarize_exit_audit(trades: pd.DataFrame) -> Dict[str, Any]:
    """汇总卖出成交的退出审计字段。

    P0：历史记录页必须能直接审计一笔 SELL 是止损、止盈、信号退出还是末日清仓，
    并展示触发价/预估净收益，避免只靠 reason 文本人工判断。
    """
    if trades is None or trades.empty:
        empty = pd.DataFrame(columns=["exit_type", "退出类型", "笔数", "总盈亏", "平均盈亏", "平均触发价", "平均预估盈亏"])
        return {"ok": False, "is_legacy_exit_audit": False, "missing_exit_columns": [], "rows": 0, "sell_rows": 0, "summary": empty, "details": pd.DataFrame()}

    df = trades.copy()
    required = ["exit_type", "exit_subtype", "trigger_price", "projected_pnl"]
    missing = [c for c in required if c not in df.columns]
    is_legacy = bool(missing)
    for col in required:
        if col not in df.columns:
            df[col] = "" if col in {"exit_type", "exit_subtype"} else 0.0

    if "action" not in df.columns:
        df["action"] = ""
    sell_df = df[df["action"].fillna("").astype(str).str.upper() == "SELL"].copy()
    if sell_df.empty:
        empty = pd.DataFrame(columns=["exit_type", "退出类型", "笔数", "总盈亏", "平均盈亏", "平均触发价", "平均预估盈亏"])
        return {"ok": False, "is_legacy_exit_audit": is_legacy, "missing_exit_columns": missing, "rows": int(len(df)), "sell_rows": 0, "summary": empty, "details": sell_df}

    for col in ("pnl", "pnl_pct", "trigger_price", "projected_pnl"):
        if col not in sell_df.columns:
            sell_df[col] = 0.0
        sell_df[col] = pd.to_numeric(sell_df[col], errors="coerce").fillna(0.0).astype(float)
    sell_df["exit_type"] = sell_df["exit_type"].fillna("").astype(str)
    sell_df["exit_subtype"] = sell_df["exit_subtype"].fillna("").astype(str)
    sell_df["退出类型"] = sell_df["exit_type"].map(lambda x: EXIT_TYPE_LABELS.get(str(x), str(x) or "未记录"))
    sell_df["退出子类"] = sell_df["exit_subtype"].map(lambda x: EXIT_SUBTYPE_LABELS.get(str(x), str(x) or "未记录"))

    summary = (
        sell_df.groupby(["exit_type", "退出类型"], dropna=False)
        .agg(
            笔数=("symbol", "count"),
            总盈亏=("pnl", "sum"),
            平均盈亏=("pnl", "mean"),
            平均触发价=("trigger_price", "mean"),
            平均预估盈亏=("projected_pnl", "mean"),
        )
        .reset_index()
        .sort_values(["笔数", "总盈亏"], ascending=[False, False])
    )

    return {
        "ok": True,
        "is_legacy_exit_audit": is_legacy,
        "missing_exit_columns": missing,
        "rows": int(len(df)),
        "sell_rows": int(len(sell_df)),
        "take_profit_rows": int((sell_df["exit_type"] == "take_profit").sum()),
        "stop_loss_rows": int((sell_df["exit_type"] == "stop_loss").sum()),
        "signal_exit_rows": int((sell_df["exit_type"] == "signal_exit").sum()),
        "final_liquidation_rows": int((sell_df["exit_type"] == "final_liquidation").sum()),
        "total_pnl": float(sell_df["pnl"].sum()),
        "projected_pnl_total": float(sell_df["projected_pnl"].sum()),
        "summary": summary.reset_index(drop=True),
        "details": sell_df,
    }


def summarize_confidence_performance(trades: pd.DataFrame) -> Dict[str, Any]:
    """按开仓 confidence 分桶汇总交易绩效。

    第一阶段用于审计，不用于改变交易结果。按成交流水逐 symbol 配对：
    BUY/ADD 建立当前持仓的 confidence 桶；SELL 发生时把该轮净盈亏归因到开仓桶。
    如果一轮中出现 ADD 或多个不同桶，标记为 `mixed_or_add`，避免把混合仓位强行归因。
    """
    bucket_cols = [
        "confidence_bucket", "confidence_action", "开仓次数", "完成轮数", "胜率",
        "总盈亏", "平均盈亏", "平均收益率", "最大单笔亏损", "平均持仓天数",
    ]
    empty = pd.DataFrame(columns=bucket_cols)
    if trades is None or trades.empty:
        return {"ok": False, "rows": 0, "buy_rows": 0, "sell_rows": 0, "rounds": empty, "details": pd.DataFrame()}

    df = trades.copy()
    for col in ("action", "event_type", "symbol", "confidence_bucket", "confidence_action"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    for col in ("confidence", "confidence_weight", "pnl", "pnl_pct", "holding_days"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "exec_date" not in df.columns:
        df["exec_date"] = df.get("date", "")
    df["_dt"] = pd.to_datetime(df["exec_date"].where(df["exec_date"].astype(str) != "", df.get("date", "")), errors="coerce")
    df = df.sort_values(["symbol", "_dt", "action"]).reset_index(drop=True)

    open_state: Dict[str, Dict[str, Any]] = {}
    completed: List[Dict[str, Any]] = []
    buy_rows = 0
    sell_rows = 0
    for _, row in df.iterrows():
        action = str(row.get("action", "")).upper()
        symbol = str(row.get("symbol", ""))
        if action == "BUY":
            buy_rows += 1
            bucket = str(row.get("confidence_bucket", "") or "unclassified")
            conf_action = str(row.get("confidence_action", "") or "")
            event_type = str(row.get("event_type", action) or action).upper()
            existing = open_state.get(symbol)
            if existing is None or event_type != "ADD":
                open_state[symbol] = {
                    "symbol": symbol,
                    "entry_date": row.get("exec_date", row.get("date", "")),
                    "confidence_bucket": bucket,
                    "confidence_action": conf_action,
                    "confidence": float(row.get("confidence", 0.0) or 0.0),
                    "has_add": event_type == "ADD",
                    "buckets": {bucket},
                }
            else:
                existing["has_add"] = True
                existing.setdefault("buckets", set()).add(bucket)
                existing["confidence_bucket"] = "mixed_or_add"
                existing["confidence_action"] = "mixed_or_add"
        elif action == "SELL":
            sell_rows += 1
            state = open_state.pop(symbol, None)
            if state is None:
                bucket = "unpaired_sell"
                conf_action = "unpaired_sell"
                entry_date = ""
                entry_conf = 0.0
            else:
                buckets = state.get("buckets", set()) or set()
                bucket = "mixed_or_add" if state.get("has_add") or len(buckets) > 1 else state.get("confidence_bucket", "unclassified")
                conf_action = "mixed_or_add" if bucket == "mixed_or_add" else state.get("confidence_action", "")
                entry_date = state.get("entry_date", "")
                entry_conf = float(state.get("confidence", 0.0) or 0.0)
            completed.append({
                "symbol": symbol,
                "entry_date": entry_date,
                "exit_date": row.get("exec_date", row.get("date", "")),
                "confidence_bucket": bucket or "unclassified",
                "confidence_action": conf_action,
                "entry_confidence": entry_conf,
                "pnl": float(row.get("pnl", 0.0) or 0.0),
                "pnl_pct": float(row.get("pnl_pct", 0.0) or 0.0),
                "holding_days": float(row.get("holding_days", 0.0) or 0.0),
                "exit_type": row.get("exit_type", ""),
                "exit_subtype": row.get("exit_subtype", ""),
                "reason": row.get("reason", ""),
            })

    details = pd.DataFrame(completed)
    if details.empty:
        return {"ok": False, "rows": int(len(df)), "buy_rows": buy_rows, "sell_rows": sell_rows, "rounds": empty, "details": details}

    summary = (
        details.groupby(["confidence_bucket", "confidence_action"], dropna=False)
        .agg(
            开仓次数=("symbol", "count"),
            完成轮数=("symbol", "count"),
            胜率=("pnl", lambda s: float((s > 0).mean()) if len(s) else 0.0),
            总盈亏=("pnl", "sum"),
            平均盈亏=("pnl", "mean"),
            平均收益率=("pnl_pct", "mean"),
            最大单笔亏损=("pnl", "min"),
            平均持仓天数=("holding_days", "mean"),
        )
        .reset_index()
        .sort_values(["confidence_bucket", "完成轮数"])
    )
    return {
        "ok": True,
        "rows": int(len(df)),
        "buy_rows": buy_rows,
        "sell_rows": sell_rows,
        "round_count": int(len(details)),
        "rounds": summary.reset_index(drop=True),
        "details": details.reset_index(drop=True),
    }


def validate_backtest_consistency(result: Any) -> Dict[str, Any]:
    """校验绩效统计、成交明细、K线点位来自同一执行事件源。"""
    executed = getattr(result, "signals_executed", None) or getattr(result, "stock_signals", {}) or {}
    points = [p for pts in executed.values() for p in (pts or [])]
    buy_points = [p for p in points if getattr(p, "action", "") == "BUY"]
    sell_points = [p for p in points if getattr(p, "action", "") == "SELL"]
    trade_details = getattr(result, "trade_details", []) or []
    trade_actions = [str(t.get("action", "")) for t in trade_details]
    detail_buy = sum(1 for a in trade_actions if a == "BUY")
    detail_sell = sum(1 for a in trade_actions if a == "SELL")

    checks = {
        "buy_count_match": int(getattr(result, "buy_count", 0) or 0) == len(buy_points) == detail_buy,
        "sell_count_match": int(getattr(result, "sell_count", 0) or 0) == len(sell_points) == detail_sell,
        "trade_detail_match": len(trade_details) == len(points),
        "buy_points": len(buy_points),
        "sell_points": len(sell_points),
        "trade_detail_rows": len(trade_details),
    }
    checks["ok"] = bool(checks["buy_count_match"] and checks["sell_count_match"] and checks["trade_detail_match"])
    return checks


def persist_backtest_run(
    *,
    result: Any,
    config: BacktestRunConfig,
    trades: Optional[pd.DataFrame] = None,
    signals_raw: Optional[pd.DataFrame] = None,
    signals_executed: Optional[pd.DataFrame] = None,
    skipped_signals: Optional[pd.DataFrame] = None,
    equity: Optional[pd.DataFrame] = None,
    positions: Optional[pd.DataFrame] = None,
    factor_snapshot: Optional[pd.DataFrame] = None,
    root: Path = BACKTEST_RUN_ROOT,
) -> Path:
    """将一次回测完整落盘。parquet 优先，失败时自动写 csv。"""
    run_dir = root / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(_json(asdict(config)), encoding="utf-8")
    metrics = {
        "total_return": getattr(result, "total_return", 0.0),
        "annual_return": getattr(result, "annual_return", 0.0),
        "sharpe_ratio": getattr(result, "sharpe_ratio", 0.0),
        "max_drawdown": getattr(result, "max_drawdown", 0.0),
        "win_rate": getattr(result, "win_rate", 0.0),
        "trade_count": getattr(result, "trade_count", 0),
        "buy_count": getattr(result, "buy_count", 0),
        "sell_count": getattr(result, "sell_count", 0),
        "final_value": getattr(result, "final_value", 0.0),
        "consistency": validate_backtest_consistency(result),
        # 数据源信息
        # 数据源信息
        "data_source": getattr(result, "data_source", ""),
        "data_adjust": getattr(result, "data_adjust", "raw"),
        "data_version": getattr(result, "data_version", ""),
        # 运行标识
        "run_id": getattr(result, "run_id", config.run_id),
    }
    (run_dir / "metrics.json").write_text(_json(metrics), encoding="utf-8")

    frames = {
        "trades": trades,
        "signals_raw": signals_raw,
        "signals_executed": signals_executed,
        "skipped_signals": skipped_signals,
        "equity": equity,
        "positions": positions,
        "factor_snapshot": factor_snapshot,
    }
    for name, df in frames.items():
        _write_frame(run_dir / f"{name}.parquet", df if df is not None else pd.DataFrame())

    report = [
        f"# Backtest Run {config.run_id}",
        "",
        f"- 策略: {config.scheme_name} ({config.scheme_id})",
        f"- 区间: {config.start_date} ~ {config.end_date}",
        f"- 初始资金: {config.initial_capital:,.0f}",
        f"- 总收益: {metrics['total_return']:.4%}",
        f"- 年化收益: {metrics['annual_return']:.4%}",
        f"- 夏普: {metrics['sharpe_ratio']:.4f}",
        f"- 最大回撤: {metrics['max_drawdown']:.4%}",
        f"- 交易: 买{metrics['buy_count']} 卖{metrics['sell_count']}（{metrics['trade_count']}轮）",
        f"- 一致性: {'PASS' if metrics['consistency']['ok'] else 'FAIL'}",
        "",
        "## 成本模型",
        "",
        "```json",
        _json(config.cost),
        "```",
        "",
        "## 策略共振配置",
        "",
        "```json",
        _json(config.resonance_config),
        "```",
    ]
    (run_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    return run_dir


def list_backtest_runs(root: Path = BACKTEST_RUN_ROOT) -> pd.DataFrame:
    """列出历史回测记录目录。"""
    root = Path(root)
    rows: List[Dict[str, Any]] = []
    if not root.exists():
        return pd.DataFrame(columns=RUN_LIST_COLUMNS)

    # 按目录 mtime 倒序，而不是按 run_id 字符串倒序。
    # run_id 使用回测 end_date，盘后/次日复盘时新记录可能仍是前一交易日日期，
    # 若按 run_id 排序会被昨晚同日期晚时刻记录压到后面，看起来像“记录丢失”。
    run_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        config = _read_json(run_dir / "config.json")
        metrics = _read_json(run_dir / "metrics.json")
        consistency = metrics.get("consistency", {}) if isinstance(metrics.get("consistency", {}), dict) else {}
        created_at = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        rows.append({
            "run_id": config.get("run_id", run_dir.name),
            "scheme_name": config.get("scheme_name", ""),
            "scheme_id": config.get("scheme_id", ""),
            "pool_mode": config.get("pool_mode", ""),
            "symbols": ",".join(config.get("symbols", []) or []),
            "start_date": config.get("start_date", ""),
            "end_date": config.get("end_date", ""),
            "total_return": _float(metrics.get("total_return", 0.0)),
            "annual_return": _float(metrics.get("annual_return", 0.0)),
            "max_drawdown": _float(metrics.get("max_drawdown", 0.0)),
            "sharpe_ratio": _float(metrics.get("sharpe_ratio", 0.0)),
            "win_rate": _float(metrics.get("win_rate", 0.0)),
            "trade_count": int(metrics.get("trade_count", 0) or 0),
            "buy_count": int(metrics.get("buy_count", 0) or 0),
            "sell_count": int(metrics.get("sell_count", 0) or 0),
            "final_value": _float(metrics.get("final_value", 0.0)),
            "consistency_ok": bool(consistency.get("ok", False)),
            # 数据源信息
            "data_source": metrics.get("data_source", ""),
            "data_adjust": metrics.get("data_adjust", "raw"),
            "data_version": metrics.get("data_version", ""),
            "created_at": created_at,
            "path": str(run_dir),
        })
    # 扩展列名
    all_columns = RUN_LIST_COLUMNS + ["data_source", "data_adjust", "data_version"]
    return pd.DataFrame(rows, columns=all_columns)


def load_backtest_run(run_id: str, root: Path = BACKTEST_RUN_ROOT) -> Dict[str, Any]:
    """读取单个回测 run 的完整落盘内容。"""
    run_dir = Path(root) / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Backtest run not found: {run_id}")
    trades = _read_frame(run_dir / "trades")
    # 兼容旧版 trades.parquet：早期记录由 TradePoint 反推，缺少成本审计字段。
    # 历史页读取时自动补齐标准列，避免旧记录触发 schema 异常。
    if not trades.empty and not validate_trade_schema(trades).get("ok", False):
        trades = trade_details_to_frame(trades.to_dict("records"), run_id=run_id, source="legacy_normalized")
    return {
        "run_id": run_id,
        "path": run_dir,
        "config": _read_json(run_dir / "config.json"),
        "metrics": _read_json(run_dir / "metrics.json"),
        "report": (run_dir / "report.md").read_text(encoding="utf-8") if (run_dir / "report.md").exists() else "",
        "trades": trades,
        "signals_executed": _read_frame(run_dir / "signals_executed"),
        "signals_raw": _read_frame(run_dir / "signals_raw"),
        "skipped_signals": _read_frame(run_dir / "skipped_signals"),
        "equity": _read_frame(run_dir / "equity"),
        "positions": _read_frame(run_dir / "positions"),
        "factor_snapshot": _read_frame(run_dir / "factor_snapshot"),
    }


def delete_backtest_run(run_id: str, root: Path = BACKTEST_RUN_ROOT, *, trash: bool = True) -> Path:
    """删除历史回测记录。

    默认软删除：移动到 data/backtest_runs/.trash/<run_id>_<ts>，避免误删不可恢复。
    """
    if not run_id or "/" in run_id or ".." in run_id:
        raise ValueError(f"Invalid run_id: {run_id}")
    root = Path(root)
    run_dir = root / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Backtest run not found: {run_id}")
    if trash:
        trash_dir = root / ".trash"
        trash_dir.mkdir(parents=True, exist_ok=True)
        dst = trash_dir / f"{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(run_dir), str(dst))
        return dst
    shutil.rmtree(run_dir)
    return run_dir


def equity_curve_to_frame(equity_curve: Mapping[str, float], run_id: str) -> pd.DataFrame:
    return pd.DataFrame([
        {"run_id": run_id, "date": _date_str(k), "equity": float(v)}
        for k, v in (equity_curve or {}).items()
    ])


def _write_frame(path: Path, df: pd.DataFrame) -> None:
    df = df.copy() if df is not None else pd.DataFrame()
    try:
        df.to_parquet(path, index=False)
    except Exception:
        df.to_csv(path.with_suffix(".csv"), index=False)


def _read_frame(stem: Path) -> pd.DataFrame:
    """读取 parquet/csv，stem 可不带后缀。"""
    parquet_path = stem.with_suffix(".parquet")
    csv_path = stem.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _date_str(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def _float(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
