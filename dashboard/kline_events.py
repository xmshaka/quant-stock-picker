"""K线复盘成交事件转换工具。

P3 约束：K线买卖点只能来自实际成交事件（signals_executed/trades），
落点优先 exec_date；signal_date 仅用于 tooltip/审计展示。
"""
from __future__ import annotations

from datetime import date
from typing import List

import pandas as pd

from signals.rules import TradePoint


def first_valid_date(*values) -> date:
    """按优先级取第一个有效日期，避免 pandas NaN/NaT 覆盖 exec_date。"""
    for value in values:
        if value is None or value == "":
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        return pd.Timestamp(value).date()
    raise ValueError("缺少有效交易日期")


def trade_point_from_executed_row(row) -> TradePoint:
    """从 signals_executed/trades 行构造 K线复盘 TradePoint。"""
    exec_dt = first_valid_date(row.get("exec_date"), row.get("date"), row.get("signal_date"))
    signal_raw = row.get("signal_date", "")
    signal_dt = None
    if signal_raw is not None and signal_raw != "":
        try:
            if not pd.isna(signal_raw):
                signal_dt = pd.Timestamp(signal_raw).date()
        except (TypeError, ValueError):
            signal_dt = None
    p = TradePoint(
        date=exec_dt,
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
        signal_date=signal_dt,
        exec_date=exec_dt,
        exit_type=str(row.get("exit_type", "") or ""),
        exit_subtype=str(row.get("exit_subtype", "") or ""),
        trigger_price=float(row.get("trigger_price", 0.0) or 0.0),
        projected_pnl=float(row.get("projected_pnl", 0.0) or 0.0),
    )
    return p


def trade_points_from_executed_frame(df: pd.DataFrame) -> List[TradePoint]:
    """批量转换成交事件；非法行跳过。"""
    points: List[TradePoint] = []
    if df is None or df.empty:
        return points
    for _, row in df.iterrows():
        try:
            points.append(trade_point_from_executed_row(row))
        except Exception:
            continue
    return points
