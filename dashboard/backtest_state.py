"""策略回测页状态辅助函数。

把易回归的 session_state 判定逻辑从 Streamlit 页面中抽出来，便于单元测试。
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping


def normalize_symbols(symbols: Iterable[Any] | None) -> tuple[str, ...]:
    """股票代码标准化：去空格、去空值、排序，避免输入顺序导致签名不稳定。"""
    cleaned = []
    for symbol in symbols or []:
        if symbol is None:
            continue
        value = str(symbol).strip()
        if value:
            cleaned.append(value)
    return tuple(sorted(cleaned))


def backtest_context_signature(
    pool_mode: str,
    custom_symbols: Iterable[Any] | None,
    lookback: int,
    top_n: int,
    capital: float,
) -> dict[str, Any]:
    """当前回测上下文签名，用于避免股票/参数切换后展示旧方案对比结果。"""
    return {
        "pool_mode": str(pool_mode or ""),
        "symbols": normalize_symbols(custom_symbols),
        "lookback_days": int(lookback),
        "top_n": int(top_n),
        "initial_capital": float(capital),
    }


def compare_is_current(session_state: Mapping[str, Any], current_signature: Mapping[str, Any]) -> bool:
    """方案对比结果是否属于当前回测上下文。"""
    return "bt_compare" in session_state and session_state.get("bt_compare_signature") == current_signature


def clear_stale_compare(session_state: dict[str, Any], current_signature: Mapping[str, Any]) -> bool:
    """若方案对比不属于当前上下文则清空；返回是否发生清理。"""
    if "bt_compare" in session_state and session_state.get("bt_compare_signature") != current_signature:
        session_state.pop("bt_compare", None)
        session_state.pop("bt_compare_signature", None)
        return True
    return False
