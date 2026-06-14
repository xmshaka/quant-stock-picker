"""回测记录页状态辅助函数。"""
from __future__ import annotations

from typing import Any, Iterable, MutableMapping


HISTORY_STATE_DEFAULTS = {
    "history_symbol_query": "",
    "history_scheme_filter": "全部",
    "history_pool_filter": "全部",
    "history_recent_days": 0,
    "history_compare_run_ids": [],
}


def ensure_history_state(session_state: MutableMapping[str, Any]) -> None:
    """初始化回测记录页持久状态。"""
    for key, default in HISTORY_STATE_DEFAULTS.items():
        if key not in session_state:
            session_state[key] = list(default) if isinstance(default, list) else default


def sync_history_state(session_state: MutableMapping[str, Any], **values: Any) -> None:
    """同步当前 widget 值到持久状态。"""
    ensure_history_state(session_state)
    for key, value in values.items():
        if key in HISTORY_STATE_DEFAULTS:
            session_state[key] = value


def reset_history_state(session_state: MutableMapping[str, Any]) -> None:
    """重置回测记录页筛选/对比状态。"""
    for key, default in HISTORY_STATE_DEFAULTS.items():
        session_state[key] = list(default) if isinstance(default, list) else default


def valid_default(value: Any, options: Iterable[Any], fallback: Any) -> Any:
    """当历史状态不在当前选项中时回落，避免 Streamlit index 报错。"""
    option_list = list(options)
    return value if value in option_list else fallback


def valid_default_list(values: Iterable[Any], options: Iterable[Any]) -> list[Any]:
    """过滤不再存在的多选默认值。"""
    option_set = set(options)
    return [v for v in values if v in option_set]
