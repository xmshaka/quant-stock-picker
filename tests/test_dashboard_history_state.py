"""回测记录页状态持久化逻辑测试。"""

from dashboard.history_state import (
    HISTORY_STATE_DEFAULTS,
    ensure_history_state,
    reset_history_state,
    sync_history_state,
    valid_default,
    valid_default_list,
)


def test_ensure_history_state_sets_defaults_without_overwriting():
    state = {"history_symbol_query": "002145"}
    ensure_history_state(state)
    assert state["history_symbol_query"] == "002145"
    assert state["history_scheme_filter"] == "全部"
    assert state["history_compare_run_ids"] == []
    assert set(HISTORY_STATE_DEFAULTS).issubset(state)


def test_sync_history_state_only_updates_known_keys():
    state = {}
    sync_history_state(
        state,
        history_symbol_query="5156150",
        history_recent_days=7,
        unknown="ignored",
    )
    assert state["history_symbol_query"] == "5156150"
    assert state["history_recent_days"] == 7
    assert "unknown" not in state


def test_reset_history_state_restores_defaults():
    state = {
        "history_symbol_query": "5156150",
        "history_scheme_filter": "趋势动量",
        "history_pool_filter": "自定义代码",
        "history_recent_days": 7,
        "history_compare_run_ids": ["run_a"],
    }
    reset_history_state(state)
    assert state == HISTORY_STATE_DEFAULTS


def test_valid_default_falls_back_when_option_missing():
    assert valid_default("趋势动量", ["全部", "低波价值"], "全部") == "全部"
    assert valid_default("低波价值", ["全部", "低波价值"], "全部") == "低波价值"


def test_valid_default_list_keeps_existing_options_order():
    assert valid_default_list(["run_c", "run_a", "missing"], ["run_a", "run_b", "run_c"]) == ["run_c", "run_a"]
