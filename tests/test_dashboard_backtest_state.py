"""策略回测页 session_state 防陈旧逻辑测试。"""

from dashboard.backtest_state import (
    backtest_context_signature,
    clear_stale_compare,
    compare_is_current,
    normalize_symbols,
)


def test_normalize_symbols_stable_order_and_strip():
    assert normalize_symbols([" 5156150 ", "002145", "", None]) == ("002145", "5156150")


def test_compare_signature_changes_when_custom_stock_changes():
    old_sig = backtest_context_signature("自定义代码", ["002145"], 60, 10, 1_000_000)
    new_sig = backtest_context_signature("自定义代码", ["5156150"], 60, 10, 1_000_000)
    assert old_sig != new_sig


def test_clear_stale_compare_when_stock_changes():
    old_sig = backtest_context_signature("自定义代码", ["002145"], 60, 10, 1_000_000)
    new_sig = backtest_context_signature("自定义代码", ["5156150"], 60, 10, 1_000_000)
    state = {"bt_compare": ["old result"], "bt_compare_signature": old_sig}

    cleared = clear_stale_compare(state, new_sig)

    assert cleared is True
    assert "bt_compare" not in state
    assert "bt_compare_signature" not in state


def test_compare_kept_when_context_unchanged():
    sig = backtest_context_signature("自定义代码", ["5156150"], 60, 10, 1_000_000)
    state = {"bt_compare": ["current result"], "bt_compare_signature": sig}

    cleared = clear_stale_compare(state, sig)

    assert cleared is False
    assert compare_is_current(state, sig) is True
    assert state["bt_compare"] == ["current result"]


def test_legacy_compare_without_signature_is_cleared():
    sig = backtest_context_signature("自定义代码", ["5156150"], 60, 10, 1_000_000)
    state = {"bt_compare": ["legacy result without signature"]}

    cleared = clear_stale_compare(state, sig)

    assert cleared is True
    assert state == {}
