"""P0 新方案信号扫描测试。"""
from datetime import date, timedelta

import pandas as pd

from signals.scanner import scan_signals
from strategy.schemes import ExitConfig, BUILTIN_SCHEMES


def _bars(symbol: str, closes, start=date(2026, 1, 1)):
    rows = []
    for i, close in enumerate(closes):
        d = start + timedelta(days=i)
        rows.append({
            "symbol": symbol,
            "trade_date": d,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1000000 + i * 1000,
        })
    return rows


def _factor_row(symbol: str, trade_date: date, **kw):
    base = {
        "symbol": symbol,
        "trade_date": trade_date,
        "momentum_5d": 0.0,
        "momentum_20d": 0.0,
        "reversal": 0.0,
        "volatility_20d": 0.2,
        "rsi14": 55.0,
        "boll_position": 0.7,
        "pb": 1.5,
        "volume_ratio": 1.0,
        "high_20d_distance": 0.0,
        "float_market_cap": 100.0,
        "amount": 200_000_000.0,
    }
    base.update(kw)
    return base


def test_trend_momentum_requires_market_score_floor():
    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(50)]
    latest = dates[-1]
    strong = [10 + i * 0.15 for i in range(50)]
    weak1 = [10 + i * 0.02 for i in range(50)]
    weak2 = [12 - i * 0.01 for i in range(50)]
    price_df = pd.DataFrame(_bars("000001", strong) + _bars("000002", weak1) + _bars("000003", weak2))
    factor_df = pd.DataFrame([
        _factor_row("000001", latest, momentum_5d=0.08, momentum_20d=0.25, volume_ratio=1.8, rsi14=65, boll_position=0.85),
        _factor_row("000002", latest, momentum_5d=0.01, momentum_20d=0.02, volume_ratio=1.0, rsi14=55, boll_position=0.5),
        _factor_row("000003", latest, momentum_5d=-0.02, momentum_20d=-0.05, volume_ratio=0.8, rsi14=45, boll_position=0.2),
    ])
    factor_names = [c for c in factor_df.columns if c not in {"symbol", "trade_date"}]

    low_buy, _ = scan_signals(factor_df, price_df, factor_names, scheme_id="trend_momentum", market_score=59, top_n=10)
    high_buy, _ = scan_signals(factor_df, price_df, factor_names, scheme_id="trend_momentum", market_score=60, top_n=10)

    assert low_buy == []
    assert any(s.symbol == "000001" for s in high_buy)
    sig = next(s for s in high_buy if s.symbol == "000001")
    assert sig.signal_date == latest
    assert sig.suggested_exec_date is None  # 当前数据只到信号日，T+1 由次日行情生成
    assert sig.scheme_id == "trend_momentum"
    assert sig.layer3_confirmations >= 3


def test_pullback_rejects_broken_down_stock():
    latest = date(2026, 2, 19)
    # 先升后健康回撤：仍高于20日低点3%以上。
    healthy = [10 + i * 0.12 for i in range(35)] + [13.8, 13.5, 13.2, 13.0, 12.8, 12.6, 12.45, 12.35, 12.30, 12.25, 12.2, 12.15, 12.1, 12.05, 12.0]
    # 破位下跌：最后阶段连续创新低。
    broken = [10 + i * 0.10 for i in range(25)] + [12.2, 11.9, 11.5, 11.1, 10.7, 10.3, 9.9, 9.5, 9.1, 8.8, 8.5, 8.2, 8.0, 7.8, 7.6, 7.4, 7.2, 7.0, 6.8, 6.6, 6.5, 6.4, 6.3, 6.2, 6.1]
    other = [9 + i * 0.01 for i in range(50)]
    price_df = pd.DataFrame(_bars("000010", healthy) + _bars("000011", broken) + _bars("000012", other))
    factor_df = pd.DataFrame([
        _factor_row("000010", latest, reversal=0.08, rsi14=35, high_20d_distance=-0.08, volume_ratio=0.9, boll_position=0.25),
        _factor_row("000011", latest, reversal=0.20, rsi14=25, high_20d_distance=-0.30, volume_ratio=0.7, boll_position=0.1),
        _factor_row("000012", latest, reversal=0.01, rsi14=50, high_20d_distance=-0.01, volume_ratio=1.0, boll_position=0.5),
    ])
    factor_names = [c for c in factor_df.columns if c not in {"symbol", "trade_date"}]

    buy, _ = scan_signals(factor_df, price_df, factor_names, scheme_id="pullback", market_score=45, top_n=10)
    symbols = {s.symbol for s in buy}

    assert "000011" not in symbols


def test_balanced_is_combiner_not_direct_scheme():
    latest = date(2026, 1, 1) + timedelta(days=49)
    strong = [10 + i * 0.15 for i in range(50)]
    mid = [10 + i * 0.05 for i in range(50)]
    low = [10 - i * 0.01 for i in range(50)]
    price_df = pd.DataFrame(_bars("000021", strong) + _bars("000022", mid) + _bars("000023", low))
    factor_df = pd.DataFrame([
        _factor_row("000021", latest, momentum_5d=0.08, momentum_20d=0.25, volume_ratio=1.8, rsi14=65, boll_position=0.85),
        _factor_row("000022", latest, momentum_5d=0.03, momentum_20d=0.08, volume_ratio=1.4, rsi14=60, boll_position=0.7),
        _factor_row("000023", latest, momentum_5d=-0.02, momentum_20d=-0.05, volume_ratio=0.8, rsi14=45, boll_position=0.2),
    ])
    factor_names = [c for c in factor_df.columns if c not in {"symbol", "trade_date"}]

    buy, _ = scan_signals(factor_df, price_df, factor_names, scheme_id="balanced", market_score=60, top_n=10)

    assert buy
    assert all(s.scheme_id in {"trend_momentum", "pullback", "breakout"} for s in buy)
    assert "balanced" not in {s.scheme_id for s in buy}


def test_sell_scan_only_for_portfolio_symbols():
    latest = date(2026, 1, 1) + timedelta(days=49)
    falling = [15 - i * 0.1 for i in range(50)]
    stable = [10 + i * 0.01 for i in range(50)]
    price_df = pd.DataFrame(_bars("000031", falling) + _bars("000032", stable) + _bars("000033", stable))
    factor_df = pd.DataFrame([
        _factor_row("000031", latest, rsi14=75),
        _factor_row("000032", latest, rsi14=55),
        _factor_row("000033", latest, rsi14=55),
    ])
    factor_names = [c for c in factor_df.columns if c not in {"symbol", "trade_date"}]

    _, sell = scan_signals(
        factor_df, price_df, factor_names,
        scheme_id="balanced", market_score=15, top_n=10,
        include_sell_symbols=["000031"],
    )

    assert {s.symbol for s in sell} == {"000031"}
    assert sell[0].signal_type == "SELL"
    assert "大盘防御减仓" in sell[0].entry_reason


def test_sell_scan_uses_exit_config_for_time_stop_context():
    latest = date(2026, 1, 1) + timedelta(days=49)
    flat = [10.0] * 50
    price_df = pd.DataFrame(_bars("000071", flat))
    factor_df = pd.DataFrame([
        _factor_row("000071", latest, rsi14=50),
    ])
    factor_names = [c for c in factor_df.columns if c not in {"symbol", "trade_date"}]

    _, sell = scan_signals(
        factor_df, price_df, factor_names,
        scheme_id="balanced", market_score=60, top_n=10,
        include_sell_symbols=["000071"],
        include_sell_context={
            "000071": {
                "add_date": str(latest - timedelta(days=5)),
                "add_reason": "强势追涨",
                "entry_price": 10.0,
            }
        },
    )

    assert sell
    sig = sell[0]
    assert "时间止损" in sig.entry_reason
    assert "time_stop" in sig.layer3_condition_keys
    assert sig.signal_type == "SELL"


def test_sell_scan_time_stop_uses_trading_days_not_calendar_days(monkeypatch):
    latest = date(2026, 1, 6)
    scheme = BUILTIN_SCHEMES["balanced"]
    original = scheme.exit_config
    monkeypatch.setattr(
        scheme,
        "exit_config",
        ExitConfig(max_holding_days=20, time_stop_days=2, time_stop_min_profit_pct=0.02, failure_window_days=0),
    )
    trade_dates = [d.date() for d in pd.bdate_range("2025-12-01", "2026-01-06")]
    rows = []
    for d in trade_dates:
        rows.append({
            "symbol": "000074",
            "trade_date": d,
            "open": 10.0,
            "high": 10.1,
            "low": 9.9,
            "close": 10.0,
            "volume": 1_000_000,
        })
    price_df = pd.DataFrame(rows)
    factor_df = pd.DataFrame([_factor_row("000074", latest, rsi14=50)])
    factor_names = [c for c in factor_df.columns if c not in {"symbol", "trade_date"}]

    _, sell = scan_signals(
        factor_df, price_df, factor_names,
        scheme_id="balanced", market_score=60, top_n=10,
        include_sell_symbols=["000074"],
        include_sell_context={"000074": {"add_date": "2026-01-02", "entry_price": 10.0}}, 
    )

    assert sell
    assert "时间止损2日" in sell[0].entry_reason
    assert "时间止损4日" not in sell[0].entry_reason
    monkeypatch.setattr(scheme, "exit_config", original)


def test_sell_scan_respects_disabled_time_stop(monkeypatch):
    latest = date(2026, 1, 1) + timedelta(days=49)
    flat = [10.0] * 50
    price_df = pd.DataFrame(_bars("000073", flat))
    factor_df = pd.DataFrame([
        _factor_row("000073", latest, rsi14=50),
    ])
    factor_names = [c for c in factor_df.columns if c not in {"symbol", "trade_date"}]
    scheme = BUILTIN_SCHEMES["balanced"]
    original = scheme.exit_config
    monkeypatch.setattr(
        scheme,
        "exit_config",
        ExitConfig(enable_time_stop=False, max_holding_days=99, time_stop_days=2, time_stop_min_profit_pct=0.02),
    )

    _, sell = scan_signals(
        factor_df, price_df, factor_names,
        scheme_id="balanced", market_score=60, top_n=10,
        include_sell_symbols=["000073"],
        include_sell_context={
            "000073": {"add_date": str(latest - timedelta(days=5)), "entry_price": 10.0}
        },
    )

    assert all("time_stop" not in s.layer3_condition_keys for s in sell)
    monkeypatch.setattr(scheme, "exit_config", original)


def test_sell_scan_strategy_failure_breakout_context():
    latest = date(2026, 1, 1) + timedelta(days=49)
    # 平台上沿约10.20，入池后跌回平台内。
    closes = [10.0] * 34 + [10.05, 10.10, 10.15, 10.20, 10.25, 10.45, 10.10, 10.00, 9.95, 9.90, 9.85, 9.80, 9.75, 9.70, 9.65, 9.60]
    price_df = pd.DataFrame(_bars("000072", closes))
    factor_df = pd.DataFrame([
        _factor_row("000072", latest, rsi14=50),
    ])
    factor_names = [c for c in factor_df.columns if c not in {"symbol", "trade_date"}]

    _, sell = scan_signals(
        factor_df, price_df, factor_names,
        scheme_id="balanced", market_score=60, top_n=10,
        include_sell_symbols=["000072"],
        include_sell_context={
            "000072": {
                "add_date": str(latest - timedelta(days=1)),
                "add_reason": "横盘突破",
            }
        },
    )

    assert sell
    assert "突破失败退出" in sell[0].entry_reason
    assert "breakout_failed" in sell[0].layer3_condition_keys


def test_scan_reason_uses_html_safe_text_without_angle_brackets():
    latest = date(2026, 1, 1) + timedelta(days=49)
    strong = [10 + i * 0.15 for i in range(50)]
    weak1 = [10 + i * 0.02 for i in range(50)]
    weak2 = [12 - i * 0.01 for i in range(50)]
    price_df = pd.DataFrame(_bars("000041", strong) + _bars("000042", weak1) + _bars("000043", weak2))
    factor_df = pd.DataFrame([
        _factor_row("000041", latest, momentum_5d=0.08, momentum_20d=0.25, volume_ratio=1.8, rsi14=65, boll_position=0.85),
        _factor_row("000042", latest, momentum_5d=0.01, momentum_20d=0.02, volume_ratio=1.0, rsi14=55, boll_position=0.5),
        _factor_row("000043", latest, momentum_5d=-0.02, momentum_20d=-0.05, volume_ratio=0.8, rsi14=45, boll_position=0.2),
    ])
    factor_names = [c for c in factor_df.columns if c not in {"symbol", "trade_date"}]

    buy, _ = scan_signals(factor_df, price_df, factor_names, scheme_id="trend_momentum", market_score=60, top_n=10)

    assert buy
    assert all("<" not in s.entry_reason and ">" not in s.entry_reason for s in buy)


def test_scan_signal_carries_resonance_config_metadata_for_ui():
    """信号页展示共振 x/y 必须来自 scanner 元数据，不能前端硬编码 /6。"""
    from strategy.schemes import BUILTIN_SCHEMES

    latest = date(2026, 1, 1) + timedelta(days=49)
    strong = [10 + i * 0.15 for i in range(50)]
    weak1 = [10 + i * 0.02 for i in range(50)]
    weak2 = [12 - i * 0.01 for i in range(50)]
    price_df = pd.DataFrame(_bars("000061", strong) + _bars("000062", weak1) + _bars("000063", weak2))
    factor_df = pd.DataFrame([
        _factor_row("000061", latest, momentum_5d=0.08, momentum_20d=0.25, volume_ratio=1.8, rsi14=65, boll_position=0.85),
        _factor_row("000062", latest, momentum_5d=0.01, momentum_20d=0.02, volume_ratio=1.0, rsi14=55, boll_position=0.5),
        _factor_row("000063", latest, momentum_5d=-0.02, momentum_20d=-0.05, volume_ratio=0.8, rsi14=45, boll_position=0.2),
    ])
    factor_names = [c for c in factor_df.columns if c not in {"symbol", "trade_date"}]

    buy, _ = scan_signals(factor_df, price_df, factor_names, scheme_id="trend_momentum", market_score=60, top_n=10)

    assert buy
    sig = buy[0]
    cfg = BUILTIN_SCHEMES[sig.scheme_id].resonance_config
    assert sig.layer3_total == len(cfg.buy_conditions)
    assert sig.layer3_min_confirmations == cfg.min_confirmations
    assert sig.layer3_condition_keys == cfg.buy_conditions
    assert f"共振{sig.layer3_confirmations}/{sig.layer3_total}" in sig.entry_reason


def test_scan_signal_carries_moneyflow_and_turnover_entry_audit_context():
    """P4.2: 信号页 BUY 应携带资金流/相对换手审计上下文，缺失字段显式标记。"""
    latest = date(2026, 1, 1) + timedelta(days=49)
    strong = [10 + i * 0.15 for i in range(50)]
    price_df = pd.DataFrame(
        _bars("000091", strong)
        + _bars("000092", [10 + i * 0.02 for i in range(50)])
        + _bars("000093", [12 - i * 0.01 for i in range(50)])
    )
    factor_df = pd.DataFrame([
        _factor_row(
            "000091", latest,
            momentum_5d=0.08, momentum_20d=0.25, volume_ratio=1.8, rsi14=65, boll_position=0.85,
            main_net_mf_pct_amount=0.0234,
            large_elg_net_mf_pct_amount=0.0189,
            main_net_mf_rank=0.8123,
            large_elg_net_mf_rank=0.7666,
            relative_turnover_5d=1.2345,
            relative_turnover_20d=1.1111,
            turnover_percentile_60d=0.6543,
            # amount_percentile_60d 刻意缺失：真实快照当前仍无历史 amount 支持，不能伪装为0。
        ),
        _factor_row("000092", latest, momentum_5d=0.01, momentum_20d=0.02, volume_ratio=1.0, rsi14=55, boll_position=0.5),
        _factor_row("000093", latest, momentum_5d=-0.02, momentum_20d=-0.05, volume_ratio=0.8, rsi14=45, boll_position=0.2),
    ])
    factor_names = [c for c in factor_df.columns if c not in {"symbol", "trade_date"}]

    buy, _ = scan_signals(factor_df, price_df, factor_names, scheme_id="trend_momentum", market_score=60, top_n=10)

    sig = next(s for s in buy if s.symbol == "000091")
    assert sig.entry_model == "trend_continuation"
    assert sig.main_trigger == "trend_momentum"
    assert "relative_turnover_20d=1.1111" in sig.factor_evidence
    assert "main_net_mf_pct_amount=0.0234" in sig.fund_flow_context
    assert "large_elg_net_mf_rank=0.7666" in sig.fund_flow_context
    assert "market_score=60.00" in sig.market_context
    assert "amount_percentile_60d" in sig.missing_fields
    assert "仅审计不硬过滤" in sig.veto_checks


def test_layered_signal_reasons_use_html_safe_text_without_angle_brackets():
    from signals.layers import TrendFilter

    latest = date(2026, 1, 1) + timedelta(days=59)
    closes = [10 + i * 0.08 for i in range(55)] + [14.0, 13.7, 13.5, 13.3, 13.1]
    bars = pd.DataFrame(_bars("000051", closes, start=latest - timedelta(days=59)))

    passed, reason, score = TrendFilter(strategy_type="pullback").check(bars, len(bars) - 1)

    assert reason
    assert "<" not in reason and ">" not in reason


def test_scan_l4_rejects_sealed_limit_up_candidate():
    """L4 风险可交易性：涨停封死候选不应进入买入榜，避免 T 日信号误导追买。"""
    latest = date(2026, 1, 1) + timedelta(days=49)
    normal = [10 + i * 0.15 for i in range(50)]
    sealed = [10 + i * 0.15 for i in range(49)] + [18.92]
    normal_rows = _bars("000081", normal)
    sealed_rows = _bars("000082", sealed)
    # FIX: 构造最后一根一字涨停K线，高低开收一致，且相对前收接近10%。
    prev_close = sealed_rows[-2]["close"]
    limit_close = round(prev_close * 1.10, 2)
    sealed_rows[-1].update({"open": limit_close, "high": limit_close, "low": limit_close, "close": limit_close})
    weak_rows = _bars("000084", [10 + i * 0.01 for i in range(50)])
    price_df = pd.DataFrame(normal_rows + sealed_rows + weak_rows)
    factor_df = pd.DataFrame([
        _factor_row("000081", latest, momentum_5d=0.08, momentum_20d=0.25, volume_ratio=1.8, rsi14=65, boll_position=0.85),
        _factor_row("000082", latest, momentum_5d=0.08, momentum_20d=0.25, volume_ratio=1.8, rsi14=65, boll_position=0.85),
        _factor_row("000084", latest, momentum_5d=0.01, momentum_20d=0.02, volume_ratio=1.0, rsi14=55, boll_position=0.5),
    ])
    factor_names = [c for c in factor_df.columns if c not in {"symbol", "trade_date"}]

    buy, _ = scan_signals(factor_df, price_df, factor_names, scheme_id="trend_momentum", market_score=60, top_n=10)

    symbols = {s.symbol for s in buy}
    assert "000081" in symbols
    assert "000082" not in symbols


def test_scan_l4_rejects_low_turnover_trend_candidate():
    """强势追涨必须有足够成交额，防止小票滑点被低估。"""
    latest = date(2026, 1, 1) + timedelta(days=49)
    strong = [10 + i * 0.15 for i in range(50)]
    price_df = pd.DataFrame(
        _bars("000083", strong)
        + _bars("000085", [10 + i * 0.02 for i in range(50)])
        + _bars("000086", [12 - i * 0.01 for i in range(50)])
    )
    price_df.loc[price_df["symbol"] == "000083", "amount"] = 20_000_000.0
    factor_df = pd.DataFrame([
        _factor_row(
            "000083", latest,
            momentum_5d=0.08, momentum_20d=0.25, volume_ratio=1.8,
            rsi14=65, boll_position=0.85, amount=20_000_000.0,
        ),
        _factor_row("000085", latest, momentum_5d=0.01, momentum_20d=0.02, volume_ratio=1.0, rsi14=55, boll_position=0.5),
        _factor_row("000086", latest, momentum_5d=-0.02, momentum_20d=-0.05, volume_ratio=0.8, rsi14=45, boll_position=0.2),
    ])
    factor_names = [c for c in factor_df.columns if c not in {"symbol", "trade_date"}]

    buy, _ = scan_signals(factor_df, price_df, factor_names, scheme_id="trend_momentum", market_score=60, top_n=10)

    assert buy == []
