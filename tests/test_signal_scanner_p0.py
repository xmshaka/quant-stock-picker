"""P0 新方案信号扫描测试。"""
from datetime import date, timedelta

import pandas as pd

from signals.scanner import scan_signals


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
    assert "大盘防御档" in sell[0].entry_reason


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


def test_layered_signal_reasons_use_html_safe_text_without_angle_brackets():
    from signals.layers import TrendFilter

    latest = date(2026, 1, 1) + timedelta(days=59)
    closes = [10 + i * 0.08 for i in range(55)] + [14.0, 13.7, 13.5, 13.3, 13.1]
    bars = pd.DataFrame(_bars("000051", closes, start=latest - timedelta(days=59)))

    passed, reason, score = TrendFilter(strategy_type="pullback").check(bars, len(bars) - 1)

    assert reason
    assert "<" not in reason and ">" not in reason
