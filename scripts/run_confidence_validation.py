#!/usr/bin/env python3
"""中样本 confidence 分桶验证。

运行指定策略/股票池，持久化新 backtest runs，再对这些 run 做 confidence 分桶审计。
默认只跑小中样本，且不改变任何交易规则。
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DASHBOARD = ROOT / "dashboard"
if str(DASHBOARD) not in sys.path:
    sys.path.insert(0, str(DASHBOARD))

from data_loader import load_data  # noqa: E402
from strategy.registry import SchemeRegistry  # noqa: E402
from backtest.scheme_backtest import SchemeBacktester  # noqa: E402
from backtest.records import BacktestRunConfig, scheme_audit_snapshot  # noqa: E402
from backtest.confidence_audit import collect_confidence_performance, persist_confidence_audit  # noqa: E402


def _parse_csv(raw: str | None) -> list[str]:
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


def _default_symbols(factor_df, limit: int) -> list[str]:
    if factor_df is None or factor_df.empty or "symbol" not in factor_df.columns:
        return []
    syms = factor_df["symbol"].astype(str).str.zfill(6).dropna().drop_duplicates().tolist()
    return syms[: max(int(limit or 0), 0)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行中样本 confidence 分桶验证（审计模式，不改交易规则）")
    parser.add_argument("--strategies", default="balanced,pullback", help="逗号分隔策略，如 balanced,pullback")
    parser.add_argument("--symbols", default="", help="逗号分隔股票代码；为空则从数据中取前N只")
    parser.add_argument("--symbol-limit", type=int, default=20)
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--data-source", default="real")
    parser.add_argument("--n-stocks", type=int, default=300)
    parser.add_argument("--n-days", type=int, default=252)
    parser.add_argument("--output-dir", default="", help="confidence audit 输出目录；为空自动生成")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    strategies = _parse_csv(args.strategies)
    if not strategies:
        raise SystemExit("至少指定一个 strategy")

    factor_df, price_df, factor_names = load_data(
        data_source=args.data_source,
        n_stocks=args.n_stocks,
        n_days=args.n_days,
    )
    symbols = _parse_csv(args.symbols) or _default_symbols(factor_df, args.symbol_limit)
    if not symbols:
        raise SystemExit("未找到可用 symbols")

    registry = SchemeRegistry()
    run_ids: list[str] = []
    for strategy_id in strategies:
        scheme = registry.get(strategy_id)
        result = SchemeBacktester().run(
            scheme=scheme,
            factor_df=factor_df,
            price_df=price_df,
            factor_names=factor_names,
            symbols=symbols,
            lookback_days=args.lookback_days,
            top_n=args.top_n,
            initial_capital=args.initial_capital,
            verbose=args.verbose,
        )
        snapshot = scheme_audit_snapshot(scheme)
        cfg = BacktestRunConfig(
            run_id=result.run_id,
            scheme_id=scheme.scheme_id,
            scheme_name=scheme.name,
            start_date=result.start_date,
            end_date=result.end_date,
            lookback_days=args.lookback_days,
            top_n=args.top_n,
            initial_capital=args.initial_capital,
            pool_mode="confidence_validation_cli",
            symbols=symbols,
            cost={"commission": 0.00025, "stamp_duty": 0.001, "transfer_fee": 0.00001, "slippage": 0.002},
            risk={"single_position_cap": 0.20, "total_position_cap": 0.90},
            scheme_config=snapshot["scheme_config"],
            resonance_config=snapshot["resonance_config"],
            data_version=result.data_version,
        )
        run_dir = result.persist(config=cfg)
        run_ids.append(result.run_id)
        print(f"RUN {strategy_id}: {result.run_id} -> {run_dir}")
        print(result.summary_text())

    out_dir = Path(args.output_dir) if args.output_dir else Path("data/confidence_audit") / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    summary_df, details_df = collect_confidence_performance(run_ids=run_ids)
    persist_confidence_audit(
        output_dir=out_dir,
        summary_df=summary_df,
        details_df=details_df,
        config={
            "run_ids": run_ids,
            "strategies": strategies,
            "symbols": symbols,
            "lookback_days": args.lookback_days,
            "top_n": args.top_n,
            "initial_capital": args.initial_capital,
            "audit_mode": "confidence_only_no_filter",
        },
    )
    print(f"AUDIT {out_dir}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
