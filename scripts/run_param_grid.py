#!/usr/bin/env python3
"""P4 参数网格轻量执行 CLI。

示例：
python scripts/run_param_grid.py \
  --strategy-id pullback \
  --factor-path data/factors_sample.parquet \
  --price-path data/prices_sample.parquet \
  --max-runs 3 \
  --lookback-days 60 \
  --top-n 10 \
  --output data/grid_results/pullback_smoke.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.param_grid_runner import (  # noqa: E402
    build_grid_audit_config,
    infer_factor_names,
    load_grid_input_frames,
    persist_grid_audit_run,
    run_scheme_parameter_grid,
    save_grid_results,
)


def _parse_symbols(raw: str | None):
    if not raw:
        return None
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_factor_names(raw: str | None):
    if not raw:
        return None
    return [x.strip() for x in raw.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 P4 参数网格轻量回测")
    parser.add_argument("--strategy-id", required=True, choices=["trend_momentum", "pullback", "breakout", "balanced"])
    parser.add_argument("--factor-path", required=True, help="因子数据 csv/parquet，需包含 symbol/trade_date")
    parser.add_argument("--price-path", required=True, help="不复权价格数据 csv/parquet，需包含 symbol/trade_date/open/high/low/close")
    parser.add_argument("--output", required=True, help="输出 csv/parquet 路径")
    parser.add_argument("--audit-root", default="", help="可选：按审计目录同时输出 grid_results/config/summary")
    parser.add_argument("--factor-names", default="", help="逗号分隔因子列；为空则自动推断数值因子列")
    parser.add_argument("--symbols", default="", help="逗号分隔股票池；为空表示全池")
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--max-runs", type=int, default=3, help="默认只跑3组，防止误触发重任务")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    factor_df, price_df = load_grid_input_frames(args.factor_path, args.price_path)
    factor_names = infer_factor_names(factor_df, _parse_factor_names(args.factor_names))
    if not factor_names:
        raise SystemExit("未找到可用因子列，请通过 --factor-names 显式指定")
    symbols = _parse_symbols(args.symbols)
    df = run_scheme_parameter_grid(
        strategy_id=args.strategy_id,
        factor_df=factor_df,
        price_df=price_df,
        factor_names=factor_names,
        symbols=symbols,
        lookback_days=args.lookback_days,
        top_n=args.top_n,
        initial_capital=args.initial_capital,
        max_runs=args.max_runs,
        verbose=args.verbose,
    )
    out = save_grid_results(df, args.output)
    print(f"saved {len(df)} grid rows to {out}")
    if args.audit_root:
        cfg = build_grid_audit_config(
            strategy_id=args.strategy_id,
            max_runs=args.max_runs,
            lookback_days=args.lookback_days,
            top_n=args.top_n,
            initial_capital=args.initial_capital,
            symbols=symbols,
            factor_names=factor_names,
            extra={"factor_path": args.factor_path, "price_path": args.price_path, "output": str(out)},
        )
        audit_dir = persist_grid_audit_run(df=df, root=args.audit_root, strategy_id=args.strategy_id, config=cfg)
        print(f"audit saved to {audit_dir}")
    if not df.empty:
        print(df.head(min(len(df), 10)).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
