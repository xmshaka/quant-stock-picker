#!/usr/bin/env python3
"""读取已落盘 backtest runs，生成 confidence 分桶绩效审计报告。"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.confidence_audit import (  # noqa: E402
    aggregate_confidence_buckets,
    collect_confidence_performance,
    persist_confidence_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 confidence 分桶绩效审计报告（只读历史 run，不改交易结果）")
    parser.add_argument("--run-root", default="data/backtest_runs", help="backtest run 根目录")
    parser.add_argument("--pattern", default="20260620_*", help="run 目录 glob，例如 20260620_*_balanced")
    parser.add_argument("--run-ids", default="", help="逗号分隔 run_id；提供后优先于 pattern")
    parser.add_argument("--output-dir", default="", help="输出目录；为空时写入 data/confidence_audit/<timestamp>")
    return parser


def _parse_run_ids(raw: str) -> list[str] | None:
    vals = [x.strip() for x in str(raw or "").split(",") if x.strip()]
    return vals or None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.run_root)
    run_ids = _parse_run_ids(args.run_ids)
    summary_df, details_df = collect_confidence_performance(
        run_ids=run_ids,
        root=root,
        pattern=args.pattern,
    )
    output_dir = Path(args.output_dir) if args.output_dir else Path("data/confidence_audit") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out = persist_confidence_audit(
        output_dir=output_dir,
        summary_df=summary_df,
        details_df=details_df,
        config={"pattern": args.pattern, "run_ids": run_ids or [], "run_root": str(root)},
    )
    agg = aggregate_confidence_buckets(details_df)
    print(f"confidence audit saved to {out}")
    print(f"summary rows={len(summary_df)}, detail rows={len(details_df)}")
    if not agg.empty:
        print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
