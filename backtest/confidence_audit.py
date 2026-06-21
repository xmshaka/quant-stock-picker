"""Confidence bucket audit utilities.

读取已落盘 backtest runs，按开仓 confidence_bucket 汇总交易绩效。
第一阶段仅审计，不改变任何回测交易结果或策略阈值。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd

from backtest.records import BACKTEST_RUN_ROOT, load_backtest_run, summarize_confidence_performance


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except Exception:
        return "0.00%"


def _fmt_num(value: Any) -> str:
    try:
        return f"{float(value):,.4f}"
    except Exception:
        return "0.0000"


def iter_run_ids(root: Path = BACKTEST_RUN_ROOT, pattern: str = "*") -> list[str]:
    """列出可审计 run_id。"""
    root = Path(root)
    if not root.exists():
        return []
    return sorted([p.name for p in root.glob(pattern) if p.is_dir()])


def collect_confidence_performance(
    *,
    run_ids: Optional[Sequence[str]] = None,
    root: Path = BACKTEST_RUN_ROOT,
    pattern: str = "*",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """汇总多个 backtest run 的 confidence 分桶绩效。

    Returns:
        summary_df: 每个 run + bucket 的绩效摘要。
        details_df: 每一轮 BUY→SELL 的明细。
    """
    selected = list(run_ids) if run_ids else iter_run_ids(root=root, pattern=pattern)
    summary_rows: list[pd.DataFrame] = []
    detail_rows: list[pd.DataFrame] = []
    for run_id in selected:
        try:
            run = load_backtest_run(run_id, root=root)
        except Exception:
            continue
        trades = run.get("trades", pd.DataFrame())
        conf = summarize_confidence_performance(trades)
        if not conf.get("ok"):
            continue
        config = run.get("config", {}) or {}
        metrics = run.get("metrics", {}) or {}
        summary = conf["rounds"].copy()
        if not summary.empty:
            summary.insert(0, "run_id", run_id)
            summary.insert(1, "scheme_id", config.get("scheme_id", ""))
            summary.insert(2, "scheme_name", config.get("scheme_name", ""))
            summary.insert(3, "pool_mode", config.get("pool_mode", ""))
            summary.insert(4, "symbols", ",".join(config.get("symbols", []) or []))
            summary["run_total_return"] = float(metrics.get("total_return", 0.0) or 0.0)
            summary["run_max_drawdown"] = float(metrics.get("max_drawdown", 0.0) or 0.0)
            summary_rows.append(summary)
        details = conf["details"].copy()
        if not details.empty:
            details.insert(0, "run_id", run_id)
            details.insert(1, "scheme_id", config.get("scheme_id", ""))
            details.insert(2, "pool_mode", config.get("pool_mode", ""))
            detail_rows.append(details)

    summary_df = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    details_df = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    return summary_df, details_df


def aggregate_confidence_buckets(details_df: pd.DataFrame) -> pd.DataFrame:
    """跨 run 汇总每个 confidence_bucket 的完成交易轮次表现。"""
    if details_df is None or details_df.empty:
        return pd.DataFrame(columns=[
            "confidence_bucket", "confidence_action", "完成轮数", "胜率", "总盈亏", "平均盈亏",
            "平均收益率", "最大单笔亏损", "平均持仓天数", "run_count",
        ])
    df = details_df.copy()
    for col in ("pnl", "pnl_pct", "holding_days"):
        df[col] = pd.to_numeric(df.get(col, 0.0), errors="coerce").fillna(0.0)
    out = (
        df.groupby(["confidence_bucket", "confidence_action"], dropna=False)
        .agg(
            完成轮数=("symbol", "count"),
            胜率=("pnl", lambda s: float((s > 0).mean()) if len(s) else 0.0),
            总盈亏=("pnl", "sum"),
            平均盈亏=("pnl", "mean"),
            平均收益率=("pnl_pct", "mean"),
            最大单笔亏损=("pnl", "min"),
            平均持仓天数=("holding_days", "mean"),
            run_count=("run_id", "nunique"),
        )
        .reset_index()
        .sort_values(["confidence_bucket", "完成轮数"], ascending=[True, False])
    )
    return out


def render_confidence_audit_report(
    *,
    summary_df: pd.DataFrame,
    details_df: pd.DataFrame,
    config: Optional[Mapping[str, Any]] = None,
) -> str:
    """渲染 Markdown 审计报告。"""
    config = dict(config or {})
    agg = aggregate_confidence_buckets(details_df)
    lines = [
        "# Confidence 分桶绩效审计报告",
        "",
        f"- 生成时间: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- run_pattern: `{config.get('pattern', '*')}`",
        f"- run_count: `{int(details_df['run_id'].nunique()) if not details_df.empty and 'run_id' in details_df else 0}`",
        f"- round_count: `{len(details_df) if details_df is not None else 0}`",
        "- 口径: 审计模式，只读取已落盘 trades，不改变交易结果，不作为硬过滤。",
        "",
        "## 跨 run 分桶汇总",
        "",
    ]
    if agg.empty:
        lines.append("无可审计 confidence 分桶数据。")
        return "\n".join(lines) + "\n"

    lines.extend([
        "| confidence_bucket | confidence_action | 完成轮数 | 胜率 | 总盈亏 | 平均盈亏 | 平均收益率 | 最大单笔亏损 | 平均持仓天数 | run_count |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in agg.iterrows():
        lines.append(
            f"| {row.get('confidence_bucket', '')} | {row.get('confidence_action', '')} | "
            f"{int(row.get('完成轮数', 0))} | {_fmt_pct(row.get('胜率', 0.0))} | "
            f"{_fmt_num(row.get('总盈亏', 0.0))} | {_fmt_num(row.get('平均盈亏', 0.0))} | "
            f"{_fmt_pct(row.get('平均收益率', 0.0))} | {_fmt_num(row.get('最大单笔亏损', 0.0))} | "
            f"{_fmt_num(row.get('平均持仓天数', 0.0))} | {int(row.get('run_count', 0))} |"
        )
    lines.extend([
        "",
        "## 专业解读约束",
        "",
        "- 本报告只能证明当前样本中的分桶表现，不能单独决定硬阈值。",
        "- 若 watch/candidate 长期呈现低胜率或负收益，可进入候选硬过滤/降仓验证。",
        "- 加仓规则必须另行验证：盈利后、同路径、高置信度、结构未破，不能只凭 BUY 重复出现。",
        "",
    ])
    if summary_df is not None and not summary_df.empty:
        lines.extend([
            "## Run 级明细位置",
            "",
            "详见 `confidence_summary.csv` 与 `confidence_details.csv`。",
            "",
        ])
    return "\n".join(lines) + "\n"


def persist_confidence_audit(
    *,
    output_dir: str | Path,
    summary_df: pd.DataFrame,
    details_df: pd.DataFrame,
    config: Optional[Mapping[str, Any]] = None,
) -> Path:
    """保存 confidence 审计结果。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out / "confidence_summary.csv", index=False)
    details_df.to_csv(out / "confidence_details.csv", index=False)
    aggregate_confidence_buckets(details_df).to_csv(out / "confidence_bucket_aggregate.csv", index=False)
    (out / "summary.md").write_text(
        render_confidence_audit_report(summary_df=summary_df, details_df=details_df, config=config),
        encoding="utf-8",
    )
    return out
