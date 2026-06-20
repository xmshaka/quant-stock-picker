"""参数网格结果页辅助函数。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pandas as pd


GRID_ROOT = Path("data/grid_results")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def list_grid_audit_runs(root: str | Path = GRID_ROOT) -> pd.DataFrame:
    """列出参数网格审计目录。"""
    base = Path(root)
    rows: List[Dict[str, Any]] = []
    if not base.exists():
        return pd.DataFrame(columns=[
            "run_id", "strategy_id", "created_at", "max_runs", "lookback_days", "top_n",
            "initial_capital", "row_count", "eligible_count", "best_return", "best_drawdown", "path",
        ])
    for d in sorted([p for p in base.iterdir() if p.is_dir()], reverse=True):
        cfg = _read_json(d / "config.json")
        results_path = d / "grid_results.csv"
        if not results_path.exists():
            continue
        try:
            df = pd.read_csv(results_path)
        except Exception:
            df = pd.DataFrame()
        rows.append({
            "run_id": d.name,
            "strategy_id": cfg.get("strategy_id", ""),
            "created_at": cfg.get("created_at", ""),
            "max_runs": cfg.get("max_runs"),
            "lookback_days": cfg.get("lookback_days"),
            "top_n": cfg.get("top_n"),
            "initial_capital": cfg.get("initial_capital", 0),
            "row_count": int(len(df)),
            "eligible_count": int(df["eligible"].astype(bool).sum()) if "eligible" in df.columns and not df.empty else 0,
            "best_return": float(pd.to_numeric(df.get("total_return", pd.Series(dtype=float)), errors="coerce").max()) if not df.empty else 0.0,
            "best_drawdown": float(pd.to_numeric(df.get("max_drawdown", pd.Series(dtype=float)), errors="coerce").min()) if not df.empty else 0.0,
            "path": str(d),
        })
    return pd.DataFrame(rows)


def load_grid_audit_run(run_path: str | Path) -> Dict[str, Any]:
    """读取单个参数网格审计目录。"""
    p = Path(run_path)
    cfg = _read_json(p / "config.json")
    summary = (p / "summary.md").read_text(encoding="utf-8") if (p / "summary.md").exists() else ""
    csv_path = p / "grid_results.csv"
    if csv_path.exists():
        results = pd.read_csv(csv_path)
    elif (p / "grid_results.parquet").exists():
        results = pd.read_parquet(p / "grid_results.parquet")
    else:
        results = pd.DataFrame()
    return {"config": cfg, "summary": summary, "results": results, "path": str(p), "run_id": p.name}


def format_grid_results_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """格式化前端结果表，突出低回撤而非高收益。"""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in ["total_return", "annual_return", "max_drawdown", "win_rate", "risk_score", "stability_score", "max_single_pct", "turnover_rate"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    rename = {
        "rank": "排名",
        "eligible": "合格",
        "risk_score": "风险分",
        "stability_score": "稳定分",
        "total_return": "总收益",
        "annual_return": "年化",
        "max_drawdown": "最大回撤",
        "sharpe_ratio": "夏普",
        "win_rate": "胜率",
        "trade_count": "交易数",
        "avg_holding_days": "平均持仓天数",
        "max_single_pct": "最大单票占比",
        "turnover_rate": "换手",
        "params_json": "参数",
    }
    keep = [c for c in [
        "rank", "eligible", "risk_score", "total_return", "max_drawdown", "sharpe_ratio",
        "win_rate", "trade_count", "avg_holding_days", "max_single_pct", "params_json",
    ] if c in out.columns]
    out = out[keep].rename(columns=rename)
    for col in ["总收益", "最大回撤", "胜率", "最大单票占比"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{float(x):.4%}")
    for col in ["风险分", "稳定分", "夏普", "平均持仓天数"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{float(x):.4f}")
    return out


def summarize_grid_run(config: Mapping[str, Any], results: pd.DataFrame) -> Dict[str, str]:
    """生成页面指标卡摘要。"""
    eligible = int(results["eligible"].astype(bool).sum()) if "eligible" in results.columns and not results.empty else 0
    best = results.sort_values("rank").iloc[0] if not results.empty and "rank" in results.columns else {}
    return {
        "策略": str(config.get("strategy_id", "—")),
        "组合数": f"{len(results)} 组",
        "合格组合": f"{eligible} 组",
        "Top1收益": f"{float(best.get('total_return', 0.0)):.4%}" if len(results) else "—",
        "Top1回撤": f"{float(best.get('max_drawdown', 0.0)):.4%}" if len(results) else "—",
    }
