"""P4 参数网格轻量执行入口。

该模块把 `backtest.param_grid` 的参数组合接入现有 SchemeBacktester，
并提供结果落盘工具。设计原则：
- runner 注入，便于单测和小样本 smoke；
- 默认不触发全 A 重任务，由调用方显式传入数据/股票池/日期窗口；
- 输出标准 grid result 表，保留低回撤优先排序口径。
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

import pandas as pd

from backtest.param_grid import GridRankingPolicy, run_parameter_grid
import backtest.scheme_backtest as scheme_backtest_module
from backtest.scheme_backtest import SchemeBacktester, SchemeBacktestResult
from strategy.schemes import StrategyScheme


def _read_frame(path: str | Path) -> pd.DataFrame:
    """读取 csv/parquet 数据文件。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(p, dtype={"symbol": str, "ts_code": str})
    raise ValueError(f"unsupported data file suffix: {suffix}")


def _normalize_trade_dates(df: pd.DataFrame) -> pd.DataFrame:
    """统一 trade_date 类型，避免 CLI 读取 csv 后日期仍是字符串。"""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    if "trade_date" in out.columns:
        out["trade_date"] = pd.to_datetime(out["trade_date"])
    return out


def load_grid_input_frames(factor_path: str | Path, price_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取参数网格输入数据。"""
    factor_df = _normalize_trade_dates(_read_frame(factor_path))
    price_df = _normalize_trade_dates(_read_frame(price_path))
    return factor_df, price_df


def infer_factor_names(factor_df: pd.DataFrame, explicit: Optional[Sequence[str]] = None) -> List[str]:
    """推断可用于回测的因子列。"""
    if explicit:
        return [x for x in explicit if x]
    excluded = {
        "symbol", "ts_code", "trade_date", "date", "open", "high", "low", "close", "volume", "amount",
        "source", "adjust", "name", "industry", "market", "list_date",
    }
    numeric_cols = factor_df.select_dtypes(include="number").columns.tolist()
    return [c for c in numeric_cols if c not in excluded]


def metrics_from_scheme_result(result: SchemeBacktestResult, *, initial_capital: float) -> Mapping[str, object]:
    """从 SchemeBacktestResult 提取 P4 标准指标。"""
    sells = [t for t in (result.trade_details or []) if str(t.get("action", "")).upper() == "SELL"]
    holding_days = [float(t.get("holding_days", 0) or 0) for t in sells]
    avg_holding_days = sum(holding_days) / len(holding_days) if holding_days else 0.0

    max_single_pct = 0.0
    for t in result.trade_details or []:
        try:
            pos_value = abs(float(t.get("position_after", 0) or 0) * float(t.get("exec_price", t.get("price", 0)) or 0))
            max_single_pct = max(max_single_pct, pos_value / float(initial_capital or 1.0))
        except (TypeError, ValueError, ZeroDivisionError):
            continue

    return {
        "total_return": result.total_return,
        "annual_return": result.annual_return,
        "max_drawdown": result.max_drawdown,
        "sharpe_ratio": result.sharpe_ratio,
        "win_rate": result.win_rate,
        "trade_count": result.trade_count,
        "avg_holding_days": avg_holding_days,
        "max_single_pct": max_single_pct,
        "turnover_rate": 0.0,
    }


def make_scheme_grid_runner(
    *,
    factor_df: pd.DataFrame,
    price_df: pd.DataFrame,
    factor_names: Sequence[str],
    symbols: Optional[Sequence[str]] = None,
    lookback_days: int = 60,
    top_n: int = 10,
    initial_capital: float = 1_000_000.0,
    verbose: bool = False,
    prefer_price_data: bool = True,
):
    """构造注入 `run_parameter_grid` 的轻量 runner。"""
    factor_df = _normalize_trade_dates(factor_df)
    price_df = _normalize_trade_dates(price_df)
    factor_names = list(factor_names)
    symbols_list = list(symbols) if symbols else None

    def _runner(scheme: StrategyScheme, params: Mapping[str, object]) -> Mapping[str, object]:
        original_fetch = scheme_backtest_module._fetch_ohlcv
        if prefer_price_data:
            # P4 smoke/CLI 默认只用传入的不复权 price_df，避免误触发外部全量行情拉取。
            scheme_backtest_module._fetch_ohlcv = lambda *args, **kwargs: pd.DataFrame()
        try:
            result = SchemeBacktester().run(
                scheme=scheme,
                factor_df=factor_df,
                price_df=price_df,
                factor_names=factor_names,
                symbols=symbols_list,
                lookback_days=lookback_days,
                top_n=top_n,
                initial_capital=initial_capital,
                verbose=verbose,
            )
        finally:
            scheme_backtest_module._fetch_ohlcv = original_fetch
        return metrics_from_scheme_result(result, initial_capital=initial_capital)

    return _runner


def run_scheme_parameter_grid(
    *,
    strategy_id: str,
    factor_df: pd.DataFrame,
    price_df: pd.DataFrame,
    factor_names: Sequence[str],
    symbols: Optional[Sequence[str]] = None,
    lookback_days: int = 60,
    top_n: int = 10,
    initial_capital: float = 1_000_000.0,
    max_runs: int | None = None,
    policy: GridRankingPolicy | None = None,
    verbose: bool = False,
    prefer_price_data: bool = True,
) -> pd.DataFrame:
    """用现有 SchemeBacktester 执行参数网格。"""
    runner = make_scheme_grid_runner(
        factor_df=factor_df,
        price_df=price_df,
        factor_names=factor_names,
        symbols=symbols,
        lookback_days=lookback_days,
        top_n=top_n,
        initial_capital=initial_capital,
        verbose=verbose,
        prefer_price_data=prefer_price_data,
    )
    return run_parameter_grid(
        strategy_id=strategy_id,
        runner=runner,
        max_runs=max_runs,
        policy=policy,
    )


def save_grid_results(df: pd.DataFrame, output_path: str | Path) -> Path:
    """保存参数网格结果。"""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(p, index=False)
    elif suffix in {".csv", ".txt", ""}:
        if suffix == "":
            p = p.with_suffix(".csv")
        df.to_csv(p, index=False)
    else:
        raise ValueError(f"unsupported output suffix: {suffix}")
    return p


def build_grid_audit_config(
    *,
    strategy_id: str,
    max_runs: int | None,
    lookback_days: int,
    top_n: int,
    initial_capital: float,
    symbols: Optional[Sequence[str]],
    factor_names: Sequence[str],
    policy: GridRankingPolicy | None = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict:
    """构造参数网格实验审计配置。"""
    policy = policy or GridRankingPolicy()
    return {
        "strategy_id": strategy_id,
        "max_runs": max_runs,
        "lookback_days": lookback_days,
        "top_n": top_n,
        "initial_capital": initial_capital,
        "symbols": list(symbols) if symbols else [],
        "factor_names": list(factor_names),
        "ranking_policy": asdict(policy),
        "prefer_price_data": True,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "extra": dict(extra or {}),
    }


def _format_pct(value: object) -> str:
    try:
        return f"{float(value) * 100:.4f}%"
    except (TypeError, ValueError):
        return "0.0000%"


def render_grid_summary(df: pd.DataFrame, config: Mapping[str, Any]) -> str:
    """渲染参数网格实验摘要。"""
    lines = [
        "# 参数网格验证摘要",
        "",
        f"- 策略: `{config.get('strategy_id', '')}`",
        f"- 组合数: `{len(df)}`",
        f"- max_runs: `{config.get('max_runs')}`",
        f"- lookback_days: `{config.get('lookback_days')}`",
        f"- top_n: `{config.get('top_n')}`",
        f"- initial_capital: `{float(config.get('initial_capital', 0.0)):.2f}`",
        f"- symbols: `{','.join(config.get('symbols') or [])}`",
        f"- factor_names: `{','.join(config.get('factor_names') or [])}`",
        f"- ranking_policy: `{json.dumps(config.get('ranking_policy', {}), ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    if df.empty:
        lines.append("无参数网格结果。")
        return "\n".join(lines) + "\n"

    top = df.sort_values("rank").head(5)
    lines.extend([
        "## Top 5",
        "",
        "| rank | eligible | total_return | max_drawdown | sharpe | trade_count | params |",
        "|---:|:---:|---:|---:|---:|---:|---|",
    ])
    for _, row in top.iterrows():
        params = str(row.get("params_json", "")).replace("|", "\\|")
        lines.append(
            f"| {int(row.get('rank', 0))} | {bool(row.get('eligible', False))} | "
            f"{_format_pct(row.get('total_return', 0.0))} | {_format_pct(row.get('max_drawdown', 0.0))} | "
            f"{float(row.get('sharpe_ratio', 0.0)):.4f} | {int(row.get('trade_count', 0))} | `{params}` |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def persist_grid_audit_run(
    *,
    df: pd.DataFrame,
    root: str | Path = "data/grid_results",
    strategy_id: str,
    config: Mapping[str, Any],
    run_id: Optional[str] = None,
) -> Path:
    """按审计目录规范持久化参数网格实验。"""
    rid = run_id or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{strategy_id}"
    run_dir = Path(root) / rid
    run_dir.mkdir(parents=True, exist_ok=False)
    df.to_csv(run_dir / "grid_results.csv", index=False)
    df.to_parquet(run_dir / "grid_results.parquet", index=False)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(dict(config), f, ensure_ascii=False, indent=2, sort_keys=True)
    (run_dir / "summary.md").write_text(render_grid_summary(df, config), encoding="utf-8")
    return run_dir
