"""P4 参数网格验证工具。

目标：在不牺牲审计口径的前提下，对不同策略的保守参数组合做稳定性验证。
本模块只负责：生成参数组合、克隆策略配置、归一化结果、低回撤优先排序。
真实回测由调用方注入 runner，避免单元测试依赖重型数据源。
"""
from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

import pandas as pd

from strategy.schemes import BUILTIN_SCHEMES, StrategyScheme


GridParams = Dict[str, object]
GridRunner = Callable[[StrategyScheme, GridParams], Mapping[str, object]]


@dataclass(frozen=True)
class GridRankingPolicy:
    """参数网格排序策略。

    低回撤优先，不允许只按收益排序。
    """
    min_trades: int = 3
    max_drawdown_limit: float = 0.20
    min_total_return: float = 0.0


DEFAULT_PARAM_SPACE: Dict[str, Dict[str, Sequence[object]]] = {
    "trend_momentum": {
        "stop_loss_atr_mult": [1.8, 2.0, 2.2],
        "take_profit_atr_mult": [2.5, 3.0],
        "trailing_atr_mult": [1.8, 2.0],
        "exit_config.max_holding_days": [8, 10],
        "exit_config.time_stop_days": [4, 5],
        "resonance_config.min_confirmations": [3, 4],
        "exit_config.market_defense_score": [20.0, 30.0],
    },
    "pullback": {
        "stop_loss_atr_mult": [1.8, 2.0, 2.2],
        "take_profit_atr_mult": [2.5, 3.0],
        "trailing_atr_mult": [1.8, 2.0],
        "exit_config.max_holding_days": [15, 20],
        "exit_config.time_stop_days": [7, 10],
        "resonance_config.min_confirmations": [3, 4],
        "exit_config.market_defense_score": [20.0, 30.0],
    },
    "breakout": {
        "stop_loss_atr_mult": [1.8, 2.0, 2.2],
        "take_profit_atr_mult": [3.0, 3.5],
        "trailing_atr_mult": [1.8, 2.0],
        "exit_config.max_holding_days": [8, 10],
        "exit_config.time_stop_days": [4, 5],
        "resonance_config.min_confirmations": [3, 4],
        "exit_config.market_defense_score": [20.0, 30.0],
    },
    "balanced": {
        "stop_loss_atr_mult": [1.8, 2.0, 2.2],
        "take_profit_atr_mult": [2.5, 3.0],
        "trailing_atr_mult": [1.8, 2.0],
        "exit_config.trailing_activation_pct": [0.05, 0.08],
        "exit_config.trailing_activation_atr_mult": [1.0, 1.5],
        "exit_config.market_defense_score": [20.0, 30.0],
        # 放在末尾，确保 CLI 默认/小样本 max_runs 能优先覆盖 15/20 与 7/10 的短线退出组合。
        "exit_config.max_holding_days": [15, 20],
        "exit_config.time_stop_days": [7, 10],
    },
}


GRID_RESULT_COLUMNS = [
    "rank",
    "scheme_id",
    "params_json",
    "eligible",
    "risk_score",
    "stability_score",
    "total_return",
    "annual_return",
    "max_drawdown",
    "sharpe_ratio",
    "win_rate",
    "trade_count",
    "avg_holding_days",
    "max_single_pct",
    "turnover_rate",
]


def _set_nested_attr(obj: object, path: str, value: object) -> None:
    """设置 StrategyScheme / nested config 的 dot-path 参数。"""
    parts = path.split(".")
    target = obj
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise AttributeError(f"unknown parameter path: {path}")
        target = getattr(target, part)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise AttributeError(f"unknown parameter path: {path}")
    setattr(target, leaf, value)


def clone_scheme_with_params(base: StrategyScheme, params: Mapping[str, object]) -> StrategyScheme:
    """克隆策略并应用参数，不污染 BUILTIN_SCHEMES。"""
    cloned = StrategyScheme.from_dict(base.to_dict())
    for path, value in params.items():
        _set_nested_attr(cloned, path, value)
    return cloned


def iter_param_grid(param_space: Mapping[str, Sequence[object]], *, max_runs: int | None = None) -> List[GridParams]:
    """将参数空间展开为确定顺序的组合列表。"""
    keys = list(param_space.keys())
    values = [list(param_space[k]) for k in keys]
    rows: List[GridParams] = []
    for combo in itertools.product(*values):
        rows.append(dict(zip(keys, combo)))
        if max_runs is not None and len(rows) >= max_runs:
            break
    return rows


def default_param_grid(strategy_id: str, *, max_runs: int | None = None) -> List[GridParams]:
    """返回内置策略的保守参数网格。"""
    if strategy_id not in DEFAULT_PARAM_SPACE:
        raise KeyError(f"unknown strategy_id for param grid: {strategy_id}")
    return iter_param_grid(DEFAULT_PARAM_SPACE[strategy_id], max_runs=max_runs)


def _json_params(params: Mapping[str, object]) -> str:
    return json.dumps(dict(params), ensure_ascii=False, sort_keys=True)


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: object, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_grid_result(
    *,
    scheme_id: str,
    params: Mapping[str, object],
    metrics: Mapping[str, object],
) -> Dict[str, object]:
    """把一次回测结果归一化为 P4 结果表行。"""
    max_drawdown = abs(_float(metrics.get("max_drawdown", 0.0)))
    total_return = _float(metrics.get("total_return", 0.0))
    annual_return = _float(metrics.get("annual_return", 0.0))
    sharpe = _float(metrics.get("sharpe_ratio", 0.0))
    trade_count = _int(metrics.get("trade_count", 0))
    win_rate = _float(metrics.get("win_rate", 0.0))
    avg_holding_days = _float(metrics.get("avg_holding_days", 0.0))
    max_single_pct = _float(metrics.get("max_single_pct", 0.0))
    turnover_rate = _float(metrics.get("turnover_rate", 0.0))
    # 风险分数越低越好：回撤是主项，集中度/换手作为次级风险惩罚。
    risk_score = max_drawdown + max(0.0, max_single_pct - 0.20) * 0.5 + max(0.0, turnover_rate - 3.0) * 0.02
    # 稳定性分数只用于同风险层内辅助排序，不能替代低回撤优先。
    stability_score = total_return - 2.0 * max_drawdown + 0.10 * sharpe + 0.05 * win_rate
    return {
        "rank": 0,
        "scheme_id": scheme_id,
        "params_json": _json_params(params),
        "eligible": False,
        "risk_score": risk_score,
        "stability_score": stability_score,
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "trade_count": trade_count,
        "avg_holding_days": avg_holding_days,
        "max_single_pct": max_single_pct,
        "turnover_rate": turnover_rate,
    }


def rank_grid_results(rows: Iterable[Mapping[str, object]], policy: GridRankingPolicy | None = None) -> pd.DataFrame:
    """按低回撤优先的口径排序参数网格结果。"""
    policy = policy or GridRankingPolicy()
    df = pd.DataFrame(list(rows))
    if df.empty:
        return pd.DataFrame(columns=GRID_RESULT_COLUMNS)

    for col in [
        "risk_score", "stability_score", "total_return", "annual_return", "max_drawdown",
        "sharpe_ratio", "win_rate", "trade_count", "avg_holding_days", "max_single_pct", "turnover_rate",
    ]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["eligible"] = (
        (df["total_return"] > policy.min_total_return)
        & (df["trade_count"] >= policy.min_trades)
        & (df["max_drawdown"].abs() <= policy.max_drawdown_limit)
    )
    df = df.sort_values(
        ["eligible", "risk_score", "max_drawdown", "total_return", "trade_count", "sharpe_ratio", "stability_score"],
        ascending=[False, True, True, False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    for col in GRID_RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col in {"scheme_id", "params_json"} else 0
    return df[GRID_RESULT_COLUMNS]


def run_parameter_grid(
    *,
    strategy_id: str,
    runner: GridRunner,
    param_grid: Sequence[Mapping[str, object]] | None = None,
    policy: GridRankingPolicy | None = None,
    max_runs: int | None = None,
) -> pd.DataFrame:
    """运行参数网格。

    runner 接收应用参数后的 StrategyScheme 与 params，返回绩效指标 dict。
    """
    if strategy_id not in BUILTIN_SCHEMES:
        raise KeyError(f"unknown strategy_id: {strategy_id}")
    base = BUILTIN_SCHEMES[strategy_id]
    grid = list(param_grid) if param_grid is not None else default_param_grid(strategy_id, max_runs=max_runs)
    rows: List[Dict[str, object]] = []
    for params in grid:
        scheme = clone_scheme_with_params(base, params)
        metrics = runner(scheme, dict(params))
        rows.append(normalize_grid_result(scheme_id=strategy_id, params=params, metrics=metrics))
    return rank_grid_results(rows, policy=policy)
