"""历史回测记录筛选辅助函数。"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable

import pandas as pd


def unique_non_empty(values: Iterable[object]) -> list[str]:
    """提取非空选项并排序，供 Streamlit selectbox 使用。"""
    cleaned = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            cleaned.add(text)
    return sorted(cleaned)


def filter_backtest_runs(
    runs: pd.DataFrame,
    *,
    symbol_query: str = "",
    scheme_name: str = "全部",
    pool_mode: str = "全部",
    recent_days: int = 0,
    now: datetime | None = None,
) -> pd.DataFrame:
    """按股票代码、策略、股票池和保存时间过滤回测记录。"""
    if runs is None or runs.empty:
        return pd.DataFrame(columns=runs.columns if runs is not None else [])

    out = runs.copy()

    query = str(symbol_query or "").strip()
    if query:
        symbols = out.get("symbols", pd.Series("", index=out.index)).fillna("").astype(str)
        run_ids = out.get("run_id", pd.Series("", index=out.index)).fillna("").astype(str)
        out = out[symbols.str.contains(query, case=False, regex=False) | run_ids.str.contains(query, case=False, regex=False)]

    if scheme_name and scheme_name != "全部" and "scheme_name" in out.columns:
        out = out[out["scheme_name"].fillna("").astype(str) == str(scheme_name)]

    if pool_mode and pool_mode != "全部" and "pool_mode" in out.columns:
        out = out[out["pool_mode"].fillna("").astype(str) == str(pool_mode)]

    if recent_days and int(recent_days) > 0 and "created_at" in out.columns:
        ref_now = now or datetime.now()
        cutoff = ref_now - timedelta(days=int(recent_days))
        created_at = pd.to_datetime(out["created_at"], errors="coerce")
        out = out[created_at >= cutoff]

    return out.reset_index(drop=True)
