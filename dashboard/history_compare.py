"""回测记录横向对比辅助函数。"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


COMPARE_COLUMNS = [
    "run_id",
    "scheme_name",
    "pool_mode",
    "symbols",
    "start_date",
    "end_date",
    "total_return",
    "annual_return",
    "max_drawdown",
    "win_rate",
    "sharpe_ratio",
    "trade_count",
    "buy_count",
    "sell_count",
    "created_at",
]


def build_run_compare_table(runs: pd.DataFrame, selected_run_ids: list[str]) -> pd.DataFrame:
    """基于 list_backtest_runs 汇总结果构造多 run 对比表。"""
    if runs is None or runs.empty or not selected_run_ids:
        return pd.DataFrame(columns=COMPARE_COLUMNS)

    selected = runs[runs["run_id"].isin(selected_run_ids)].copy()
    if selected.empty:
        return pd.DataFrame(columns=[c for c in COMPARE_COLUMNS if c in runs.columns])

    # 保持用户勾选顺序，避免表格跳动。
    order = {run_id: idx for idx, run_id in enumerate(selected_run_ids)}
    selected["_order"] = selected["run_id"].map(order).fillna(9999).astype(int)
    selected = selected.sort_values("_order").drop(columns=["_order"])

    cols = [c for c in COMPARE_COLUMNS if c in selected.columns]
    out = selected[cols].copy()

    numeric_cols = ["total_return", "annual_return", "max_drawdown", "win_rate", "sharpe_ratio", "trade_count", "buy_count", "sell_count"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out.reset_index(drop=True)


def best_run_summary(compare_table: pd.DataFrame) -> dict[str, object]:
    """给出横向对比的关键最优项。"""
    if compare_table is None or compare_table.empty:
        return {}

    summary: dict[str, object] = {"count": int(len(compare_table))}
    if "total_return" in compare_table.columns and compare_table["total_return"].notna().any():
        idx = compare_table["total_return"].idxmax()
        summary["best_return_run_id"] = compare_table.loc[idx, "run_id"]
        summary["best_return"] = float(compare_table.loc[idx, "total_return"])
    if "max_drawdown" in compare_table.columns and compare_table["max_drawdown"].notna().any():
        idx = compare_table["max_drawdown"].idxmin()
        summary["best_drawdown_run_id"] = compare_table.loc[idx, "run_id"]
        summary["best_drawdown"] = float(compare_table.loc[idx, "max_drawdown"])
    if "sharpe_ratio" in compare_table.columns and compare_table["sharpe_ratio"].notna().any():
        idx = compare_table["sharpe_ratio"].idxmax()
        summary["best_sharpe_run_id"] = compare_table.loc[idx, "run_id"]
        summary["best_sharpe"] = float(compare_table.loc[idx, "sharpe_ratio"])
    return summary


def format_compare_table(compare_table: pd.DataFrame) -> pd.DataFrame:
    """格式化给 Streamlit 展示的多 run 对比表。"""
    if compare_table is None or compare_table.empty:
        return pd.DataFrame()

    out = compare_table.copy()
    percent_cols = ["total_return", "annual_return", "max_drawdown", "win_rate"]
    for col in percent_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:+.2%}" if col in ("total_return", "annual_return") else f"{x:.2%}")
    if "sharpe_ratio" in out.columns:
        out["sharpe_ratio"] = pd.to_numeric(out["sharpe_ratio"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.3f}")

    rename = {
        "run_id": "Run ID",
        "scheme_name": "策略",
        "pool_mode": "股票池",
        "symbols": "代码",
        "start_date": "开始",
        "end_date": "结束",
        "total_return": "总收益",
        "annual_return": "年化",
        "max_drawdown": "最大回撤",
        "win_rate": "胜率",
        "sharpe_ratio": "夏普",
        "trade_count": "交易轮数",
        "buy_count": "买入",
        "sell_count": "卖出",
        "created_at": "保存时间",
    }
    return out.rename(columns=rename)


def compare_table_to_csv(compare_table: pd.DataFrame) -> bytes:
    """导出多 run 对比原始数值 CSV，保留数值精度方便二次分析。"""
    if compare_table is None or compare_table.empty:
        return b""
    return compare_table.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def equity_to_cumulative_return(equity: pd.DataFrame) -> pd.DataFrame:
    """将 equity 曲线统一转为累计收益率百分比。

    兼容两种历史格式：
    - equity 为绝对权益值，如 1000000；
    - equity 为逐日收益率/累计小数序列，数值绝对值较小。
    """
    if equity is None or equity.empty or "date" not in equity.columns or "equity" not in equity.columns:
        return pd.DataFrame(columns=["date", "cum_return_pct"])

    df = equity[["date", "equity"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
    df = df.dropna(subset=["date", "equity"]).sort_values("date")
    if df.empty:
        return pd.DataFrame(columns=["date", "cum_return_pct"])

    first = float(df["equity"].iloc[0])
    if abs(first) > 100:
        base = first if first != 0 else 1.0
        df["cum_return_pct"] = (df["equity"] / base - 1.0) * 100.0
    else:
        # 小数格式按收益率序列处理，逐日连乘。
        df["cum_return_pct"] = ((1.0 + df["equity"]).cumprod() - 1.0) * 100.0
    return df[["date", "cum_return_pct"]].reset_index(drop=True)


def plot_multi_equity_curves(equity_map: dict[str, pd.DataFrame], title: str = "多Run权益曲线对比") -> go.Figure:
    """绘制多 run 累计收益曲线。"""
    fig = go.Figure()
    palette = ["#1e88e5", "#ef5350", "#26a69a", "#f0b90b", "#ab47bc", "#29b6f6", "#ff7043"]
    for idx, (run_id, equity) in enumerate(equity_map.items()):
        curve = equity_to_cumulative_return(equity)
        if curve.empty:
            continue
        fig.add_trace(go.Scatter(
            x=curve["date"],
            y=curve["cum_return_pct"],
            mode="lines",
            name=run_id,
            line=dict(width=2, color=palette[idx % len(palette)]),
        ))

    fig.update_layout(
        title=title,
        height=420,
        template="plotly_white",
        paper_bgcolor="#ede7e0",
        plot_bgcolor="#e6ded3",
        font=dict(color="#2f2a22"),
        yaxis_title="累计收益 (%)",
        margin=dict(l=55, r=15, t=40, b=30),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#f6f3ed",
            bordercolor="#c2b39f",
            font=dict(color="#2f2a22", size=11),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
    )
    fig.update_xaxes(gridcolor="#d2c7b8", tickfont=dict(color="#5f5648"))
    fig.update_yaxes(tickformat=".1f", gridcolor="#d2c7b8", tickfont=dict(color="#5f5648"))
    fig.add_hline(y=0, line_dash="dash", line_color="#c2b39f")
    return fig
