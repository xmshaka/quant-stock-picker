"""多数据源日 K 对照工具。

用于同一标的、同一区间、同一复权口径下比较 Tencent / Tushare / AKShare / Baostock：
- 行数、起止日期
- close 与基准源差异
- volume / amount 字段可用性与量级差异
- 字段风险提示，辅助决定数据源是否可用于回测撮合
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd
from loguru import logger

from data.fetchers import AKShareFetcher, BaostockFetcher, TencentFetcher, TushareFetcher
from data.bars_normalizer import normalize_daily_bars


SOURCE_FACTORIES = {
    "tencent": TencentFetcher,
    "tushare": TushareFetcher,
    "akshare": AKShareFetcher,
    "baostock": BaostockFetcher,
}


@dataclass
class SourceCompareResult:
    symbol: str
    start_date: str
    end_date: str
    adjust: str
    summary: pd.DataFrame
    aligned: pd.DataFrame
    warnings: list[str]


def _normalize_bars(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    # Fetcher 新输出已带 source/adjust 且单位标准化；此时只做类型整理，禁止二次换算。
    if "source" in out.columns and "adjust" in out.columns:
        out["trade_date"] = pd.to_datetime(out["trade_date"])
        for col in ["open", "high", "low", "close", "volume", "amount", "pct_change", "change", "turnover"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        keep = ["source", "symbol", "trade_date", "open", "high", "low", "close", "volume", "amount", "amount_estimated", "pct_change", "change", "turnover", "adjust"]
        return out[[c for c in keep if c in out.columns]].sort_values("trade_date").reset_index(drop=True)

    # 旧测试桩/外部注入无 source/adjust 时，按传入 source 标准化一次。
    return normalize_daily_bars(out, source=source, adjust="unknown")


def _summarize_source(source: str, df: pd.DataFrame, error: str = "") -> dict:
    if df is None or df.empty:
        return {
            "source": source,
            "ok": False,
            "rows": 0,
            "start": "",
            "end": "",
            "close_last": None,
            "volume_median": None,
            "amount_median": None,
            "has_amount": False,
            "error": error,
        }
    return {
        "source": source,
        "ok": True,
        "rows": len(df),
        "start": df["trade_date"].min().strftime("%Y-%m-%d"),
        "end": df["trade_date"].max().strftime("%Y-%m-%d"),
        "close_last": round(float(df["close"].dropna().iloc[-1]), 4) if df["close"].notna().any() else None,
        "volume_median": round(float(df["volume"].median()), 4) if "volume" in df.columns and df["volume"].notna().any() else None,
        "amount_median": round(float(df["amount"].median()), 4) if "amount" in df.columns and df["amount"].notna().any() else None,
        "has_amount": bool("amount" in df.columns and df["amount"].notna().any() and float(df["amount"].fillna(0).abs().sum()) > 0),
        "error": error,
    }


def _build_aligned(frames: dict[str, pd.DataFrame], baseline: str) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    ok_sources = [s for s, df in frames.items() if df is not None and not df.empty]
    if len(ok_sources) < 2:
        return pd.DataFrame(), ["可用数据源少于2个，无法做交叉差异对照"]

    base = baseline if baseline in ok_sources else ok_sources[0]
    aligned = frames[base][["trade_date", "close"]].rename(columns={"close": f"close_{base}"})
    for src in ok_sources:
        if src == base:
            continue
        part = frames[src][["trade_date", "close"]].rename(columns={"close": f"close_{src}"})
        aligned = aligned.merge(part, on="trade_date", how="inner")

    if aligned.empty:
        return aligned, ["各源无共同交易日期，无法比较 close 差异"]

    base_col = f"close_{base}"
    for src in ok_sources:
        if src == base:
            continue
        col = f"close_{src}"
        if col not in aligned.columns:
            continue
        diff_col = f"close_diff_pct_{src}_vs_{base}"
        aligned[diff_col] = (aligned[col] - aligned[base_col]).abs() / aligned[base_col].abs().clip(lower=1e-9) * 100
        max_diff = float(aligned[diff_col].max())
        mean_diff = float(aligned[diff_col].mean())
        if max_diff > 0.5:
            warnings.append(f"{src} vs {base} close 最大差异 {max_diff:.4f}%：需确认复权口径/字段单位")
        logger.info(f"[SourceCompare] {src} vs {base}: mean_diff={mean_diff:.4f}%, max_diff={max_diff:.4f}%")

    # 字段风险提示。
    for src, df in frames.items():
        if df is None or df.empty:
            continue
        if "amount" not in df.columns or df["amount"].fillna(0).abs().sum() == 0:
            warnings.append(f"{src} 缺少有效 amount，不能直接用于成交额分层滑点")
        elif "amount_estimated" in df.columns and df["amount_estimated"].fillna(False).astype(bool).any():
            warnings.append(f"{src} 缺少有效 amount，当前 amount 为估算值，只能用于流动性粗分层")
        if src == "tencent":
            warnings.append("tencent K线 amount 当前由原始volume(手)*close*100估算，成交额只适合流动性粗分层")
        if src == "tushare":
            warnings.append("tushare daily 默认不复权；标准化层已标记 adjust=raw")
        if src == "baostock":
            warnings.append("baostock 适合日K兜底；无实时估值，盘中行情/估值字段不可依赖")

    return aligned, warnings


def compare_daily_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
    sources: Optional[Iterable[str]] = None,
    baseline: str = "baostock",
    fetchers: Optional[dict[str, object]] = None,
    use_cache: bool = False,
) -> SourceCompareResult:
    """比较多源日 K。

    Args:
        symbol: 6位 A 股代码。
        start_date/end_date: YYYYMMDD。
        adjust: qfq/hfq/""。注意 Tushare daily 当前按不复权返回。
        sources: 默认 tencent,tushare,akshare,baostock。
        baseline: close 差异基准源。
        fetchers: 单测注入用。
        use_cache: 默认 False。多源对照必须优先真实接口，避免通用 K 线缓存混入不同复权口径。
    """
    selected = [s.lower() for s in (sources or ["tencent", "tushare", "akshare", "baostock"])]
    frames: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict] = []

    for src in selected:
        if src not in SOURCE_FACTORIES and not (fetchers and src in fetchers):
            summary_rows.append(_summarize_source(src, pd.DataFrame(), error="unsupported_source"))
            continue
        try:
            fetcher = fetchers[src] if fetchers and src in fetchers else SOURCE_FACTORIES[src]()
            if use_cache and hasattr(fetcher, "get_daily_bars_cached"):
                df = fetcher.get_daily_bars_cached(symbol, start_date, end_date, adjust=adjust)
            else:
                df = fetcher.get_daily_bars(symbol, start_date=start_date, end_date=end_date, adjust=adjust)
            norm = _normalize_bars(df, src)
            frames[src] = norm
            summary_rows.append(_summarize_source(src, norm))
        except Exception as e:
            frames[src] = pd.DataFrame()
            summary_rows.append(_summarize_source(src, pd.DataFrame(), error=str(e)[:200]))
            logger.warning(f"[SourceCompare] {src} 拉取失败: {e}")

    summary = pd.DataFrame(summary_rows)
    aligned, warnings = _build_aligned(frames, baseline=baseline)
    return SourceCompareResult(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        adjust=adjust or "raw",
        summary=summary,
        aligned=aligned,
        warnings=warnings,
    )


def compare_to_markdown(result: SourceCompareResult, max_rows: int = 8) -> str:
    """生成简洁 Markdown 报告。"""
    lines = [
        f"# 多源日K对照 {result.symbol}",
        f"区间: {result.start_date} ~ {result.end_date} | adjust={result.adjust}",
        "",
        "## 源摘要",
        result.summary.to_markdown(index=False),
    ]
    if not result.aligned.empty:
        diff_cols = [c for c in result.aligned.columns if c.startswith("close_diff_pct_")]
        close_cols = [c for c in result.aligned.columns if c.startswith("close_") and not c.startswith("close_diff_pct_")]
        view_cols = ["trade_date"] + close_cols + diff_cols
        lines += ["", "## Close 对齐样本", result.aligned[view_cols].tail(max_rows).to_markdown(index=False)]
    if result.warnings:
        lines += ["", "## 风险提示"] + [f"- {w}" for w in result.warnings]
    return "\n".join(lines)
