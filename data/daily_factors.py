"""每日全池因子预计算

每天收盘后批量计算全A股票池的因子，存为 Parquet 缓存，
供看板和策略引擎直接读取，避免实时逐个计算。
"""
from __future__ import annotations

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from data.universe import Universe


DAILY_FACTOR_DIR = settings.project_root / "data" / "daily_factors"
DAILY_FACTOR_DIR.mkdir(parents=True, exist_ok=True)


def _today_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def _factor_path(date_str: Optional[str] = None) -> Path:
    d = date_str or _today_str()
    return DAILY_FACTOR_DIR / f"factors_{d}.parquet"


def _price_path(date_str: Optional[str] = None) -> Path:
    d = date_str or _today_str()
    return DAILY_FACTOR_DIR / f"prices_{d}.parquet"


def _meta_path(date_str: Optional[str] = None) -> Path:
    d = date_str or _today_str()
    return DAILY_FACTOR_DIR / f"meta_{d}.json"


def has_daily_factors(date_str: Optional[str] = None) -> bool:
    """检查指定日期的全池因子是否存在。"""
    return _factor_path(date_str).exists() and _price_path(date_str).exists()


def latest_snapshot_date() -> Optional[str]:
    """返回最新一份全池快照的日期字符串 (YYYYMMDD)。不存在返回 None。"""
    files = sorted(DAILY_FACTOR_DIR.glob("factors_*.parquet"))
    if not files:
        return None
    # 取文件名中的 YYYYMMDD
    name = files[-1].stem  # factors_20260526
    try:
        return name.split("_")[1]
    except Exception:
        return None


def load_snapshot_meta(date_str: Optional[str] = None) -> Optional[dict]:
    """读取指定日期的快照 meta。"""
    d = date_str or latest_snapshot_date()
    if not d:
        return None
    p = _meta_path(d)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def latest_data_source_meta() -> dict:
    """返回最新快照/缓存可追溯的数据来源摘要，供状态页展示。"""
    d = latest_snapshot_date()
    meta = load_snapshot_meta(d) if d else None
    if meta:
        return {
            "snapshot_date": d,
            "snapshot_source": meta.get("data_source", "daily_factor_snapshot"),
            "primary_source": meta.get("universe_source", ""),
            "quote_source": meta.get("quote_source", ""),
            "daily_basic_source": meta.get("daily_basic_source", ""),
            "daily_basic_date": meta.get("daily_basic_date", ""),
            "computed_at": meta.get("computed_at", ""),
        }
    return {"snapshot_date": d or "", "snapshot_source": "none"}


def load_daily_factors(date_str: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """读取指定日期的全池因子。

    Returns:
        (factor_df, price_df, factor_names)
    """
    d = date_str or _today_str()
    factor_df = pd.read_parquet(_factor_path(d))
    price_df = pd.read_parquet(_price_path(d))
    factor_names = [c for c in factor_df.columns if c not in ("symbol", "trade_date")]
    logger.info(f"[DailyFactors] 读取 {d} 全池因子: {len(factor_df)} 条, {len(factor_names)} 个因子")
    return factor_df, price_df, factor_names


MONEYFLOW_FACTOR_COLUMNS = [
    "main_net_mf_amount",
    "large_net_mf_amount",
    "elg_net_mf_amount",
    "large_elg_net_mf_amount",
    "main_net_mf_pct_amount",
    "large_elg_net_mf_pct_amount",
    "main_net_mf_rank",
    "large_elg_net_mf_rank",
]

RELATIVE_TURNOVER_FACTOR_COLUMNS = [
    "relative_turnover_5d",
    "relative_turnover_20d",
    "turnover_percentile_60d",
    "amount_percentile_60d",
]


def _normalize_trade_date(series: pd.Series) -> pd.Series:
    """统一 trade_date 到 YYYY-MM-DD 字符串，便于跨源合并。"""
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def _moneyflow_trade_date(series: pd.Series) -> pd.Series:
    """Tushare moneyflow 日期通常为 YYYYMMDD，统一为 YYYY-MM-DD。"""
    s = series.astype(str).str.replace("-", "", regex=False)
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")


def add_relative_turnover_factors(
    factor_df: pd.DataFrame,
    price_df: Optional[pd.DataFrame] = None,
    turnover_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """新增相对换手/成交额分位因子。

    只使用每个交易日及其之前的历史窗口：rolling/expanding 均按 symbol 内时间排序，
    不引用未来数据。第一阶段作为审计/排序因子，不直接硬过滤。
    """
    if factor_df is None or factor_df.empty or "symbol" not in factor_df.columns or "trade_date" not in factor_df.columns:
        return factor_df
    out = factor_df.copy()
    out["_trade_date_norm"] = _normalize_trade_date(out["trade_date"])
    out = out.sort_values(["symbol", "_trade_date_norm"]).reset_index(drop=True)

    turnover = None
    turnover_col = "turnover_ratio" if "turnover_ratio" in out.columns else "turnover" if "turnover" in out.columns else None
    if turnover_col:
        turnover = pd.to_numeric(out[turnover_col], errors="coerce")
    elif turnover_df is not None and not turnover_df.empty:
        t = _standardize_turnover_frame(turnover_df)
        if not t.empty:
            out = out.merge(
                t.rename(columns={"turnover": "_daily_turnover"}),
                on=["symbol", "_trade_date_norm"],
                how="left",
            )
            turnover = pd.to_numeric(out["_daily_turnover"], errors="coerce")
    elif price_df is not None and not price_df.empty and {"symbol", "trade_date", "turnover"}.issubset(price_df.columns):
        p_turnover = price_df[["symbol", "trade_date", "turnover"]].copy()
        p_turnover["_trade_date_norm"] = _normalize_trade_date(p_turnover["trade_date"])
        p_turnover = p_turnover.drop_duplicates(["symbol", "_trade_date_norm"], keep="last")
        out = out.merge(
            p_turnover[["symbol", "_trade_date_norm", "turnover"]].rename(columns={"turnover": "_daily_turnover"}),
            on=["symbol", "_trade_date_norm"],
            how="left",
        )
        turnover = pd.to_numeric(out["_daily_turnover"], errors="coerce")
    if turnover is not None:
        g = turnover.groupby(out["symbol"], sort=False)
        avg5 = g.transform(lambda s: s.rolling(5, min_periods=3).mean())
        avg20 = g.transform(lambda s: s.rolling(20, min_periods=5).mean())
        out["relative_turnover_5d"] = turnover / avg5.replace(0, pd.NA)
        out["relative_turnover_20d"] = turnover / avg20.replace(0, pd.NA)
        out["turnover_percentile_60d"] = g.transform(
            lambda s: s.rolling(60, min_periods=10).apply(_last_value_percentile, raw=True)
        )
    else:
        for col in ["relative_turnover_5d", "relative_turnover_20d", "turnover_percentile_60d"]:
            out[col] = pd.NA

    amount_series = None
    if price_df is not None and not price_df.empty and {"symbol", "trade_date", "amount"}.issubset(price_df.columns):
        p = price_df[["symbol", "trade_date", "amount"]].copy()
        p["_trade_date_norm"] = _normalize_trade_date(p["trade_date"])
        p = p.drop_duplicates(["symbol", "_trade_date_norm"], keep="last")
        out = out.merge(p[["symbol", "_trade_date_norm", "amount"]].rename(columns={"amount": "_daily_amount"}),
                        on=["symbol", "_trade_date_norm"], how="left")
        amount_series = pd.to_numeric(out["_daily_amount"], errors="coerce")
    elif "amount" in out.columns:
        amount_series = pd.to_numeric(out["amount"], errors="coerce")

    if amount_series is not None:
        out["amount_percentile_60d"] = amount_series.groupby(out["symbol"], sort=False).transform(
            lambda s: s.rolling(60, min_periods=10).apply(_last_value_percentile, raw=True)
        )
    else:
        out["amount_percentile_60d"] = pd.NA

    return out.drop(columns=[c for c in ["_trade_date_norm", "_daily_amount", "_daily_turnover"] if c in out.columns])


def _standardize_turnover_frame(turnover_df: pd.DataFrame) -> pd.DataFrame:
    """标准化换手率历史表为 symbol/_trade_date_norm/turnover。

    支持 Tushare daily_basic(`ts_code`, `trade_date`, `turnover_rate`)、
    已归一化表(`symbol`, `trade_date`, `turnover_ratio/turnover`)。仅用于计算相对换手，
    不把缺失值伪造成 0。
    """
    if turnover_df is None or turnover_df.empty:
        return pd.DataFrame(columns=["symbol", "_trade_date_norm", "turnover"])
    t = turnover_df.copy()
    if "symbol" not in t.columns:
        if "ts_code" in t.columns:
            t["symbol"] = t["ts_code"].astype(str).str.slice(0, 6)
        else:
            return pd.DataFrame(columns=["symbol", "_trade_date_norm", "turnover"])
    if "trade_date" not in t.columns:
        return pd.DataFrame(columns=["symbol", "_trade_date_norm", "turnover"])
    turnover_col = None
    for c in ["turnover_ratio", "turnover_rate", "turnover", "turnover_rate_f"]:
        if c in t.columns:
            turnover_col = c
            break
    if turnover_col is None:
        return pd.DataFrame(columns=["symbol", "_trade_date_norm", "turnover"])
    t["symbol"] = t["symbol"].astype(str).str.zfill(6)
    t["_trade_date_norm"] = _normalize_trade_date(t["trade_date"])
    t["turnover"] = pd.to_numeric(t[turnover_col], errors="coerce")
    t = t.dropna(subset=["symbol", "_trade_date_norm", "turnover"])
    return t[["symbol", "_trade_date_norm", "turnover"]].drop_duplicates(["symbol", "_trade_date_norm"], keep="last")


def _last_value_percentile(values) -> float:
    """rolling 窗口内最后一个值的历史分位；raw=True，避免构造 Series 拖慢全池。"""
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    if len(arr) == 0:
        return float("nan")
    last = arr[-1]
    return float((arr <= last).sum() / len(arr))


def add_moneyflow_factors(factor_df: pd.DataFrame, moneyflow_df: Optional[pd.DataFrame], price_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """把 Tushare moneyflow 合成为资金流因子。

    单位约定：Tushare moneyflow amount 为“万元”，本系统标准化行情 amount 为“元”。
    `*_amount` 字段保留万元口径，`*_pct_amount` 统一用万元*10000 / 当日成交额(元)。
    """
    if factor_df is None or factor_df.empty:
        return factor_df
    out = factor_df.copy()
    for col in MONEYFLOW_FACTOR_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    if moneyflow_df is None or moneyflow_df.empty or "ts_code" not in moneyflow_df.columns:
        return out
    if "symbol" not in out.columns or "trade_date" not in out.columns:
        return out

    mf = moneyflow_df.copy()
    mf["symbol"] = mf["ts_code"].astype(str).str.slice(0, 6)
    mf["_trade_date_norm"] = _moneyflow_trade_date(mf["trade_date"] if "trade_date" in mf.columns else pd.Series([None] * len(mf)))
    amount_cols = [
        "buy_sm_amount", "buy_md_amount", "buy_lg_amount", "buy_elg_amount",
        "sell_sm_amount", "sell_md_amount", "sell_lg_amount", "sell_elg_amount",
    ]
    for c in ["net_mf_amount", *amount_cols]:
        mf[c] = pd.to_numeric(mf.get(c, 0.0), errors="coerce").fillna(0.0)
    mf["main_net_mf_amount"] = mf["net_mf_amount"]
    mf["large_net_mf_amount"] = mf["buy_lg_amount"] - mf["sell_lg_amount"]
    mf["elg_net_mf_amount"] = mf["buy_elg_amount"] - mf["sell_elg_amount"]
    mf["large_elg_net_mf_amount"] = mf["large_net_mf_amount"] + mf["elg_net_mf_amount"]
    mf["_mf_turnover_amount_yuan"] = mf[["buy_sm_amount", "buy_md_amount", "buy_lg_amount", "buy_elg_amount"]].sum(axis=1) * 10000.0
    # 若买方拆单缺失，回退卖方拆单估算。Tushare moneyflow amount 为万元。
    sell_turnover_yuan = mf[["sell_sm_amount", "sell_md_amount", "sell_lg_amount", "sell_elg_amount"]].sum(axis=1) * 10000.0
    mf["_mf_turnover_amount_yuan"] = mf["_mf_turnover_amount_yuan"].where(mf["_mf_turnover_amount_yuan"] > 0, sell_turnover_yuan)
    keep = [
        "symbol", "_trade_date_norm", "main_net_mf_amount", "large_net_mf_amount",
        "elg_net_mf_amount", "large_elg_net_mf_amount", "_mf_turnover_amount_yuan",
    ]
    mf = mf[keep].drop_duplicates(["symbol", "_trade_date_norm"], keep="last")

    out["_trade_date_norm"] = _normalize_trade_date(out["trade_date"])
    if price_df is not None and not price_df.empty and {"symbol", "trade_date", "amount"}.issubset(price_df.columns):
        p = price_df[["symbol", "trade_date", "amount"]].copy()
        p["_trade_date_norm"] = _normalize_trade_date(p["trade_date"])
        p = p.drop_duplicates(["symbol", "_trade_date_norm"], keep="last")
        out = out.merge(p[["symbol", "_trade_date_norm", "amount"]].rename(columns={"amount": "_daily_amount"}),
                        on=["symbol", "_trade_date_norm"], how="left")
    elif "amount" in out.columns:
        out["_daily_amount"] = out["amount"]
    else:
        out["_daily_amount"] = pd.NA

    out = out.merge(mf, on=["symbol", "_trade_date_norm"], how="left", suffixes=("", "_mf"))
    for col in ["main_net_mf_amount", "large_net_mf_amount", "elg_net_mf_amount", "large_elg_net_mf_amount"]:
        mf_col = f"{col}_mf"
        if mf_col in out.columns:
            out[col] = out[mf_col].combine_first(out[col])
            out = out.drop(columns=[mf_col])

    amount_yuan = pd.to_numeric(out["_daily_amount"], errors="coerce")
    if "_mf_turnover_amount_yuan" in out.columns:
        amount_yuan = amount_yuan.where(amount_yuan > 0, pd.to_numeric(out["_mf_turnover_amount_yuan"], errors="coerce"))
    out["main_net_mf_pct_amount"] = pd.to_numeric(out["main_net_mf_amount"], errors="coerce") * 10000.0 / amount_yuan.replace(0, pd.NA)
    out["large_elg_net_mf_pct_amount"] = pd.to_numeric(out["large_elg_net_mf_amount"], errors="coerce") * 10000.0 / amount_yuan.replace(0, pd.NA)
    out["main_net_mf_rank"] = out.groupby("_trade_date_norm")["main_net_mf_pct_amount"].rank(pct=True)
    out["large_elg_net_mf_rank"] = out.groupby("_trade_date_norm")["large_elg_net_mf_pct_amount"].rank(pct=True)
    return out.drop(columns=[c for c in ["_trade_date_norm", "_daily_amount", "_mf_turnover_amount_yuan"] if c in out.columns])


def enrich_short_term_factors(
    factor_df: pd.DataFrame,
    price_df: Optional[pd.DataFrame] = None,
    moneyflow_df: Optional[pd.DataFrame] = None,
    turnover_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """短线买卖点上下文因子增强入口。"""
    enriched = add_relative_turnover_factors(factor_df, price_df, turnover_df)
    enriched = add_moneyflow_factors(enriched, moneyflow_df, price_df)
    return enriched


def fetch_tushare_daily_basic_turnover_history(trade_dates) -> pd.DataFrame:
    """按交易日批量读取 Tushare daily_basic 换手率历史。

    用于相对换手率 rolling 计算。只读取实际交易日，优先命中
    `TushareFetcher.get_daily_basic()` 的 L2 parquet 缓存；接口失败/空表时跳过，
    不生成假 0。
    """
    if trade_dates is None:
        return pd.DataFrame()
    dates = pd.to_datetime(pd.Series(trade_dates).dropna().unique(), errors="coerce")
    dates = sorted({d.strftime("%Y%m%d") for d in dates if pd.notna(d)})
    if not dates:
        return pd.DataFrame()
    try:
        from data.fetchers.tushare_fetcher import TushareFetcher
        fetcher = TushareFetcher()
    except Exception as e:
        logger.warning(f"[DailyFactors] 初始化 Tushare daily_basic 换手率源失败: {e}")
        return pd.DataFrame()

    frames = []
    for td in dates:
        try:
            df = fetcher.get_daily_basic(trade_date=td)
            if df is None or df.empty:
                continue
            cols = [c for c in ["ts_code", "symbol", "trade_date", "turnover_rate", "turnover_rate_f", "turnover"] if c in df.columns]
            if not {"trade_date"}.issubset(cols) or not any(c in cols for c in ["ts_code", "symbol"]):
                continue
            if not any(c in cols for c in ["turnover_rate", "turnover_rate_f", "turnover"]):
                continue
            frames.append(df[cols].copy())
        except Exception as e:
            logger.debug(f"[DailyFactors] daily_basic turnover {td} 失败: {e}")
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    logger.info(f"[DailyFactors] Tushare daily_basic 换手率历史: {len(out)} 条, {len(frames)} 个交易日")
    return out


def snapshot_coverage_report(date_str: Optional[str] = None) -> dict:
    """返回每日因子快照覆盖质量摘要。

    注意：快照文件日期不等于所有股票的最新交易日。某些个股可能因停牌、
    数据源缓存未补尾、临时接口异常而只保留自身较早的最新 K 线。
    该报告供数据状态页区分“扫描日志更新数量”和“快照实际最新覆盖度”。
    """
    d = date_str or latest_snapshot_date()
    if not d or not _factor_path(d).exists():
        return {}

    factor_df = pd.read_parquet(_factor_path(d), columns=["symbol", "trade_date"])
    if factor_df.empty:
        return {"snapshot_date": d, "symbols": 0, "global_latest_date": None, "fresh_symbols": 0, "stale_symbols": 0}

    latest_by_symbol = factor_df.groupby("symbol")["trade_date"].max()
    global_latest = latest_by_symbol.max()
    fresh = int((latest_by_symbol == global_latest).sum())
    stale = int((latest_by_symbol < global_latest).sum())
    dist = latest_by_symbol.astype(str).value_counts().sort_index(ascending=False)
    return {
        "snapshot_date": d,
        "symbols": int(latest_by_symbol.size),
        "global_latest_date": str(global_latest)[:10],
        "fresh_symbols": fresh,
        "stale_symbols": stale,
        "coverage_pct": round(fresh / max(int(latest_by_symbol.size), 1) * 100, 2),
        "date_distribution": {str(k)[:10]: int(v) for k, v in dist.head(10).items()},
    }


def compute_daily_factors(max_workers: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """计算今日全池因子并保存。

    流程:
    1. 从 Universe 加载全A股票池
    2. 用 DataLoader 批量拉取行情并计算因子
    3. 保存 Parquet + meta
    """
    from dashboard.data_loader import DataLoader

    logger.info("[DailyFactors] 开始全池因子预计算...")
    start = datetime.now()

    # 1. 加载股票池
    universe_df = Universe().load(use_cache=True)
    source_meta = dict(universe_df.attrs.get("source_meta", {}))
    symbols = universe_df["symbol"].tolist()
    logger.info(f"[DailyFactors] 股票池: {len(symbols)} 只")

    # 2. 批量计算因子 (强制不走看板级 pickle 缓存，走数据源)
    loader = DataLoader(preferred="tencent")
    factor_df, price_df, factor_names = loader.load(
        n_stocks=len(symbols),
        n_days=120,
        use_cache=False,  # 强制重新从数据源拉取
        include_symbols=symbols,
    )

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(
        f"[DailyFactors] 计算完成: {len(factor_df)} 条因子, "
        f"{len(price_df)} 条价格, 耗时 {elapsed:.1f}s"
    )

    # 2.1 增强短线买卖点上下文因子：相对换手 + Tushare moneyflow。
    moneyflow_df = pd.DataFrame()
    turnover_df = pd.DataFrame()
    latest_trade_date = ""
    try:
        latest_trade_date = pd.to_datetime(factor_df["trade_date"], errors="coerce").max().strftime("%Y%m%d")
        from data.fetchers.tushare_fetcher import TushareFetcher
        moneyflow_df = TushareFetcher().get_money_flow(trade_date=latest_trade_date)
        if moneyflow_df is not None and not moneyflow_df.empty:
            logger.info(f"[DailyFactors] Tushare moneyflow {latest_trade_date}: {len(moneyflow_df)} 条")
        else:
            logger.warning(f"[DailyFactors] Tushare moneyflow {latest_trade_date} 为空，资金流因子保留缺失")
    except Exception as e:
        logger.warning(f"[DailyFactors] moneyflow 增强失败，跳过资金流因子: {e}")
        moneyflow_df = pd.DataFrame()
    try:
        turnover_df = fetch_tushare_daily_basic_turnover_history(factor_df["trade_date"])
        if turnover_df is None or turnover_df.empty:
            logger.warning("[DailyFactors] daily_basic 换手率历史为空，相对换手因子保留缺失")
    except Exception as e:
        logger.warning(f"[DailyFactors] daily_basic 换手率增强失败，跳过相对换手因子: {e}")
        turnover_df = pd.DataFrame()
    factor_df = enrich_short_term_factors(factor_df, price_df, moneyflow_df, turnover_df)
    factor_names = [c for c in factor_df.columns if c not in ("symbol", "trade_date")]

    # 3. 保存
    today = _today_str()
    factor_df.to_parquet(_factor_path(today))
    price_df.to_parquet(_price_path(today))

    meta = {
        "date": today,
        "data_source": "daily_factor_snapshot",
        "universe_source": source_meta.get("primary_source", "unknown"),
        "quote_source": source_meta.get("quote_source", "unknown"),
        "daily_basic_source": source_meta.get("daily_basic_source", "unknown"),
        "daily_basic_date": source_meta.get("daily_basic_date"),
        "moneyflow_source": "tushare_moneyflow_api" if moneyflow_df is not None and not moneyflow_df.empty else "missing",
        "moneyflow_date": latest_trade_date,
        "universe_size": len(symbols),
        "factor_rows": len(factor_df),
        "price_rows": len(price_df),
        "factor_names": factor_names,
        "elapsed_seconds": elapsed,
        "computed_at": datetime.now().isoformat(),
    }
    _meta_path(today).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"[DailyFactors] 已保存到 {DAILY_FACTOR_DIR}")

    return factor_df, price_df, factor_names


def clean_old_factors(keep_days: int = 7) -> int:
    """清理过期 parquet，默认保留 7 天。返回删除文件数。"""
    cutoff = datetime.now() - timedelta(days=keep_days)
    removed = 0
    for p in DAILY_FACTOR_DIR.glob("*_*"):
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime)
            if mtime < cutoff:
                p.unlink()
                removed += 1
        except Exception:
            continue
    if removed:
        logger.info(f"[DailyFactors] 清理过期文件: {removed} 个")
    return removed


def main():
    """CLI 入口:  python -m data.daily_factors"""
    import argparse
    parser = argparse.ArgumentParser(description="每日全池因子预计算")
    parser.add_argument("--once", action="store_true", help="立即执行一次")
    parser.add_argument("--clean", action="store_true", help="清理过期文件")
    args = parser.parse_args()

    if args.clean:
        clean_old_factors()
        return

    compute_daily_factors()
    clean_old_factors()


if __name__ == "__main__":
    main()
