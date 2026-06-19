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
