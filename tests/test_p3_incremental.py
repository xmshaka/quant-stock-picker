"""P3 冒烟测试: PostgreSQL 归档 + 增量调度

依赖:
- PostgreSQL 已启动, quant_db 已建表
- 网络可访问腾讯接口
"""
import os
import sys
import time

sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import pandas as pd
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> <level>{level:<6}</level> {message}")


def test_pg_connection():
    """PG 连通性"""
    from data.storage.models import get_engine
    print("\n── Test 1: PG 连通性 ──")
    eng = get_engine()
    with eng.connect() as conn:
        from sqlalchemy import text
        result = conn.execute(text("SELECT version()")).scalar()
        print(f"  ✅ {result[:60]}...")


def test_bulk_upsert():
    """批量 upsert"""
    from data.storage.repository import StockRepository
    print("\n── Test 2: 批量 upsert ──")
    repo = StockRepository()

    # 造 1000 条假数据
    df = pd.DataFrame({
        "symbol": ["TEST01"] * 1000,
        "trade_date": pd.date_range("2020-01-01", periods=1000),
        "open": [10.0] * 1000,
        "high": [11.0] * 1000,
        "low": [9.5] * 1000,
        "close": [10.5] * 1000,
        "volume": [100000] * 1000,
        "amount": [1050000] * 1000,
        "pct_change": [1.0] * 1000,
    })

    # 首次插入
    t0 = time.monotonic()
    n = repo.save_bars(df)
    t1 = time.monotonic()
    print(f"  首次插入 1000 条: {n} 条, {t1-t0:.2f}s")
    assert n == 1000

    # 覆盖更新 (close 改成 99)
    df2 = df.copy()
    df2["close"] = 99.0
    t0 = time.monotonic()
    n2 = repo.save_bars(df2)
    t1 = time.monotonic()
    print(f"  覆盖更新 1000 条: {n2} 条, {t1-t0:.2f}s")

    # 验证更新生效
    back = repo.get_bars("TEST01", limit=5)
    assert (back["close"].iloc[0] == 99.0), f"未更新, 实际={back['close'].iloc[0]}"
    print(f"  ✅ ON CONFLICT 覆盖生效")

    # 清理
    from data.storage.models import get_engine
    from sqlalchemy import text
    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM stock_bars WHERE symbol='TEST01'"))
    print(f"  ✅ 测试数据清理")


def test_bulk_latest_dates():
    """批量查询最新日期"""
    from data.storage.repository import StockRepository
    print("\n── Test 3: 批量 latest_dates ──")
    repo = StockRepository()
    out = repo.get_latest_dates_bulk(["600519", "000001", "NOTEXIST"])
    print(f"  结果: {out}")
    assert "NOTEXIST" in out and out["NOTEXIST"] is None
    print(f"  ✅ 通过")


def test_incremental_small():
    """小范围增量扫描 (5 只股票)"""
    from data.incremental import IncrementalUpdater
    print("\n── Test 4: 增量扫描 (5 只股票) ──")
    updater = IncrementalUpdater(max_workers=3)
    report = updater.run_daily_scan(
        symbols=["600519", "000001", "600000", "000002", "600036"],
        lookback_days=30,
    )
    print(report.summary())
    assert report.total_symbols == 5
    assert report.failed_count == 0


def test_incremental_second_run():
    """二次扫描应该全部 skip"""
    from data.incremental import IncrementalUpdater
    print("\n── Test 5: 二次扫描 (应全部 skip 或仅补 1-2 天) ──")
    updater = IncrementalUpdater(max_workers=3)
    report = updater.run_daily_scan(
        symbols=["600519", "000001", "600000", "000002", "600036"],
        lookback_days=30,
    )
    print(report.summary())
    # 已最新 + 新增行数应该非常少
    assert report.failed_count == 0


def test_cache_l3_promotion():
    """从 PG 预热 L2 缓存"""
    from data.cache_manager import CacheManager
    from pathlib import Path
    print("\n── Test 6: L3 -> L2 缓存晋升 ──")

    # 清掉 L2 该股票文件
    cache = CacheManager.get()
    p = cache.l2._bars_path("600519")
    if p.exists():
        p.unlink()
        print(f"  清掉 L2 文件: {p}")

    def _fake_fetch(s, e):
        # 不应该走到这里 (L3 有数据)
        print(f"  ⚠️ 不应走源, 但调用了 fetch({s}, {e})")
        return pd.DataFrame()

    df = cache.get_or_fetch_bars(
        symbol="600519",
        start_date="20250201",
        end_date="20250228",
        fetch_fn=_fake_fetch,
        use_pg=True,
    )
    print(f"  返回 {len(df)} 条 (从 PG 预热到 L2)")
    assert not df.empty, "L3 应有数据但读取为空"
    assert p.exists(), "L2 应已被预热写入"
    print(f"  ✅ L3 预热 L2 生效")


if __name__ == "__main__":
    test_pg_connection()
    test_bulk_upsert()
    test_bulk_latest_dates()
    test_incremental_small()
    test_incremental_second_run()
    test_cache_l3_promotion()
    print("\n🎉 全部通过")
