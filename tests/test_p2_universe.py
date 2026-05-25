"""P2 冒烟测试: 全 A 股票池 + 基本面过滤

测试 Universe 模块的加载/过滤链路
依赖网络: 需要 AKShare + 腾讯接口
"""
import os
import sys
import time

sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> <level>{level:<6}</level> {message}")


def test_universe_load():
    """测试全 A 加载 + 默认过滤"""
    from data.universe import Universe
    print("\n── Test 1: 全 A 股票池加载 (默认配置) ──")

    t0 = time.monotonic()
    df = Universe.load(use_cache=False)
    t1 = time.monotonic()
    print(f"  耗时: {t1-t0:.1f}s")
    print(f"  结果: {len(df)} 只股票")
    print(f"  列: {df.columns.tolist()}")

    # 检查 ST 是否被排除
    if "name" in df.columns:
        st_left = df[df["name"].str.contains("ST", na=False)]
        print(f"  ST 残留: {len(st_left)} 只 (应为 0)")

    # 基本断言
    assert not df.empty, "股票池为空"
    assert "symbol" in df.columns
    print("  ✅ 通过")


def test_universe_filter_report():
    """测试过滤报告"""
    from data.universe import Universe
    print("\n── Test 2: 过滤报告 ──")

    df = Universe.load(use_cache=False)
    report = df.attrs.get("filter_report", {})
    if report:
        print(f"  原始: {report['original']}")
        print(f"  最终: {report['final']}")
        for step in report.get("steps", []):
            if step["removed"] > 0:
                print(f"    - {step['name']}: -{step['removed']} ({step['reason']})")
    print("  ✅ 通过")


def test_universe_with_overrides():
    """测试自定义过滤参数"""
    from data.universe import Universe
    print("\n── Test 3: 自定义过滤 (流通市值≥50亿) ──")

    df = Universe.load(use_cache=False, min_float_mv_yi=50.0)
    print(f"  流通市值≥50亿: {len(df)} 只")

    if "float_mv" in df.columns and not df.empty:
        mv_yi = df["float_mv"].dropna() / 1e8
        print(f"  流通市值范围: {mv_yi.min():.1f} ~ {mv_yi.max():.1f} 亿")
        assert mv_yi.min() >= 49.9, f"最小市值 {mv_yi.min()} < 50 亿"
    print("  ✅ 通过")


def test_universe_symbols():
    """测试便捷 symbol 列表"""
    from data.universe import Universe
    print("\n── Test 4: symbol 列表 ──")

    symbols = Universe.list_symbols(exclude_st=False, exclude_new_stock_days=0)
    print(f"  不排除 ST + 次新: {len(symbols)} 只")
    assert len(symbols) > 5000, f"全 A 不排除应该 > 5000, 实际 {len(symbols)}"
    print("  ✅ 通过")


def test_universe_cache_hit():
    """测试缓存命中"""
    from data.universe import Universe
    print("\n── Test 5: 缓存命中 ──")

    # 第一次: 写缓存
    t0 = time.monotonic()
    df1 = Universe.load(use_cache=True)
    t1 = time.monotonic()
    print(f"  第1次 (可能写缓存): {len(df1)} 只, {t1-t0:.1f}s")

    # 第二次: 应命中
    t2 = time.monotonic()
    df2 = Universe.load(use_cache=True)
    t3 = time.monotonic()
    print(f"  第2次 (应命中): {len(df2)} 只, {t3-t2:.2f}s")
    assert len(df2) == len(df1)
    print("  ✅ 通过")


def test_tencent_fetcher_all_a():
    """测试 TencentFetcher.get_stock_list 走 all_a"""
    from data.fetchers.tencent_fetcher import TencentFetcher
    print("\n── Test 6: TencentFetcher.get_stock_list(all_a) ──")

    f = TencentFetcher()
    df = f.get_stock_list(universe="all_a")
    print(f"  结果: {len(df)} 只")
    print(f"  列: {df.columns.tolist()}")
    assert not df.empty
    print("  ✅ 通过")


if __name__ == "__main__":
    test_universe_load()
    test_universe_filter_report()
    test_universe_with_overrides()
    test_universe_symbols()
    test_universe_cache_hit()
    test_tencent_fetcher_all_a()
    print("\n🎉 全部通过")
