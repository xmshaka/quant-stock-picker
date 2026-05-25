"""数据源测试脚本"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import pytest
from loguru import logger
logger.remove()
logger.add(sys.stdout, level="INFO")

from data.fetchers import TencentFetcher, AKShareFetcher, TushareFetcher
from dashboard.data_loader import DataLoader

def test_tencent():
    print("\n" + "="*50)
    print("测试腾讯数据源")
    print("="*50)

    tf = TencentFetcher()

    # 1. 实时行情
    print("\n1. 实时行情 (贵州茅台+五粮液)")
    quotes = tf.get_realtime_quotes(["600519", "000858"])
    print(f"   获取到 {len(quotes)} 条")
    if not quotes.empty:
        print(quotes[["symbol", "name", "close", "pct_change", "pe_ttm", "pb", "turnover"]].to_string(index=False))
        # 验证 pct_change 是否在合理范围
        if quotes["pct_change"].abs().max() > 100:
            print("   ⚠️ pct_change 异常大，可能有字段错位")
        else:
            print("   ✅ pct_change 正常")

    # 2. 历史K线
    print("\n2. 历史K线 (贵州茅台 最近30天)")
    bars = tf.get_daily_bars("600519", start_date="20250501")
    print(f"   获取到 {len(bars)} 条")
    if not bars.empty:
        print(bars.tail(3)[["trade_date", "open", "close", "high", "low", "volume", "pct_change"]].to_string(index=False))

    # 3. 股票列表
    print("\n3. 股票列表 (沪深300成分)")
    stocks = tf.get_stock_list()
    print(f"   获取到 {len(stocks)} 条")
    if not stocks.empty:
        print(stocks.head(5)[["symbol", "name", "close", "pct_change"]].to_string(index=False))

    assert bars is not None and not bars.empty


def test_akshare():
    print("\n" + "="*50)
    print("测试 AKShare 数据源")
    print("="*50)

    af = AKShareFetcher()

    # 1. 股票列表
    print("\n1. 股票列表")
    stocks = af.get_stock_list()
    print(f"   获取到 {len(stocks)} 条")
    if not stocks.empty:
        print(stocks.head(5)[["symbol", "name", "close", "pct_change"]].to_string(index=False))

    # 2. 历史K线
    print("\n2. 历史K线 (贵州茅台 最近30天)")
    bars = af.get_daily_bars("600519", start_date="20250501")
    print(f"   获取到 {len(bars)} 条")
    if not bars.empty:
        print(bars.tail(3)[["trade_date", "open", "close", "high", "low", "volume", "pct_change"]].to_string(index=False))

    # 3. 指数成分
    print("\n3. 沪深300成分")
    components = af.get_index_components("000300")
    print(f"   获取到 {len(components)} 条")
    if not components.empty:
        print(components.head(5)[["symbol", "name", "weight"]].to_string(index=False))

    # 4. 涨停股
    print("\n4. 今日涨停股")
    try:
        zt = af.get_limit_up_stocks()
        print(f"   获取到 {len(zt)} 条")
        if not zt.empty:
            print(zt.head(3)[["symbol", "name", "close", "consecutive_boards"]].to_string(index=False))
    except Exception as e:
        print(f"   获取失败 (可能是非交易日): {e}")

    if bars is None or bars.empty:
        pytest.skip("AKShare K线接口当前不可用")


def test_tushare():
    print("\n" + "="*50)
    print("测试 Tushare 数据源")
    print("="*50)

    tf = TushareFetcher()
    if not tf._has_token():
        print("   ⚠️ Tushare Token未配置，跳过测试")
        print("   在 .env 中添加: TUSHARE_TOKEN=你的token")
        pytest.skip("Tushare Token未配置")

    # 1. 股票列表
    print("\n1. 股票列表")
    stocks = tf.get_stock_list()
    print(f"   获取到 {len(stocks)} 条")
    if not stocks.empty:
        print(stocks.head(5)[["symbol", "name", "industry"]].to_string(index=False))

    # 2. 历史K线
    print("\n2. 历史K线 (贵州茅台 最近30天)")
    bars = tf.get_daily_bars("600519", start_date="20250501")
    print(f"   获取到 {len(bars)} 条")
    if not bars.empty:
        print(bars.tail(3)[["trade_date", "open", "close", "high", "low", "volume", "pct_change"]].to_string(index=False))

    # 3. 每日指标
    print("\n3. 每日估值指标")
    try:
        basic = tf.get_daily_basic(trade_date="20250516")
        print(f"   获取到 {len(basic)} 条")
    except Exception as e:
        print(f"   获取失败: {e}")

    assert bars is not None and not bars.empty


def test_dataloader():
    print("\n" + "="*50)
    print("测试 DataLoader 多源回退")
    print("="*50)

    for pref in ["tencent", "akshare", "tushare", "mock"]:
        print(f"\n优先: {pref} (5只, 20天)")
        dl = DataLoader(preferred=pref)
        f, p, names = dl.load(n_stocks=5, n_days=20)
        status = "✅" if not f.empty else "❌"
        src = "mock" if f.empty else pref
        print(f"   {status} 来源={src}, 因子={len(f)}条, 价格={len(p)}条, 因子={names[:3]}...")
        if not f.empty:
            print(f.head(2)[["symbol", "trade_date", "momentum_20d", "reversal"]].to_string(index=False))

    assert True


if __name__ == "__main__":
    ok1 = test_tencent()
    ok2 = test_akshare()
    ok3 = test_tushare()
    ok4 = test_dataloader()

    print("\n" + "="*50)
    print("测试结果汇总")
    print("="*50)
    print(f"  腾讯API:   {'✅ 通过' if ok1 else '❌ 失败'}")
    print(f"  AKShare:   {'✅ 通过' if ok2 else '❌ 失败'}")
    print(f"  Tushare:   {'✅ 通过' if ok3 else ('⏭️ 未配置' if ok3 is None else '❌ 失败')}")
    print(f"  DataLoader: {'✅ 通过' if ok4 else '❌ 失败'}")
