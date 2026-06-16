#!/usr/bin/env python3
"""运行实际验证回测"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
from pathlib import Path
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("🚀 运行实际验证回测 - 测试数据源追踪修复")
print("=" * 70)

# 导入必要的模块
try:
    from backtest.scheme_backtest import run_multi_scheme_backtest
    from data.fetchers.fallback_fetcher import get_daily_bars
    from data.bars_normalizer import normalize_bars
    print("✅ 模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

# 1. 获取实际数据（带数据源字段）
print(f"\n1. 📡 获取实际K线数据...")
symbol = "000001.SZ"  # 平安银行
start_date = "2026-06-10"
end_date = "2026-06-12"

try:
    # 使用fallback fetcher获取数据（会自动添加source/adjust字段）
    raw_bars = get_daily_bars(
        symbols=[symbol],
        start_date=start_date,
        end_date=end_date,
        adjust="raw"
    )
    
    if raw_bars.empty:
        print(f"❌ 无法获取数据，使用模拟数据")
        # 创建模拟数据
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        raw_bars = pd.DataFrame({
            'trade_date': dates,
            'symbol': symbol,
            'open': [10.0, 10.1, 10.2],
            'high': [10.5, 10.6, 10.7],
            'low': [9.5, 9.6, 9.7],
            'close': [10.0, 10.1, 10.2],
            'volume': [1000000, 1100000, 1200000],
            'amount': [10000000, 11100000, 12200000],
            'source': 'tencent',
            'adjust': 'raw'
        })
    else:
        print(f"✅ 获取到实际数据: {len(raw_bars)} 行")
        print(f"   数据源: {raw_bars['source'].iloc[0] if 'source' in raw_bars.columns else 'N/A'}")
        print(f"   复权口径: {raw_bars['adjust'].iloc[0] if 'adjust' in raw_bars.columns else 'N/A'}")
    
except Exception as e:
    print(f"❌ 数据获取失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. 准备回测参数
print(f"\n2. ⚙️ 准备回测参数...")

schemes = [{
    'scheme_id': 'verification',
    'name': '验证修复',
    'signal_rules': {
        'buy': [{'rule': 'close > 0', 'weight': 1.0}],  # 简单规则
        'sell': [{'rule': 'close < 0', 'weight': 1.0}]   # 不会触发
    },
    'factor_weights': {},
    'description': '测试数据源追踪修复'
}]

# 创建因子数据
factor_dates = pd.date_range(start=start_date, end=end_date, freq='D')
factor_df = pd.DataFrame({
    'trade_date': factor_dates,
    'symbol': symbol,
    'pe_ratio': [10.0, 10.1, 10.2],
    'pb_ratio': [1.5, 1.6, 1.7],
    'market_cap': [1000000000, 1010000000, 1020000000]
})

print(f"   回测区间: {start_date} 到 {end_date}")
print(f"   测试股票: {symbol}")
print(f"   模式: 单股模式")
print(f"   初始资金: 1,000,000")

# 3. 运行回测
print(f"\n3. 🚀 运行验证回测...")

try:
    # 注意：这里使用单股模式
    result = run_multi_scheme_backtest(
        schemes=schemes,
        start_date=start_date,
        end_date=end_date,
        factor_df=factor_df,
        factor_names=['pe_ratio', 'pb_ratio'],
        price_df=raw_bars,
        pool_mode='single',
        symbols=[symbol],
        lookback_days=60,
        top_n=10,
        initial_capital=1000000.0,
        verbose=False
    )
    
    print(f"✅ 回测执行成功!")
    print(f"   数据源: {result.data_source}/{result.data_adjust}")
    print(f"   数据版本: {result.data_version}")
    print(f"   总收益: {result.total_return:+.2%}")
    print(f"   交易次数: {result.trade_count}")
    print(f"   run_id: {result.run_id}")
    
    # 4. 验证数据源字段
    print(f"\n4. ✅ 数据源字段验证:")
    
    if result.data_source:
        print(f"   ✅ data_source字段非空: '{result.data_source}'")
    else:
        print(f"   ❌ data_source字段为空")
    
    if result.data_adjust:
        print(f"   ✅ data_adjust字段: '{result.data_adjust}'")
    else:
        print(f"   ❌ data_adjust字段为空")
    
    if result.data_version:
        print(f"   ✅ data_version字段: '{result.data_version}'")
    else:
        print(f"   ❌ data_version字段为空")
    
    # 5. 检查summary_text
    print(f"\n5. 📝 summary_text验证:")
    summary = result.summary_text()
    if "数据源:" in summary:
        print(f"   ✅ summary_text包含'数据源:'")
        # 提取数据源部分
        lines = summary.split('\n')
        for line in lines:
            if "数据源:" in line:
                print(f"      {line.strip()}")
    else:
        print(f"   ❌ summary_text不包含'数据源:'")
    
    # 6. 生成报告
    print(f"\n" + "=" * 70)
    print("📋 验证报告")
    print("=" * 70)
    
    print(f"回测ID: {result.run_id}")
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据源状态: {result.data_source}/{result.data_adjust}")
    print(f"修复验证: {'成功' if result.data_source else '失败'}")
    
    if result.data_source:
        print(f"\n🎉 数据源追踪修复验证成功!")
        print(f"   单股模式现在可以正确记录数据源")
        print(f"   新回测将显示: 数据源: {result.data_source}/{result.data_adjust}")
    else:
        print(f"\n❌ 数据源追踪修复验证失败!")
        print(f"   可能原因:")
        print(f"   1. price_df没有source/adjust字段")
        print(f"   2. 代码没有正确加载")
        print(f"   3. 需要重启Python进程")
    
    print(f"\n💡 下一步:")
    print(f"   1. 访问回测记录页面查看新回测")
    print(f"   2. 验证数据源列显示正确")
    print(f"   3. 运行更多回测测试不同场景")
    
except Exception as e:
    print(f"❌ 回测执行失败: {e}")
    import traceback
    traceback.print_exc()

print(f"\n⏰ 验证完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)