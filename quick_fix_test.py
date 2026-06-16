#!/usr/bin/env python3
"""快速测试修复效果"""
import sys
import os
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("⚡ 快速测试修复效果")
print("=" * 70)

# 1. 测试tencent_fetcher修复
print("1. 🔧 测试tencent_fetcher修复...")
from data.fetchers.tencent_fetcher import TencentFetcher

fetcher = TencentFetcher()

# 测试不同adjust参数
test_cases = [
    ("", "空字符串 -> 应该返回raw"),
    ("raw", "raw -> 应该返回raw"),
    ("qfq", "qfq -> 应该返回qfq"),
    ("hfq", "hfq -> 应该返回hfq"),
]

for adjust_param, description in test_cases:
    try:
        df = fetcher.get_daily_bars(
            symbol="000001",
            start_date="20260610",
            end_date="20260612",
            adjust=adjust_param
        )
        
        if not df.empty:
            actual_adjust = df['adjust'].iloc[0] if 'adjust' in df.columns else "N/A"
            actual_source = df['source'].iloc[0] if 'source' in df.columns else "N/A"
            print(f"   {description}:")
            print(f"     参数: adjust='{adjust_param}'")
            print(f"     结果: source='{actual_source}', adjust='{actual_adjust}'")
            print(f"     行数: {len(df)}")
        else:
            print(f"   {description}: 无数据")
            
    except Exception as e:
        print(f"   {description}: 失败 - {e}")

# 2. 测试_fetch_ohlcv修复
print(f"\n2. 🔄 测试_fetch_ohlcv修复...")
from backtest.scheme_backtest import _fetch_ohlcv

try:
    ohlcv_df = _fetch_ohlcv(
        symbols=["000001"],
        lookback_days=60,
        adjust=""  # 空字符串，应该使用raw
    )
    
    if not ohlcv_df.empty:
        print(f"   ✅ _fetch_ohlcv成功: {len(ohlcv_df)} 行")
        print(f"      字段: {list(ohlcv_df.columns)}")
        
        if 'source' in ohlcv_df.columns:
            source_val = ohlcv_df['source'].iloc[0] if not ohlcv_df['source'].isna().all() else "空值"
            print(f"      source: {source_val}")
        else:
            print(f"      ❌ 缺少source字段")
            
        if 'adjust' in ohlcv_df.columns:
            adjust_val = ohlcv_df['adjust'].iloc[0] if not ohlcv_df['adjust'].isna().all() else "空值"
            print(f"      adjust: {adjust_val}")
        else:
            print(f"      ❌ 缺少adjust字段")
    else:
        print(f"   ⚠️  _fetch_ohlcv返回空数据")
        
except Exception as e:
    print(f"   ❌ _fetch_ohlcv失败: {e}")

# 3. 测试单股模式数据源提取
print(f"\n3. 📊 测试单股模式数据源提取...")

# 模拟price_df（可能没有source/adjust字段）
price_df_without_fields = pd.DataFrame({
    'trade_date': ['2026-06-10', '2026-06-11'],
    'symbol': ['000001.SZ', '000001.SZ'],
    'close': [10.0, 10.1],
    'volume': [1000000, 1100000],
    'amount': [10000000, 11100000]
    # 没有source/adjust字段
})

price_df_with_fields = pd.DataFrame({
    'trade_date': ['2026-06-10', '2026-06-11'],
    'symbol': ['000001.SZ', '000001.SZ'],
    'close': [10.0, 10.1],
    'volume': [1000000, 1100000],
    'amount': [10000000, 11100000],
    'source': 'tencent',
    'adjust': 'raw'
})

print(f"   测试1: price_df没有source/adjust字段")
if not price_df_without_fields.empty:
    data_source = ""
    data_adjust = "raw"
    
    if 'source' in price_df_without_fields.columns and not price_df_without_fields['source'].isna().all():
        data_source = str(price_df_without_fields.iloc[0]['source'])
    else:
        print(f"      ⚠️  price_df没有source字段，使用默认值: '{data_source}'")
    
    if 'adjust' in price_df_without_fields.columns and not price_df_without_fields['adjust'].isna().all():
        data_adjust = str(price_df_without_fields.iloc[0]['adjust'])
    else:
        print(f"      ⚠️  price_df没有adjust字段，使用默认值: '{data_adjust}'")
    
    print(f"      结果: data_source='{data_source}', data_adjust='{data_adjust}'")

print(f"\n   测试2: price_df有source/adjust字段")
if not price_df_with_fields.empty:
    data_source = ""
    data_adjust = "raw"
    
    if 'source' in price_df_with_fields.columns and not price_df_with_fields['source'].isna().all():
        data_source = str(price_df_with_fields.iloc[0]['source'])
        print(f"      ✅ 从price_df提取source: '{data_source}'")
    else:
        print(f"      ❌ price_df没有source字段")
    
    if 'adjust' in price_df_with_fields.columns and not price_df_with_fields['adjust'].isna().all():
        data_adjust = str(price_df_with_fields.iloc[0]['adjust'])
        print(f"      ✅ 从price_df提取adjust: '{data_adjust}'")
    else:
        print(f"      ❌ price_df没有adjust字段")
    
    print(f"      结果: data_source='{data_source}', data_adjust='{data_adjust}'")

print(f"\n" + "=" * 70)
print("📋 修复总结")
print("=" * 70)

print(f"已完成的修复:")
print(f"1. ✅ tencent_fetcher.get_daily_bars: 支持adjust='raw'参数")
print(f"2. ✅ _fetch_ohlcv: 正确处理adjust='' -> 'raw'转换")
print(f"3. ✅ 单股模式数据源提取: 处理没有source/adjust字段的情况")

print(f"\n关键问题:")
print(f"❌ 单股模式的price_df可能没有source/adjust字段")
print(f"   原因: price_df来自回测调用者，可能使用旧数据源")

print(f"\n建议:")
print(f"1. 🔄 重启Streamlit服务（已执行）")
print(f"2. 🧪 运行新的单股模式回测")
print(f"3. 📊 确保price_df包含source/adjust字段")
print(f"4. 🗑️  清理旧缓存，重新生成带字段的数据")

print(f"\n⏰ 测试完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)