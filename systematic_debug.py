#!/usr/bin/env python3
"""系统性调试数据源追踪问题"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("🔧 系统性调试数据源追踪问题")
print("=" * 80)

# 1. 分析最新回测
print("1. 📊 分析最新回测: 20260615_220911_composite")
run_dir = Path("data/backtest_runs/20260615_220911_composite")

if run_dir.exists():
    # 读取配置
    with open(run_dir / "config.json", 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"   模式: {config.get('pool_mode')}")
    print(f"   股票: {config.get('symbols')}")
    print(f"   时间: {config.get('start_date')} 到 {config.get('end_date')}")

# 2. 模拟单股模式回测流程
print(f"\n2. 🔄 模拟单股模式回测流程")

# 创建测试数据 - 模拟可能的情况
symbols = ["600021", "002167"]
start_date = "2026-03-18"
end_date = "2026-06-15"

print(f"   测试股票: {symbols}")
print(f"   回测区间: {start_date} 到 {end_date}")

# 3. 测试数据获取
print(f"\n3. 📡 测试数据获取流程")

# 方案A: 通过_fetch_ohlcv获取（全池模式）
print(f"   方案A: 通过_fetch_ohlcv获取")
from backtest.scheme_backtest import _fetch_ohlcv

try:
    ohlcv_df = _fetch_ohlcv(symbols, lookback_days=60, adjust="")
    if not ohlcv_df.empty:
        print(f"      ✅ 成功获取: {len(ohlcv_df)} 行")
        print(f"          字段: {list(ohlcv_df.columns)}")
        if 'source' in ohlcv_df.columns:
            source_val = ohlcv_df['source'].iloc[0] if not ohlcv_df['source'].isna().all() else "空值"
            print(f"          source: {source_val}")
        if 'adjust' in ohlcv_df.columns:
            adjust_val = ohlcv_df['adjust'].iloc[0] if not ohlcv_df['adjust'].isna().all() else "空值"
            print(f"          adjust: {adjust_val}")
    else:
        print(f"      ❌ 获取失败: 空数据")
except Exception as e:
    print(f"      ❌ 获取失败: {e}")

# 方案B: 通过tencent_fetcher直接获取
print(f"\n   方案B: 通过tencent_fetcher直接获取")
from data.fetchers.tencent_fetcher import TencentFetcher

fetcher = TencentFetcher()
for symbol in symbols:
    try:
        df = fetcher.get_daily_bars(
            symbol=symbol,
            start_date="20260318",
            end_date="20260615",
            adjust="raw"
        )
        if not df.empty:
            print(f"      ✅ {symbol}: {len(df)} 行")
            if 'source' in df.columns:
                print(f"          source: {df['source'].iloc[0]}")
            else:
                print(f"          ❌ 缺少source字段")
        else:
            print(f"      ❌ {symbol}: 无数据")
    except Exception as e:
        print(f"      ❌ {symbol}: 失败 - {e}")

# 4. 检查缓存状态
print(f"\n4. 💾 检查缓存状态")
cache_dir = Path("data/parquet/bars")

# 检查新缓存
new_cache_files = list(cache_dir.glob("*/*/*/*.parquet"))
print(f"   新缓存文件数: {len(new_cache_files)}")

if new_cache_files:
    for i, file in enumerate(new_cache_files[:3]):  # 检查前3个
        try:
            df = pd.read_parquet(file)
            print(f"      {file.name}:")
            print(f"          路径: {file}")
            print(f"          字段: {list(df.columns)}")
            if 'source' in df.columns:
                source_val = df['source'].iloc[0] if not df['source'].isna().all() else "空值"
                print(f"          source: {source_val}")
            if 'adjust' in df.columns:
                adjust_val = df['adjust'].iloc[0] if not df['adjust'].isna().all() else "空值"
                print(f"          adjust: {adjust_val}")
        except Exception as e:
            print(f"      读取失败: {e}")

# 5. 检查单股模式代码路径
print(f"\n5. 🔍 检查单股模式代码路径")

# 读取_run_single_stock_backtest函数
scheme_file = Path("backtest/scheme_backtest.py")
if scheme_file.exists():
    with open(scheme_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找数据源提取代码
    if 'data_source = ""' in content and 'data_adjust = "raw"' in content:
        print(f"    ✅ 找到单股模式数据源提取代码")
        
        # 提取相关代码段
        lines = content.split('\n')
        found = False
        for i, line in enumerate(lines):
            if 'if not price_df.empty:' in line:
                found = True
                print(f"       数据源提取代码 (行 {i+1}-{i+20}):")
                for j in range(i, min(i+20, len(lines))):
                    print(f"         {j+1}: {lines[j]}")
                break
        if not found:
            print(f"    ❌ 未找到数据源提取关键代码")
    else:
        print(f"    ❌ 未找到单股模式数据源提取代码")

# 6. 创建诊断测试
print(f"\n6. 🧪 创建诊断测试")

# 模拟price_df的几种情况
test_cases = [
    {
        "name": "有source/adjust字段",
        "df": pd.DataFrame({
            'trade_date': ['2026-06-10'],
            'symbol': ['600021'],
            'close': [10.0],
            'volume': [1000000],
            'amount': [10000000],
            'source': 'tencent',
            'adjust': 'raw'
        })
    },
    {
        "name": "有source字段但为空",
        "df": pd.DataFrame({
            'trade_date': ['2026-06-10'],
            'symbol': ['600021'],
            'close': [10.0],
            'volume': [1000000],
            'amount': [10000000],
            'source': '',
            'adjust': 'raw'
        })
    },
    {
        "name": "没有source/adjust字段",
        "df": pd.DataFrame({
            'trade_date': ['2026-06-10'],
            'symbol': ['600021'],
            'close': [10.0],
            'volume': [1000000],
            'amount': [10000000]
        })
    }
]

print(f"   测试数据源提取逻辑:")
for case in test_cases:
    price_df = case["df"]
    print(f"\n   {case['name']}:")
    print(f"      字段: {list(price_df.columns)}")
    
    # 模拟修复后的提取逻辑
    data_source = ""
    data_adjust = "raw"
    
    if not price_df.empty:
        if 'source' in price_df.columns and not price_df['source'].isna().all():
            data_source = str(price_df.iloc[0]['source'])
            print(f"      提取source: '{data_source}'")
        else:
            print(f"      ⚠️  没有source字段或全部为空")
        
        if 'adjust' in price_df.columns and not price_df['adjust'].isna().all():
            data_adjust = str(price_df.iloc[0]['adjust'])
            print(f"      提取adjust: '{data_adjust}'")
        else:
            print(f"      ⚠️  没有adjust字段或全部为空")
    
    print(f"      结果: data_source='{data_source}', data_adjust='{data_adjust}'")

print(f"\n" + "=" * 80)
print("📋 系统性诊断结论")
print("=" * 80)

print(f"基于最新回测 20260615_220911_composite 的诊断:")
print(f"1. ✅ 修复代码已生效: data_version格式为 'source=, adjust=raw, single_stock_mode'")
print(f"2. ❌ 但data_source为空字符串: 说明price_df的source字段是空字符串或不存在")
print(f"\n根本原因:")
print(f"  🔴 price_df包含source字段，但值为空字符串('')")
print(f"  或")
print(f"  🔴 price_df没有source字段，提取逻辑使用默认空字符串")

print(f"\n验证方法:")
print(f"1. 在_run_single_stock_backtest函数中添加调试日志")
print(f"2. 打印price_df的字段和source值")
print(f"3. 追踪price_df的来源")

print(f"\n立即修复:")
print(f"1. 检查price_df来源，确保包含正确的source字段")
print(f"2. 如果source为空，使用数据源名称填充")
print(f"3. 添加数据源字段验证")

print(f"\n⏰ 诊断时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)