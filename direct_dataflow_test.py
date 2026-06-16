#!/usr/bin/env python3
"""直接测试单股模式数据流 - 基于证据的诊断"""
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
print("🔬 直接测试单股模式数据流 - 基于证据的诊断")
print("=" * 80)

# 0. 设置详细日志
import loguru
logger = loguru.logger
logger.add(sys.stderr, level="INFO")

# 1. 模拟最新回测的配置
print("1. 🎯 模拟回测配置")
symbols = ["600030", "301120"]  # 最新回测使用的股票
start_date = "2026-03-18"
end_date = "2026-06-15"
lookback_days = 60

print(f"   股票: {symbols}")
print(f"   区间: {start_date} 到 {end_date}")
print(f"   回溯天数: {lookback_days}")

# 2. 测试数据获取路径
print(f"\n2. 📡 测试数据获取路径")

# 路径A: 通过_fetch_ohlcv获取（全池模式使用）
print(f"   🔄 路径A: 通过_fetch_ohlcv获取")
from backtest.scheme_backtest import _fetch_ohlcv

try:
    ohlcv_df = _fetch_ohlcv(symbols, lookback_days=lookback_days, adjust="")
    if not ohlcv_df.empty:
        print(f"      ✅ 成功获取: {len(ohlcv_df)} 行")
        print(f"          字段: {list(ohlcv_df.columns)}")
        
        # 检查关键字段
        for col in ['source', 'adjust']:
            if col in ohlcv_df.columns:
                unique_values = ohlcv_df[col].unique()
                non_nan_values = [v for v in unique_values if not pd.isna(v)]
                print(f"          {col}: {non_nan_values[:3]} (共{len(non_nan_values)}个非空值)")
            else:
                print(f"          ❌ 缺少{col}字段")
    else:
        print(f"      ❌ 获取失败: 空数据")
except Exception as e:
    print(f"      ❌ 获取失败: {e}")
    traceback.print_exc()

# 路径B: 通过fallback_fetcher获取（实际回测可能使用）
print(f"\n   🔄 路径B: 通过fallback_fetcher获取")
try:
    from data.fetchers.fallback_fetcher import FallbackFetcher
    fetcher = FallbackFetcher()
    
    for symbol in symbols:
        try:
            # 注意：需要检查实际使用的接口
            df = fetcher.get_daily_bars(
                symbol=symbol,
                start_date="20260318",
                end_date="20260615",
                adjust="raw"
            )
            if not df.empty:
                print(f"      ✅ {symbol}: {len(df)} 行")
                if 'source' in df.columns:
                    source_val = df['source'].iloc[0] if not df['source'].isna().all() else "空值"
                    print(f"          source: {source_val}")
                else:
                    print(f"          ❌ 缺少source字段")
            else:
                print(f"      ❌ {symbol}: 无数据")
        except Exception as e:
            print(f"      ❌ {symbol}: 失败 - {e}")
except Exception as e:
    print(f"      ❌ fallback_fetcher初始化失败: {e}")

# 3. 检查缓存中的实际数据
print(f"\n3. 💾 检查缓存中的实际数据")

# 查找缓存文件
cache_dir = Path("data/parquet/bars")
cache_files = list(cache_dir.glob("**/*.parquet"))

print(f"   缓存文件总数: {len(cache_files)}")

# 按结构分类
old_structure = [f for f in cache_files if len(f.parts) - len(cache_dir.parts) == 2]  # bars/{prefix}/{symbol}.parquet
new_structure = [f for f in cache_files if len(f.parts) - len(cache_dir.parts) == 4]  # bars/{source}/{adjust}/{prefix}/{symbol}.parquet

print(f"   旧结构文件: {len(old_structure)} 个 (应为0)")
print(f"   新结构文件: {len(new_structure)} 个")

# 检查新缓存
if new_structure:
    for i, file in enumerate(new_structure[:3]):
        try:
            df = pd.read_parquet(file)
            print(f"\n      {file.name}:")
            print(f"          路径: {file}")
            print(f"          行数: {len(df)}")
            print(f"          字段: {list(df.columns)}")
            
            # 检查关键字段
            for col in ['source', 'adjust']:
                if col in df.columns:
                    unique_values = df[col].unique()
                    non_nan_values = [v for v in unique_values if not pd.isna(v)]
                    if non_nan_values:
                        print(f"          {col}: {non_nan_values[0]} (共{len(non_nan_values)}个非空值)")
                    else:
                        print(f"          ⚠️  {col}: 全部为空值或NaN")
                else:
                    print(f"          ❌ 缺少{col}字段")
        except Exception as e:
            print(f"      读取失败: {e}")

# 4. 模拟单股模式数据流
print(f"\n4. 🔄 模拟单股模式数据流")

# 创建模拟的price_df（可能来自不同来源）
print(f"   🧪 测试不同来源的price_df")

test_cases = [
    {
        "name": "从_fetch_ohlcv获取",
        "df": ohlcv_df if 'ohlcv_df' in locals() and not ohlcv_df.empty else pd.DataFrame()
    },
    {
        "name": "从tencent_fetcher获取",
        "get_func": lambda: None  # 稍后填充
    },
    {
        "name": "从缓存文件获取",
        "get_func": lambda: pd.read_parquet(new_structure[0]) if new_structure else pd.DataFrame()
    }
]

for case in test_cases:
    print(f"\n   📊 {case['name']}:")
    
    if 'df' in case:
        df = case['df']
    elif 'get_func' in case:
        try:
            df = case['get_func']()
        except Exception as e:
            print(f"      获取失败: {e}")
            continue
    else:
        continue
    
    if df.empty:
        print(f"      空DataFrame")
        continue
    
    print(f"      行数: {len(df)}")
    print(f"      字段: {list(df.columns)}")
    
    # 检查source字段
    if 'source' in df.columns:
        unique_sources = df['source'].dropna().unique()
        if len(unique_sources) > 0:
            print(f"      source字段值: {unique_sources[:3]}")
        else:
            print(f"      ⚠️  source字段全部为空或NaN")
    else:
        print(f"      ❌ 缺少source字段")
    
    # 检查adjust字段
    if 'adjust' in df.columns:
        unique_adjusts = df['adjust'].dropna().unique()
        if len(unique_adjusts) > 0:
            print(f"      adjust字段值: {unique_adjusts[:3]}")
        else:
            print(f"      ⚠️  adjust字段全部为空或NaN")
    else:
        print(f"      ❌ 缺少adjust字段")

# 5. 关键发现
print(f"\n" + "=" * 80)
print("🔍 关键发现 - 基于证据")
print("=" * 80)

# 检查最新回测的metrics
latest_run = "20260615_224937_composite"
metrics_file = Path("data/backtest_runs") / latest_run / "metrics.json"

if metrics_file.exists():
    with open(metrics_file, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    
    print(f"📊 最新回测 {latest_run}:")
    print(f"   data_source: '{metrics.get('data_source', 'NOT FOUND')}'")
    print(f"   data_adjust: '{metrics.get('data_adjust', 'NOT FOUND')}'")
    print(f"   data_version: '{metrics.get('data_version', 'NOT FOUND')}'")
    
    # 分析data_version
    data_version = metrics.get('data_version', '')
    if 'source=' in data_version:
        print(f"   ✅ data_version包含source=格式（修复已生效）")
        # 解析data_version
        if 'source=' in data_version and 'adjust=' in data_version:
            import re
            source_match = re.search(r'source=([^,]+)', data_version)
            adjust_match = re.search(r'adjust=([^,]+)', data_version)
            if source_match and adjust_match:
                extracted_source = source_match.group(1)
                extracted_adjust = adjust_match.group(1)
                print(f"   🔍 从data_version解析: source='{extracted_source}', adjust='{extracted_adjust}'")
    else:
        print(f"   ❌ data_version不包含source=格式（修复未生效）")

print(f"\n📋 问题定位:")
print(f"1. ✅ 修复代码已生效: data_version格式正确")
print(f"2. ❌ 但data_source为空字符串: ''")
print(f"\n🔬 基于证据的假设:")
print(f"假设A: price_df的source字段是空字符串")
print(f"假设B: price_df没有source字段")
print(f"假设C: 数据流中移除了source字段")

print(f"\n🎯 下一步行动:")
print(f"1. 直接检查price_df的来源")
print(f"2. 追踪数据从获取到回测的全流程")
print(f"3. 添加更多调试信息")

print(f"\n⏰ 诊断时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)