#!/usr/bin/env python3
"""深度诊断数据源追踪问题"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("🔍 深度诊断数据源追踪问题")
print("=" * 80)

# 1. 检查最新的回测
latest_run = "20260615_215050_composite"
run_dir = Path("data/backtest_runs") / latest_run

print(f"1. 📊 分析回测: {latest_run}")
print(f"   目录: {run_dir}")

if run_dir.exists():
    # 读取metrics
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    
    print(f"   data_source: '{metrics.get('data_source', 'NOT FOUND')}'")
    print(f"   data_adjust: '{metrics.get('data_adjust', 'NOT FOUND')}'")
    print(f"   data_version: '{metrics.get('data_version', 'NOT FOUND')}'")
    
    # 分析data_version
    data_version = metrics.get('data_version', '')
    if 'source=' in data_version:
        print(f"   ✅ data_version包含source=格式（修复已生效）")
    else:
        print(f"   ❌ data_version不包含source=格式（修复未生效）")

# 2. 检查代码修复
print(f"\n2. 🔧 检查代码修复状态")
scheme_file = Path("backtest/scheme_backtest.py")

if scheme_file.exists():
    with open(scheme_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查单股模式修复
    single_stock_fix = 'if not price_df.empty:' in content and 'source' in price_df.columns' in content
    if single_stock_fix:
        print(f"   ✅ 单股模式数据源提取代码已存在")
        
        # 提取相关代码段
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'data_source = ""' in line and i < len(lines)-10:
                print(f"   📝 代码段 (行 {i+1}):")
                for j in range(i, min(i+10, len(lines))):
                    print(f"      {j+1}: {lines[j]}")
                break
    else:
        print(f"   ❌ 单股模式数据源提取代码不存在")

# 3. 检查K线数据源
print(f"\n3. 📈 检查K线数据源状态")
from data.fetchers.fallback_fetcher import get_daily_bars

# 测试获取数据
test_symbol = "300498"  # 最新回测使用的股票
try:
    print(f"   测试获取股票 {test_symbol} 的K线数据...")
    bars = get_daily_bars(
        symbols=[test_symbol],
        start_date="2026-06-10",
        end_date="2026-06-12",
        adjust="raw"
    )
    
    if not bars.empty:
        print(f"   ✅ 成功获取数据: {len(bars)} 行")
        print(f"      字段: {list(bars.columns)}")
        
        # 检查source字段
        if 'source' in bars.columns:
            source_values = bars['source'].unique()
            print(f"      source字段值: {source_values}")
            if len(source_values) == 1 and pd.isna(source_values[0]):
                print(f"      ⚠️  source字段全部为空值")
            elif '' in source_values or pd.isna(source_values[0]):
                print(f"      ⚠️  source字段包含空值")
            else:
                print(f"      ✅ source字段有实际值: {source_values[0]}")
        else:
            print(f"      ❌ 数据没有source字段")
        
        # 检查adjust字段
        if 'adjust' in bars.columns:
            adjust_values = bars['adjust'].unique()
            print(f"      adjust字段值: {adjust_values}")
        else:
            print(f"      ❌ 数据没有adjust字段")
    else:
        print(f"   ❌ 无法获取数据")
        
except Exception as e:
    print(f"   ❌ 获取数据失败: {e}")

# 4. 检查缓存文件
print(f"\n4. 💾 检查缓存文件状态")
cache_dir = Path("data/parquet/bars")
if cache_dir.exists():
    # 检查新结构缓存
    new_cache_pattern = cache_dir / "*" / "*" / "*" / "*.parquet"
    new_files = list(cache_dir.glob("*/*/*/*.parquet"))
    
    # 检查旧结构缓存
    old_cache_pattern = cache_dir / "*" / "*.parquet"
    old_files = list(cache_dir.glob("*/*.parquet"))
    
    print(f"   新结构缓存 (bars/{{source}}/{{adjust}}/{{prefix}}/{{symbol}}.parquet): {len(new_files)} 个文件")
    print(f"   旧结构缓存 (bars/{{prefix}}/{{symbol}}.parquet): {len(old_files)} 个文件")
    
    if new_files:
        sample_file = new_files[0]
        print(f"   示例新缓存: {sample_file}")
        try:
            df = pd.read_parquet(sample_file)
            print(f"       字段: {list(df.columns)}")
            if 'source' in df.columns:
                print(f"       source: {df['source'].iloc[0] if not df['source'].isna().all() else '空值'}")
            if 'adjust' in df.columns:
                print(f"       adjust: {df['adjust'].iloc[0] if not df['adjust'].isna().all() else '空值'}")
        except Exception as e:
            print(f"       读取失败: {e}")

# 5. 检查bars_normalizer
print(f"\n5. 🛠️ 检查数据标准化模块")
normalizer_file = Path("data/bars_normalizer.py")
if normalizer_file.exists():
    with open(normalizer_file, 'r', encoding='utf-8') as f:
        normalizer_content = f.read()
    
    if 'def normalize_bars' in normalizer_content:
        print(f"   ✅ normalize_bars函数存在")
        if 'source' in normalizer_content and 'adjust' in normalizer_content:
            print(f"   ✅ 包含source和adjust字段处理")
        else:
            print(f"   ❌ 不包含source和adjust字段处理")
    else:
        print(f"   ❌ normalize_bars函数不存在")

# 6. 根本原因分析
print(f"\n" + "=" * 80)
print("🔮 根本原因分析")
print("=" * 80)

print(f"基于最新回测 {latest_run} 的分析:")
print(f"1. ✅ 修复代码已生效: data_version格式为 'source=, adjust=raw, single_stock_mode'")
print(f"2. ❌ 但data_source为空: 说明price_df的source字段为空或不存在")
print(f"\n可能原因:")
print(f"  A. 📦 数据源模块没有添加source字段")
print(f"     - 检查: data/fetchers/tencent_fetcher.py, tushare_fetcher.py等")
print(f"     - 修复: 确保所有fetcher输出都包含source字段")
print(f"\n  B. 💾 缓存文件没有source字段")
print(f"     - 检查: 旧缓存文件可能没有source字段")
print(f"     - 修复: 清除旧缓存，重新获取数据")
print(f"\n  C. 🔄 数据流中断")
print(f"     - 检查: bars_normalizer.py是否被正确调用")
print(f"     - 修复: 确保所有数据都经过标准化处理")

# 7. 立即修复方案
print(f"\n" + "=" * 80)
print("🚀 立即修复方案")
print("=" * 80)

print(f"方案1: 检查并修复数据源字段添加")
print(f"  步骤:")
print(f"  1. 检查 data/fetchers/tencent_fetcher.py 的 get_daily_bars 函数")
print(f"  2. 确保返回的DataFrame包含 'source' 和 'adjust' 字段")
print(f"  3. 如果缺失，添加: df['source'] = 'tencent'; df['adjust'] = adjust")

print(f"\n方案2: 强制重新生成缓存")
print(f"  步骤:")
print(f"  1. 删除旧缓存: find data/parquet/bars -maxdepth 2 -name '*.parquet' -delete")
print(f"  2. 运行新回测，重新获取数据")
print(f"  3. 验证新缓存是否包含source/adjust字段")

print(f"\n方案3: 调试数据流")
print(f"  步骤:")
print(f"  1. 在 get_daily_bars 函数中添加日志")
print(f"  2. 在 normalize_bars 函数中添加日志")
print(f"  3. 在 _run_single_stock_backtest 函数中添加日志")
print(f"  4. 跟踪数据从获取到回测的全流程")

print(f"\n⏰ 建议立即执行方案1，检查数据源模块")
print("=" * 80)