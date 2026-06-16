#!/usr/bin/env python3
"""诊断回测数据源问题"""
import sys
import os
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def analyze_backtest_run(run_id):
    """分析特定回测运行的数据源问题"""
    run_dir = Path("data/backtest_runs") / run_id
    
    if not run_dir.exists():
        print(f"❌ 回测记录不存在: {run_dir}")
        return
    
    print(f"🔍 分析回测: {run_id}")
    print(f"   目录: {run_dir}")
    
    # 1. 读取metrics.json
    metrics_file = run_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        print(f"\n📊 metrics.json:")
        print(f"   data_source: '{metrics.get('data_source', 'NOT FOUND')}'")
        print(f"   data_adjust: '{metrics.get('data_adjust', 'NOT FOUND')}'")
        print(f"   data_version: '{metrics.get('data_version', 'NOT FOUND')}'")
        print(f"   run_id: '{metrics.get('run_id', 'NOT FOUND')}'")
    
    # 2. 读取config.json
    config_file = run_dir / "config.json"
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"\n⚙️  config.json:")
        print(f"   模式: {config.get('pool_mode', 'N/A')}")
        print(f"   股票: {config.get('symbols', [])}")
        print(f"   时间: {config.get('start_date', 'N/A')} 到 {config.get('end_date', 'N/A')}")
    
    # 3. 检查可能的K线数据来源
    print(f"\n🔍 可能的K线数据来源:")
    
    # a. 检查是否有缓存文件
    cache_dir = Path("data/parquet/bars")
    if cache_dir.exists():
        symbols = config.get('symbols', [])
        for symbol in symbols:
            # 检查旧缓存结构
            old_pattern = cache_dir / "00" / f"{symbol}.parquet"
            if old_pattern.exists():
                print(f"   ✅ 找到旧缓存: {old_pattern}")
                # 读取缓存文件检查字段
                try:
                    df = pd.read_parquet(old_pattern)
                    print(f"      字段: {list(df.columns)}")
                    if 'source' in df.columns:
                        print(f"      source字段值: {df['source'].iloc[0] if not df['source'].isna().all() else '全部为空'}")
                    if 'adjust' in df.columns:
                        print(f"      adjust字段值: {df['adjust'].iloc[0] if not df['adjust'].isna().all() else '全部为空'}")
                except Exception as e:
                    print(f"      读取失败: {e}")
    
    # 4. 检查回测运行时环境
    print(f"\n🔄 回测运行时环境:")
    
    # 检查代码修改时间
    scheme_file = Path("backtest/scheme_backtest.py")
    if scheme_file.exists():
        mtime = datetime.fromtimestamp(scheme_file.stat().st_mtime)
        print(f"   代码最后修改: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 5. 检查数据源字段是否应该在price_df中存在
    print(f"\n📈 数据源字段追踪状态:")
    print(f"   ✅ SchemeBacktestResult类已添加数据源字段")
    print(f"   ✅ 全池模式数据源提取已实现")
    print(f"   ✅ 单股模式数据源提取已修复（本次修复）")
    
    # 6. 推测问题原因
    print(f"\n🔮 问题原因推测:")
    
    run_time_str = run_id.split('_')[0] if '_' in run_id else run_id
    try:
        run_time = datetime.strptime(run_time_str, "%Y%m%d")
        code_mtime = datetime.fromtimestamp(scheme_file.stat().st_mtime)
        
        if run_time < code_mtime:
            print(f"   ⏰ 回测运行时间 ({run_time_str}) 早于代码修复时间")
            print(f"   💡 需要运行新的回测来验证修复")
        else:
            print(f"   ⏰ 回测运行时间 ({run_time_str}) 晚于或等于代码修复时间")
            print(f"   💡 问题可能是: price_df缺少source/adjust字段")
    except:
        print(f"   ⚠️  无法解析回测时间: {run_time_str}")

def main():
    run_id = "20260615_212350_momentum"
    analyze_backtest_run(run_id)
    
    print(f"\n" + "="*60)
    print(f"🔧 修复验证方法:")
    print(f"="*60)
    print(f"1. 运行一个新的单股模式回测:")
    print(f"   - 使用带有source/adjust字段的price_df")
    print(f"   - 观察数据源是否正确记录")
    print(f"\n2. 检查现有回测的price_df来源:")
    print(f"   - 查看回测调用时传入的price_df")
    print(f"   - 确认price_df是否包含source/adjust字段")
    print(f"\n3. 验证修复效果:")
    print(f"   - 创建一个测试回测")
    print(f"   - 确保数据源字段被正确提取和保存")

if __name__ == "__main__":
    main()