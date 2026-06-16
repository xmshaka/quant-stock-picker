#!/usr/bin/env python3
"""运行测试回测验证数据源追踪"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
from pathlib import Path
import json

print("🚀 运行测试回测验证数据源追踪...")

# 创建测试数据
start_date = "2026-06-10"
end_date = "2026-06-15"
symbol = "000001.SZ"

# 创建带数据源的测试K线数据
dates = pd.date_range(start=start_date, end=end_date, freq='D')

test_data = pd.DataFrame({
    'trade_date': dates,
    'symbol': symbol,
    'open': [10.0 + i*0.1 for i in range(len(dates))],
    'high': [10.5 + i*0.1 for i in range(len(dates))],
    'low': [9.5 + i*0.1 for i in range(len(dates))],
    'close': [10.0 + i*0.1 for i in range(len(dates))],
    'volume': [1000000 + i*10000 for i in range(len(dates))],
    'amount': [10000000 + i*100000 for i in range(len(dates))],
    'source': 'test_source',
    'adjust': 'raw'
})

print(f"📊 测试数据: {len(test_data)} 行")
print(f"   数据源字段: source={test_data['source'].iloc[0]}, adjust={test_data['adjust'].iloc[0]}")

# 直接测试SchemeBacktestResult创建
from backtest.scheme_backtest import SchemeBacktestResult

print("\n🧪 测试SchemeBacktestResult创建:")
result = SchemeBacktestResult(
    scheme_id="test_backtest",
    scheme_name="测试回测",
    start_date=start_date,
    end_date=end_date,
    data_source=test_data['source'].iloc[0],
    data_adjust=test_data['adjust'].iloc[0],
    data_version=f"source={test_data['source'].iloc[0]}, adjust={test_data['adjust'].iloc[0]}, timestamp={datetime.now().strftime('%Y%m%d_%H%M%S')}",
    total_return=0.05,
    annual_return=0.12,
    sharpe_ratio=1.2,
    max_drawdown=0.03,
    win_rate=0.55,
    trade_count=5,
    buy_count=3,
    sell_count=2,
    final_value=1050000.0,
    run_id=f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)

print(f"✅ SchemeBacktestResult创建成功:")
print(f"   数据源: {result.data_source}/{result.data_adjust}")
print(f"   数据版本: {result.data_version}")

# 测试持久化
from backtest.records import persist_backtest_run, BacktestRunConfig

print("\n💾 测试回测记录持久化...")

with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir) / "backtest_runs"
    tmp_path.mkdir(parents=True, exist_ok=True)
    
    config = BacktestRunConfig(
        run_id=result.run_id,
        scheme_id="test_backtest",
        scheme_name="测试回测",
        start_date=start_date,
        end_date=end_date,
        lookback_days=60,
        top_n=20,
        initial_capital=1000000.0,
    )
    
    run_dir = persist_backtest_run(
        result=result,
        config=config,
        trades=pd.DataFrame([{
            'trade_date': '2026-06-12',
            'symbol': symbol,
            'side': 'buy',
            'price': 10.1,
            'volume': 100,
            'amount': 1010.0,
            'pnl': 50.0
        }]),
        signals_raw=pd.DataFrame(),
        signals_executed=pd.DataFrame(),
        equity=pd.DataFrame({'date': dates, 'value': [1000000 + i*10000 for i in range(len(dates))]}),
        positions=pd.DataFrame(),
        factor_snapshot=pd.DataFrame(),
        root=tmp_path,
    )
    
    print(f"✅ 回测记录已保存到: {run_dir}")
    
    # 检查保存的文件
    metrics_file = run_dir / "metrics.json"
    assert metrics_file.exists(), "metrics.json不存在"
    
    with open(metrics_file, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    
    print(f"✅ metrics.json验证:")
    print(f"   data_source: {metrics.get('data_source', 'NOT FOUND')}")
    print(f"   data_adjust: {metrics.get('data_adjust', 'NOT FOUND')}")
    print(f"   data_version: {metrics.get('data_version', 'NOT FOUND')}")
    
    # 验证字段存在且正确
    assert 'data_source' in metrics, "metrics.json缺少data_source字段"
    assert 'data_adjust' in metrics, "metrics.json缺少data_adjust字段"
    assert 'data_version' in metrics, "metrics.json缺少data_version字段"
    assert metrics['data_source'] == 'test_source', f"data_source应为'test_source', 实际为'{metrics['data_source']}'"
    assert metrics['data_adjust'] == 'raw', f"data_adjust应为'raw', 实际为'{metrics['data_adjust']}'"
    
    # 检查其他文件
    files = list(run_dir.glob("*"))
    print(f"\n📁 生成的文件:")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"   {f.name}: {size_kb:.1f} KB")

print("\n🎉 数据源追踪验证测试成功完成！")
print("\n📋 验证结果:")
print("1. ✅ SchemeBacktestResult类支持数据源字段")
print("2. ✅ 回测记录持久化包含数据源信息")
print("3. ✅ metrics.json正确保存数据源字段")
print("4. ✅ 所有验证测试通过")

print("\n💡 建议:")
print("1. 重启Streamlit服务清除缓存")
print("2. 运行实际回测验证数据源追踪")
print("3. 检查回测记录页是否显示正确的数据源")