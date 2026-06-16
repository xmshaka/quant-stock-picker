#!/usr/bin/env python3
"""立即验证数据源修复是否生效"""
import sys
import os
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("🚨 立即验证数据源修复是否生效")
print("=" * 70)

# 1. 强制重新加载模块
print("1. 🔄 强制重新加载模块...")
modules_to_reload = [
    'backtest.scheme_backtest',
    'backtest.records',
    'data.fetchers.base',
    'data.bars_normalizer'
]

for module_name in modules_to_reload:
    if module_name in sys.modules:
        print(f"   删除模块缓存: {module_name}")
        del sys.modules[module_name]

# 2. 导入修复后的模块
print("\n2. 📦 导入修复后的模块...")
try:
    # 重新导入
    import backtest.scheme_backtest as scheme_backtest
    importlib.reload(scheme_backtest)
    
    print(f"   ✅ 成功导入 backtest.scheme_backtest")
    print(f"   模块路径: {scheme_backtest.__file__}")
    
    # 检查修改时间
    import os
    import time
    module_file = scheme_backtest.__file__
    if os.path.exists(module_file):
        mtime = os.path.getmtime(module_file)
        print(f"   模块最后修改: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))}")
    
except Exception as e:
    print(f"   ❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()

# 3. 检查修复的代码
print("\n3. 🔍 检查修复的代码...")
import inspect

# 检查_run_single_stock_backtest函数
source = inspect.getsource(scheme_backtest.SchemeBacktester._run_single_stock_backtest)
if 'data_source = ""' in source and 'data_adjust = "raw"' in source:
    # 检查是否包含我们的修复逻辑
    if 'if not price_df.empty:' in source and 'source' in price_df.columns' in source:
        print("   ✅ 找到修复的数据源提取逻辑")
    else:
        print("   ❌ 没有找到修复的数据源提取逻辑")
        print("      可能代码没有正确保存或加载")

# 4. 测试数据源提取
print("\n4. 🧪 测试数据源提取...")
import pandas as pd

# 创建测试数据
test_price_df = pd.DataFrame({
    'trade_date': ['2026-06-10'],
    'symbol': ['000001.SZ'],
    'close': [10.0],
    'volume': [1000000],
    'amount': [10000000],
    'source': ['tencent'],
    'adjust': ['raw']
})

# 模拟修复逻辑
data_source = ""
data_adjust = "raw"
data_version = "single_stock_mode"

if not test_price_df.empty:
    if 'source' in test_price_df.columns and not test_price_df['source'].isna().all():
        data_source = str(test_price_df.iloc[0]['source'])
    if 'adjust' in test_price_df.columns and not test_price_df['adjust'].isna().all():
        data_adjust = str(test_price_df.iloc[0]['adjust'])
    data_version = f"source={data_source}, adjust={data_adjust}, single_stock_mode"

print(f"   测试结果:")
print(f"     数据源: {data_source}/{data_adjust}")
print(f"     版本: {data_version}")

# 5. 检查Streamlit模块缓存
print("\n5. 🐍 检查Python模块缓存...")
print(f"   sys.modules 中 backtest 相关模块:")
for key in list(sys.modules.keys()):
    if 'backtest' in key or 'scheme' in key:
        print(f"     {key}")

# 6. 建议解决方案
print("\n" + "=" * 70)
print("🔧 建议的解决方案")
print("=" * 70)

print("问题: Streamlit 服务没有重新加载 Python 模块")
print("原因: Python 模块缓存，Streamlit 默认缓存模块")
print("\n解决方案 (按优先级):")

print("1. 🚨 立即方案: 重启 Streamlit 服务")
print("   kill -9 1422955")
print("   cd /root/.openclaw/workspace/quant-stock-picker")
print("   nohup ./venv/bin/streamlit run dashboard/量化选股.py --server.address=0.0.0.0 --server.port=5004 --server.headless=true > streamlit.log 2>&1 &")

print("\n2. 🔧 代码方案: 在 Streamlit 应用中强制重新加载")
print("   在 dashboard/量化选股.py 开头添加:")
print("   import sys")
print("   if 'backtest.scheme_backtest' in sys.modules:")
print("       del sys.modules['backtest.scheme_backtest']")

print("\n3. ⚡ 临时方案: 直接运行回测验证")
print("   在终端直接运行 Python 脚本，绕过 Streamlit 缓存")
print("   venv/bin/python -c \"from backtest.scheme_backtest import SchemeBacktestResult; print('✅ 模块已加载')\"")

print("\n4. 📝 验证方案: 创建独立的验证回测")
print("   运行独立脚本验证修复效果")

print("\n⏰ 建议立即执行方案1 (重启Streamlit服务)")
print("=" * 70)