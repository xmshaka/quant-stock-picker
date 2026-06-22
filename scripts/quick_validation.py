#!/usr/bin/env python3
"""
快速验证优化后的趋势动量策略
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tests.test_signals_layers import make_bars
from signals.layers import ResonanceChecker

def create_realistic_test_data(n_days=60, n_stocks=20):
    """创建更真实的测试数据"""
    all_bars = []
    
    for stock_idx in range(n_stocks):
        # 基础走势
        if stock_idx % 3 == 0:
            trend = "up"  # 上涨
        elif stock_idx % 3 == 1:
            trend = "down"  # 下跌
        else:
            trend = "sideways"  # 横盘
        
        bars = make_bars(n_days, trend=trend)
        
        # 添加股票代码
        stock_code = f"000{stock_idx+1:03d}.SH"
        bars["symbol"] = stock_code
        
        # 添加更真实的因子数据
        np.random.seed(stock_idx)
        
        # 资金流：与趋势相关
        if trend == "up":
            mf_mean = 50000
        elif trend == "down":
            mf_mean = -20000
        else:
            mf_mean = 10000
            
        bars["main_net_mf_amount"] = np.random.normal(mf_mean, abs(mf_mean)*0.3, n_days)
        bars["large_elg_net_mf_amount"] = np.random.normal(mf_mean*2, abs(mf_mean)*0.3, n_days)
        bars["large_elg_net_mf_rank"] = np.random.uniform(0.5 if trend=="down" else 0.6, 0.9, n_days)
        
        # 量能因子
        if trend == "up":
            turnover_mean = 1.3
        elif trend == "down":
            turnover_mean = 0.8
        else:
            turnover_mean = 1.0
            
        bars["relative_turnover_5d"] = np.random.normal(turnover_mean, 0.2, n_days)
        bars["amount_percentile_60d"] = np.random.uniform(0.4 if trend=="down" else 0.6, 0.85, n_days)
        
        all_bars.append(bars)
    
    return pd.concat(all_bars, ignore_index=True)

def analyze_buy_conditions(strategy_id="trend_momentum"):
    """分析买点条件"""
    print(f"\n=== 分析 {strategy_id} 策略买点条件 ===")
    
    # 创建测试数据
    print("创建测试数据...")
    all_bars = create_realistic_test_data(n_days=60, n_stocks=20)
    
    # 按股票分析
    results = []
    symbols = all_bars["symbol"].unique()
    
    for symbol in symbols[:10]:  # 只分析前10只股票
        symbol_bars = all_bars[all_bars["symbol"] == symbol].copy()
        
        if len(symbol_bars) < 30:
            continue
            
        # 分析最后10个交易日
        for idx in range(len(symbol_bars)-10, len(symbol_bars)):
            day_bars = symbol_bars.iloc[:idx+1]
            
            rc = ResonanceChecker.from_strategy(strategy_id)
            buy_result, conditions = rc.check_buy(day_bars, idx)
            
            if buy_result:
                # 统计条件满足情况
                quant_conditions = [c for c in conditions if any(f in c.key for f in ['mf_', 'turnover', 'percentile', 'rank'])]
                tech_conditions = [c for c in conditions if c not in quant_conditions]
                
                quant_met = sum(1 for c in quant_conditions if c.met)
                tech_met = sum(1 for c in tech_conditions if c.met)
                
                results.append({
                    "symbol": symbol,
                    "date_idx": idx,
                    "buy_signal": buy_result,
                    "total_conditions": len(conditions),
                    "quant_conditions": len(quant_conditions),
                    "quant_met": quant_met,
                    "tech_conditions": len(tech_conditions),
                    "tech_met": tech_met,
                    "quant_ratio": quant_met / len(quant_conditions) if len(quant_conditions) > 0 else 0,
                })
    
    # 汇总分析
    if results:
        df = pd.DataFrame(results)
        print(f"\n买点信号总数: {len(df)}")
        print(f"涉及股票数: {df['symbol'].nunique()}")
        
        print("\n量化因子条件分析:")
        print(f"  平均满足率: {df['quant_ratio'].mean():.1%}")
        print(f"  高满足率(>80%): {sum(df['quant_ratio'] > 0.8)} 个信号")
        print(f"  中满足率(50-80%): {sum((df['quant_ratio'] >= 0.5) & (df['quant_ratio'] <= 0.8))} 个信号")
        print(f"  低满足率(<50%): {sum(df['quant_ratio'] < 0.5)} 个信号")
        
        print("\n技术指标条件分析:")
        if df['tech_conditions'].sum() > 0:
            tech_ratio = (df['tech_met'] / df['tech_conditions']).mean()
            print(f"  平均满足率: {tech_ratio:.1%}")
        else:
            print(f"  无技术指标条件")
        
        # 确定性评估
        high_confidence = sum((df['quant_ratio'] > 0.7) & (df['tech_met'] >= 3))
        medium_confidence = sum((df['quant_ratio'] > 0.5) & (df['tech_met'] >= 2))
        low_confidence = len(df) - high_confidence - medium_confidence
        
        print(f"\n交易确定性评估:")
        print(f"  高确定性: {high_confidence} ({high_confidence/len(df):.1%})")
        print(f"  中确定性: {medium_confidence} ({medium_confidence/len(df):.1%})")
        print(f"  低确定性: {low_confidence} ({low_confidence/len(df):.1%})")
        
        return df
    else:
        print("未生成买点信号")
        return None

def main():
    """主函数"""
    print("=== 趋势动量策略优化验证 ===")
    
    # 1. 验证趋势动量策略
    df_trend = analyze_buy_conditions("trend_momentum")
    
    # 2. 对比原始逻辑（如果需要）
    # 这里可以添加对比逻辑
    
    # 3. 专业评估
    if df_trend is not None:
        quant_avg = df_trend['quant_ratio'].mean()
        
        print(f"\n=== 专业评估 ===")
        if quant_avg > 0.7:
            print("✅ 优秀：量化因子主导买点决策")
            print("   建议：继续优化其他策略")
        elif quant_avg > 0.5:
            print("⚠️ 良好：量化因子参与决策")
            print("   建议：微调阈值，然后优化其他策略")
        else:
            print("❌ 待改进：量化因子参与度不足")
            print("   建议：重新评估条件设计")
    
    print("\n=== 建议下一步 ===")
    print("1. 如果量化因子满足率>70%：继续优化pullback策略")
    print("2. 如果量化因子满足率50-70%：微调趋势动量策略阈值")
    print("3. 如果量化因子满足率<50%：重新设计条件逻辑")
    print("4. 完成后汇总Git提交")

if __name__ == "__main__":
    main()