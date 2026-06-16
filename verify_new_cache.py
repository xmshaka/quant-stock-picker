#!/usr/bin/env python3
"""验证新缓存结构生成"""
import sys
sys.path.insert(0, '.')

from config.settings import settings
from data.cache_manager import CacheManager
from data.fetchers.fallback_fetcher import FallbackFetcher
import pandas as pd
from datetime import datetime

def verify_new_cache_structure():
    """验证新缓存结构"""
    print("🔍 验证新缓存结构...")
    
    # 1. 检查缓存管理器
    cm = CacheManager()
    print(f"缓存目录: {settings.parquet_dir}")
    
    # 2. 尝试获取一些数据（这会生成新缓存）
    symbol = "000001.SZ"
    start_date = "2026-06-10"
    end_date = "2026-06-15"
    
    print(f"\n尝试获取 {symbol} 的K线数据 ({start_date} 到 {end_date})...")
    
    try:
        # 使用FallbackFetcher获取数据
        fetcher = FallbackFetcher()
        bars = fetcher.get_daily_bars(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="raw"
        )
        
        if bars is not None and not bars.empty:
            print(f"✅ 成功获取 {len(bars)} 条K线数据")
            print(f"   数据源: {bars['source'].iloc[0] if 'source' in bars.columns else '未知'}")
            print(f"   复权口径: {bars['adjust'].iloc[0] if 'adjust' in bars.columns else 'raw'}")
            
            # 检查是否生成了新缓存
            new_cache_path = settings.parquet_dir / "bars" / ""  # 这里应该显示新结构
            print(f"\n检查新缓存目录结构...")
            
            # 查找新结构缓存
            import os
            for root, dirs, files in os.walk(settings.parquet_dir / "bars"):
                if len(root.split('/')) >= 6:  # 新结构至少有4层
                    print(f"  找到新结构目录: {root}")
                    if files:
                        print(f"    文件数: {len(files)}")
                        for f in files[:3]:
                            print(f"      {f}")
            
        else:
            print("⚠️  获取K线数据失败或为空")
            
    except Exception as e:
        print(f"❌ 获取数据时出错: {e}")
    
    # 3. 检查缓存健康
    print(f"\n检查缓存健康状态...")
    from data.cache_health import cache_health_summary
    health = cache_health_summary()
    
    l2 = health.get('l2_kline', {})
    print(f"  L2缓存:")
    print(f"    总文件数: {l2.get('total_files', 0)}")
    print(f"    旧文件数: {l2.get('old_files', 0)}")
    print(f"    新文件数: {l2.get('new_files', 0)}")
    print(f"    分布: {l2.get('distribution', [])}")
    
    # 4. 检查PG状态
    print(f"\n检查PG状态...")
    from data.storage.repository import stock_repo
    repo = stock_repo()
    pg_stats = repo.count_bars_by_source_adjust()
    
    if not pg_stats.empty:
        print(f"  PG分布:")
        for _, row in pg_stats.iterrows():
            source = row['source'] if pd.notna(row['source']) else '(空)'
            print(f"    {source}/{row['adjust']}: {row['count']} 行")
    else:
        print("  PG无数据或查询失败")
    
    print(f"\n✅ 验证完成")

if __name__ == "__main__":
    verify_new_cache_structure()