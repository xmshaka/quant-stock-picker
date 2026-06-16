#!/usr/bin/env python3
"""安全清理旧版K线缓存"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from config.settings import settings

def scan_old_bars():
    """扫描旧版缓存"""
    bars_dir = settings.parquet_dir / "bars"
    if not bars_dir.exists():
        print(f"缓存目录不存在: {bars_dir}")
        return []
    
    old_files = []
    for p in bars_dir.rglob("*.parquet"):
        rel = p.relative_to(bars_dir)
        parts = rel.parts
        
        # 新结构：bars/{source}/{adjust}/{prefix}/{symbol}.parquet (至少4层)
        # 旧结构：bars/{prefix}/{symbol}.parquet (2层)
        if len(parts) == 2:
            # 检查prefix是否是2位数字
            prefix = parts[0]
            if len(prefix) == 2 and prefix.isdigit():
                old_files.append(p)
    
    return old_files

def main():
    print("🔍 扫描旧版K线缓存...")
    
    old_files = scan_old_bars()
    print(f"找到 {len(old_files)} 个旧缓存文件")
    
    if old_files:
        print(f"\n示例文件:")
        for f in old_files[:10]:
            print(f"  {f.relative_to(settings.parquet_dir / 'bars')}")
        
        if len(old_files) > 10:
            print(f"  ... 还有 {len(old_files) - 10} 个文件")
        
        print(f"\n📊 按目录统计:")
        dir_stats = {}
        for f in old_files:
            dir_name = f.parent.name
            dir_stats[dir_name] = dir_stats.get(dir_name, 0) + 1
        
        for dir_name, count in sorted(dir_stats.items()):
            print(f"  bars/{dir_name}/: {count} 个文件")
        
        print(f"\n⚠️ 注意事项:")
        print(f"  1. 旧缓存路径: bars/{{prefix}}/{{symbol}}.parquet")
        print(f"  2. 新缓存路径: bars/{{source}}/{{adjust}}/{{prefix}}/{{symbol}}.parquet")
        print(f"  3. 新链路不会读取旧缓存")
        print(f"  4. 清理前请确认无回滚需求")
        
        print(f"\n💡 清理命令:")
        print(f"  # 预览清理（不实际删除）")
        print(f"  venv/bin/python scripts/cache_admin.py clean-old-bars --dry-run")
        print(f"  # 实际清理")
        print(f"  venv/bin/python scripts/cache_admin.py clean-old-bars --confirm")
    else:
        print("✅ 没有找到旧缓存文件")

if __name__ == "__main__":
    main()