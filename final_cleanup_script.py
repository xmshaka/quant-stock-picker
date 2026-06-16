#!/usr/bin/env python3
"""最终清理脚本 - 安全清理旧版K线缓存"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from config.settings import settings

def get_old_cache_info():
    """获取旧缓存信息"""
    bars_dir = settings.parquet_dir / "bars"
    if not bars_dir.exists():
        return {"total_files": 0, "total_size": 0, "files": []}
    
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
    
    total_size = sum(f.stat().st_size for f in old_files) if old_files else 0
    
    return {
        "total_files": len(old_files),
        "total_size": total_size,
        "files": old_files
    }

def main():
    print("=" * 60)
    print("🧹 旧版K线缓存安全清理工具")
    print("=" * 60)
    
    info = get_old_cache_info()
    
    if info["total_files"] == 0:
        print("✅ 没有找到需要清理的旧缓存文件")
        return
    
    print(f"📊 扫描结果:")
    print(f"   旧缓存文件数: {info['total_files']:,} 个")
    print(f"   总大小: {info['total_size'] / (1024*1024):.1f} MB")
    print(f"   目录: {settings.parquet_dir / 'bars'}")
    
    # 按目录统计
    dir_stats = {}
    for f in info["files"]:
        dir_name = f.parent.name
        dir_stats[dir_name] = dir_stats.get(dir_name, 0) + 1
    
    print(f"\n📁 按目录分布:")
    for dir_name, count in sorted(dir_stats.items()):
        dir_size = sum(f.stat().st_size for f in info["files"] if f.parent.name == dir_name)
        print(f"   bars/{dir_name}/: {count:,} 个文件 ({dir_size / (1024*1024):.1f} MB)")
    
    print(f"\n⚠️  安全警告:")
    print(f"   1. 旧缓存路径: bars/{{prefix}}/{{symbol}}.parquet")
    print(f"   2. 新缓存路径: bars/{{source}}/{{adjust}}/{{prefix}}/{{symbol}}.parquet")
    print(f"   3. 新链路不会读取旧缓存，但保留可用于回滚")
    print(f"   4. 清理前请确认: 近期没有回滚到旧版本的需求")
    
    print(f"\n💾 磁盘空间影响:")
    current_usage = sum(f.stat().st_size for f in (settings.parquet_dir / "bars").rglob("*") if f.is_file())
    new_usage = current_usage - info["total_size"]
    print(f"   当前占用: {current_usage / (1024*1024):.1f} MB")
    print(f"   清理后可释放: {info['total_size'] / (1024*1024):.1f} MB")
    print(f"   预计剩余: {new_usage / (1024*1024):.1f} MB")
    
    print(f"\n🔧 清理命令（二选一）:")
    print(f"   # 方案1: 使用原工具（需要修复导入问题）")
    print(f"   venv/bin/python scripts/cache_admin.py clean-old-bars --confirm")
    print(f"\n   # 方案2: 直接执行清理（推荐）")
    print(f"   cd /root/.openclaw/workspace/quant-stock-picker")
    print(f"   find data/parquet/bars -maxdepth 2 -name \"*.parquet\" -type f | head -5")
    print(f"   # 确认文件列表后执行:")
    print(f"   find data/parquet/bars -maxdepth 2 -name \"*.parquet\" -type f -delete")
    
    print(f"\n📝 注意事项:")
    print(f"   1. 确认无回滚需求后再执行实际清理")
    print(f"   2. 建议先备份重要数据")
    print(f"   3. 清理后无法恢复，请谨慎操作")
    print(f"   4. 清理后运行新回测生成新结构缓存")
    
    print(f"\n✅ 扫描完成，等待用户确认执行清理")
    print("=" * 60)

if __name__ == "__main__":
    main()