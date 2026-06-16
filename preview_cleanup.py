#!/usr/bin/env python3
"""预览清理旧版K线缓存（dry-run）"""
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

def preview_cleanup():
    """预览清理操作（dry-run）"""
    print("🔍 预览清理旧版K线缓存（dry-run模式）")
    print("=" * 60)
    
    old_files = scan_old_bars()
    total_files = len(old_files)
    total_size = sum(f.stat().st_size for f in old_files) if old_files else 0
    
    print(f"📊 扫描结果:")
    print(f"  旧缓存文件数: {total_files:,} 个")
    print(f"  总大小: {total_size / (1024*1024):.1f} MB")
    print(f"  目录: {settings.parquet_dir / 'bars'}")
    
    if not old_files:
        print("\n✅ 没有需要清理的旧缓存文件")
        return
    
    # 按目录统计
    print(f"\n📁 按目录分布:")
    dir_stats = {}
    for f in old_files:
        dir_name = f.parent.name
        dir_stats[dir_name] = dir_stats.get(dir_name, 0) + 1
    
    for dir_name, count in sorted(dir_stats.items()):
        dir_size = sum(f.stat().st_size for f in old_files if f.parent.name == dir_name)
        print(f"  bars/{dir_name}/: {count:,} 个文件 ({dir_size / (1024*1024):.1f} MB)")
    
    # 显示示例文件
    print(f"\n📄 示例文件（前10个）:")
    for i, f in enumerate(old_files[:10]):
        rel_path = f.relative_to(settings.parquet_dir / 'bars')
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  {i+1:2d}. {rel_path} ({size_mb:.2f} MB)")
    
    if total_files > 10:
        print(f"  ... 还有 {total_files - 10:,} 个文件")
    
    # 显示最大的文件
    print(f"\n🏆 最大的5个文件:")
    large_files = sorted(old_files, key=lambda f: f.stat().st_size, reverse=True)[:5]
    for i, f in enumerate(large_files):
        rel_path = f.relative_to(settings.parquet_dir / 'bars')
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  {i+1:2d}. {rel_path} ({size_mb:.2f} MB)")
    
    # 安全警告
    print(f"\n⚠️ 安全警告:")
    print(f"  1. 这是 DRY-RUN 模式，不会实际删除任何文件")
    print(f"  2. 旧缓存路径: bars/{{prefix}}/{{symbol}}.parquet")
    print(f"  3. 新缓存路径: bars/{{source}}/{{adjust}}/{{prefix}}/{{symbol}}.parquet")
    print(f"  4. 新链路不会读取旧缓存，但保留可用于回滚")
    print(f"  5. 清理前请确认: 近期没有回滚到旧版本的需求")
    
    print(f"\n💾 磁盘空间影响:")
    current_usage = sum(f.stat().st_size for f in (settings.parquet_dir / "bars").rglob("*") if f.is_file())
    new_usage = current_usage - total_size
    print(f"  当前占用: {current_usage / (1024*1024):.1f} MB")
    print(f"  清理后可释放: {total_size / (1024*1024):.1f} MB")
    print(f"  预计剩余: {new_usage / (1024*1024):.1f} MB")
    
    print(f"\n🔧 清理命令:")
    print(f"  # 实际清理（需要确认）")
    print(f"  venv/bin/python scripts/cache_admin.py clean-old-bars --confirm")
    
    print(f"\n📝 注意事项:")
    print(f"  1. 确认无回滚需求后再执行实际清理")
    print(f"  2. 建议先备份重要数据")
    print(f"  3. 清理后无法恢复，请谨慎操作")
    
    print(f"\n✅ 预览完成（dry-run模式）")
    print(f"   实际清理需要运行带 --confirm 参数的命令")

if __name__ == "__main__":
    preview_cleanup()