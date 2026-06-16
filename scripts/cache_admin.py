#!/usr/bin/env python3
"""缓存管理 CLI 工具（安全、可审计、人工确认）。

** 旧 K线缓存清理 **
新版缓存路径：bars/{source}/{adjust}/{prefix}/{symbol}.parquet
旧版路径：bars/{prefix}/{symbol}.parquet

此工具用于：
- 扫描旧缓存
- 提供 dry-run 预览
- 在显式确认后删除

** 使用 **
# 扫描所有旧缓存
python scripts/cache_admin.py scan-old-bars

# 预览会删除哪些文件
python scripts/cache_admin.py clean-old-bars --dry-run

# 在确认无回滚需求后，执行清理
python scripts/cache_admin.py clean-old-bars --confirm
"""
import sys
import argparse
from pathlib import Path
from typing import List

from config.settings import settings


def _is_old_layout(p: Path, bars_root: Path) -> bool:
    """判断是否为旧版缓存结构。

    Args:
        p: 绝对路径，如 /root/.../parquet/bars/60/600519.parquet
        bars_root: 缓存根目录，如 /root/.../parquet/bars

    Returns:
        旧版缓存结构：bars/{prefix}/{symbol}.parquet，其中 prefix 为 2 位数字/字母。
    """
    rel = p.relative_to(bars_root)
    parts = rel.parts
    if len(parts) != 2:
        return False
    # 第一部分是前缀，通常是 2 位数字/字母
    prefix = parts[0]
    if not (len(prefix) == 2 and (prefix.isdigit() or prefix.isalpha())):
        return False
    # 第二部分是以 .parquet 结尾的文件
    if not parts[1].endswith(".parquet"):
        return False
    return True


def _find_old_cache_files(bars_root: Path) -> List[Path]:
    """扫描所有旧版 K线缓存文件。"""
    if not bars_root.exists():
        return []
    old = []
    for p in bars_root.rglob("*.parquet"):
        try:
            if _is_old_layout(p, bars_root):
                old.append(p)
        except Exception:
            continue
    # 按路径排序，输出可读
    old.sort()
    return old


def cmd_scan_old_bars(args):
    """扫描旧版缓存，打印统计与示例。"""
    bars_root = settings.parquet_dir / "bars"
    if not bars_root.exists():
        print(f"[!] 缓存目录不存在：{bars_root}")
        return 1

    old_files = _find_old_cache_files(bars_root)
    if not old_files:
        print(f"[✓] 未发现旧版 K线缓存（{bars_root}）")
        return 0

    print(f"[!] 发现 {len(old_files)} 个旧版 K线缓存文件。")
    print(f"    根目录：{bars_root}")
    print("    示例：")
    for p in old_files[:10]:
        rel = p.relative_to(bars_root)
        print(f"      - {rel}")
    if len(old_files) > 10:
        print(f"      和 {len(old_files) - 10} 个其他文件")
    print()
    print("** 后续操作 **")
    print("  python scripts/cache_admin.py clean-old-bars --dry-run   (预览删除)")
    print("  python scripts/cache_admin.py clean-old-bars --confirm   (确认清理)")
    return 0


def cmd_clean_old_bars(args):
    """清理旧版缓存。"""
    bars_root = settings.parquet_dir / "bars"
    if not bars_root.exists():
        print(f"[!] 缓存目录不存在：{bars_root}")
        return 1

    old_files = _find_old_cache_files(bars_root)
    if not old_files:
        print(f"[✓] 无需清理，旧版缓存为空。")
        return 0

    print(f"[!] 发现 {len(old_files)} 个待清理旧版 K线缓存文件。")
    total_mb = sum(p.stat().st_size for p in old_files if p.exists()) / (1024 * 1024)
    print(f"    总大小：{total_mb:.2f} MB")
    print("    前 20 个文件：")
    for i, p in enumerate(old_files[:20]):
        rel = p.relative_to(bars_root)
        print(f"      {i + 1:2d}. {rel}")
    if len(old_files) > 20:
        print(f"      和 {len(old_files) - 20} 个其他文件")

    if args.dry_run:
        print()
        print("[DRY‑RUN] 仅预览，未删除任何文件。")
        return 0

    if not args.confirm:
        print()
        print("[!] 未提供 --confirm，请再次确认后执行。")
        print("    确认选项：")
        print("        python scripts/cache_admin.py clean-old-bars --confirm")
        print("    否则请添加 --dry-run 仅预览。")
        return 1

    # 确认删除
    print()
    confirm = input(f"确认删除 {len(old_files)} 个旧版缓存文件？(输入 yes 继续): ")
    if confirm.lower().strip() != "yes":
        print("取消清理。")
        return 2

    deleted = []
    errors = []
    for p in old_files:
        try:
            p.unlink()
            deleted.append(p)
        except Exception as e:
            errors.append((p, e))

    if deleted:
        print(f"[✓] 已删除 {len(deleted)} 个旧缓存文件。")
    if errors:
        print(f"[!] 删除失败 {len(errors)} 个文件：")
        for p, e in errors:
            rel = p.relative_to(bars_root)
            print(f"  - {rel}: {e}")
        return 3

    print("[✓] 清理完成。")
    return 0


def main():
    parser = argparse.ArgumentParser(description="缓存管理 CLI")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # scan-old-bars
    sub = subparsers.add_parser("scan-old-bars", help="扫描旧版 K线缓存")
    sub.set_defaults(func=cmd_scan_old_bars)

    # clean-old-bars
    sub = subparsers.add_parser("clean-old-bars", help="清理旧版 K线缓存")
    sub.add_argument("--dry-run", action="store_true", help="预览，不实际删除")
    sub.add_argument("--confirm", action="store_true", help="显式确认执行清理")
    sub.set_defaults(func=cmd_clean_old_bars)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
