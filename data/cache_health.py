"""缓存健康统计。

用于数据状态页展示：
- L2 K线缓存总数
- 按 source/adjust 的文件分布
- L3 PG K线缓存开关
- 旧版 bars/{prefix}/{symbol}.parquet 是否残留
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

from config.settings import settings


def _iter_parquet_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return root.rglob("*.parquet")


def l2_kline_cache_health(bars_dir: Path | None = None) -> dict:
    """返回 L2 K线缓存健康摘要。

    新版路径：bars/{source}/{adjust}/{prefix}/{symbol}.parquet
    旧版路径：bars/{prefix}/{symbol}.parquet，其中 prefix 通常为 2 位数字。
    """
    root = Path(bars_dir) if bars_dir is not None else settings.parquet_dir / "bars"
    by_source_adjust: Counter[tuple[str, str]] = Counter()
    old_files: list[Path] = []
    malformed_files: list[Path] = []

    for p in _iter_parquet_files(root):
        rel = p.relative_to(root)
        parts = rel.parts
        if len(parts) >= 4:
            source, adjust = parts[0], parts[1]
            by_source_adjust[(source, adjust)] += 1
        elif len(parts) == 2:
            # 旧结构 bars/60/600519.parquet
            old_files.append(p)
        else:
            malformed_files.append(p)

    distribution = [
        {"source": source, "adjust": adjust, "files": count}
        for (source, adjust), count in sorted(by_source_adjust.items())
    ]
    total_files = sum(item["files"] for item in distribution)

    return {
        "bars_dir": str(root),
        "l3_enabled": bool(getattr(settings, "cache_l3_kline_enabled", False)),
        "new_files": total_files,
        "old_files": len(old_files),
        "malformed_files": len(malformed_files),
        "distribution": distribution,
        "old_examples": [str(p.relative_to(root)) for p in old_files[:5]],
        "malformed_examples": [str(p.relative_to(root)) for p in malformed_files[:5]],
    }


def cache_health_summary() -> dict:
    """页面聚合入口。"""
    return {
        "l2_kline": l2_kline_cache_health(),
        "snapshot_files": sum(1 for _ in (settings.parquet_dir / "snapshots").rglob("*.parquet"))
        if (settings.parquet_dir / "snapshots").exists() else 0,
    }
