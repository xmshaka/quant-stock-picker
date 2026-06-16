"""缓存管理 CLI 工具测试。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import tempfile

import pytest


def _run_cli(cmd: str, cwd: Path, stdin: str = "") -> tuple[str, int]:
    # 复制脚本到临时目录，因为 cwd 是临时测试目录，没有 scripts/cache_admin.py
    import shutil
    from pathlib import Path
    script_src = Path(__file__).parent.parent / "scripts" / "cache_admin.py"
    script_dst = cwd / "cache_admin.py"
    shutil.copy(script_src, script_dst)
    
    args = [sys.executable, "cache_admin.py"] + cmd.split()
    result = subprocess.run(args, cwd=cwd, input=stdin, capture_output=True, text=True, timeout=30)
    return result.stdout + result.stderr, result.returncode


def test_scan_old_bars(tmp_path):
    # 创建旧结构
    (tmp_path / "bars" / "60").mkdir(parents=True)
    (tmp_path / "bars" / "60" / "600519.parquet").write_text("x")
    (tmp_path / "bars" / "00").mkdir()
    (tmp_path / "bars" / "00" / "000001.parquet").write_text("x")

    # 创建新结构，不应被识别为旧缓存
    (tmp_path / "bars" / "tencent" / "raw" / "60").mkdir(parents=True)
    (tmp_path / "bars" / "tencent" / "raw" / "60" / "600519.parquet").write_text("x")

    # 模拟 settings 的 parquet_dir
    import sys
    sys.path.insert(0, str(tmp_path))
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "settings.py").write_text("""
from pathlib import Path
class Settings:
    parquet_dir = Path(__file__).parent.parent / "cache"
settings = Settings()
""")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "bars").symlink_to(tmp_path / "bars")

    # 执行扫描
    out, code = _run_cli("scan-old-bars", cwd=tmp_path)
    assert "发现 2 个旧版 K线缓存文件" in out
    assert "60/600519.parquet" in out
    assert "00/000001.parquet" in out
    assert "tencent/raw/60/600519.parquet" not in out
    assert code == 0


def test_scan_old_bars_empty(tmp_path):
    # 无旧结构
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "bars").mkdir()
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "settings.py").write_text("""
from pathlib import Path
class Settings:
    parquet_dir = Path(__file__).parent.parent / "cache"
settings = Settings()
""")

    out, code = _run_cli("scan-old-bars", cwd=tmp_path)
    assert "未发现旧版 K线缓存" in out
    assert code == 0


def test_clean_old_bars_dry_run(tmp_path):
    (tmp_path / "bars" / "60").mkdir(parents=True)
    (tmp_path / "bars" / "60" / "600519.parquet").write_text("x")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "bars").symlink_to(tmp_path / "bars")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "settings.py").write_text("""
from pathlib import Path
class Settings:
    parquet_dir = Path(__file__).parent.parent / "cache"
settings = Settings()
""")

    out, code = _run_cli("clean-old-bars --dry-run", cwd=tmp_path)
    assert "DRY‑RUN" in out
    assert "60/600519.parquet" in out
    assert (tmp_path / "bars" / "60" / "600519.parquet").exists()
    assert code == 0


def test_clean_old_bars_requires_confirm(tmp_path):
    (tmp_path / "bars" / "60").mkdir(parents=True)
    (tmp_path / "bars" / "60" / "600519.parquet").write_text("x")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "bars").symlink_to(tmp_path / "bars")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "settings.py").write_text("""
from pathlib import Path
class Settings:
    parquet_dir = Path(__file__).parent.parent / "cache"
settings = Settings()
""")

    out, code = _run_cli("clean-old-bars", cwd=tmp_path, stdin="no\n")
    assert "未提供 --confirm" in out or "取消清理" in out
    assert (tmp_path / "bars" / "60" / "600519.parquet").exists()
    assert code != 0


