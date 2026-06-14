"""Dashboard 静态回归防线。

这些测试覆盖已出现过的系统性 UI/配置回归：
- Streamlit `use_container_width` 已弃用，后续版本会移除；统一使用 `width="stretch"`。
- `server.enableCORS=false` 与默认 `server.enableXsrfProtection=true` 冲突，会在启动日志产生 Warning，且被 Streamlit 覆盖为 true。
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = ROOT / "dashboard"
CONFIG_PATH = ROOT / ".streamlit" / "config.toml"


def _python_sources() -> list[Path]:
    return sorted(
        p for p in DASHBOARD_DIR.rglob("*.py")
        if "__pycache__" not in p.parts
    )


def test_dashboard_does_not_use_deprecated_use_container_width():
    """禁止 Streamlit 弃用参数回流，避免升级后 UI 组件报错。"""
    offenders: list[str] = []
    for path in _python_sources():
        text = path.read_text(encoding="utf-8")
        if "use_container_width" in text:
            offenders.append(str(path.relative_to(ROOT)))

    assert offenders == [], "请改用 width='stretch'，残留文件: " + ", ".join(offenders)


def test_streamlit_config_does_not_disable_cors_while_xsrf_enabled():
    """禁止配置 enableCORS=false；默认 XSRF 开启时该配置会被覆盖并产生 Warning。"""
    text = CONFIG_PATH.read_text(encoding="utf-8")
    normalized = "".join(line.split("#", 1)[0].strip().lower() for line in text.splitlines())

    assert "enablecors=false" not in normalized
