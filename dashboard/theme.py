"""共享主题 — Financial Professional 风格业务组件。

参考 awesome-streamlit-themes/financial，并调整为米白暖色系：深海军蓝主色、米白背景、细边框、
保守圆角和高对比排版。框架层面由 .streamlit/config.toml 控制，
这里只管业务组件：信号卡、指标卡、徽章、进度条等。
"""
import re
from html import escape

import streamlit as st
from typing import List, Dict, Optional, Any

# ── 配色常量（供 Python 逻辑引用）──
C = {
    "bg":        "#ede7e0",
    "surface":   "#e6ded3",
    "surface2":  "#f6f3ed",
    "surface3":  "#d8ccbd",
    "border":    "#d2c7b8",
    "border2":   "#c2b39f",
    "text":      "#2f2a22",
    "text2":     "#5f5648",
    "accent":    "#1e3a8a",
    "accent2":   "#1e40af",
    "green":     "#047857",
    "green_bg":  "rgba(4,120,87,0.10)",
    "red":       "#b91c1c",
    "red_bg":    "rgba(185,28,28,0.10)",
    "yellow":    "#b45309",
    "yellow_bg": "rgba(180,83,9,0.12)",
    "orange":    "#c2410c",
    "orange_bg": "rgba(194,65,12,0.10)",
    "blue_bg":   "rgba(30,58,138,0.08)",
    "shadow":    "0 1px 2px rgba(47,42,34,0.10)",
}


def inject_theme():
    """注入业务组件 CSS。框架样式由 .streamlit/config.toml 管。"""
    st.markdown(f"""
    <style>
    /* ── 布局微调 ── */
    .block-container {{
        padding-top: 3.2rem !important;
        max-width: 1280px;
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {C['text']} !important;
        font-weight: 700 !important;
        letter-spacing: -0.015em;
    }}
    [data-testid="stCaptionContainer"], .stCaption {{
        color: {C['text2']} !important;
    }}
    .stApp {{
        background: {C['bg']};
    }}

    /* ── 指标卡 ── */
    .qsp-metric {{
        background: {C['surface']};
        border: 1px solid {C['border']};
        border-radius: 0.375rem;
        min-height: 92px;
        padding: 12px 14px;
        text-align: center;
        box-shadow: {C['shadow']};
        transition: border-color 0.15s, box-shadow 0.15s;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-sizing: border-box;
    }}
    .qsp-metric:hover {{
        border-color: {C['accent']};
        box-shadow: 0 2px 6px rgba(47,42,34,0.12);
    }}
    .qsp-metric .label {{
        font-size: 0.72rem;
        color: {C['text2']};
        letter-spacing: 0.05em;
        margin-bottom: 4px;
        min-height: 1rem;
        white-space: nowrap;
    }}
    .qsp-metric .value {{
        font-size: 1.18rem;
        font-weight: 700;
        color: {C['text']};
        line-height: 1.22;
        min-height: 1.6rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.08em;
        white-space: nowrap;
        font-variant-numeric: tabular-nums;
        font-family: Inter, "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif;
    }}
    .qsp-metric .value.green {{ color: {C['green']}; }}
    .qsp-metric .value.red   {{ color: {C['red']}; }}
    .qsp-metric .value.yellow {{ color: {C['yellow']}; }}
    .qsp-metric .value.compact,
    .qsp-metric .value.small {{ font-size: 1.02rem; line-height: 1.32; }}
    .qsp-metric .sub {{
        font-size: 0.68rem;
        color: {C['text2']};
        margin-top: 2px;
    }}

    /* ── 信号卡 ── */
    .qsp-signal {{
        background: {C['surface']};
        border: 1px solid {C['border']};
        border-radius: 0.375rem;
        padding: 12px 14px;
        margin-bottom: 8px;
        box-shadow: {C['shadow']};
        transition: border-color 0.15s, box-shadow 0.15s;
    }}
    .qsp-signal:hover {{
        border-color: {C['accent']};
    }}
    .qsp-signal .name {{
        font-size: 0.88rem;
        font-weight: 600;
        color: {C['text']};
    }}
    .qsp-signal .meta {{
        font-size: 0.72rem;
        color: {C['text2']};
        margin-top: 2px;
    }}

    /* ── 徽章 ── */
    .qsp-badge {{
        display: inline-block;
        font-size: 0.65rem;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 0.375rem;
        letter-spacing: 0.02em;
        vertical-align: middle;
    }}
    .qsp-badge.buy     {{ background: {C['green_bg']}; color: {C['green']}; }}
    .qsp-badge.sell    {{ background: {C['red_bg']}; color: {C['red']}; }}
    .qsp-badge.hold    {{ background: {C['blue_bg']}; color: {C['accent']}; }}
    .qsp-badge.hot     {{ background: {C['orange_bg']}; color: {C['orange']}; }}
    .qsp-badge.regime  {{ background: {C['blue_bg']}; color: {C['accent']}; }}
    .qsp-badge.risk    {{ background: {C['red_bg']}; color: {C['red']}; }}
    .qsp-badge.neutral {{ background: {C['surface2']}; color: {C['text2']}; }}

    /* ── 顶部信息条 ── */
    .qsp-topbar {{
        background: {C['surface']};
        border: 1px solid {C['border']};
        border-radius: 0.375rem;
        padding: 10px 16px;
        margin-bottom: 12px;
        box-shadow: {C['shadow']};
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.75rem;
        color: {C['text2']};
    }}
    .qsp-topbar .date {{
        font-weight: 600;
        color: {C['text']};
    }}

    /* ── 进度条 ── */
    .qsp-progress {{
        background: {C['surface3']};
        border-radius: 0.375rem;
        height: 6px;
        overflow: hidden;
        margin-top: 4px;
    }}
    .qsp-progress .fill {{
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s;
    }}

    /* ── 空状态 ── */
    .qsp-empty {{
        text-align: center;
        padding: 32px 16px;
        color: {C['text2']};
        font-size: 0.85rem;
        background: {C['surface2']};
        border: 1px dashed {C['border']};
        border-radius: 0.375rem;
    }}
    .qsp-empty .icon {{
        font-size: 2rem;
        margin-bottom: 8px;
        opacity: 0.5;
    }}

    /* ── 移动端 ── */
    @media (max-width: 768px) {{
        .block-container {{ padding-left: 0.6rem; padding-right: 0.6rem; }}
        .qsp-metric .value {{ font-size: 1.08rem; }}
        .qsp-metric .value.compact,
        .qsp-metric .value.small {{ font-size: 0.96rem; }}
    }}
    </style>
    """, unsafe_allow_html=True)


# ── 组件函数 ──

_CJK_OR_UNIT_RE = re.compile(r"[\u4e00-\u9fff]|[¥￥%％]|\d\s*[A-Za-z]+")


def _metric_value_needs_compact(value: Any) -> bool:
    """判断指标值是否需要紧凑字号。

    根因：旧逻辑按单个字符串长度自动给 ``small``，导致同一行里
    ``5819``、``0.5000%``、``6笔`` 字号不一致；中文单位又会触发字体 fallback，
    视觉上更明显。现在只判断“整行是否应紧凑”，再统一应用到同一行。
    """
    text = str(value or "").strip()
    if not text:
        return False
    return len(text) >= 6 or bool(_CJK_OR_UNIT_RE.search(text)) or "," in text


def _metric_row_compact(metrics: List[Dict[str, Any]]) -> bool:
    """同一 metric_row 内只允许一种值字号，避免数字/单位混排跳动。"""
    return any(
        str(m.get("value_class", "")).strip() in {"small", "compact"}
        or _metric_value_needs_compact(m.get("value", ""))
        for m in metrics
    )


def _metric_value_classes(metric: Dict[str, Any], row_compact: bool) -> str:
    """生成指标值 class，保留颜色 class，字号由行级 compact 统一控制。"""
    classes = []
    color_cls = str(metric.get("color", "") or "").strip()
    if color_cls:
        classes.append(color_cls)
    legacy_value_cls = str(metric.get("value_class", "") or "").strip()
    if row_compact or legacy_value_cls in {"small", "compact"}:
        classes.append("compact")
    elif legacy_value_cls:
        classes.append(legacy_value_cls)
    return " ".join(classes)

def metric_row(metrics: List[Dict[str, Any]], cols: int = 0):
    """渲染指标卡行。"""
    n = cols or len(metrics)
    columns = st.columns(n)
    row_compact = _metric_row_compact(metrics)
    for i, m in enumerate(metrics):
        with columns[i % n]:
            classes = _metric_value_classes(m, row_compact)
            sub = m.get("sub", "")
            sub_html = f'<div class="sub">{sub}</div>' if sub else ""
            st.markdown(f"""
            <div class="qsp-metric">
                <div class="label">{m['label']}</div>
                <div class="value {classes}">{m['value']}</div>
                {sub_html}
            </div>
            """, unsafe_allow_html=True)


def section_header(title: str, subtitle: str = ""):
    """章节标题。"""
    if subtitle:
        st.markdown(f"### {title}")
        st.caption(subtitle)
    else:
        st.markdown(f"### {title}")


def badge_html(text: str, kind: str = "neutral") -> str:
    """返回徽章 HTML（仅用于 unsafe_allow_html=True 场景）"""
    return f'<span class="qsp-badge {escape(str(kind), quote=True)}">{escape(str(text))}</span>'


def badge(text: str, kind: str = "neutral") -> str:
    """返回徽章纯文本（安全，可用于任何 markdown）"""
    icons = {
        "buy": "▲", "sell": "▼", "hold": "●", "hot": "🔥",
        "regime": "◆", "risk": "⚠", "neutral": "○",
    }
    icon = icons.get(kind, "○")
    return f"{icon} {text}"


def empty_state(icon: str, message: str):
    """空状态。"""
    st.markdown(f"""
    <div class="qsp-empty">
        <div class="icon">{icon}</div>
        <div>{message}</div>
    </div>
    """, unsafe_allow_html=True)


def signal_card(name: str, meta: str, badges_html: str = ""):
    """信号卡。badges_html 应使用 badge_html() 生成。"""
    safe_meta = escape(str(meta))
    st.markdown(f"""
    <div class="qsp-signal">
        <div class="name">{name} {badges_html}</div>
        <div class="meta">{safe_meta}</div>
    </div>
    """, unsafe_allow_html=True)


def topbar(date_str: str, left_html: str = "", right_html: str = ""):
    """顶部信息条。"""
    st.markdown(f"""
    <div class="qsp-topbar">
        <span>{left_html} <span class="date">{date_str}</span></span>
        <span>{right_html}</span>
    </div>
    """, unsafe_allow_html=True)


def progress_bar(value: float, color: str = C["accent"], width: str = "100%"):
    """进度条。value: 0-100。"""
    st.markdown(f"""
    <div class="qsp-progress" style="width:{width}">
        <div class="fill" style="width:{min(100, max(0, value))}%;background:{color};"></div>
    </div>
    """, unsafe_allow_html=True)
