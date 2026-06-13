"""共享主题 — 业务组件样式 + 配色常量

框架层面（侧边栏/菜单/输入框/表格/按钮）交给 Streamlit 原生深色方案，
这里只管业务组件：信号卡、指标卡、徽章、进度条等。
"""
import streamlit as st
from typing import List, Dict, Optional, Any

# ── 配色常量（供 Python 逻辑引用）──
C = {
    "bg":        "#0e1117",
    "surface":   "#1a1d24",
    "surface2":  "#24272f",
    "border":    "#2d3139",
    "text":      "#e0e0e0",
    "text2":     "#8b8fa3",
    "accent":    "#4f8cff",
    "accent2":   "#3b6fd4",
    "green":     "#22c55e",
    "green_bg":  "rgba(34,197,94,0.12)",
    "red":       "#ef4444",
    "red_bg":    "rgba(239,68,68,0.12)",
    "yellow":    "#f59e0b",
    "yellow_bg": "rgba(245,158,11,0.12)",
    "orange":    "#f97316",
    "orange_bg": "rgba(249,115,22,0.12)",
    "blue_bg":   "rgba(79,140,255,0.10)",
}


def inject_theme():
    """注入业务组件 CSS。框架样式由 .streamlit/config.toml 的 theme.base=dark 管。"""
    st.markdown(f"""
    <style>
    /* ── 布局微调 ── */
    .block-container {{
        padding-top: 3.2rem !important;
        max-width: 1200px;
    }}

    /* ── 指标卡 ── */
    .qsp-metric {{
        background: {C['surface']};
        border: 1px solid {C['border']};
        border-radius: 10px;
        padding: 14px 16px;
        text-align: center;
        transition: border-color 0.15s;
    }}
    .qsp-metric:hover {{
        border-color: {C['accent']};
    }}
    .qsp-metric .label {{
        font-size: 0.72rem;
        color: {C['text2']};
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }}
    .qsp-metric .value {{
        font-size: 1.4rem;
        font-weight: 700;
        color: {C['text']};
        line-height: 1.2;
    }}
    .qsp-metric .value.green {{ color: {C['green']}; }}
    .qsp-metric .value.red   {{ color: {C['red']}; }}
    .qsp-metric .value.yellow {{ color: {C['yellow']}; }}
    .qsp-metric .value.small {{ font-size: 1.02rem; line-height: 1.35; white-space: nowrap; }}
    .qsp-metric .sub {{
        font-size: 0.68rem;
        color: {C['text2']};
        margin-top: 2px;
    }}

    /* ── 信号卡 ── */
    .qsp-signal {{
        background: {C['surface']};
        border: 1px solid {C['border']};
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 8px;
        transition: border-color 0.15s;
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
        border-radius: 6px;
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
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 12px;
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
        background: {C['surface2']};
        border-radius: 4px;
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
    }}
    .qsp-empty .icon {{
        font-size: 2rem;
        margin-bottom: 8px;
        opacity: 0.5;
    }}

    /* ── 移动端 ── */
    @media (max-width: 768px) {{
        .block-container {{ padding-left: 0.6rem; padding-right: 0.6rem; }}
        .qsp-metric .value {{ font-size: 1.15rem; }}
    }}
    </style>
    """, unsafe_allow_html=True)


# ── 组件函数 ──

def metric_row(metrics: List[Dict[str, Any]], cols: int = 0):
    """渲染指标卡行。"""
    n = cols or len(metrics)
    columns = st.columns(n)
    for i, m in enumerate(metrics):
        with columns[i % n]:
            color_cls = m.get("color", "")
            value_cls = m.get("value_class", "")
            if not value_cls and len(str(m.get("value", ""))) >= 7:
                value_cls = "small"
            classes = " ".join(x for x in [color_cls, value_cls] if x)
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
    return f'<span class="qsp-badge {kind}">{text}</span>'


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
    st.markdown(f"""
    <div class="qsp-signal">
        <div class="name">{name} {badges_html}</div>
        <div class="meta">{meta}</div>
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
