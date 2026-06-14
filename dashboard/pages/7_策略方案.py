"""策略方案管理页面"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import streamlit as st
import copy

from strategy.registry import SchemeRegistry
from strategy.schemes import BUILTIN_SCHEMES, StrategyScheme, RuleType
from theme import inject_theme, metric_row, section_header, badge, empty_state, C

st.set_page_config(page_title="策略方案", page_icon="📋", layout="wide")
inject_theme()

registry = SchemeRegistry()

section_header("策略方案库")
st.caption("选择或自定义因子方案，用于选股和回测。")

# ========== 方案列表 ==========
schemes = registry.list_all()

# 推荐当前行情方案
from signals.engine import SignalEngine
# 简化：不检测行情，直接展示所有
for scheme in schemes:
    regime_tags = " ".join([badge(r[:4], "regime") for r in scheme.regime_fit if r != "*"])
    if "*" in scheme.regime_fit:
        regime_tags = badge("全行情", "hold")

    builtin_tag = badge("内置", "neutral") if scheme.is_builtin else badge("自定义", "buy")

    col_info, col_action = st.columns([4, 1])
    with col_info:
        st.markdown(f"**{scheme.name}** {builtin_tag} {regime_tags}")
        st.caption(scheme.description)
    with col_action:
        if st.button("选用", key=f"select_{scheme.scheme_id}", width="stretch"):
            st.session_state.selected_scheme = scheme.scheme_id
            st.toast(f"已选用: {scheme.name}")

# ========== 当前方案详情 ==========
st.divider()
selected_id = st.session_state.get("selected_scheme", "composite")
selected = registry.get(selected_id)
if not selected:
    selected = schemes[0] if schemes else None

if selected:
    section_header("当前方案", selected.name)

    # 因子权重展示
    with st.expander("📊 因子权重", expanded=True):
        from data_loader import FACTOR_NAME_MAP
        weight_data = []
        for f, w in sorted(selected.factor_weights.items(), key=lambda x: abs(x[1]), reverse=True):
            cn = FACTOR_NAME_MAP.get(f, f)
            bar_color = C['green'] if w > 0 else C['red'] if w < 0 else C['text2']
            weight_data.append({"因子": cn, "权重": w})

        if weight_data:
            import pandas as pd
            df = pd.DataFrame(weight_data)
            st.dataframe(df, width="stretch", hide_index=True,
                         column_config={"权重": st.column_config.NumberColumn(format="%.2f")})

    # 信号规则
    with st.expander("📐 信号规则", expanded=True):
        rule_names = {
            RuleType.RSI_REVERSAL: "RSI 超买超卖反转",
            RuleType.MA_CROSS: "均线金叉/死叉",
            RuleType.MACD_TREND: "MACD 趋势",
            RuleType.BOLL_BREAK: "布林带突破/回归",
            RuleType.VOLUME_BREAKOUT: "放量突破",
            RuleType.KDJ_CROSS: "KDJ 金叉/死叉",
        }
        for i, rule_cfg in enumerate(selected.signal_rules):
            name = rule_names.get(rule_cfg.rule_type, rule_cfg.rule_type.value)
            params_str = ", ".join(f"{k}={v}" for k, v in rule_cfg.params.items())
            st.markdown(f"**{i+1}. {name}** — `{params_str}`")

    # 自定义方案编辑
    st.divider()
    section_header("创建自定义方案")

    with st.form("custom_scheme"):
        c1, c2 = st.columns(2)
        with c1:
            new_name = st.text_input("方案名称", value=f"{selected.name} (副本)")
            new_id = st.text_input("方案ID", value=f"custom_{selected.scheme_id}")
        with c2:
            new_desc = st.text_area("描述", value=selected.description, height=68)

        st.caption("因子权重（基于当前方案微调）")
        edited_weights = {}
        weight_cols = st.columns(3)
        for idx, (f, w) in enumerate(sorted(selected.factor_weights.items())):
            with weight_cols[idx % 3]:
                cn = FACTOR_NAME_MAP.get(f, f)
                edited_weights[f] = st.slider(cn, -1.0, 1.0, w, 0.05, key=f"cw_{f}")

        if st.form_submit_button("💾 保存自定义方案", type="primary", width="stretch"):
            new_scheme = StrategyScheme(
                scheme_id=new_id,
                name=new_name,
                description=new_desc,
                factor_weights=edited_weights,
                signal_rules=copy.deepcopy(selected.signal_rules),
                regime_fit=selected.regime_fit,
                is_builtin=False,
            )
            registry.save(new_scheme)
            st.success(f"✅ 已保存: {new_name}")
            st.rerun()
else:
    empty_state("📋", "暂无方案")
