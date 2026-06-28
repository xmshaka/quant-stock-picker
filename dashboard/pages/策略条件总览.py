"""
策略条件总览页面 - 集中展示四个策略的买卖点条件
解决用户反馈的"各策略的买卖点条件要清晰可见，目前设置分散到两个页面"问题
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategy.schemes import BUILTIN_SCHEMES

st.set_page_config(
    page_title="策略条件总览",
    page_icon="📊",
    layout="wide"
)

st.title("📊 策略条件总览")
st.caption("集中展示四个策略的买卖点条件，解决条件分散问题")

# 简介
st.markdown("""
### 页面目的
本页面集中展示quant-stock-picker四个核心策略的买卖点条件，包括：
- **资金流条件**（主力/超大单净流入、排名等）
- **相对换手条件**（相对换手率、成交额分位等）
- **技术条件**（动量、均线、RSI、布林等）

### 策略定位
""")

# 策略概览
st.subheader("🔍 策略概览")

strategy_overview = []
for scheme_id in ["trend_momentum", "pullback", "breakout", "balanced"]:
    scheme = BUILTIN_SCHEMES[scheme_id]
    rc = scheme.resonance_config
    ec = scheme.exit_config
    
    # 统计条件类型
    mf_conditions = [c for c in rc.buy_conditions if "mf_" in c]
    turnover_conditions = [c for c in rc.buy_conditions if "turnover_" in c or "amount_" in c]
    tech_conditions = [c for c in rc.buy_conditions if c not in mf_conditions + turnover_conditions]
    
    strategy_overview.append({
        "策略名称": scheme.name,
        "策略ID": scheme_id,
        "定位": scheme.description,
        "买入条件总数": len(rc.buy_conditions),
        "需满足条件数": rc.min_confirmations,
        "资金流条件": len(mf_conditions),
        "相对换手条件": len(turnover_conditions),
        "技术条件": len(tech_conditions),
        "最大持仓天数": ec.max_holding_days,
        "时间止损天数": ec.time_stop_days,
    })

overview_df = pd.DataFrame(strategy_overview)
st.dataframe(overview_df, use_container_width=True, hide_index=True)

# 详细条件展示
st.subheader("📋 详细条件展示")

# 为每个策略创建标签页
tabs = st.tabs(["强势追涨", "回调低吸", "横盘突破", "均衡择时"])

strategy_mapping = {
    "强势追涨": "trend_momentum",
    "回调低吸": "pullback", 
    "横盘突破": "breakout",
    "均衡择时": "balanced"
}

for tab_name, tab in zip(strategy_mapping.keys(), tabs):
    with tab:
        scheme_id = strategy_mapping[tab_name]
        scheme = BUILTIN_SCHEMES[scheme_id]
        rc = scheme.resonance_config
        ec = scheme.exit_config
        
        # 策略基本信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("策略ID", scheme_id)
        with col2:
            st.metric("买入条件总数", len(rc.buy_conditions))
        with col3:
            st.metric("需满足条件数", rc.min_confirmations)
        
        # 条件分类展示
        st.markdown("#### 买入条件")
        
        # 分类条件
        mf_conditions = [c for c in rc.buy_conditions if "mf_" in c]
        turnover_conditions = [c for c in rc.buy_conditions if "turnover_" in c or "amount_" in c]
        tech_conditions = [c for c in rc.buy_conditions if c not in mf_conditions + turnover_conditions]
        
        # 资金流条件
        if mf_conditions:
            with st.expander(f"💰 资金流条件 ({len(mf_conditions)}个)", expanded=True, key=f"{scheme_id}_mf_expander"):
                for cond in mf_conditions:
                    # 添加简单的条件解释
                    explanation = {
                        "large_elg_net_mf_positive": "超大单净流入 > 5万元",
                        "main_net_mf_positive": "主力净流入 > 1万元", 
                        "large_elg_net_mf_rank_high": "超大单流入排名 > 70%",
                        "main_net_mf_negative_improving": "主力净流出改善（流出 < 5万元）",
                        "large_elg_net_mf_negative_improving": "超大单净流出改善（流出 < 10万元）",
                        "large_elg_net_mf_positive_strong": "超大单净流入强劲 > 5万元",
                        "main_net_mf_positive_strong": "主力净流入强劲 > 3万元",
                        "main_net_mf_not_negative": "主力不净流出（流出 > -2万元）",
                    }.get(cond, "资金流相关条件")
                    
                    st.markdown(f"- **{cond}**: {explanation}")
        
        # 相对换手条件
        if turnover_conditions:
            with st.expander(f"📈 相对换手条件 ({len(turnover_conditions)}个)", expanded=True, key=f"{scheme_id}_turnover_expander"):
                for cond in turnover_conditions:
                    explanation = {
                        "relative_turnover_5d_high": "5日相对换手率 > 1.0x（活跃）",
                        "amount_percentile_60d_high": "60日成交额分位 > 0.6",
                        "relative_turnover_5d_low": "5日相对换手率 < 0.9x（缩量）",
                        "turnover_percentile_60d_low": "60日换手率分位 < 0.4",
                        "relative_turnover_5d_not_low": "5日相对换手率 > 0.8x",
                    }.get(cond, "相对换手相关条件")
                    
                    st.markdown(f"- **{cond}**: {explanation}")
        
        # 技术条件
        if tech_conditions:
            with st.expander(f"⚙️ 技术条件 ({len(tech_conditions)}个)", expanded=True, key=f"{scheme_id}_tech_expander"):
                for cond in tech_conditions:
                    explanation = {
                        "volume_expand": "放量 > 1.0x",
                        "ma5_above_ma20": "MA5高于MA20",
                        "momentum_5d_strong": "5日动量强劲 > 2.5%",
                        "momentum_20d_strong": "20日动量强劲 > 4%",
                        "rsi_not_extreme": "RSI不过热 < 70",
                        "rsi_oversold": "RSI超卖 < 45",
                        "boll_lower": "布林位置 < 0.35",
                        "pullback_range": "回撤幅度 5%-15%",
                        "not_break_20d_low": "不破20日低点",
                        "volume_calm": "成交量平稳 < 1.0x",
                        "near_support": "接近支撑位",
                        "break_platform": "突破平台上沿",
                        "volume_surge": "成交量激增 > 1.4x",
                        "narrow_range": "平台振幅 < 10%",
                        "boll_upper_break": "突破布林上轨 > 0.7",
                        "momentum_5d_positive": "5日动量为正",
                    }.get(cond, "技术分析条件")
                    
                    st.markdown(f"- **{cond}**: {explanation}")
        
        # 卖出条件（L3 共振层）
        if rc.sell_conditions:
            st.markdown("#### 卖出条件（L3 共振层）")
            with st.expander(f"卖出条件 ({len(rc.sell_conditions)}个)", key=f"{scheme_id}_sell_expander"):
                sell_explanations = {
                    "main_net_mf_negative": "主力净流出",
                    "large_elg_net_mf_negative": "超大单净流出",
                    "main_net_mf_negative_worsening": "主力净流出恶化",
                    "large_elg_net_mf_negative_worsening": "超大单净流出恶化",
                    "relative_turnover_5d_low": "5日相对换手率 < 0.9x（缩量走弱）",
                    "relative_turnover_5d_high": "5日相对换手率 > 1.1x（放量下跌）",
                    "ma5_below_ma20": "MA5低于MA20（短期均线下穿）",
                    "macd_bearish": "MACD转弱（死叉/柱状图转负）",
                    "volume_price_down": "放量下跌",
                    "rsi_overbought": "RSI超买 > 70",
                    "boll_upper": "布林上轨 > 0.7",
                }
                for cond in rc.sell_conditions:
                    expl = sell_explanations.get(cond, "")
                    st.markdown(f"- **{cond}** {(' — ' + expl) if expl else ''}")
        
        # ── 策略失败退出规则（ExitConfig 层） ──
        strategy_failure_rules = {
            "trend_momentum": "动量失效退出（开盘价跌破 MA20）",
            "pullback": "回调破位退出（开盘价跌破 20日最低价）",
            "breakout": "突破失败退出（开盘价跌破 建仓时平台上沿）",
            "balanced": "无专属策略失败退出（走共振层卖出条件）",
        }
        st.markdown("#### 🚨 策略失败退出规则（ExitConfig 层）")
        with st.expander("策略失败退出（开仓后 n 天内跌破关键位则强制退出）", key=f"{scheme_id}_failure_exit"):
            st.markdown(f"- **规则**: {strategy_failure_rules.get(scheme_id, '无')}")
            st.markdown(f"- **观察窗口**: 建仓后 {ec.failure_window_days} 天内")
            st.markdown(f"- **开关**: {'✅ 启用' if ec.enable_strategy_failure_exit else '❌ 禁用'}")
        
        # 退出配置
        st.markdown("#### ⏱️ 退出配置")
        exit_cols = st.columns(4)
        with exit_cols[0]:
            st.metric("最大持仓天数", ec.max_holding_days)
        with exit_cols[1]:
            st.metric("时间止损天数", ec.time_stop_days)
        with exit_cols[2]:
            st.metric("时间止损最低收益", f"{ec.time_stop_min_profit_pct*100:.1f}%")
        with exit_cols[3]:
            st.metric("策略失败观察窗口", f"{ec.failure_window_days}天")
        
        # 退出开关状态
        st.markdown("##### 退出开关")
        switch_cols = st.columns(5)
        with switch_cols[0]:
            st.checkbox("大盘防御减仓", value=ec.enable_market_defense_exit, disabled=True, key=f"{scheme_id}_market_defense")
        with switch_cols[1]:
            st.checkbox("策略失败退出", value=ec.enable_strategy_failure_exit, disabled=True, key=f"{scheme_id}_strategy_failure")
        with switch_cols[2]:
            st.checkbox("跟踪止盈/回撤", value=ec.enable_trailing_exit, disabled=True, key=f"{scheme_id}_trailing_exit")
        with switch_cols[3]:
            st.checkbox("时间止损", value=ec.enable_time_stop, disabled=True, key=f"{scheme_id}_time_stop")
        with switch_cols[4]:
            st.checkbox("最长持仓退出", value=ec.enable_max_holding_exit, disabled=True, key=f"{scheme_id}_max_holding")

# 数据来源说明
st.markdown("---")
st.caption("""
**数据来源说明**:
- 策略配置: `strategy/schemes.py`
- 条件实现: `signals/layers.py`, `signals/scanner.py`
- 资金流数据: Tushare moneyflow 接口
- 相对换手因子: 基于历史成交额/换手率计算
- 最后更新: 2026-06-25
""")

# 添加页面链接
st.sidebar.markdown("""
### 相关页面
- [策略回测](/8_策略回测) - 运行策略回测
- [量化选股](/量化选股) - 查看每日信号
- [回测记录](/9_回测记录) - 查看历史回测结果
- [参数网格](/10_参数网格) - 策略参数优化

### 使用说明
1. 本页面展示四个核心策略的完整买卖点条件
2. 条件已按资金流、相对换手、技术条件分类
3. 点击展开按钮查看详细条件说明
4. 退出配置和开关状态为默认值，可在回测页面调整
""")