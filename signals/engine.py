"""信号生成引擎 - 基于因子打分生成买入/卖出信号

Phase 1 (2026-05-27):
- 新增行情分类（Market Regime）：强势单边上涨/弱势单边上涨/震荡/弱势单边下跌/强势单边下跌
- RSI 三态处理：震荡反转、趋势跟随、极端衰减（>=75 追高风险）
- 动量/反转因子随行情自适应调整方向
"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import date

import pandas as pd
import numpy as np
from loguru import logger


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Signal:
    """单个交易信号"""
    symbol: str
    signal_type: SignalType
    strategy_name: str       # 触发信号的因子/策略名
    strength: float          # 信号强度 0-10
    score: float             # 综合得分
    raw_factors: Dict[str, float]  # 各因子原始值
    trade_date: date
    regime: str = "震荡整理"      # 行情分类
    risk_tags: List[str] = field(default_factory=list)  # 风险提示标签

    @property
    def emoji(self) -> str:
        if self.strength >= 4:
            return "🟢" if self.signal_type == SignalType.BUY else "🔴"
        elif self.strength >= 2:
            return "🟡" if self.signal_type == SignalType.BUY else "🟠"
        else:
            return "⚪"

    @property
    def type_label(self) -> str:
        return "买入" if self.signal_type == SignalType.BUY else "卖出"

    @property
    def risk_badge(self) -> str:
        """风险提示标签 HTML"""
        if not self.risk_tags:
            return ""
        tags = " ".join([f'<span style="background:#ff5252;color:white;padding:1px 4px;border-radius:4px;font-size:0.65rem;">{t}</span>' for t in self.risk_tags])
        return f" {tags}"


class SignalEngine:
    """信号生成引擎 - 支持行情分类自适应"""

    # 已废弃的因子（快照中可能残留，但不再参与打分）
# TODO: margin_change已废弃，考虑移除引用  # [已移除 20260627_172424]
    DEPRECATED_FACTORS = {'boll_width', 'margin_change'}

    # 行情分类阈值
    REGIME_TREND_THRESHOLD = 0.8   # 80分位视为强趋势
    RSI_EXTREME_HIGH = 75          # 强势上涨中 RSI>=75 触发追高风险
    RSI_EXTREME_LOW = 25           # 强势下跌中 RSI<=25 触发超卖保护

    def __init__(
        self,
        buy_threshold: float = 0.7,
        sell_threshold: float = 0.3,
        min_strength: float = 2.0,
    ):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_strength = min_strength

    # ──────────────────────────────────────────
    # 主入口
    # ──────────────────────────────────────────
    def generate_signals(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_names: List[str],
        factor_weights: Optional[Dict[str, float]] = None,
        top_n: int = 20,
        include_symbols: Optional[List[str]] = None,
    ) -> Tuple[List[Signal], List[Signal]]:
        # 过滤废弃因子
        factor_names = [f for f in factor_names if f not in self.DEPRECATED_FACTORS]
        factor_df = factor_df.drop(
            columns=list(self.DEPRECATED_FACTORS & set(factor_df.columns)),
            errors='ignore',
        )

        latest_date = factor_df['trade_date'].max()
        day_data = factor_df[factor_df['trade_date'] == latest_date].copy()
        if day_data.empty:
            return [], []

        # 若指定了股票子集，只保留这些
        if include_symbols is not None:
            day_data = day_data[day_data['symbol'].isin(include_symbols)]
            if day_data.empty:
                return [], []

        # 1. 行情分类（批量——只 groupby 一次）
        regimes = self._detect_regimes(
            day_data['symbol'].unique().tolist(), price_df, day_data
        )

        # 2. 计算综合得分（regime-aware），同时获取各因子标准化贡献
        scores, contributions = self._calc_scores(day_data, factor_names, factor_weights, regimes)

        # 3. 预构建 symbol->原始因子值（用于展示）
        factor_cols = [f for f in factor_names if f in day_data.columns]
        day_indexed = day_data.set_index('symbol')
        symbol_factors = {
            sym: {f: day_indexed.loc[sym, f] for f in factor_cols}
            for sym in day_indexed.index
        }

        # 4. 生成信号
        buy_signals = []
        sell_signals = []

        buy_candidates = scores.nlargest(top_n)
        for symbol, score in buy_candidates.items():
            strength = self._calc_strength(score, scores, SignalType.BUY)
            if strength >= self.min_strength:
                # 用标准化贡献决定策略名（避免原始值尺度差异导致偏差）
                contrib_dict = contributions.loc[symbol].to_dict() if symbol in contributions.index else {}
                strategy_name = self._determine_strategy(
                    contrib_dict, factor_weights, factor_names
                )
                regime = regimes.get(symbol, '震荡整理')
                risk_tags = self._build_risk_tags(symbol, regime, day_data)

                buy_signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strategy_name=strategy_name,
                    strength=strength,
                    score=score,
                    raw_factors={k: v for k, v in symbol_factors.get(symbol, {}).items() if k in factor_names},
                    trade_date=latest_date,
                    regime=regime,
                    risk_tags=risk_tags,
                ))

        sell_candidates = scores.nsmallest(top_n)
        for symbol, score in sell_candidates.items():
            strength = self._calc_strength(score, scores, SignalType.SELL)
            if strength >= self.min_strength:
                contrib_dict = contributions.loc[symbol].to_dict() if symbol in contributions.index else {}
                strategy_name = self._determine_strategy(
                    contrib_dict, factor_weights, factor_names, reverse=True
                )
                regime = regimes.get(symbol, '震荡整理')
                risk_tags = self._build_risk_tags(symbol, regime, day_data)

                sell_signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strategy_name=strategy_name,
                    strength=strength,
                    score=score,
                    raw_factors={k: v for k, v in symbol_factors.get(symbol, {}).items() if k in factor_names},
                    trade_date=latest_date,
                    regime=regime,
                    risk_tags=risk_tags,
                ))

        return buy_signals, sell_signals

    # ──────────────────────────────────────────
    # 行情分类
    # ──────────────────────────────────────────
    def _detect_regimes(self, symbols: List[str], price_df: pd.DataFrame, day_data: pd.DataFrame) -> Dict[str, str]:
        """向量化批量行情分类——只 groupby 一次"""
        if 'momentum_20d' not in day_data.columns:
            return {sym: '震荡整理' for sym in symbols}

        day_indexed = day_data.set_index('symbol')

        # 1. 趋势强度阈值（80 分位）
        m20_all = day_indexed['momentum_20d'].abs()
        threshold = m20_all.quantile(self.REGIME_TREND_THRESHOLD)
        if threshold == 0:
            return {sym: '震荡整理' for sym in symbols}

        # 2. 均线——一次 groupby 算完
        price_sorted = price_df.sort_values(['symbol', 'trade_date'])
        g = price_sorted.groupby('symbol')['close']
        ma5  = g.apply(lambda x: x.iloc[-5:].mean()  if len(x) >= 5  else np.nan, include_groups=False)
        ma10 = g.apply(lambda x: x.iloc[-10:].mean() if len(x) >= 10 else np.nan, include_groups=False)
        ma20 = g.apply(lambda x: x.iloc[-20:].mean() if len(x) >= 20 else np.nan, include_groups=False)
        latest = g.last()

        # 对齐为 DataFrame
        ma_df = pd.DataFrame({'ma5': ma5, 'ma10': ma10, 'ma20': ma20, 'latest': latest})

        # 3. 逐只分类（向量化取值，无循环过滤）
        regimes = {}
        for sym in symbols:
            if sym not in day_indexed.index:
                regimes[sym] = '震荡整理'
                continue

            sym_m20 = day_indexed.loc[sym, 'momentum_20d']
            trend_strength = min(abs(sym_m20) / threshold, 1.0)
            if trend_strength < 0.5:
                regimes[sym] = '震荡整理'
                continue

            row = ma_df.loc[sym] if sym in ma_df.index else None
            if row is None or pd.isna(row['ma20']):
                regimes[sym] = '震荡整理'
                continue

            sym_m5 = day_indexed.loc[sym, 'momentum_5d'] if 'momentum_5d' in day_indexed.columns else sym_m20
            l, m5, m10, m20v = row['latest'], row['ma5'], row['ma10'], row['ma20']

            if l > m5 > m10 > m20v and sym_m20 > 0:
                regimes[sym] = '弱势单边上涨' if sym_m5 < sym_m20 * 0.5 else '强势单边上涨'
            elif l < m5 < m10 < m20v and sym_m20 < 0:
                regimes[sym] = '弱势单边下跌' if sym_m5 > sym_m20 * 0.5 else '强势单边下跌'
            else:
                regimes[sym] = '震荡整理'

        return regimes

    def _detect_regime(self, symbol: str, price_df: pd.DataFrame, day_data: pd.DataFrame) -> str:
        """单只包装——复用批量方法"""
        return self._detect_regimes([symbol], price_df, day_data).get(symbol, '震荡整理')

    # ──────────────────────────────────────────
    # 得分计算（regime-aware）
    # ──────────────────────────────────────────
    def _calc_scores(
        self,
        day_data: pd.DataFrame,
        factor_names: List[str],
        factor_weights: Optional[Dict[str, float]],
        regimes: Dict[str, str],
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """计算综合得分，同时返回各因子的标准化贡献明细

        Returns:
            (scores: 综合得分 Series,
             contributions: 各因子标准化贡献 DataFrame [symbol × factor])
        """
        scores = pd.Series(0.0, index=day_data['symbol'].unique())
        weights = factor_weights or {}
        total_weight = 0

        regime_series = pd.Series(regimes)
        # 记录每只股票的各因子标准化贡献（Z-score × 有效权重）
        contributions = pd.DataFrame(0.0, index=day_data['symbol'].unique(), columns=factor_names)

        for f in factor_names:
            if f not in day_data.columns:
                continue

            vals = day_data.set_index('symbol')[f]

            # NaN 填充
            if f == 'pe_ttm' and vals.isna().any():
                vals = vals.fillna(vals.median())

            mean, std = vals.mean(), vals.std()
            z = (vals - mean) / std if std > 0 else pd.Series(0, index=vals.index)

            w = weights.get(f, 1.0)

            # regime 适配
            if f == 'rsi14':
                effective_w = self._apply_rsi_rules(vals, regime_series, w)
            elif f in ('momentum_5d', 'momentum_20d'):
                effective_w = self._apply_momentum_rules(regime_series, w)
            elif f == 'reversal':
                effective_w = self._apply_reversal_rules(regime_series, w)
            else:
                effective_w = pd.Series(w, index=vals.index)

            contrib = z * effective_w
            scores = scores.add(contrib, fill_value=0)
            contributions[f] = contrib.reindex(contributions.index, fill_value=0)
            total_weight += abs(w)

        if total_weight > 0:
            scores = scores / total_weight
            contributions = contributions / total_weight

        return scores.dropna(), contributions.reindex(scores.index).dropna()

    def _apply_rsi_rules(self, vals: pd.Series, regime_series: pd.Series, base_w: float) -> pd.Series:
        """RSI 三态处理：震荡反转 / 趋势跟随 / 极端衰减"""
        result = pd.Series(base_w, index=vals.index)

        for sym in vals.index:
            regime = regime_series.get(sym, '震荡整理')
            rsi = vals[sym]

            if regime == '震荡整理':
                result[sym] = -abs(base_w)  # 反转：低RSI加分

            elif regime == '强势单边上涨':
                effective = abs(base_w) * 1.5  # 跟随，boost
                if rsi >= self.RSI_EXTREME_HIGH:
                    effective *= 0.5  # 极端衰减50%
                result[sym] = effective  # 高RSI加分（但衰减）

            elif regime == '弱势单边上涨':
                result[sym] = 0  # 中性，不参考

            elif regime == '强势单边下跌':
                effective = abs(base_w) * 1.5
                if rsi <= self.RSI_EXTREME_LOW:
                    effective *= 0.5
                result[sym] = effective  # 高RSI扣分（跟随弱势）

            elif regime == '弱势单边下跌':
                result[sym] = 0  # 中性

        return result

    def _apply_momentum_rules(self, regime_series: pd.Series, base_w: float) -> pd.Series:
        """动量因子适配：震荡反转 / 趋势跟随"""
        result = pd.Series(base_w, index=regime_series.index)

        for sym in regime_series.index:
            regime = regime_series[sym]
            if regime in ('强势单边上涨', '强势单边下跌'):
                result[sym] = abs(base_w) * 1.3  # 跟随，boost
            elif regime in ('弱势单边上涨', '弱势单边下跌'):
                result[sym] = abs(base_w) * 0.3  # 弱化
            else:
                result[sym] = -abs(base_w)  # 震荡反转

        return result

    def _apply_reversal_rules(self, regime_series: pd.Series, base_w: float) -> pd.Series:
        """反转因子适配：震荡强化 / 趋势弱化"""
        result = pd.Series(base_w, index=regime_series.index)

        for sym in regime_series.index:
            regime = regime_series[sym]
            if regime == '震荡整理':
                result[sym] = abs(base_w) * 1.2  # 震荡市中反转因子更重要
            elif regime in ('弱势单边上涨', '弱势单边下跌'):
                result[sym] = abs(base_w) * 0.8  # 弱势中等待反转
            else:
                result[sym] = abs(base_w) * 0.2  # 强趋势中反转信号弱化

        return result

    # ──────────────────────────────────────────
    # 风险提示标签
    # ──────────────────────────────────────────
    def _build_risk_tags(self, symbol: str, regime: str, day_data: pd.DataFrame) -> List[str]:
        tags = []
        sym_data = day_data[day_data['symbol'] == symbol]
        if sym_data.empty:
            return tags

        if regime == '强势单边上涨' and 'rsi14' in sym_data.columns:
            rsi = sym_data['rsi14'].iloc[0]
            if rsi >= self.RSI_EXTREME_HIGH:
                tags.append('追高风险')

        if regime == '强势单边下跌' and 'rsi14' in sym_data.columns:
            rsi = sym_data['rsi14'].iloc[0]
            if rsi <= self.RSI_EXTREME_LOW:
                tags.append('超卖保护')

        return tags

    # ──────────────────────────────────────────
    # 辅助方法
    # ──────────────────────────────────────────
    def _get_next_returns(self, price_df: pd.DataFrame, current_date: date, symbols: List[str]) -> pd.Series:
        next_date = pd.to_datetime(current_date) + pd.Timedelta(days=1)
        next_data = price_df[
            (price_df['trade_date'] == next_date.date()) &
            (price_df['symbol'].isin(symbols))
        ]
        if next_data.empty:
            return pd.Series()
        return next_data.set_index('symbol')['close'].pct_change()

    def _calc_strength(self, score: float, all_scores: pd.Series, signal_type: SignalType) -> float:
        if all_scores.empty:
            return 0

        cache_key = id(all_scores)
        if not hasattr(self, '_strength_cache'):
            self._strength_cache = {}

        if cache_key not in self._strength_cache:
            std = all_scores.std()
            if std == 0 or pd.isna(std):
                self._strength_cache[cache_key] = (0, 0)
            else:
                self._strength_cache[cache_key] = (all_scores.mean(), std)

        mean, std = self._strength_cache[cache_key]
        if std == 0:
            return 5.0

        z = (score - mean) / std
        if signal_type == SignalType.BUY:
            strength = min(10, max(0, z * 2 + 5))
        else:
            strength = min(10, max(0, -z * 2 + 5))
        return round(strength, 1)

    def _determine_strategy(
        self,
        contrib_dict: Dict[str, float],
        factor_weights: Optional[Dict[str, float]] = None,
        active_factors: Optional[List[str]] = None,
        reverse: bool = False,
    ) -> str:
        """根据各因子的标准化贡献决定策略名

        Args:
            contrib_dict: 各因子的标准化贡献值 {factor_name: z_score * effective_weight}
            reverse: False=买入（找最大正贡献），True=卖出（找最大负贡献）
        """
        if not contrib_dict:
            return "综合打分"

        weights = factor_weights or {}
        active = set(active_factors) if active_factors else set(contrib_dict.keys())

        best_factor = None
        best_contrib = -float('inf')

        for f, contrib in contrib_dict.items():
            if f not in active:
                continue
            if pd.isna(contrib):
                continue
            # 买入：找最大正贡献；卖出：找最大负贡献（绝对值最大的负值）
            effective_contrib = -contrib if reverse else contrib
            if effective_contrib > best_contrib:
                best_contrib = effective_contrib
                best_factor = f

        if best_factor:
            cn_name = FACTOR_NAME_MAP.get(best_factor, best_factor)
            return f"{cn_name}触发"
        return "综合打分"


# ──────────────────────────────────────────
# 因子中文名映射
# ──────────────────────────────────────────
FACTOR_NAME_MAP = {
    'rsi14': 'RSI14',
    'macd_hist': 'MACD柱状线',
    'boll_position': '布林带位置',
    'momentum_5d': '5日动量',
    'volatility_20d': '20日波动率',
    'max_dd_60d': '60日最大回撤',
    'north_hold_change': '北向资金20日变化',
    'turnover_ratio': '换手率比率',
    'volume_ratio': '量比',
    'pe_ttm': '市盈率TTM',
    'pb': '市净率',
    'ep': '盈利收益率',
    'roe': '净资产收益率',
    'gross_margin': '毛利率',
    'revenue_growth': '营收增长率',
    'profit_growth': '利润增长率',
    'momentum_20d': '20日动量',
    'momentum_60d': '60日动量',
    'liquidity': '流动性综合',
    'reversal': '反转因子',
}


class SignalFormatter:
    """信号格式化输出"""

    @staticmethod
    def to_text(buy_signals: List[Signal], sell_signals: List[Signal]) -> str:
        lines = []
        lines.append("🎯 【今日信号】")
        lines.append("")

        lines.append(f"🟢 买入信号 ({len(buy_signals)}个)")
        for s in sorted(buy_signals, key=lambda x: x.strength, reverse=True):
            risk = f" [{','.join(s.risk_tags)}]" if s.risk_tags else ""
            lines.append(f"{s.emoji} {s.symbol} | {s.strategy_name} | 强度{s.strength}{risk} | {s.regime}")
        lines.append("")

        lines.append(f"🔴 卖出信号 ({len(sell_signals)}个)")
        for s in sorted(sell_signals, key=lambda x: x.strength, reverse=True):
            risk = f" [{','.join(s.risk_tags)}]" if s.risk_tags else ""
            lines.append(f"{s.emoji} {s.symbol} | {s.strategy_name} | 强度{s.strength}{risk} | {s.regime}")
        lines.append("")

        lines.append("━━━━━━━━━━━━━━━━━")
        return "\n".join(lines)
