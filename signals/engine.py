"""信号生成引擎 - 基于因子打分生成买入/卖出信号"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
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


class SignalEngine:
    """信号生成引擎"""
    
    def __init__(
        self,
        buy_threshold: float = 0.7,      # 买入信号阈值（得分前30%）
        sell_threshold: float = 0.3,     # 卖出信号阈值（得分后30%）
        min_strength: float = 2.0,       # 最小信号强度
    ):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_strength = min_strength
    
    def generate_signals(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_names: List[str],
        factor_weights: Optional[Dict[str, float]] = None,
        top_n: int = 20,
    ) -> Tuple[List[Signal], List[Signal]]:
        """
        生成买入/卖出信号
        
        Returns:
            (buy_signals, sell_signals)
        """
        latest_date = factor_df['trade_date'].max()
        day_data = factor_df[factor_df['trade_date'] == latest_date].copy()
        
        if day_data.empty:
            return [], []
        
        # 计算综合得分
        scores = self._calc_scores(day_data, factor_names, factor_weights)
        
        # 计算次日收益用于验证（如果有）
        next_returns = self._get_next_returns(price_df, latest_date, day_data['symbol'].unique())
        
        # 生成信号
        buy_signals = []
        sell_signals = []
        
        # 买入信号：得分高的
        buy_candidates = scores.nlargest(top_n)
        for symbol, score in buy_candidates.items():
            strength = self._calc_strength(score, scores, SignalType.BUY)
            if strength >= self.min_strength:
                raw_factors = {f: day_data[day_data['symbol']==symbol][f].values[0] 
                              for f in factor_names if f in day_data.columns}
                
                # 确定触发策略
                strategy_name = self._determine_strategy(raw_factors, factor_weights)
                
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strategy_name=strategy_name,
                    strength=strength,
                    score=score,
                    raw_factors=raw_factors,
                    trade_date=latest_date
                )
                buy_signals.append(signal)
        
        # 卖出信号：得分低的
        sell_candidates = scores.nsmallest(top_n)
        for symbol, score in sell_candidates.items():
            strength = self._calc_strength(score, scores, SignalType.SELL)
            if strength >= self.min_strength:
                raw_factors = {f: day_data[day_data['symbol']==symbol][f].values[0] 
                              for f in factor_names if f in day_data.columns}
                
                strategy_name = self._determine_strategy(raw_factors, factor_weights, reverse=True)
                
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strategy_name=strategy_name,
                    strength=strength,
                    score=score,
                    raw_factors=raw_factors,
                    trade_date=latest_date
                )
                sell_signals.append(signal)
        
        return buy_signals, sell_signals
    
    def _calc_scores(
        self,
        day_data: pd.DataFrame,
        factor_names: List[str],
        factor_weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """计算综合得分"""
        scores = pd.Series(0.0, index=day_data['symbol'].unique())
        
        weights = factor_weights or {}
        total_weight = 0
        
        for f in factor_names:
            if f not in day_data.columns:
                continue
            
            w = weights.get(f, 1.0)
            # 截面z-score
            vals = day_data.set_index('symbol')[f]
            mean, std = vals.mean(), vals.std()
            if std > 0:
                z = (vals - mean) / std
            else:
                z = pd.Series(0, index=vals.index)
            
            scores = scores.add(z * w, fill_value=0)
            total_weight += abs(w)
        
        if total_weight > 0:
            scores = scores / total_weight
        
        return scores.dropna()
    
    def _get_next_returns(
        self,
        price_df: pd.DataFrame,
        current_date: date,
        symbols: List[str],
    ) -> pd.Series:
        """获取次日收益"""
        next_date = pd.to_datetime(current_date) + pd.Timedelta(days=1)
        next_data = price_df[
            (price_df['trade_date'] == next_date.date()) &
            (price_df['symbol'].isin(symbols))
        ]
        
        if next_data.empty:
            return pd.Series()
        
        return next_data.set_index('symbol')['close'].pct_change()
    
    def _calc_strength(
        self,
        score: float,
        all_scores: pd.Series,
        signal_type: SignalType,
    ) -> float:
        """计算信号强度 0-10"""
        if all_scores.empty or all_scores.std() == 0:
            return 0
        
        z = (score - all_scores.mean()) / all_scores.std()
        
        if signal_type == SignalType.BUY:
            # 买入：z-score越高越强
            strength = min(10, max(0, z * 2 + 5))
        else:
            # 卖出：z-score越低越强
            strength = min(10, max(0, -z * 2 + 5))
        
        return round(strength, 1)
    
    def _determine_strategy(
        self,
        raw_factors: Dict[str, float],
        factor_weights: Optional[Dict[str, float]] = None,
        reverse: bool = False,
    ) -> str:
        """确定触发策略名称"""
        if not raw_factors:
            return "综合打分"

        weights = factor_weights or {}

        # 找贡献最大的因子
        best_factor = None
        best_contrib = -float('inf')

        for f, v in raw_factors.items():
            w = weights.get(f, 1.0)
            contrib = v * w * (-1 if reverse else 1)
            if contrib > best_contrib:
                best_contrib = contrib
                best_factor = f

        if best_factor:
            cn_name = FACTOR_NAME_MAP.get(best_factor, best_factor)
            return f"{cn_name}触发"
        return "综合打分"


# 因子中文名映射
FACTOR_NAME_MAP = {
    # 技术因子
    'rsi14': 'RSI14',
    'macd_hist': 'MACD柱状线',
    'boll_position': '布林带位置',
    'boll_width': '布林带宽度',
    'volatility_20d': '20日波动率',
    'max_dd_60d': '60日最大回撤',
    # 情绪因子
    'north_hold_change': '北向资金20日变化',
    'margin_change': '融资融券20日变化',
    'turnover_ratio': '换手率比率',
    # 估值因子
    'pe_ttm': '市盈率TTM',
    'pb': '市净率',
    'ep': '盈利收益率',
    # 质量因子
    'roe': '净资产收益率',
    'gross_margin': '毛利率',
    'revenue_growth': '营收增长率',
    'profit_growth': '利润增长率',
    # 动量/流动性
    'momentum_20d': '20日动量',
    'momentum_60d': '60日动量',
    'liquidity': '流动性综合',
    'reversal': '反转因子',
}


class SignalFormatter:
    """信号格式化输出"""
    
    @staticmethod
    def to_text(buy_signals: List[Signal], sell_signals: List[Signal]) -> str:
        """格式化为文本报告"""
        lines = []
        lines.append("🎯 【今日信号】")
        lines.append("")
        
        # 买入信号
        lines.append(f"🟢 买入信号 ({len(buy_signals)}个)")
        for s in sorted(buy_signals, key=lambda x: x.strength, reverse=True):
            lines.append(f"{s.emoji} {s.symbol} | {s.strategy_name} | 强度{s.strength}")
        lines.append("")
        
        # 卖出信号
        lines.append(f"🔴 卖出信号 ({len(sell_signals)}个)")
        for s in sorted(sell_signals, key=lambda x: x.strength, reverse=True):
            lines.append(f"{s.emoji} {s.symbol} | {s.strategy_name} | 强度{s.strength}")
        lines.append("")
        
        lines.append("━━━━━━━━━━━━━━━━━")
        
        return "\n".join(lines)
