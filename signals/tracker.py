"""策略表现追踪 - 统计各因子策略的历史表现"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import numpy as np
from loguru import logger

# 因子中文映射
FACTOR_CN = {
    # 技术因子
    'rsi14': 'RSI14',
    'macd_hist': 'MACD柱状线',
    'boll_position': '布林带位置',
    'momentum_5d': '5日动量',
    'volatility_20d': '20日波动率',
    'max_dd_60d': '60日最大回撤',
    # 情绪因子
    'north_hold_change': '北向资金20日变化',
    # 'margin_change': '融资融券20日变化',  # 数据缺失99.2%，已移除  # [已移除 20260627_172424]
    'turnover_ratio': '换手率比率',
    'volume_ratio': '量比',
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


@dataclass
class StrategyStats:
    """策略统计结果"""
    strategy_name: str
    avg_return_5d: float      # 5日平均收益
    avg_return_10d: float     # 10日平均收益
    win_rate: float           # 胜率
    sharpe: float             # 夏普比率
    score: float              # 综合评分
    trade_count: int          # 交易次数
    
    @property
    def emoji(self) -> str:
        if self.score >= 8:
            return "✅"
        elif self.score >= 5:
            return "🟡"
        else:
            return "❌"


class StrategyTracker:
    """策略表现追踪器"""
    
    def __init__(self, hold_days: int = 5):
        self.hold_days = hold_days
    
    def track_all_strategies(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_names: List[str],
        top_n: int = 20,
        lookback_days: int = 30,
    ) -> pd.DataFrame:
        """
        追踪所有因子策略的表现
        
        对每个因子，模拟"每天选该因子排名前N，持有hold_days天"的策略
        
        Returns:
            DataFrame [strategy_name, avg_return_5d, win_rate, sharpe, score, trade_count]
        """
        results = []
        
        for factor_name in factor_names:
            if factor_name not in factor_df.columns:
                continue
            
            stats = self._track_single_strategy(
                factor_df, price_df, factor_name, top_n, lookback_days
            )
            if stats:
                results.append({
                    'strategy_name': stats.strategy_name,
                    'avg_return_5d': stats.avg_return_5d,
                    'avg_return_10d': stats.avg_return_10d,
                    'win_rate': stats.win_rate,
                    'sharpe': stats.sharpe,
                    'score': stats.score,
                    'trade_count': stats.trade_count,
                })
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        # 按评分排序
        return df.sort_values('score', ascending=False).reset_index(drop=True)
    
    def _track_single_strategy(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_name: str,
        top_n: int,
        lookback_days: int,
    ) -> Optional[StrategyStats]:
        """追踪单个因子策略的表现"""
        
        dates = sorted(factor_df['trade_date'].unique())
        if len(dates) < lookback_days + self.hold_days:
            return None
        
        # 只取最近lookback_days个交易日
        trade_dates = dates[-(lookback_days + self.hold_days):-self.hold_days]
        
        returns = []
        win_count = 0
        
        for d in trade_dates:
            day_data = factor_df[factor_df['trade_date'] == d]
            if day_data.empty or factor_name not in day_data.columns:
                continue
            
            # 选该因子排名前N的股票
            top_stocks = day_data.nlargest(top_n, factor_name)['symbol'].tolist()
            
            # 计算持有期收益
            hold_return = self._calc_hold_return(price_df, d, top_stocks)
            if hold_return is not None:
                returns.append(hold_return)
                if hold_return > 0:
                    win_count += 1
        
        if len(returns) < 3:
            return None
        
        returns = np.array(returns)
        avg_ret = float(np.mean(returns))
        win_rate = win_count / len(returns)
        
        # 夏普（简化版，假设无风险利率为0）
        std = np.std(returns)
        sharpe = avg_ret / std * np.sqrt(252) if std > 0 else 0
        
        # 综合评分 = 收益率权重 + 胜率权重 + 夏普权重
        score = (avg_ret * 100 * 0.4) + (win_rate * 10 * 0.3) + (sharpe * 0.3)
        
        # 10日收益
        returns_10d = []
        trade_dates_10d = dates[-(lookback_days + 10):-10] if len(dates) >= lookback_days + 10 else dates[:-10]
        for d in trade_dates_10d:
            day_data = factor_df[factor_df['trade_date'] == d]
            if day_data.empty or factor_name not in day_data.columns:
                continue
            top_stocks = day_data.nlargest(top_n, factor_name)['symbol'].tolist()
            hold_return = self._calc_hold_return(price_df, d, top_stocks, hold_days=10)
            if hold_return is not None:
                returns_10d.append(hold_return)
        
        avg_ret_10d = float(np.mean(returns_10d)) if returns_10d else 0
        
        return StrategyStats(
            strategy_name=factor_name,
            avg_return_5d=avg_ret,
            avg_return_10d=avg_ret_10d,
            win_rate=win_rate,
            sharpe=sharpe,
            score=round(score, 1),
            trade_count=len(returns)
        )
    
    def _calc_hold_return(
        self,
        price_df: pd.DataFrame,
        start_date: date,
        symbols: List[str],
        hold_days: int = None,
    ) -> Optional[float]:
        """计算持有期收益"""
        hold_days = hold_days or self.hold_days
        
        start_data = price_df[
            (price_df['trade_date'] == start_date) &
            (price_df['symbol'].isin(symbols))
        ]
        
        if start_data.empty:
            return None
        
        # 找hold_days后的日期
        all_dates = sorted(price_df['trade_date'].unique())
        try:
            start_idx = all_dates.index(start_date)
            end_idx = start_idx + hold_days
            if end_idx >= len(all_dates):
                return None
            end_date = all_dates[end_idx]
        except ValueError:
            return None
        
        end_data = price_df[
            (price_df['trade_date'] == end_date) &
            (price_df['symbol'].isin(symbols))
        ]
        
        if end_data.empty:
            return None
        
        # 计算等权收益
        start_prices = start_data.set_index('symbol')['close']
        end_prices = end_data.set_index('symbol')['close']
        
        common = start_prices.index.intersection(end_prices.index)
        if len(common) == 0:
            return None
        
        returns = (end_prices[common] / start_prices[common] - 1).mean()
        return float(returns)
    
    def format_ranking(self, stats_df: pd.DataFrame) -> str:
        """格式化为排行榜文本"""
        if stats_df.empty:
            return "暂无策略数据"

        lines = []
        lines.append("🏆 【策略排行榜】")
        lines.append("")

        for _, row in stats_df.iterrows():
            emoji = "✅" if row['score'] >= 8 else "🟡" if row['score'] >= 5 else "❌"
            cn_name = FACTOR_CN.get(row['strategy_name'], row['strategy_name'])
            lines.append(
                f"{emoji} {cn_name}"
            )
            lines.append(
                f"   5日均涨 {row['avg_return_5d']:+.2%} | "
                f"胜率 {row['win_rate']:.0%} | "
                f"评分 {row['score']}"
            )
            lines.append("")

        return "\n".join(lines)
