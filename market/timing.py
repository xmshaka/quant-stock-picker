"""
大盘择时模型 — 6指标综合评分 → 仓位映射

指标（各0-25分，满分100）：
1. 趋势强度: 上证指数 20日线位置 + 涨跌幅
2. 资金情绪: 北向资金 5日净流向
3. 杠杆热度: 融资余额 5日变化率
4. 市场活跃度: 上证成交额 5日均量比 + 涨停家数比

仓位映射：
  ≥80: 满仓 90%    60-80: 高仓 70%    40-60: 中等 50%
  20-40: 低仓 30%   <20: 防御 10%
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# ──────────────────────────────────────────
# 仓位映射
# ──────────────────────────────────────────
@dataclass
class PositionBracket:
    """仓位档位"""
    label: str
    min_score: float
    max_score: float
    position_pct: float      # 总仓位比例
    per_entry_mult: float     # 单次建仓倍数


POSITION_BRACKETS = [
    PositionBracket("防御",   0,  20, 0.10, 0.11),
    PositionBracket("低仓",  20,  40, 0.30, 0.33),
    PositionBracket("中等",  40,  60, 0.50, 0.56),
    PositionBracket("高仓",  60,  80, 0.70, 0.78),
    PositionBracket("满仓",  80, 100, 0.90, 1.00),
]


# ──────────────────────────────────────────
# 市场择时模型
# ──────────────────────────────────────────
class MarketTimingModel:
    """大盘择时模型

    用法:
        model = MarketTimingModel()
        model.fetch_all(start_date, end_date)  # 拉取市场数据
        score = model.score_on(d)              # 获取某日评分
        mult = model.position_multiplier_on(d)  # 获取仓位倍数
    """

    # ── 指标权重 ──
    WEIGHT_TREND = 0.25      # 趋势强度
    WEIGHT_CAPITAL = 0.25    # 北向资金
    WEIGHT_LEVERAGE = 0.25  # 融资热度
    WEIGHT_ACTIVITY = 0.25  # 市场活跃度

    # ── 参数 ──
    MA_PERIOD = 20           # 均线周期
    CAPITAL_LOOKBACK = 5     # 北向资金回看日数
    LEVERAGE_LOOKBACK = 5    # 融资余额回看日数
    VOLUME_LOOKBACK = 5      # 量比回看日数
    LIMIT_UP_NORMAL = 50     # 涨停家数基准

    def __init__(self, tushare_token: Optional[str] = None):
        self._token = tushare_token or os.getenv("TUSHARE_TOKEN", "")
        self._pro = None
        self._scores: Dict[date, float] = {}       # date → score (0-100)
        self._details: Dict[date, dict] = {}       # date → {trend, capital, leverage, activity}
        self._brackets: Dict[date, PositionBracket] = {}
        self._loaded = False

    @property
    def pro(self):
        if self._pro is None:
            import tushare as ts
            self._pro = ts.pro_api(self._token)
        return self._pro

    # ── 数据拉取 ──────────────────────────

    def fetch_all(self, start_date: str, end_date: str):
        """拉取所有市场数据并计算评分

        Args:
            start_date: YYYYMMDD
            end_date: YYYYMMDD
        """
        logger.info(f"[MarketTiming] 拉取市场数据 {start_date} → {end_date}")

        # 1. 上证指数日线
        sh_index = self._fetch_sh_index(start_date, end_date)
        if sh_index.empty:
            logger.warning("[MarketTiming] 上证指数数据为空，使用默认中性评分")
            self._loaded = True
            return

        # 2. 北向资金
        north_flow = self._fetch_north_flow(start_date, end_date)

        # 3. 融资余额
        margin_data = self._fetch_margin(start_date, end_date)

        # 4. 涨停家数
        limit_up_counts = self._fetch_limit_up_counts(start_date, end_date)

        # 逐日计算评分
        for _, row in sh_index.iterrows():
            d = row['trade_date']
            if isinstance(d, str):
                d = datetime.strptime(d, '%Y%m%d').date()

            trend_score = self._calc_trend_score(sh_index, d)
            capital_score = self._calc_capital_score(north_flow, d)
            leverage_score = self._calc_leverage_score(margin_data, d)
            activity_score = self._calc_activity_score(sh_index, limit_up_counts, d)

            total = trend_score + capital_score + leverage_score + activity_score
            total = max(0, min(100, total))

            self._scores[d] = total
            self._details[d] = {
                'trend': trend_score, 'capital': capital_score,
                'leverage': leverage_score, 'activity': activity_score,
            }

        # 计算仓位档位
        self._compute_brackets()
        self._loaded = True
        logger.info(
            f"[MarketTiming] 评分计算完成, {len(self._scores)} 个交易日, "
            f"均值={np.mean(list(self._scores.values())):.1f}"
        )

    def _fetch_sh_index(self, start: str, end: str) -> pd.DataFrame:
        """拉取上证指数日线"""
        try:
            df = self.pro.index_daily(ts_code='000001.SH', start_date=start, end_date=end)
            if df is not None and not df.empty:
                df = df.sort_values('trade_date').reset_index(drop=True)
                df['close'] = df['close'].astype(float)
                df['pct_chg'] = df['pct_chg'].astype(float)
                df['amount'] = df['amount'].astype(float)
                return df
        except Exception as e:
            logger.error(f"[MarketTiming] 上证指数拉取失败: {e}")
        return pd.DataFrame()

    def _fetch_north_flow(self, start: str, end: str) -> pd.DataFrame:
        """拉取北向资金流向"""
        try:
            df = self.pro.moneyflow_hsgt(start_date=start, end_date=end)
            if df is not None and not df.empty:
                df = df.sort_values('trade_date').reset_index(drop=True)
                df['north_money'] = df['north_money'].astype(float)
                return df
        except Exception as e:
            logger.warning(f"[MarketTiming] 北向资金拉取失败: {e}")
        return pd.DataFrame()

    def _fetch_margin(self, start: str, end: str) -> pd.DataFrame:
        """拉取融资余额 (上交所)"""
        try:
            df = self.pro.margin(start_date=start, end_date=end)
            if df is not None and not df.empty:
                sse = df[df['exchange_id'] == 'SSE'].copy()
                sse = sse.sort_values('trade_date').reset_index(drop=True)
                sse['rzye'] = sse['rzye'].astype(float)  # 融资余额
                sse['rzrqye'] = sse['rzrqye'].astype(float)  # 两融余额
                return sse
        except Exception as e:
            logger.warning(f"[MarketTiming] 融资余额拉取失败: {e}")
        return pd.DataFrame()

    def _fetch_limit_up_counts(self, start: str, end: str) -> pd.DataFrame:
        """拉取每日涨停家数"""
        records = []
        try:
            # 逐个交易日拉取
            trade_cal = self.pro.trade_cal(
                exchange='SSE', start_date=start, end_date=end, is_open='1'
            )
            if trade_cal is None or trade_cal.empty:
                return pd.DataFrame()

            for _, row in trade_cal.iterrows():
                cal_date = row['cal_date']
                try:
                    df = self.pro.limit_list_d(trade_date=cal_date, limit_type='U')
                    if df is not None:
                        records.append({
                            'trade_date': cal_date,
                            'limit_up_count': len(df),
                        })
                except Exception:
                    records.append({'trade_date': cal_date, 'limit_up_count': 0})

        except Exception as e:
            logger.warning(f"[MarketTiming] 涨停家数拉取失败: {e}")

        return pd.DataFrame(records)

    # ── 指标计算 ──────────────────────────

    def _calc_trend_score(self, sh_index: pd.DataFrame, d: date) -> float:
        """趋势强度评分 0-25

        逻辑:
        - 价格 > MA20 且 MA20 向上 → 强势
        - 价格 > MA20 但 MA20 走平 → 中性偏强
        - 价格 < MA20 → 弱势
        """
        mask = pd.to_datetime(sh_index['trade_date']).dt.date <= d
        window = sh_index[mask].tail(self.MA_PERIOD + 5)
        if len(window) < self.MA_PERIOD:
            return 12.5  # 数据不足，中性

        closes = window['close'].values
        ma20 = np.mean(closes[-self.MA_PERIOD:])
        prev_ma20 = np.mean(closes[-(self.MA_PERIOD+1):-1])

        current = closes[-1]
        pct_above = (current - ma20) / ma20 * 100  # 偏离百分比
        ma_slope = (ma20 - prev_ma20) / prev_ma20 * 100  # MA斜率

        # 趋势方向分 (0-15)
        if pct_above > 3:
            trend_dir = 15
        elif pct_above > 1:
            trend_dir = 12
        elif pct_above > -1:
            trend_dir = 7.5
        elif pct_above > -3:
            trend_dir = 3
        else:
            trend_dir = 0

        # MA方向分 (0-10)
        if ma_slope > 0.3:
            ma_score = 10
        elif ma_slope > 0:
            ma_score = 6
        elif ma_slope > -0.3:
            ma_score = 3
        else:
            ma_score = 0

        return trend_dir + ma_score

    def _calc_capital_score(self, north_flow: pd.DataFrame, d: date) -> float:
        """北向资金情绪评分 0-25

        逻辑:
        - 近5日累计净流入 → 强势
        - 近5日累计净流出 → 弱势
        """
        if north_flow.empty:
            return 12.5

        mask = pd.to_datetime(north_flow['trade_date']).dt.date <= d
        window = north_flow[mask].tail(self.CAPITAL_LOOKBACK)
        if len(window) < 3:
            return 12.5

        total_flow = window['north_money'].sum()
        # 日均净流入（亿）
        daily_avg = total_flow / len(window) / 1e8

        # 分段评分
        if daily_avg > 50:
            return 25
        elif daily_avg > 20:
            return 20
        elif daily_avg > 5:
            return 15
        elif daily_avg > -5:
            return 12.5
        elif daily_avg > -20:
            return 8
        elif daily_avg > -50:
            return 4
        else:
            return 0

    def _calc_leverage_score(self, margin_data: pd.DataFrame, d: date) -> float:
        """融资热度评分 0-25

        逻辑:
        - 融资余额温和增长 → 健康
        - 融资余额暴涨 → 过热，减分
        - 融资余额持续下降 → 恐慌，减分
        """
        if margin_data.empty:
            return 12.5

        mask = pd.to_datetime(margin_data['trade_date']).dt.date <= d
        window = margin_data[mask].tail(self.LEVERAGE_LOOKBACK + 1)
        if len(window) < 5:
            return 12.5

        rzye = window['rzye'].values
        # 5日变化率
        change_5d = (rzye[-1] - rzye[0]) / rzye[0] * 100

        # 最优区间: 0~2% 温和增长
        if 0 <= change_5d <= 2:
            return 25
        elif 2 < change_5d <= 4:
            return 20  # 偏热
        elif change_5d > 4:
            return 10  # 过热
        elif -2 <= change_5d < 0:
            return 18  # 小幅下降
        elif -5 <= change_5d < -2:
            return 10  # 明显下降
        else:
            return 5   # 恐慌去杠杆

    def _calc_activity_score(
        self, sh_index: pd.DataFrame, limit_up_counts: pd.DataFrame, d: date
    ) -> float:
        """市场活跃度评分 0-25

        逻辑:
        - 成交额温和放大 → 活跃健康
        - 涨停家数适中 → 情绪正常
        - 量能萎缩 → 清淡
        - 涨停过多 → 过热
        """
        # 成交额 5日均量比 (0-15)
        mask = pd.to_datetime(sh_index['trade_date']).dt.date <= d
        window = sh_index[mask].tail(self.VOLUME_LOOKBACK + 20)
        if len(window) < self.VOLUME_LOOKBACK + 5:
            return 12.5

        amounts = window['amount'].values
        recent_avg = np.mean(amounts[-self.VOLUME_LOOKBACK:])
        baseline_avg = np.mean(amounts[-(self.VOLUME_LOOKBACK+20):-self.VOLUME_LOOKBACK])

        if baseline_avg > 0:
            vol_ratio = recent_avg / baseline_avg
        else:
            vol_ratio = 1.0

        if 1.2 <= vol_ratio <= 1.8:
            vol_score = 15  # 温和放量
        elif 1.0 <= vol_ratio < 1.2:
            vol_score = 12
        elif 0.8 <= vol_ratio < 1.0:
            vol_score = 8
        elif vol_ratio > 1.8:
            vol_score = 6  # 过度放量，可能见顶
        else:
            vol_score = 3  # 缩量

        # 涨停家数 (0-10)
        limit_score = 5
        if not limit_up_counts.empty:
            limit_mask = pd.to_datetime(limit_up_counts['trade_date']).dt.date == d
            limit_row = limit_up_counts[limit_mask]
            if len(limit_row) > 0:
                count = limit_row.iloc[0]['limit_up_count']
                ratio = count / self.LIMIT_UP_NORMAL
                if 0.8 <= ratio <= 1.5:
                    limit_score = 10
                elif 0.5 <= ratio < 0.8:
                    limit_score = 7
                elif 1.5 < ratio <= 2.5:
                    limit_score = 5  # 偏热
                elif ratio > 2.5:
                    limit_score = 2  # 过热
                else:
                    limit_score = 3  # 冷清

        return vol_score + limit_score

    # ── 仓位映射 ──────────────────────────

    def _compute_brackets(self):
        """计算每日仓位档位"""
        self._brackets.clear()
        for d, score in self._scores.items():
            for bracket in POSITION_BRACKETS:
                if bracket.min_score <= score < bracket.max_score:
                    self._brackets[d] = bracket
                    break
            else:
                self._brackets[d] = POSITION_BRACKETS[-1]  # 满仓兜底

    def score_on(self, d: date) -> float:
        """获取某日市场评分 (0-100)"""
        if not self._loaded:
            return 50.0
        # 找最近交易日
        if d in self._scores:
            return self._scores[d]
        # 向前找最近交易日
        sorted_dates = sorted(self._scores.keys())
        for sd in reversed(sorted_dates):
            if sd <= d:
                return self._scores[sd]
        return 50.0

    def detail_on(self, d: date) -> dict:
        """获取某日评分明细"""
        if d in self._details:
            return self._details[d]
        sorted_dates = sorted(self._details.keys())
        for sd in reversed(sorted_dates):
            if sd <= d:
                return self._details[sd]
        return {'trend': 12.5, 'capital': 12.5, 'leverage': 12.5, 'activity': 12.5}

    def position_multiplier_on(self, d: date) -> float:
        """获取某日仓位倍数 (0~1)

        用于回测中调制单次建仓金额:
            alloc = cash * pos_pct_per_entry * multiplier
        """
        if not self._loaded:
            return 0.78  # 默认高仓位

        if d in self._brackets:
            return self._brackets[d].per_entry_mult

        sorted_dates = sorted(self._brackets.keys())
        for sd in reversed(sorted_dates):
            if sd <= d:
                return self._brackets[sd].per_entry_mult
        return 0.78

    def bracket_on(self, d: date) -> PositionBracket:
        """获取某日仓位档位"""
        if not self._loaded:
            return POSITION_BRACKETS[3]  # 默认高仓

        if d in self._brackets:
            return self._brackets[d]

        sorted_dates = sorted(self._brackets.keys())
        for sd in reversed(sorted_dates):
            if sd <= d:
                return self._brackets[sd]
        return POSITION_BRACKETS[3]

    def to_dataframe(self) -> pd.DataFrame:
        """导出评分时间序列"""
        if not self._scores:
            return pd.DataFrame()
        rows = []
        for d, score in self._scores.items():
            detail = self._details.get(d, {})
            bracket = self._brackets.get(d)
            rows.append({
                'date': d,
                'score': score,
                'trend': detail.get('trend', 0),
                'capital': detail.get('capital', 0),
                'leverage': detail.get('leverage', 0),
                'activity': detail.get('activity', 0),
                'bracket': bracket.label if bracket else 'unknown',
                'position_pct': bracket.position_pct if bracket else 0,
                'multiplier': bracket.per_entry_mult if bracket else 0,
            })
        return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)


# ──────────────────────────────────────────
# 便捷函数
# ──────────────────────────────────────────
def create_market_timing(
    start_date: str, end_date: str, tushare_token: Optional[str] = None
) -> MarketTimingModel:
    """创建并初始化大盘择时模型"""
    model = MarketTimingModel(tushare_token=tushare_token)
    model.fetch_all(start_date, end_date)
    return model
