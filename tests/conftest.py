"""Pytest 测试夹具

⚠️ 以下测试数据为**人工构造的简化数据集**，仅用于验证数学公式的正确性，
不用于任何实际投资决策。真实运行时必须使用 AKShare/Tushare 真实数据。

数据构造方式：
- 5只虚拟股票代码（参考A股格式）
- 30个交易日（2025-04-01 ~ 2025-05-15）
- 收盘价用确定性公式生成（带趋势+震荡），确保测试结果可复现
- 财务指标在A股真实分布范围内人工设定
"""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 确保项目根目录在路径中
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============== 构造确定性价格序列 ==============

def _make_price_series(symbol: str, n_days: int, base_price: float, trend: float, freq: float) -> np.ndarray:
    """生成确定性价格序列
    
    close = base + trend * day + sin(day * freq) * amp
    确保结果可复现（无随机数）
    """
    days = np.arange(n_days)
    # 每只股票用不同的相位，避免完全一样
    phase_map = {"000001": 0, "000002": 1.2, "000333": 2.5, "600000": 0.8, "600519": 3.1}
    phase = phase_map.get(symbol, 0)
    amp = base_price * 0.03  # 振幅约3%
    close = base_price + trend * days + np.sin(days * freq + phase) * amp
    return np.round(close, 2)


SYMBOLS = ["000001", "000002", "000333", "600000", "600519"]
BASE_PRICES = {"000001": 10.0, "000002": 25.0, "000333": 45.0, "600000": 8.5, "600519": 1200.0}
TRENDS = {"000001": 0.05, "000002": 0.02, "000333": -0.03, "600000": 0.01, "600519": 0.10}
FREQS = {"000001": 0.3, "000002": 0.5, "000333": 0.2, "600000": 0.4, "600519": 0.15}


# ============== 财务指标（人工设定，在A股真实范围内）=============

FINANCIALS = {
    "000001": {"pe_ttm": 8.5, "pb": 0.85, "ps": 2.1, "roe": 0.12, "roa": 0.008,
               "gross_margin": 0.35, "net_margin": 0.28,
               "revenue_growth": 0.05, "profit_growth": 0.08},
    "000002": {"pe_ttm": 15.0, "pb": 1.2, "ps": 1.5, "roe": 0.10, "roa": 0.015,
               "gross_margin": 0.22, "net_margin": 0.12,
               "revenue_growth": -0.03, "profit_growth": -0.05},
    "000333": {"pe_ttm": 12.0, "pb": 2.5, "ps": 1.8, "roe": 0.22, "roa": 0.08,
               "gross_margin": 0.28, "net_margin": 0.14,
               "revenue_growth": 0.12, "profit_growth": 0.15},
    "600000": {"pe_ttm": 6.0, "pb": 0.55, "ps": 1.2, "roe": 0.09, "roa": 0.006,
               "gross_margin": 0.40, "net_margin": 0.35,
               "revenue_growth": 0.02, "profit_growth": 0.03},
    "600519": {"pe_ttm": 28.0, "pb": 8.5, "ps": 12.0, "roe": 0.30, "roa": 0.18,
               "gross_margin": 0.92, "net_margin": 0.52,
               "revenue_growth": 0.18, "profit_growth": 0.20},
}


def _make_trade_dates(start: str, n_days: int) -> list:
    """生成交易日列表（简化版：跳过周末，不处理节假日）"""
    dates = []
    current = datetime.strptime(start, "%Y-%m-%d").date()
    while len(dates) < n_days:
        if current.weekday() < 5:  # 周一到周五
            dates.append(current)
        current += timedelta(days=1)
    return dates


@pytest.fixture
def sample_bars_30d() -> pd.DataFrame:
    """30天历史K线数据（5只股票 × 约30天 = ~150条）"""
    trade_dates = _make_trade_dates("2025-04-01", 30)
    rows = []
    for symbol in SYMBOLS:
        close_arr = _make_price_series(symbol, len(trade_dates),
                                       BASE_PRICES[symbol], TRENDS[symbol], FREQS[symbol])
        for i, td in enumerate(trade_dates):
            c = close_arr[i]
            # 用确定性方式生成 OHLCV
            h = round(c + 0.1 + abs(np.sin(i * 0.5)) * 0.3, 2)
            l = round(c - 0.1 - abs(np.cos(i * 0.5)) * 0.3, 2)
            o = round(close_arr[max(0, i - 1)] + np.sin(i * 0.7) * 0.1, 2) if i > 0 else round(c, 2)
            rows.append({
                "symbol": symbol,
                "trade_date": td,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": int(100000 + i * 5000),
                "amount": int((100000 + i * 5000) * c),
            })
    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


@pytest.fixture
def sample_snapshot() -> pd.DataFrame:
    """单截面数据（5只股票 × 1天）— 模拟 stock_zh_a_spot_em 返回格式"""
    trade_date = datetime(2025, 5, 15).date()
    rows = []
    for symbol in SYMBOLS:
        fin = FINANCIALS[symbol]
        # 最新收盘价用构造序列的最后一天
        close_arr = _make_price_series(symbol, 30, BASE_PRICES[symbol], TRENDS[symbol], FREQS[symbol])
        c = close_arr[-1]
        rows.append({
            "symbol": symbol,
            "trade_date": trade_date,
            "open": round(c * 0.998, 2),
            "high": round(c * 1.005, 2),
            "low": round(c * 0.995, 2),
            "close": c,
            "volume": int(150000),
            "amount": int(150000 * c),
            "pct_change": round((c / close_arr[-2] - 1) * 100, 2) if len(close_arr) > 1 else 0,
            "turnover": 0.015,
            "pe_ttm": fin["pe_ttm"],
            "pb": fin["pb"],
            "ps": fin["ps"],
            "roe": fin["roe"],
            "roa": fin["roa"],
            "gross_margin": fin["gross_margin"],
            "net_margin": fin["net_margin"],
            "revenue_growth": fin["revenue_growth"],
            "profit_growth": fin["profit_growth"],
            "total_mv": int(c * 1000000000),  # 假设10亿股
            "float_mv": int(c * 700000000),   # 流通7亿股
        })
    df = pd.DataFrame(rows)
    return df


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """空 DataFrame"""
    return pd.DataFrame()
