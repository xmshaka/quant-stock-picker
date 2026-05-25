"""回测引擎 - Backtrader 封装

核心约束（数据红线）：
- 所有价格数据必须来自真实行情（AKShare/Tushare）
- 信号在 T 日收盘后产生，最早 T+1 日开盘成交（禁止未来函数）
- 手续费按国内真实费率：佣金万2.5（最低5元）、印花税千1（卖出）
"""
from typing import Dict, List, Optional, Tuple
from datetime import date, timedelta
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

# Backtrader
import backtrader as bt

from config.settings import settings


@dataclass
class BacktestParams:
    """回测参数"""
    start_date: date
    end_date: date
    initial_capital: float = 1_000_000.0
    commission_rate: float = 0.00025  # 万2.5
    min_commission: float = 5.0       # 最低佣金
    stamp_duty: float = 0.001         # 千1，仅卖出
    slippage: float = 0.001           # 滑点 0.1%
    position_pct: float = 0.95        # 仓位上限 95%
    max_stocks: int = 20              # 最大持股数
    rebalance_freq: int = 5           # 调仓频率（交易日）


class StockData(bt.feeds.PandasData):
    """股票数据 feed - 适配 Backtrader"""
    params = (
        ("datetime", None),  # 使用索引作为日期
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),
    )


class CommissionScheme(bt.CommInfoBase):
    """国内A股佣金方案"""
    params = (
        ("commission", 0.00025),   # 万2.5
        ("min_commission", 5.0),   # 最低5元
        ("stamp_duty", 0.001),     # 千1印花税（卖出）
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        # 买入：佣金
        # 卖出：佣金 + 印花税
        commission = abs(size) * price * self.p.commission
        commission = max(commission, self.p.min_commission)
        if size < 0:  # 卖出
            stamp = abs(size) * price * self.p.stamp_duty
            commission += stamp
        return commission


class MultiFactorStrategy(bt.Strategy):
    """多因子选股策略

    调仓逻辑：
    1. 每 N 个交易日调仓一次
    2. 根据选股信号买入 Top N
    3. 不在信号列表中的持仓卖出
    4. 等权分配资金（考虑手续费预留）
    """
    params = (
        ("signals", {}),           # {trade_date: [symbol1, symbol2, ...]}
        ("max_stocks", 20),
        ("position_pct", 0.95),
        ("verbose", False),
    )

    def __init__(self):
        self.order_dict = {}       # 跟踪订单
        self.trade_dates = sorted(self.p.signals.keys())
        self.current_signal_idx = 0
        self.rebalance_count = 0

    def next(self):
        """每个交易日调用"""
        current_date = self.datas[0].datetime.date(0)

        # 检查是否需要调仓
        if self.current_signal_idx >= len(self.trade_dates):
            return

        signal_date = self.trade_dates[self.current_signal_idx]
        # 信号在 signal_date 收盘后产生，T+1 日开盘执行
        # 所以当前日期 > signal_date 时才执行
        if current_date <= signal_date:
            return

        target_symbols = self.p.signals[signal_date]
        self.rebalance_count += 1

        if self.p.verbose:
            logger.info(f"[{current_date}] 第{self.rebalance_count}次调仓，目标: {target_symbols}")

        # === 卖出不在目标列表的持仓 ===
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size > 0 and data._name not in target_symbols:
                self.close(data=data)
                if self.p.verbose:
                    logger.info(f"  卖出 {data._name}")

        # === 计算可用资金 ===
        total_value = self.broker.getvalue()
        cash = self.broker.getcash()
        target_value = total_value * self.p.position_pct / len(target_symbols) if target_symbols else 0

        # === 买入目标股票 ===
        for symbol in target_symbols:
            data = self.getdatabyname(symbol)
            if data is None:
                logger.warning(f"  数据缺失: {symbol}")
                continue

            # 检查是否已持仓且市值足够
            pos = self.getposition(data)
            current_value = pos.size * data.close[0] if pos.size > 0 else 0

            # 目标市值 - 当前市值 = 需买入金额
            diff = target_value - current_value
            if diff > 1000:  # 最小买入金额1000元
                size = int(diff / data.close[0] / 100) * 100  # A股100股整数倍
                if size > 0 and cash >= size * data.close[0] * 1.005:  # 预留手续费
                    self.buy(data=data, size=size)
                    cash -= size * data.close[0]
                    if self.p.verbose:
                        logger.info(f"  买入 {symbol} {size}股 @ {data.close[0]:.2f}")

        self.current_signal_idx += 1

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                if self.p.verbose:
                    logger.info(f"  买入执行: {order.data._name} {order.size}股 @ {order.executed.price:.2f}")
            else:
                if self.p.verbose:
                    logger.info(f"  卖出执行: {order.data._name} {order.size}股 @ {order.executed.price:.2f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(f"  订单失败: {order.data._name} 状态={order.status}")

    def stop(self):
        """回测结束"""
        final_value = self.broker.getvalue()
        pnl = final_value - self.broker.startingcash
        logger.info(f"回测结束: 初始资金={self.broker.startingcash:.0f}, "
                   f"最终资金={final_value:.0f}, 盈亏={pnl:.0f}, 调仓{self.rebalance_count}次")


class BacktestEngine:
    """回测引擎封装"""

    def __init__(self, params: BacktestParams):
        self.params = params
        self.cerebro = bt.Cerebro()
        self._setup_broker()

    def _setup_broker(self):
        """配置券商"""
        self.cerebro.broker.setcash(self.params.initial_capital)

        # 自定义佣金方案
        comm = CommissionScheme(
            commission=self.params.commission_rate,
            min_commission=self.params.min_commission,
            stamp_duty=self.params.stamp_duty,
        )
        self.cerebro.broker.addcommissioninfo(comm)

        # 滑点
        self.cerebro.broker.set_slippage_perc(self.params.slippage)

    def add_data(self, symbol: str, df: pd.DataFrame):
        """添加单只股票数据

        Args:
            symbol: 股票代码
            df: DataFrame, columns=[open, high, low, close, volume], index=date
        """
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # 过滤日期范围
        mask = (df.index.date >= self.params.start_date) & (df.index.date <= self.params.end_date)
        df = df.loc[mask]

        if df.empty:
            logger.warning(f"{symbol} 在回测区间无数据")
            return

        data = StockData(dataname=df, name=symbol)
        self.cerebro.adddata(data)

    def add_signals(self, signals: Dict[date, List[str]]):
        """添加选股信号

        Args:
            signals: {trade_date: [symbol1, symbol2, ...]}
                     trade_date 是信号产生日期（T日收盘），T+1日开盘执行
        """
        self.signals = signals

    def run(self, verbose: bool = False) -> Dict:
        """运行回测

        Returns:
            {
                "total_return": float,
                "annual_return": float,
                "sharpe_ratio": float,
                "max_drawdown": float,
                "trade_count": int,
                "final_value": float,
            }
        """
        self.cerebro.addstrategy(
            MultiFactorStrategy,
            signals=self.signals,
            max_stocks=self.params.max_stocks,
            position_pct=self.params.position_pct,
            verbose=verbose,
        )

        # 添加分析器
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.03)
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        self.cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="returns", timeframe=bt.TimeFrame.Days)

        logger.info(f"开始回测: {self.params.start_date} ~ {self.params.end_date}")
        results = self.cerebro.run()
        strat = results[0]

        # 提取结果
        final_value = self.cerebro.broker.getvalue()
        total_return = (final_value - self.params.initial_capital) / self.params.initial_capital

        # 计算年化收益
        days = (self.params.end_date - self.params.start_date).days
        years = days / 365.0
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 and total_return > -1 else 0

        # 分析器结果
        sharpe = strat.analyzers.sharpe.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        trades = strat.analyzers.trades.get_analysis()
        returns = strat.analyzers.returns.get_analysis()

        # 权益曲线
        equity_curve = {k.strftime("%Y-%m-%d") if hasattr(k, "strftime") else str(k): v
                       for k, v in returns.items()}

        result = {
            "start_date": str(self.params.start_date),
            "end_date": str(self.params.end_date),
            "initial_capital": self.params.initial_capital,
            "final_value": final_value,
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe.get("sharperatio", 0) or 0,
            "max_drawdown": drawdown.get("max", {}).get("drawdown", 0) or 0,
            "max_drawdown_duration": drawdown.get("max", {}).get("len", 0) or 0,
            "trade_count": trades.get("total", {}).get("total", 0) if trades else 0,
            "win_rate": trades.get("won", {}).get("total", 0) / trades.get("total", {}).get("total", 1) if trades and trades.get("total", {}).get("total", 0) > 0 else 0,
            "equity_curve": equity_curve,
        }

        logger.info(f"回测结果: 总收益={total_return*100:.2f}%, "
                   f"年化={annual_return*100:.2f}%, "
                   f"夏普={result['sharpe_ratio']:.3f}, "
                   f"最大回撤={result['max_drawdown']*100:.2f}%")

        return result

    def plot(self, filepath: Optional[str] = None):
        """绘制回测结果"""
        if filepath:
            # Backtrader 默认不支持直接保存图片，需要matplotlib
            import matplotlib
            matplotlib.use("Agg")
            self.cerebro.plot(style="candlestick", savefig=filepath)
        else:
            self.cerebro.plot(style="candlestick")
