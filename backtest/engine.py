"""回测引擎 - Backtrader 封装

核心约束（数据红线）：
- 所有价格数据必须来自真实行情（AKShare/Tushare）
- 信号在 T 日收盘后产生，最早 T+1 日开盘成交（禁止未来函数）
- 手续费按国内真实费率：佣金万2.5（最低5元）、印花税千1（卖出）、过户费双向万0.1、滑点
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
    transfer_fee: float = 0.00001     # 过户费，双向万0.1
    slippage: float = 0.002           # 默认蓝筹单边滑点 0.2%
    position_pct: float = 0.90        # 账户总仓位上限 90%
    max_stocks: int = 20              # 最大持股数
    rebalance_freq: int = 5           # 调仓频率（交易日）


def get_liquidity_slippage_rate(turnover_amount: float, default_rate: float = 0.002) -> Tuple[float, str]:
    """按成交额分层返回A股单边滑点。

    规则统一使用 MEMORY.md 红线：
    - 成交额 > 5亿：大盘蓝筹，0.002
    - 1亿 <= 成交额 <= 5亿：中盘，0.005
    - 成交额 < 1亿：小盘，0.010
    缺失/非法成交额时保守回退到传入默认值，避免误把未知流动性当蓝筹。
    """
    try:
        amount = float(turnover_amount or 0.0)
    except (TypeError, ValueError):
        amount = 0.0

    if amount > 500_000_000:
        return 0.002, "large_cap_gt_5e"
    if amount >= 100_000_000:
        return 0.005, "mid_cap_1e_5e"
    if amount > 0:
        return 0.010, "small_cap_lt_1e"
    return float(default_rate), "unknown_default"


def estimate_turnover_amount(amount: float = 0.0, volume: float = 0.0, close: float = 0.0) -> float:
    """估算用于滑点分层的市场成交额。

    优先使用行情 amount；当 amount 缺失或为 0 时，用 volume * close * 100 兜底。
    腾讯日 K 的 volume 单位是“手”，1手=100股。
    """
    try:
        amt = float(amount or 0.0)
    except (TypeError, ValueError):
        amt = 0.0
    if amt > 0:
        return amt
    try:
        vol = float(volume or 0.0)
        px = float(close or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if vol <= 0 or px <= 0:
        return 0.0
    return vol * px * 100.0


class StockData(bt.feeds.PandasData):
    """股票数据 feed - 适配 Backtrader"""
    lines = ("amount",)
    params = (
        ("datetime", None),  # 使用索引作为日期
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("amount", "amount"),
        ("openinterest", -1),
    )


class CommissionScheme(bt.CommInfoBase):
    """国内A股佣金方案"""
    params = (
        ("commission", 0.00025),   # 万2.5
        ("min_commission", 5.0),   # 最低5元
        ("stamp_duty", 0.001),     # 千1印花税（卖出）
        ("transfer_fee", 0.00001), # 双向过户费
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        # 买入：佣金
        # 卖出：佣金 + 印花税
        turnover = abs(size) * price
        commission = turnover * self.p.commission
        commission = max(commission, self.p.min_commission)
        commission += turnover * self.p.transfer_fee
        if size < 0:  # 卖出
            stamp = turnover * self.p.stamp_duty
            commission += stamp
        return commission


class LiquiditySlippageBackBroker(bt.brokers.BackBroker):
    """按数据源 amount 动态分层滑点的 Backtrader Broker。

    Backtrader 原生 `set_slippage_perc` 是全局滑点，会导致小票/中盘成交价仍按
    蓝筹 0.2% 撮合。这里在 broker 撮合函数层读取当前订单 data 的当日成交额，
    对买入 `_slip_up` / 卖出 `_slip_down` 使用 per-symbol/per-date 滑点。
    """
    params = (
        ("default_slippage", 0.002),
        ("liquidity_slippage", True),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slip_data = None

    def _try_exec(self, order):
        self._slip_data = order.data
        try:
            return super()._try_exec(order)
        finally:
            self._slip_data = None

    def _current_turnover_amount(self) -> float:
        data = self._slip_data
        if data is None:
            return 0.0
        try:
            amount = float(getattr(data, "amount")[0])
        except Exception:
            amount = 0.0
        try:
            volume = float(data.volume[0])
            close = float(data.close[0])
        except Exception:
            volume = 0.0
            close = 0.0
        return estimate_turnover_amount(amount=amount, volume=volume, close=close)

    def _dynamic_slip_perc(self) -> float:
        if not self.p.liquidity_slippage:
            return float(self.p.slip_perc or self.p.default_slippage or 0.0)
        rate, _ = get_liquidity_slippage_rate(self._current_turnover_amount(), self.p.default_slippage)
        return float(rate)

    def _slip_up(self, pmax, price, doslip=True, lim=False):
        if not doslip:
            return price
        slip_perc = self._dynamic_slip_perc()
        slip_fixed = self.p.slip_fixed
        if slip_perc:
            pslip = price * (1 + slip_perc)
        elif slip_fixed:
            pslip = price + slip_fixed
        else:
            return price
        if pslip <= pmax:
            return pslip
        elif self.p.slip_match or (lim and self.p.slip_limit):
            if not self.p.slip_out:
                return pmax
            return pslip
        return None

    def _slip_down(self, pmin, price, doslip=True, lim=False):
        if not doslip:
            return price
        slip_perc = self._dynamic_slip_perc()
        slip_fixed = self.p.slip_fixed
        if slip_perc:
            pslip = price * (1 - slip_perc)
        elif slip_fixed:
            pslip = price - slip_fixed
        else:
            return price
        if pslip >= pmin:
            return pslip
        elif self.p.slip_match or (lim and self.p.slip_limit):
            if not self.p.slip_out:
                return pmin
            return pslip
        return None


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
        ("position_pct", 0.90),
        ("commission_rate", 0.00025),
        ("min_commission", 5.0),
        ("stamp_duty", 0.001),
        ("transfer_fee", 0.00001),
        ("slippage", 0.002),
        ("liquidity_slippage", True),
        ("verbose", False),
    )

    def __init__(self):
        self.order_dict = {}       # 跟踪订单
        self.trade_dates = sorted(self.p.signals.keys())
        self.current_signal_idx = 0
        self.rebalance_count = 0
        self.buy_count = 0         # 买入次数（单边）
        self.sell_count = 0        # 卖出次数（单边）
        self.executed_points = {}  # {symbol: [TradePoint]}
        self.trade_details = []
        self.position_state = {}   # symbol -> {avg_cost, first_buy_date, shares}

    def _turnover_amount(self, data) -> float:
        """估算当日成交额：优先用数据源 amount，缺失时 volume * close。"""
        try:
            amount = float(getattr(data, "amount")[0])
        except Exception:
            amount = 0.0
        try:
            volume = float(data.volume[0])
            close = float(data.close[0])
        except Exception:
            volume = 0.0
            close = 0.0
        return estimate_turnover_amount(amount=amount, volume=volume, close=close)

    def _slippage_info(self, data) -> Tuple[float, str, float]:
        turnover_amount = self._turnover_amount(data)
        if self.p.liquidity_slippage:
            rate, bucket = get_liquidity_slippage_rate(turnover_amount, self.p.slippage)
        else:
            rate, bucket = float(self.p.slippage), "fixed"
        return rate, bucket, turnover_amount

    def next(self):
        """每个交易日调用"""
        current_date = self.datas[0].datetime.date(0)

        # 找到当前日期对应的最新信号（跳过已过期的）
        # 信号在 signal_date 收盘后产生，current_date > signal_date 时执行
        best_idx = None
        while self.current_signal_idx < len(self.trade_dates):
            signal_date = self.trade_dates[self.current_signal_idx]
            if current_date > signal_date:
                best_idx = self.current_signal_idx
                self.current_signal_idx += 1
            else:
                break

        if best_idx is None:
            return

        signal_date = self.trade_dates[best_idx]
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
        if not target_symbols:
            return
        target_value = total_value * self.p.position_pct / len(target_symbols)

        # === 买入目标股票 ===
        for symbol in target_symbols:
            data = self.getdatabyname(symbol)
            if data is None:
                logger.warning(f"  数据缺失: {symbol}")
                continue

            pos = self.getposition(data)
            current_value = pos.size * data.close[0] if pos.size > 0 else 0
            diff = target_value - current_value
            if diff > 1000:
                size = int(diff / data.close[0] / 100) * 100
                if size > 0 and cash >= size * data.close[0] * 1.005:
                    self.buy(data=data, size=size)
                    cash -= size * data.close[0]
                    if self.p.verbose:
                        logger.info(f"  买入 {symbol} {size}股 @ {data.close[0]:.2f}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            exec_dt = bt.num2date(order.executed.dt).date() if order.executed.dt else self.datas[0].datetime.date(0)
            symbol = order.data._name
            action = "BUY" if order.isbuy() else "SELL"
            size = abs(int(order.executed.size))
            exec_price = float(order.executed.price)
            comm = float(order.executed.comm or 0.0)
            turnover = size * exec_price
            slippage_rate, liquidity_bucket, turnover_amount = self._slippage_info(order.data)
            commission = max(turnover * self.p.commission_rate, self.p.min_commission)
            stamp_duty = turnover * self.p.stamp_duty if action == "SELL" else 0.0
            transfer_fee = turnover * self.p.transfer_fee
            slippage_cost = turnover * slippage_rate
            cash_after = float(self.broker.getcash())
            position_after = int(self.getposition(order.data).size)
            state = self.position_state.get(symbol, {"avg_cost": 0.0, "first_buy_date": exec_dt, "shares": 0})
            avg_cost = float(state.get("avg_cost") or 0.0)
            pnl = 0.0
            pnl_pct = 0.0
            holding_days = 0

            if action == "BUY":
                prev_shares = max(int(state.get("shares", 0) or 0), 0)
                prev_cost_value = prev_shares * avg_cost
                new_cost_value = turnover + commission + transfer_fee
                total_shares = max(position_after, prev_shares + size)
                avg_cost = (prev_cost_value + new_cost_value) / total_shares if total_shares > 0 else exec_price
                first_buy_date = state.get("first_buy_date") or exec_dt
                self.position_state[symbol] = {
                    "avg_cost": avg_cost,
                    "first_buy_date": first_buy_date,
                    "shares": position_after,
                }
            else:
                first_buy_date = state.get("first_buy_date") or exec_dt
                holding_days = max((exec_dt - first_buy_date).days, 0) if hasattr(first_buy_date, "day") else 0
                trade_cost = commission + stamp_duty + transfer_fee + slippage_cost
                pnl = (exec_price - avg_cost) * size - trade_cost if avg_cost else 0.0
                pnl_pct = pnl / (avg_cost * size) if avg_cost and size else 0.0
                if position_after <= 0:
                    self.position_state.pop(symbol, None)
                else:
                    self.position_state[symbol] = {
                        "avg_cost": avg_cost,
                        "first_buy_date": first_buy_date,
                        "shares": position_after,
                    }
            try:
                from signals.rules import TradePoint
                tp = TradePoint(
                    date=exec_dt,
                    action=action,
                    reason="调仓执行",
                    confidence=1.0,
                    price=exec_price,
                    rule_name="Backtrader调仓",
                    exec_price=exec_price,
                    shares=size,
                    cash_after=cash_after,
                    position_shares=position_after,
                    avg_cost=avg_cost,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    holding_days=holding_days,
                )
                self.executed_points.setdefault(symbol, []).append(tp)
            except Exception:
                pass
            self.trade_details.append({
                "symbol": symbol,
                "date": exec_dt,
                "action": action,
                "price": exec_price,
                "exec_price": exec_price,
                "shares": size,
                # P0: 细分四项成本，commission_total 保留 broker 实际扣费总额
                "amount": turnover,
                "commission": commission,
                "stamp_duty": stamp_duty,
                "transfer_fee": transfer_fee,
                "slippage": slippage_cost,
                "slippage_rate": slippage_rate,
                "liquidity_bucket": liquidity_bucket,
                "turnover_amount": turnover_amount,
                "commission_total": comm,
                "cash_after": cash_after,
                "position_shares": position_after,
                "avg_cost": avg_cost,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "holding_days": holding_days,
                "reason": "调仓执行",
            })
            if order.isbuy():
                self.buy_count += 1
                if self.p.verbose:
                    logger.info(f"  买入执行: {order.data._name} {order.size}股 @ {order.executed.price:.2f}")
            else:
                self.sell_count += 1
                if self.p.verbose:
                    logger.info(f"  卖出执行: {order.data._name} {order.size}股 @ {order.executed.price:.2f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(f"  订单失败: {order.data._name} 状态={order.status}")

    def stop(self):
        """回测结束"""
        final_value = self.broker.getvalue()
        pnl = final_value - self.broker.startingcash
        logger.info(f"回测结束: 初始资金={self.broker.startingcash:.0f}, "
                   f"最终资金={final_value:.0f}, 盈亏={pnl:.0f}, "
                   f"买入{self.buy_count}次, 卖出{self.sell_count}次")


class BacktestEngine:
    """回测引擎封装"""

    def __init__(self, params: BacktestParams):
        self.params = params
        self.cerebro = bt.Cerebro()
        self._setup_broker()

    def _setup_broker(self):
        """配置券商"""
        broker = LiquiditySlippageBackBroker(
            default_slippage=self.params.slippage,
            liquidity_slippage=True,
            slip_open=True,
            slip_match=True,
            slip_out=False,
        )
        self.cerebro.setbroker(broker)
        self.cerebro.broker.setcash(self.params.initial_capital)

        # 自定义佣金方案
        comm = CommissionScheme(
            commission=self.params.commission_rate,
            min_commission=self.params.min_commission,
            stamp_duty=self.params.stamp_duty,
            transfer_fee=self.params.transfer_fee,
        )
        self.cerebro.broker.addcommissioninfo(comm)

        # P1.3: 滑点由 LiquiditySlippageBackBroker 按单票当日成交额动态撮合

    def add_data(self, symbol: str, df: pd.DataFrame):
        """添加单只股票数据

        Args:
            symbol: 股票代码
            df: DataFrame, columns=[open, high, low, close, volume], index=date
        """
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        if "amount" not in df.columns:
            df["amount"] = 0.0
        if "volume" in df.columns and "close" in df.columns:
            df["amount"] = df.apply(
                lambda r: estimate_turnover_amount(r.get("amount", 0.0), r.get("volume", 0.0), r.get("close", 0.0)),
                axis=1,
            )

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
            commission_rate=self.params.commission_rate,
            min_commission=self.params.min_commission,
            stamp_duty=self.params.stamp_duty,
            transfer_fee=self.params.transfer_fee,
            slippage=self.params.slippage,
            liquidity_slippage=True,
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

        # trade_count: 用 Backtrader round-trip 统计（一个完整的买+卖 = 1 笔交易）
        # Backtrader 末日自动平仓的交易也会被计入，这符合「交易次数」的直觉
        rt_total = trades.get("total", {}).get("total", 0) if trades else 0
        rt_won = trades.get("won", {}).get("total", 0) if trades else 0
        win_rate = rt_won / rt_total if rt_total > 0 else 0

        result = {
            "start_date": str(self.params.start_date),
            "end_date": str(self.params.end_date),
            "initial_capital": self.params.initial_capital,
            "final_value": final_value,
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe.get("sharperatio", 0) or 0,
            "max_drawdown": (drawdown.get("max", {}).get("drawdown", 0) or 0) / 100,
            "max_drawdown_duration": drawdown.get("max", {}).get("len", 0) or 0,
            "trade_count": rt_total,
            "win_rate": win_rate,
            "buy_count": strat.buy_count,
            "sell_count": strat.sell_count,
            "equity_curve": equity_curve,
            # P0: 执行事件源
            "executed_points": getattr(strat, "executed_points", {}),
            "trade_details": getattr(strat, "trade_details", []),
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
