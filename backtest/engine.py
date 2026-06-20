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
from strategy.schemes import BUILTIN_SCHEMES, ExitConfig


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
    strategy_id: str = "balanced"     # P3: 全池退出规则使用的策略模板
    exit_config: Optional[ExitConfig] = None  # P3: 可覆盖内置短线退出配置
    market_scores: Optional[Dict[date, float]] = None  # {date: market_score}
    cheat_on_open: bool = True        # P3: 信号T日收盘后，T+1开盘成交
    trailing_atr_mult: float = 2.0     # 跟踪止盈 = 持仓最高价 - N×ATR


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
        ("strategy_id", "balanced"),
        ("exit_config", None),
        ("market_scores", None),
        ("cheat_on_open", True),
        ("trailing_atr_mult", 2.0),
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

    def _holding_trading_days(self, current_date: date, first_buy_date: date) -> int:
        """按已加载行情交易日计算持仓天数，避免周末/节假日被计入时间止损。"""
        if first_buy_date is None:
            return 0
        try:
            # Backtrader 在策略运行中只能安全读取当前及历史 bar：0 为当前，-1/-2 为历史。
            dates = sorted({data.datetime.date(-ago) for data in self.datas for ago in range(len(data))})
            if current_date in dates and first_buy_date in dates:
                return max(dates.index(current_date) - dates.index(first_buy_date), 0)
        except Exception:
            pass
        return max((current_date - first_buy_date).days, 0) if hasattr(first_buy_date, "day") else 0

    def _exit_config(self) -> ExitConfig:
        cfg = self.p.exit_config
        if isinstance(cfg, ExitConfig):
            return cfg
        if isinstance(cfg, dict):
            return ExitConfig.from_dict(cfg)
        scheme = BUILTIN_SCHEMES.get(str(self.p.strategy_id or "balanced"))
        return getattr(scheme, "exit_config", ExitConfig()) if scheme else ExitConfig()

    def _market_score_on(self, d: date) -> float:
        scores = self.p.market_scores or {}
        if not scores:
            return 50.0
        if d in scores:
            return float(scores[d])
        sorted_dates = sorted(scores.keys())
        for sd in reversed(sorted_dates):
            if sd <= d:
                return float(scores[sd])
        return 50.0

    def _failure_exit_meta(self, symbol: str, data, current_date: date, state: dict, cfg: ExitConfig, current_price: float):
        strategy_id = str(state.get("strategy_id") or self.p.strategy_id or "balanced")
        first_buy_date = state.get("first_buy_date") or current_date
        holding_days = self._holding_trading_days(current_date, first_buy_date)
        if holding_days > int(getattr(cfg, "failure_window_days", 3) or 3):
            return None
        try:
            ma20 = float(np.mean([data.close[-i] for i in range(1, 21)]))
            low20 = float(min(data.close[-i] for i in range(1, 21)))
        except Exception:
            return None
        if strategy_id == "trend_momentum" and current_price < ma20:
            return "动量失效退出", "动量失效退出", ma20
        if strategy_id == "pullback" and current_price < low20:
            return "回调破位退出", "回调破位退出", low20
        if strategy_id == "breakout":
            platform_high = float(state.get("platform_high") or 0.0)
            if platform_high > 0 and current_price < platform_high:
                return "突破失败退出", "突破失败退出", platform_high
        return None

    def _trailing_exit_meta(self, symbol: str, data, state: dict, cfg: ExitConfig, current_price: float):
        """盈利保护型跟踪退出：最高浮盈达到激活区间后才生效。"""
        if not bool(getattr(cfg, "enable_trailing_exit", True)):
            return None
        avg_cost = float(state.get("avg_cost") or 0.0)
        if avg_cost <= 0:
            return None
        try:
            current_high = float(data.high[0])
        except Exception:
            current_high = current_price
        highest = max(float(state.get("highest") or 0.0), current_high, current_price)
        try:
            atr_val = float(state.get("atr") or 0.0)
        except Exception:
            atr_val = 0.0
        if atr_val <= 0:
            return None
        activation_pct = float(getattr(cfg, "trailing_activation_pct", 0.05) or 0.0)
        activation_atr_mult = float(getattr(cfg, "trailing_activation_atr_mult", 1.0) or 0.0)
        pct_ok = activation_pct <= 0 or highest >= avg_cost * (1 + activation_pct)
        atr_ok = activation_atr_mult <= 0 or highest >= avg_cost + activation_atr_mult * atr_val
        if not (pct_ok or atr_ok):
            state["highest"] = highest
            return None
        trailing_mult = float(getattr(self.p, "trailing_atr_mult", 2.0) or 2.0)
        prev_trailing = float(state.get("trailing_stop") or 0.0)
        trailing_stop = max(prev_trailing, highest - trailing_mult * atr_val)
        state["highest"] = highest
        state["trailing_stop"] = trailing_stop
        if current_price > trailing_stop:
            return None
        projected_pct = current_price / avg_cost - 1
        if projected_pct > 0:
            return {
                "reason": f"跟踪止盈(最高{highest:.2f})",
                "rule_name": "ATR跟踪止盈",
                "exit_type": "take_profit",
                "exit_subtype": "atr_trailing_profit",
                "trigger_price": trailing_stop,
            }
        return {
            "reason": f"跟踪止盈失效-回撤止损(最高{highest:.2f})",
            "rule_name": "ATR跟踪回撤止损",
            "exit_type": "stop_loss",
            "exit_subtype": "atr_trailing_profit_failed",
            "trigger_price": trailing_stop,
        }

    def _atr_from_history(self, data, period: int = 14) -> float:
        """用当前开盘前已完成K线计算ATR，避免使用当日 high/low/close 未来信息。"""
        try:
            if len(data) <= period + 1:
                return 0.0
            trs = []
            max_ago = min(len(data) - 1, period)
            for ago in range(max_ago, 0, -1):
                high = float(data.high[-ago])
                low = float(data.low[-ago])
                prev_close = float(data.close[-ago - 1]) if ago + 1 < len(data) else float(data.close[-ago])
                trs.append(max(abs(high - low), abs(high - prev_close), abs(low - prev_close)))
            return float(np.mean(trs)) if trs else 0.0
        except Exception:
            return 0.0

    def _p2_exit_meta(self, symbol: str, data, current_date: date, target_symbols: List[str], state: dict):
        """P3第一批：全池路径复用P2短线退出审计，不改变撮合机制。"""
        cfg = self._exit_config()
        avg_cost = float(state.get("avg_cost") or 0.0)
        first_buy_date = state.get("first_buy_date") or current_date
        holding_days = self._holding_trading_days(current_date, first_buy_date)
        current_price = float(data.open[0]) if self.p.cheat_on_open else float(data.close[0])
        pnl_pct = current_price / avg_cost - 1 if avg_cost > 0 else 0.0
        market_score = self._market_score_on(current_date)
        if bool(getattr(cfg, "enable_market_defense_exit", True)) and market_score < float(getattr(cfg, "market_defense_score", 20.0) or 20.0):
            return {
                "reason": f"大盘防御减仓(评分{market_score:.0f})",
                "rule_name": "大盘防御减仓",
                "exit_type": "market_exit",
                "exit_subtype": "market_defense",
                "trigger_price": current_price,
            }
        failure = self._failure_exit_meta(symbol, data, current_date, state, cfg, current_price) if bool(getattr(cfg, "enable_strategy_failure_exit", True)) else None
        if failure:
            reason, rule_name, trigger = failure
            subtype_map = {
                "动量失效退出": "trend_momentum_failed",
                "回调破位退出": "pullback_breakdown",
                "突破失败退出": "breakout_failed",
            }
            return {
                "reason": reason,
                "rule_name": rule_name,
                "exit_type": "strategy_failure",
                "exit_subtype": subtype_map.get(rule_name, "generic_strategy_failure"),
                "trigger_price": trigger,
            }
        trailing = self._trailing_exit_meta(symbol, data, state, cfg, current_price)
        if trailing:
            return trailing
        max_holding_days = int(getattr(cfg, "max_holding_days", 20) or 20)
        time_stop_days = int(getattr(cfg, "time_stop_days", 7) or 7)
        min_profit = float(getattr(cfg, "time_stop_min_profit_pct", 0.0) or 0.0)
        if bool(getattr(cfg, "enable_max_holding_exit", True)) and holding_days >= max_holding_days:
            return {
                "reason": f"最长持仓退出({holding_days}日)",
                "rule_name": "最长持仓退出",
                "exit_type": "time_exit",
                "exit_subtype": "max_holding_days",
                "trigger_price": current_price,
            }
        if bool(getattr(cfg, "enable_time_stop", True)) and holding_days >= time_stop_days and pnl_pct < min_profit:
            return {
                "reason": f"时间止损({holding_days}日收益{pnl_pct:.2%})",
                "rule_name": "时间止损",
                "exit_type": "stop_loss",
                "exit_subtype": "time_stop",
                "trigger_price": current_price,
            }
        if symbol not in target_symbols:
            return {
                "reason": "调仓卖出",
                "rule_name": "Backtrader调仓",
                "exit_type": "signal_exit",
                "exit_subtype": "rule_signal",
                "trigger_price": 0.0,
            }
        return None

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

    def next_open(self):
        """P3: 信号T日收盘后，T+1开盘前下单并以当日开盘撮合。"""
        if self.p.cheat_on_open:
            self._rebalance_current_bar(use_open_price=True)

    def next(self):
        """每个交易日调用；非 cheat-on-open 模式下保留旧撮合路径。"""
        if self.p.cheat_on_open:
            return
        self._rebalance_current_bar(use_open_price=False)

    def _rebalance_current_bar(self, *, use_open_price: bool):
        """执行当日调仓；开盘撮合时只用 open 估算下单，避免当日收盘未来函数。"""
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

        # === 卖出不在目标列表或触发P2短线退出的持仓 ===
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size > 0:
                exit_meta = self._p2_exit_meta(data._name, data, current_date, target_symbols, self.position_state.get(data._name, {}))
                if exit_meta is None:
                    continue
                order = self.close(data=data)
                if order is not None:
                    self.order_dict[order.ref] = {
                        "signal_date": signal_date,
                        **exit_meta,
                    }
                if self.p.verbose:
                    logger.info(f"  卖出 {data._name}: {exit_meta['reason']}")

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
            ref_price = float(data.open[0]) if use_open_price else float(data.close[0])
            current_value = pos.size * ref_price if pos.size > 0 else 0
            diff = target_value - current_value
            if diff > 1000:
                size = int(diff / ref_price / 100) * 100
                if size > 0 and cash >= size * ref_price * 1.005:
                    order = self.buy(data=data, size=size)
                    if order is not None:
                        platform_high = 0.0
                        try:
                            platform_high = float(max(data.close[-i] for i in range(1, min(16, len(data)))) )
                        except Exception:
                            platform_high = 0.0
                        self.order_dict[order.ref] = {
                            "signal_date": signal_date,
                            "reason": "调仓买入",
                            "rule_name": "Backtrader调仓",
                            "strategy_id": self.p.strategy_id,
                            "platform_high": platform_high,
                            "atr": self._atr_from_history(data),
                        }
                    cash -= size * ref_price
                    if self.p.verbose:
                        logger.info(f"  买入 {symbol} {size}股 @ {ref_price:.2f}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            exec_dt = bt.num2date(order.executed.dt).date() if order.executed.dt else self.datas[0].datetime.date(0)
            symbol = order.data._name
            action = "BUY" if order.isbuy() else "SELL"
            order_meta = self.order_dict.pop(order.ref, {})
            signal_date = order_meta.get("signal_date", "")
            reason = order_meta.get("reason", "调仓执行")
            rule_name = order_meta.get("rule_name", "Backtrader调仓")
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
                    "strategy_id": order_meta.get("strategy_id", self.p.strategy_id),
                    "platform_high": order_meta.get("platform_high", state.get("platform_high", 0.0)),
                    "highest": max(float(state.get("highest") or 0.0), exec_price),
                    "atr": float(order_meta.get("atr", state.get("atr", 0.0)) or 0.0),
                    "trailing_stop": float(state.get("trailing_stop", 0.0) or 0.0),
                }
            else:
                first_buy_date = state.get("first_buy_date") or exec_dt
                holding_days = self._holding_trading_days(exec_dt, first_buy_date)
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
                        "strategy_id": state.get("strategy_id", self.p.strategy_id),
                        "platform_high": state.get("platform_high", 0.0),
                        "highest": state.get("highest", 0.0),
                        "atr": state.get("atr", 0.0),
                        "trailing_stop": state.get("trailing_stop", 0.0),
                    }
            exit_type = order_meta.get("exit_type", "signal_exit" if action == "SELL" else "")
            exit_subtype = order_meta.get("exit_subtype", "rule_signal" if action == "SELL" else "")
            trigger_price = float(order_meta.get("trigger_price", exec_price if action == "SELL" else 0.0) or 0.0)
            if action == "SELL" and trigger_price <= 0:
                trigger_price = exec_price
            projected_pnl = pnl if action == "SELL" else 0.0
            try:
                from signals.rules import TradePoint
                tp = TradePoint(
                    date=exec_dt,
                    action=action,
                    reason=reason,
                    confidence=1.0,
                    price=exec_price,
                    rule_name=rule_name,
                    exec_price=exec_price,
                    shares=size,
                    cash_after=cash_after,
                    position_shares=position_after,
                    avg_cost=avg_cost,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    holding_days=holding_days,
                    signal_date=signal_date,
                    exec_date=exec_dt,
                    exit_type=exit_type,
                    exit_subtype=exit_subtype,
                    trigger_price=trigger_price,
                    projected_pnl=projected_pnl,
                )
                self.executed_points.setdefault(symbol, []).append(tp)
            except Exception:
                pass
            self.trade_details.append({
                "symbol": symbol,
                "date": exec_dt,
                "exec_date": exec_dt,
                "signal_date": signal_date,
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
                "reason": reason,
                "rule_name": rule_name,
                "exit_type": exit_type,
                "exit_subtype": exit_subtype,
                "trigger_price": trigger_price,
                "projected_pnl": projected_pnl,
            })
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order_dict.pop(order.ref, None)
            logger.warning(f"  订单失败: {order.data._name} 状态={order.status}")
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_count += 1
                if self.p.verbose:
                    logger.info(f"  买入执行: {order.data._name} {order.size}股 @ {order.executed.price:.2f}")
            else:
                self.sell_count += 1
                if self.p.verbose:
                    logger.info(f"  卖出执行: {order.data._name} {order.size}股 @ {order.executed.price:.2f}")

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
        if self.params.cheat_on_open and hasattr(self.cerebro.broker, "set_coo"):
            self.cerebro.broker.set_coo(True)
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
            strategy_id=self.params.strategy_id,
            exit_config=self.params.exit_config,
            market_scores=self.params.market_scores or {},
            cheat_on_open=self.params.cheat_on_open,
            trailing_atr_mult=self.params.trailing_atr_mult,
            verbose=verbose,
        )

        # 添加分析器
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.03)
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        self.cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="returns", timeframe=bt.TimeFrame.Days)

        logger.info(f"开始回测: {self.params.start_date} ~ {self.params.end_date}")
        results = self.cerebro.run(cheat_on_open=self.params.cheat_on_open)
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
