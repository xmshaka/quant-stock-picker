"""Phase 3: 方案回测引擎 — 串联方案→选股→信号→回测→落盘

流程：
1. 按 scheme.factor_weights 对全池截面打分 → 选 Top N
2. 对 Top N 每只股票，用 scheme.signal_rules 在K线上生成买卖点
3. 汇总为调仓信号序列 {date: [symbols]}
4. 喂入 BacktestEngine 执行
5. 返回绩效 + 交易明细 + 权益曲线
6. 持久化到 parquet (通过 backtest.records)

P0 一致性原则：
- K线默认展示 signals_executed (实际成交)
- signals_raw (原始规则信号) 为可选叠加层，不参与默认买卖次数统计
- 绩效统计、交易明细、K线买卖点必须来自同一执行事件源
"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

from strategy.schemes import StrategyScheme
from signals.rules import TradePoint, evaluate_all_rules
from signals.engine import SignalEngine
from backtest.engine import BacktestEngine, BacktestParams, estimate_turnover_amount, get_liquidity_slippage_rate
from backtest.records import (
    BacktestRunConfig, make_run_id, trade_points_to_frame,
    trade_details_to_frame, equity_curve_to_frame, persist_backtest_run,
)
from config.settings import settings
import copy


def _fetch_ohlcv(symbols: list, lookback_days: int, adjust: str = "") -> pd.DataFrame:
    """从腾讯数据源拉取 OHLCV，补齐快照缺失的 open/high/low/volume。

    P0 数据口径：撮合 / 默认K线买卖点必须使用不复权价格。
    - adjust=""   : 不复权，默认用于回测撮合和默认K线
    - adjust="qfq": 前复权，仅允许用于趋势展示叠加，不参与成交统计
    """
    try:
        from data.fetchers.tencent_fetcher import TencentFetcher
        from datetime import datetime, timedelta
        fetcher = TencentFetcher()
        end = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - timedelta(days=lookback_days + 60)).strftime('%Y%m%d')
        frames = []
        for sym in symbols:
            try:
                df = fetcher.get_daily_bars(sym, start_date=start, end_date=end, adjust=adjust)
                if df is not None and not df.empty:
                    df["adjust"] = adjust or "none"
                    frames.append(df)
            except Exception as e:
                logger.debug(f"[SchemeBacktest] 拉取{sym} OHLCV失败: {e}")
                continue
        if frames:
            result = pd.concat(frames, ignore_index=True)
            logger.info(f"[SchemeBacktest] OHLCV拉取成功: {len(frames)}只, {len(result)}条")
            return result
    except Exception as e:
        logger.warning(f"[SchemeBacktest] 拉取OHLCV失败: {e}")
    return pd.DataFrame()


@dataclass
class SchemeBacktestResult:
    """方案回测结果"""
    scheme_id: str
    scheme_name: str
    start_date: str
    end_date: str
    # 绩效指标
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    trade_count: int = 0       # round-trip 交易次数（买+卖=1）
    buy_count: int = 0         # 买入次数（单边）
    sell_count: int = 0        # 卖出次数（单边）
    final_value: float = 0.0
    run_id: str = ""
    # 事件源（P0：K线默认展示 signals_executed）
    signals_executed: Dict[str, List[TradePoint]] = field(default_factory=dict)
    signals_raw: Dict[str, List[TradePoint]] = field(default_factory=dict)
    # 详情
    equity_curve: Dict[str, float] = field(default_factory=dict)
    trade_details: List[Dict] = field(default_factory=list)
    _stock_signals_backup: Dict[str, List[TradePoint]] = field(default_factory=dict)

    @property
    def stock_signals(self) -> Dict[str, List[TradePoint]]:
        """兼容旧版：优先使用 signals_executed"""
        return self.signals_executed if self.signals_executed else self._stock_signals_backup

    @stock_signals.setter
    def stock_signals(self, val: Dict[str, List[TradePoint]]):
        self._stock_signals_backup = val or {}

    def fmt(self, key: str) -> str:
        """安全格式化，避免除零"""
        try:
            if key == 'total_return': return f"{self.total_return:+.2%}"
            if key == 'annual_return': return f"{self.annual_return:+.2%}"
            if key == 'sharpe_ratio': return f"{self.sharpe_ratio:.3f}"
            if key == 'max_drawdown': return f"{self.max_drawdown:.2%}"
            if key == 'win_rate': return f"{self.win_rate:.0%}" if self.trade_count > 0 else "N/A"
            if key == 'trade_count': return str(self.trade_count)
            if key == 'final_value': return f"{self.final_value:,.0f}"
        except (ZeroDivisionError, ValueError, TypeError):
            return "N/A"
        return ""

    @property
    def trade_summary(self) -> str:
        """交易次数摘要：买X卖Y（N轮）"""
        return f"买{self.buy_count} 卖{self.sell_count}（{self.trade_count}轮）"

    def summary_text(self) -> str:
        lines = [
            f"策略: {self.scheme_name}",
            f"区间: {self.start_date} ~ {self.end_date}",
            f"总收益: {self.fmt('total_return')}",
            f"年化收益: {self.fmt('annual_return')}",
            f"夏普比率: {self.fmt('sharpe_ratio')}",
            f"最大回撤: {self.fmt('max_drawdown')}",
            f"胜率: {self.fmt('win_rate')}",
            f"交易: {self.trade_summary}",
            f"最终资金: {self.fmt('final_value')}",
            f"run_id: {self.run_id}",
        ]
        return "\n".join(lines)

    def persist(self, config=None, **overrides) -> Path:
        """持久化回测结果到 parquet。"""
        from backtest.records import (
            BacktestRunConfig, trade_points_to_frame,
            trade_details_to_frame, equity_curve_to_frame, persist_backtest_run,
        )
        if config is None:
            config = BacktestRunConfig(
                run_id=self.run_id,
                scheme_id=self.scheme_id,
                scheme_name=self.scheme_name,
                start_date=self.start_date,
                end_date=self.end_date,
                lookback_days=0,
                top_n=0,
                initial_capital=1_000_000.0,
            )
        # P0: trades.parquet 以真实成交明细为准，避免丢失成本/PnL字段
        trades_df = trade_details_to_frame(self.trade_details, run_id=config.run_id, source="executed")
        raw_df = trade_points_to_frame(self.signals_raw, source="raw_rule")
        executed_df = trade_points_to_frame(self.signals_executed, source="executed")
        equity_df = equity_curve_to_frame(self.equity_curve, config.run_id)
        return persist_backtest_run(
            result=self, config=config,
            trades=trades_df, signals_raw=raw_df, signals_executed=executed_df,
            equity=equity_df,
            **overrides,
        )


class SchemeBacktester:
    """方案回测器"""

    def __init__(self):
        self._signal_engine = SignalEngine(
            buy_threshold=0.7, sell_threshold=0.3, min_strength=2.0,
        )

    @staticmethod
    def _calc_atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算 ATR (Average True Range)

        Returns: Series indexed by trade_date, value = ATR
        """
        high = bars['high'].astype(float)
        low = bars['low'].astype(float)
        close = bars['close'].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        atr.index = bars['trade_date']
        return atr

    def _run_single_stock_backtest(
        self,
        symbols: list,
        signal_rules,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        bt_dates,
        start_date,
        end_date,
        lookback_days: int,
        initial_capital: float,
        verbose: bool,
        scheme: 'StrategyScheme' = None,
    ) -> SchemeBacktestResult:
        """单股/少量股票回测：直接用信号规则驱动买卖模拟

        支持：加仓、ATR止盈止损、资金校验、持仓记录。
        """
        # ── 方案参数 ──
        max_add_times = scheme.max_add_times if scheme else 2
        pos_pct_per_entry = scheme.position_pct_per_entry if scheme else 0.30
        # 风控硬约束：单票最大仓位 20%，不能被方案参数放宽
        max_single_pct = min(scheme.max_single_pct if scheme else 0.20, 0.20)
        sl_mult = scheme.stop_loss_atr_mult if scheme else 2.0
        tp_mult = scheme.take_profit_atr_mult if scheme else 3.0
        trail_mult = scheme.trailing_atr_mult if scheme else 2.0
        atr_period = scheme.atr_period if scheme else 14

        # 拉取 OHLCV
        ohlcv_df = _fetch_ohlcv(symbols, lookback_days)
        have_ohlcv = not ohlcv_df.empty
        if have_ohlcv:
            ohlcv_df['trade_date'] = pd.to_datetime(ohlcv_df['trade_date'])

        # 为每只股票生成信号 + ATR
        stock_signal_details: Dict[str, List[TradePoint]] = {}
        sym_bars_map: Dict[str, pd.DataFrame] = {}
        atr_lookup: Dict[str, Dict] = {}  # {sym: {date: atr_value}}
        for sym in symbols:
            if have_ohlcv:
                sym_bars = ohlcv_df[ohlcv_df['symbol'] == sym].copy()
            else:
                sym_bars = price_df[price_df['symbol'] == sym].copy()
            sym_bars = sym_bars.sort_values('trade_date')
            # 需要额外前置数据来计算 ATR
            sym_bars_full = sym_bars.copy()
            sym_bars = sym_bars[
                (sym_bars['trade_date'] >= start_date) &
                (sym_bars['trade_date'] <= end_date)
            ]
            if sym_bars.empty:
                continue
            for col in ('open', 'high', 'low', 'volume'):
                if col not in sym_bars.columns:
                    sym_bars[col] = sym_bars['close']
            for col in ('open', 'high', 'low', 'volume'):
                if col not in sym_bars_full.columns:
                    sym_bars_full[col] = sym_bars_full['close']
            sym_bars_map[sym] = sym_bars

            # 计算 ATR（用完整数据，取回测区间内的部分）
            atr_series = self._calc_atr(sym_bars_full, atr_period)
            for _, row in sym_bars.iterrows():
                dt = row['trade_date']
                if hasattr(dt, 'date'):
                    dt = dt.date()
                atr_val = atr_series.get(row['trade_date'], np.nan) if hasattr(atr_series, 'get') else np.nan
                if pd.isna(atr_val) and hasattr(atr_series, 'loc'):
                    try:
                        atr_val = atr_series.loc[row['trade_date']]
                    except (KeyError, IndexError):
                        atr_val = np.nan
                atr_lookup.setdefault(sym, {})[dt] = float(atr_val) if not pd.isna(atr_val) else 0.0

            points = evaluate_all_rules(sym_bars, signal_rules)
            if points:
                stock_signal_details[sym] = points

        # ── 直接模拟交易 ──
        # 规则：
        #   1. 同日同股票同动作 → 已去重（规则层）
        #   2. 同日有买有卖 → SELL 优先（防当日冲销）
        #   3. T日信号 → T+1日开盘执行（防未来函数）
        #   4. 支持加仓（同向BUY再触发）
        #   5. 每日检查止盈止损

        cash = initial_capital
        positions: Dict[str, dict] = {}  # {sym: PositionRecord}
        equity_curve = {}
        buy_count = 0
        sell_count = 0
        trades_pnl: List[float] = []
        actual_trades: List[tuple] = []  # (symbol, TradePoint)
        trade_details: List[Dict] = []   # 每笔交易明细

        # 价格查找表：open + close
        open_lookup: Dict[str, Dict] = {}
        close_lookup: Dict[str, Dict] = {}
        high_lookup: Dict[str, Dict] = {}
        amount_lookup: Dict[str, Dict] = {}
        for sym, bars in sym_bars_map.items():
            for _, row in bars.iterrows():
                dt = row['trade_date']
                if hasattr(dt, 'date'):
                    dt = dt.date()
                open_lookup.setdefault(sym, {})[dt] = float(row.get('open', row['close']))
                close_lookup.setdefault(sym, {})[dt] = float(row['close'])
                high_lookup.setdefault(sym, {})[dt] = float(row.get('high', row['close']))
                amount_lookup.setdefault(sym, {})[dt] = estimate_turnover_amount(
                    row.get('amount', 0.0),
                    row.get('volume', 0.0),
                    row.get('close', 0.0),
                )

        # 合并同日信号（SELL 优先）
        daily_signals: Dict[str, Dict] = {}
        for sym, points in stock_signal_details.items():
            by_date: Dict = {}
            for p in points:
                d = p.date.date() if hasattr(p.date, 'date') else p.date
                if d not in by_date:
                    by_date[d] = p
                elif p.action == 'SELL':
                    by_date[d] = p
            daily_signals[sym] = by_date

        # T+1 日期映射
        date_list = sorted(bt_dates)
        next_date_map: Dict = {}
        for i, dt in enumerate(date_list):
            dt_key = dt.date() if hasattr(dt, 'date') else dt
            if i + 1 < len(date_list):
                next_dt = date_list[i + 1]
                next_dt_key = next_dt.date() if hasattr(next_dt, 'date') else next_dt
                next_date_map[dt_key] = next_dt_key

        # 收集待执行信号（信号日 → 执行日），保留 TradePoint 信息
        pending_actions: Dict[str, Dict] = {}  # {exec_date: {sym: TradePoint}}
        for sym in symbols:
            sig_map = daily_signals.get(sym, {})
            for sig_date, tp in sig_map.items():
                exec_date = next_date_map.get(sig_date)
                if exec_date is None:
                    continue
                if exec_date not in pending_actions:
                    pending_actions[exec_date] = {}
                existing = pending_actions[exec_date].get(sym)
                if existing and existing.action == 'SELL':
                    pass  # 已有SELL，不覆盖
                elif tp.action == 'SELL':
                    pending_actions[exec_date][sym] = tp
                elif existing is None:
                    pending_actions[exec_date][sym] = tp
                # else: 已有BUY，保留第一个

        # 成本模型：佣金、印花税、过户费、滑点四项齐全
        DEFAULT_SLIPPAGE = 0.002  # 缺失成交额时默认蓝筹单边滑点 0.2%
        COMMISSION = 0.00025   # 佣金万2.5
        MIN_COMMISSION = 5.0
        STAMP_DUTY = 0.001     # 卖出单边千1
        TRANSFER_FEE = 0.00001 # 双向过户费万0.1

        def _slippage_info(sym, dt_key):
            turnover_amount = amount_lookup.get(sym, {}).get(dt_key, 0.0)
            rate, bucket = get_liquidity_slippage_rate(turnover_amount, DEFAULT_SLIPPAGE)
            return rate, bucket, turnover_amount

        def _sell(sym, dt_key, price, reason, rule_name):
            """统一卖出逻辑"""
            nonlocal cash, sell_count
            pos = positions.pop(sym)
            slippage_rate, liquidity_bucket, turnover_amount = _slippage_info(sym, dt_key)
            exec_price = price * (1 - slippage_rate)  # 卖出滑点：成交价略低
            revenue = pos['shares'] * exec_price
            commission = max(revenue * COMMISSION, MIN_COMMISSION)
            stamp = revenue * STAMP_DUTY
            transfer_fee = revenue * TRANSFER_FEE
            slippage_cost = pos['shares'] * price * slippage_rate
            net = revenue - commission - stamp - transfer_fee
            pnl = net - pos['shares'] * pos['avg_cost']
            pnl_pct = pnl / (pos['shares'] * pos['avg_cost']) if pos['avg_cost'] > 0 else 0
            trades_pnl.append(pnl)
            cash += net
            sell_count += 1
            holding_days = (dt_key - pos['entry_dt']).days if hasattr(dt_key, 'days') else 0
            tp_out = TradePoint(
                date=dt_key, action='SELL', reason=reason,
                confidence=1.0, price=exec_price, rule_name=rule_name,
                exec_price=exec_price, shares=pos['shares'],
                cash_after=cash, position_shares=0,
                avg_cost=pos['avg_cost'],
                pnl=pnl, pnl_pct=pnl_pct, holding_days=holding_days,
            )
            actual_trades.append((sym, tp_out))
            trade_details.append({
                'symbol': sym, 'date': dt_key, 'action': 'SELL',
                'price': exec_price, 'exec_price': exec_price, 'shares': pos['shares'],
                'cost': pos['avg_cost'], 'pnl': pnl, 'pnl_pct': pnl_pct,
                'commission': commission, 'stamp_duty': stamp,
                'transfer_fee': transfer_fee, 'slippage': slippage_cost,
                'slippage_rate': slippage_rate,
                'liquidity_bucket': liquidity_bucket,
                'turnover_amount': turnover_amount,
                'cash_after': cash, 'reason': reason,
                'holding_days': holding_days,
                'entries': list(pos.get('entries', [])),
            })
            if verbose:
                logger.info(f"  [{dt_key}] 卖出 {sym} {pos['shares']}股 @ {price:.2f} 盈亏={pnl:+.0f} ({reason})")
            return tp_out

        def _buy(sym, dt_key, price, reason, rule_name, confidence=1.0, is_add=False):
            """统一买入/加仓逻辑，含资金校验"""
            nonlocal cash, buy_count
            total_value = cash + sum(
                p['shares'] * close_lookup.get(s, {}).get(dt_key, p['avg_cost'])
                for s, p in positions.items()
            )
            max_alloc = total_value * max_single_pct

            if is_add:
                # 加仓：受单股上限约束
                existing = positions[sym]
                current_val = existing['shares'] * price
                remaining = max_alloc - current_val
                alloc = min(cash * pos_pct_per_entry, remaining)
            else:
                # 建仓
                alloc = min(cash * pos_pct_per_entry, max_alloc)

            shares = int(alloc / price / 100) * 100
            if shares <= 0:
                if verbose:
                    logger.info(f"  [{dt_key}] 跳过 {sym} 余额不足 (cash={cash:.0f}, alloc={alloc:.0f})")
                return None

            slippage_rate, liquidity_bucket, turnover_amount = _slippage_info(sym, dt_key)
            exec_price = price * (1 + slippage_rate)  # 买入滑点：成交价略高
            cost = shares * exec_price
            commission = max(cost * COMMISSION, MIN_COMMISSION)
            transfer_fee = cost * TRANSFER_FEE
            slippage_cost = shares * price * slippage_rate
            total_cost = cost + commission + transfer_fee
            if total_cost > cash:
                if verbose:
                    logger.info(f"  [{dt_key}] 跳过 {sym} 资金不足 ({total_cost:.0f} > {cash:.0f})")
                return None

            atr_val = atr_lookup.get(sym, {}).get(dt_key, 0.0)

            if is_add:
                existing = positions[sym]
                old_total = existing['shares'] * existing['avg_cost']
                new_total = old_total + cost
                new_shares = existing['shares'] + shares
                new_avg = new_total / new_shares if new_shares > 0 else price
                existing['shares'] = new_shares
                existing['avg_cost'] = new_avg
                existing['entries'].append({'date': dt_key, 'price': exec_price, 'shares': shares, 'reason': reason})
                # 止盈止损用加权ATR
                old_atr = existing.get('atr', atr_val)
                old_shares = max(new_shares - shares, 0)
                existing['atr'] = (old_atr * old_shares + atr_val * shares) / new_shares if new_shares > 0 else atr_val
                existing['stop_loss'] = new_avg - sl_mult * existing['atr']
                existing['take_profit'] = new_avg + tp_mult * existing['atr']
            else:
                stop_loss = price - sl_mult * atr_val if atr_val > 0 else price * 0.92
                take_profit = price + tp_mult * atr_val if atr_val > 0 else price * 1.15
                positions[sym] = {
                    'shares': shares, 'avg_cost': exec_price, 'entry_dt': dt_key,
                    'highest': price, 'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'trailing_stop': stop_loss,
                    'atr': atr_val, 'add_count': 0,
                    'entries': [{'date': dt_key, 'price': exec_price, 'shares': shares, 'reason': reason}],
                }

            cash -= total_cost
            buy_count += 1
            pos = positions[sym]
            tp_out = TradePoint(
                date=dt_key, action='BUY', reason=reason,
                confidence=confidence, price=exec_price, rule_name=rule_name,
                exec_price=exec_price, shares=shares,
                cash_after=cash,
                position_shares=pos['shares'],
                avg_cost=pos['avg_cost'],
                stop_loss=pos['stop_loss'],
                take_profit=pos['take_profit'],
            )
            actual_trades.append((sym, tp_out))
            trade_details.append({
                # FIX:P0: 加仓也属于 BUY 执行事件，K线/明细/统计必须一致
                'symbol': sym, 'date': dt_key, 'action': 'BUY', 'event_type': 'ADD' if is_add else 'BUY',
                'price': exec_price, 'exec_price': exec_price, 'shares': shares,
                'commission': commission, 'stamp_duty': 0.0,
                'transfer_fee': transfer_fee, 'slippage': slippage_cost,
                'slippage_rate': slippage_rate,
                'liquidity_bucket': liquidity_bucket,
                'turnover_amount': turnover_amount,
                'cash_after': cash, 'reason': reason,
                'position_shares': pos['shares'],
                'avg_cost': pos['avg_cost'],
                'stop_loss': pos['stop_loss'],
                'take_profit': pos['take_profit'],
            })
            if verbose:
                tag = '加仓' if is_add else '买入'
                logger.info(f"  [{dt_key}] {tag} {sym} {shares}股 @ {price:.2f} "
                           f"持仓={pos['shares']} 均价={pos['avg_cost']:.2f} "
                           f"止损={pos['stop_loss']:.2f} 止盈={pos['take_profit']:.2f}")
            return tp_out

        # ── 主循环 ──
        for dt in date_list:
            dt_key = dt.date() if hasattr(dt, 'date') else dt

            # 1. 止盈止损检查（开盘价，优先于信号执行）
            for sym in list(positions.keys()):
                pos = positions[sym]
                open_price = open_lookup.get(sym, {}).get(dt_key, 0)
                if open_price <= 0:
                    continue

                # 更新最高价
                high_price = high_lookup.get(sym, {}).get(dt_key, open_price)
                if high_price > pos['highest']:
                    pos['highest'] = high_price
                    # 跟踪止盈只升不降
                    new_trailing = pos['highest'] - trail_mult * pos['atr']
                    if new_trailing > pos['trailing_stop']:
                        pos['trailing_stop'] = new_trailing

                # 止损优先
                if open_price <= pos['stop_loss']:
                    _sell(sym, dt_key, open_price, f'止损({pos["stop_loss"]:.2f})', 'ATR止损')
                    continue
                # 跟踪止盈
                if open_price <= pos['trailing_stop'] and pos['highest'] > pos['avg_cost']:
                    _sell(sym, dt_key, open_price, f'跟踪止盈(最高{pos["highest"]:.2f})', 'ATR跟踪止盈')
                    continue
                # 固定止盈
                if open_price >= pos['take_profit']:
                    _sell(sym, dt_key, open_price, f'止盈({pos["take_profit"]:.2f})', 'ATR止盈')
                    continue

            # 2. 执行今日的待处理信号
            actions = pending_actions.get(dt_key, {})
            for sym, tp in actions.items():
                price = close_lookup.get(sym, {}).get(dt_key, 0)
                if price <= 0:
                    continue

                if tp.action == 'SELL' and sym in positions:
                    _sell(sym, dt_key, price, tp.reason or '信号卖出', tp.rule_name or '信号卖出')

                elif tp.action == 'BUY':
                    if sym in positions:
                        # 已持仓 → 加仓（检查次数上限）
                        if positions[sym].get('add_count', 0) < max_add_times:
                            result_tp = _buy(sym, dt_key, price, tp.reason or '加仓', tp.rule_name or '加仓',
                                            confidence=tp.confidence, is_add=True)
                            if result_tp:
                                positions[sym]['add_count'] = positions[sym].get('add_count', 0) + 1
                        elif verbose:
                            logger.info(f"  [{dt_key}] {sym} 已达最大加仓次数({max_add_times})")
                    else:
                        # 新建仓
                        _buy(sym, dt_key, price, tp.reason or '信号买入', tp.rule_name or '信号买入',
                             confidence=tp.confidence)

            # 3. 计算当日总权益
            total_value = cash
            for sym, pos in positions.items():
                price = close_lookup.get(sym, {}).get(dt_key, pos['avg_cost'])
                total_value += pos['shares'] * price
            equity_curve[dt_key] = total_value

        # 末日清仓
        final_dt = date_list[-1].date() if hasattr(date_list[-1], 'date') else date_list[-1]
        for sym in list(positions.keys()):
            price = close_lookup.get(sym, {}).get(final_dt, positions[sym]['avg_cost'])
            _sell(sym, final_dt, price, '末日清仓', '末日清仓')

        # 原始规则信号仅作为可选叠加层，不能参与默认 K线/绩效统计
        raw_signal_details = copy.deepcopy(stock_signal_details)

        # 用实际执行的交易记录作为K线展示信号（与回测绩效完全一致）
        stock_signal_details = {}
        for sym, tp in actual_trades:
            stock_signal_details.setdefault(sym, []).append(tp)
        for sym in symbols:
            if sym not in stock_signal_details:
                stock_signal_details[sym] = []

        final_value = cash
        total_return = (final_value - initial_capital) / initial_capital
        days = (end_date - start_date).days
        years = days / 365.0
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 and total_return > -1 else 0
        win_count = sum(1 for p in trades_pnl if p > 0)
        win_rate = win_count / len(trades_pnl) if trades_pnl else 0

        # 最大回撤
        eq_values = list(equity_curve.values())
        max_dd = 0
        peak = eq_values[0] if eq_values else initial_capital
        for v in eq_values:
            peak = max(peak, v)
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)

        # 简化夏普
        if len(eq_values) > 1:
            daily_returns = [(eq_values[i] - eq_values[i-1]) / eq_values[i-1]
                            for i in range(1, len(eq_values)) if eq_values[i-1] > 0]
            if daily_returns:
                avg_ret = np.mean(daily_returns)
                std_ret = np.std(daily_returns)
                sharpe = (avg_ret - 0.03/252) / std_ret * np.sqrt(252) if std_ret > 0 else 0
            else:
                sharpe = 0
        else:
            sharpe = 0

        # 权益曲线：直接输出每日绝对权益值（避免连乘浮点误差）
        eq_curve = {str(dt): val for dt, val in equity_curve.items()}

        trade_count = len(trades_pnl)  # 完成的交易轮数

        result = SchemeBacktestResult(
            scheme_id=scheme.scheme_id if scheme else '',
            scheme_name=scheme.name if scheme else '',
            start_date=str(start_date),
            end_date=str(end_date),
            run_id=make_run_id(scheme.scheme_id if scheme else 'single', end_date),
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            trade_count=trade_count,
            buy_count=buy_count,
            sell_count=sell_count,
            final_value=final_value,
            equity_curve=eq_curve,
            # FIX:P0 — K线默认事件源必须是实际成交，而不是原始规则信号
            signals_executed=stock_signal_details,
            signals_raw=raw_signal_details,
            trade_details=trade_details,
        )

        logger.info(f"[SingleStock] 收益={total_return:+.2%}, 交易={result.trade_summary}, "
                    f"胜率={win_rate:.0%}, 最终={final_value:,.0f}")

        return result

    @staticmethod
    def _apply_universe_filter(factor_df: pd.DataFrame, price_df: pd.DataFrame) -> tuple:
        """应用股票配置中的过滤条件（ST/退市/停牌/北交所/市值/换手率）

        Returns:
            (filtered_factor_df, filtered_price_df, filter_report)
        """
        try:
            from data.universe import Universe
            uni = Universe().load(use_cache=True)
            if uni.empty:
                logger.warning("[SchemeBacktest] Universe 为空，跳过过滤")
                return factor_df, price_df, "Universe 为空，未过滤"

            allowed = set(uni['symbol'].astype(str))
            before = factor_df['symbol'].nunique()
            factor_df = factor_df[factor_df['symbol'].isin(allowed)].copy()
            price_df = price_df[price_df['symbol'].isin(allowed)].copy()
            after = factor_df['symbol'].nunique()
            report = f"过滤前 {before} 只 → 过滤后 {after} 只（淘汰 {before - after}）"
            logger.info(f"[SchemeBacktest] {report}")
            return factor_df, price_df, report
        except Exception as e:
            logger.warning(f"[SchemeBacktest] Universe 过滤失败: {e}")
            return factor_df, price_df, f"过滤失败: {e}"

    def run(
        self,
        scheme: StrategyScheme,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_names: List[str],
        symbols: Optional[List[str]] = None,
        lookback_days: int = 60,
        top_n: int = 10,
        initial_capital: float = 1_000_000.0,
        verbose: bool = False,
        progress_callback=None,
    ) -> SchemeBacktestResult:
        """运行方案回测

        Args:
            scheme: 策略方案
            factor_df: 因子数据
            price_df: 价格数据
            factor_names: 因子名列表
            symbols: 指定股票子集（None=全池）
            lookback_days: 回测回看天数
            top_n: 每期选股数量
            initial_capital: 初始资金
            progress_callback: 进度回调 fn(step, total, msg)，用于 UI 进度条
        """
        # 0. 全池模式时应用股票配置过滤
        filter_report = ""
        if not symbols:
            factor_df, price_df, filter_report = self._apply_universe_filter(factor_df, price_df)
            if progress_callback:
                progress_callback(0, 100, f"过滤完成: {filter_report}")

        # 1. 确定回测区间
        all_dates = sorted(factor_df['trade_date'].unique())
        if len(all_dates) < lookback_days:
            lookback_days = len(all_dates)
        bt_dates = all_dates[-lookback_days:]
        start_date = bt_dates[0]
        end_date = bt_dates[-1]

        logger.info(f"[SchemeBacktest] {scheme.name}: {start_date} ~ {end_date}, {len(bt_dates)} 天")

        # 2. 确定选股模式
        target_pool = set(symbols) if symbols else None
        is_single_stock = target_pool and len(target_pool) <= 3

        signals_map: Dict[date, List[str]] = {}
        stock_signal_details: Dict[str, List[TradePoint]] = {}

        if is_single_stock:
            selected_syms = list(target_pool)
            if progress_callback:
                progress_callback(5, 100, "单股模式: 生成信号规则...")

            # 单股模式：用信号规则驱动买卖，直接模拟回测
            result = self._run_single_stock_backtest(
                selected_syms, scheme.signal_rules,
                factor_df, price_df, bt_dates, start_date, end_date,
                lookback_days, initial_capital, verbose,
                scheme=scheme,
            )
            result.scheme_id = scheme.scheme_id
            result.scheme_name = scheme.name
            if progress_callback:
                progress_callback(100, 100, "单股回测完成")
            return result
        else:
            # === 全池模式：因子截面打分 → 选 Top N ===
            total_days = len(bt_dates)
            for idx, dt in enumerate(bt_dates):
                if progress_callback and idx % 5 == 0:
                    pct = 10 + int(50 * idx / total_days)
                    progress_callback(pct, 100, f"因子打分 {idx+1}/{total_days}")

                day_factors = factor_df[factor_df['trade_date'] == dt].copy()
                if day_factors.empty:
                    continue

                scores = self._score_day(day_factors, factor_names, scheme.factor_weights)
                if scores.empty:
                    continue

                top_symbols = scores.nlargest(top_n).index.tolist()

                if target_pool:
                    selected = [s for s in top_symbols if s in target_pool]
                    if not selected:
                        for sym in target_pool:
                            if sym in scores.index:
                                rank = int((scores > scores[sym]).sum())
                                total = len(scores)
                                if rank < total * 0.5:
                                    selected.append(sym)
                    if selected:
                        signals_map[dt] = selected
                else:
                    signals_map[dt] = top_symbols

            if progress_callback:
                progress_callback(60, 100, f"选股完成: {len(signals_map)} 个调仓日")

        if not signals_map:
            logger.warning("[SchemeBacktest] 无有效信号")
            return SchemeBacktestResult(
                scheme_id=scheme.scheme_id, scheme_name=scheme.name,
                start_date=str(start_date), end_date=str(end_date),
            )

        # 2b. 去重：相邻交易日目标股票相同时跳过，避免无效调仓
        deduped_map: Dict[date, List[str]] = {}
        prev_targets = None
        for dt in sorted(signals_map.keys()):
            targets = tuple(sorted(signals_map[dt]))
            if targets != prev_targets:
                deduped_map[dt] = signals_map[dt]
                prev_targets = targets
        signals_map = deduped_map
        logger.info(f"[SchemeBacktest] 去重后 {len(signals_map)} 个调仓日")

        # 3. 对选中股票运行信号规则（全池模式才需要，单股模式已在步骤2完成）
        all_selected = set()
        for syms in signals_map.values():
            all_selected.update(syms)

        # 单股模式已在步骤2拉取了 ohlcv_df，此处做全池模式的初始化
        if not is_single_stock:
            ohlcv_df = pd.DataFrame()
            have_ohlcv = False

        if not is_single_stock:
            if progress_callback:
                progress_callback(65, 100, f"拉取 {len(all_selected)} 只股票 OHLCV...")

            ohlcv_df = _fetch_ohlcv(list(all_selected), lookback_days)
            have_ohlcv = not ohlcv_df.empty
            if have_ohlcv:
                ohlcv_df['trade_date'] = pd.to_datetime(ohlcv_df['trade_date'])

            if progress_callback:
                progress_callback(75, 100, "生成买卖点信号...")

            for sym in all_selected:
                if have_ohlcv:
                    sym_bars = ohlcv_df[ohlcv_df['symbol'] == sym].copy()
                else:
                    sym_bars = price_df[price_df['symbol'] == sym].copy()
                sym_bars = sym_bars.sort_values('trade_date')
                sym_bars = sym_bars[
                    (sym_bars['trade_date'] >= start_date) &
                    (sym_bars['trade_date'] <= end_date)
                ]
                if sym_bars.empty:
                    continue
                for col in ('open', 'high', 'low', 'volume'):
                    if col not in sym_bars.columns:
                        sym_bars[col] = sym_bars['close']
                points = evaluate_all_rules(sym_bars, scheme.signal_rules)
                if points:
                    stock_signal_details[sym] = points

        # 4. 运行 Backtrader 回测
        bt_params = BacktestParams(
            start_date=pd.Timestamp(start_date).date() if hasattr(start_date, 'date') else start_date,
            end_date=pd.Timestamp(end_date).date() if hasattr(end_date, 'date') else end_date,
            initial_capital=initial_capital,
            max_stocks=top_n,
            rebalance_freq=5,
        )

        engine = BacktestEngine(bt_params)

        # ohlcv_df: 单股模式在步骤2已拉取，全池模式在步骤3已拉取
        added_symbols = set()
        for sym in all_selected:
            if have_ohlcv:
                sym_data = ohlcv_df[ohlcv_df['symbol'] == sym].copy()
            else:
                sym_data = price_df[price_df['symbol'] == sym].copy()
            if sym_data.empty:
                continue
            for col in ('open', 'high', 'low', 'volume'):
                if col not in sym_data.columns:
                    sym_data[col] = sym_data['close']
            if 'amount' not in sym_data.columns:
                sym_data['amount'] = 0.0
            sym_data['amount'] = sym_data.apply(
                lambda r: estimate_turnover_amount(r.get('amount', 0.0), r.get('volume', 0.0), r.get('close', 0.0)),
                axis=1,
            )
            sym_data = sym_data.set_index('trade_date')[['open', 'high', 'low', 'close', 'volume', 'amount']]
            sym_data.index = pd.to_datetime(sym_data.index)
            engine.add_data(sym, sym_data)
            added_symbols.add(sym)

        if not added_symbols:
            logger.warning("[SchemeBacktest] 无可用股票数据")
            return SchemeBacktestResult(
                scheme_id=scheme.scheme_id, scheme_name=scheme.name,
                start_date=str(start_date), end_date=str(end_date),
            )

        signals_for_bt = {}
        for dt, syms in signals_map.items():
            dt_key = pd.Timestamp(dt).date() if hasattr(dt, 'date') else dt
            signals_for_bt[dt_key] = [s for s in syms if s in added_symbols]

        engine.add_signals(signals_for_bt)

        if progress_callback:
            progress_callback(85, 100, "Backtrader 回测执行中...")

        try:
            bt_result = engine.run(verbose=verbose)
        except Exception as e:
            logger.error(f"[SchemeBacktest] Backtrader 执行失败: {e}")
            return SchemeBacktestResult(
                scheme_id=scheme.scheme_id, scheme_name=scheme.name,
                start_date=str(start_date), end_date=str(end_date),
            )

        # 5. 汇总结果
        result = SchemeBacktestResult(
            scheme_id=scheme.scheme_id,
            scheme_name=scheme.name,
            start_date=str(start_date),
            end_date=str(end_date),
            run_id=make_run_id(scheme.scheme_id, end_date),
            total_return=bt_result.get("total_return", 0),
            annual_return=bt_result.get("annual_return", 0),
            sharpe_ratio=bt_result.get("sharpe_ratio", 0),
            max_drawdown=bt_result.get("max_drawdown", 0),
            win_rate=bt_result.get("win_rate", 0),
            trade_count=bt_result.get("trade_count", 0),
            buy_count=bt_result.get("buy_count", 0),
            sell_count=bt_result.get("sell_count", 0),
            final_value=bt_result.get("final_value", 0),
            equity_curve=bt_result.get("equity_curve", {}),
            # FIX:P0 — 全池模式默认K线使用 Backtrader 实际成交点
            signals_executed=bt_result.get("executed_points", {}),
            signals_raw=stock_signal_details,
            trade_details=bt_result.get("trade_details", []),
        )

        logger.info(f"[SchemeBacktest] {scheme.name}: 收益={result.total_return:+.2%}, "
                    f"夏普={result.sharpe_ratio:.3f}, 回撤={result.max_drawdown:.2%}, "
                    f"交易={result.trade_summary}")

        return result

    def _score_day(
        self,
        day_data: pd.DataFrame,
        factor_names: List[str],
        weights: Dict[str, float],
    ) -> pd.Series:
        """单日截面打分（简化版，不含 regime 自适应）"""
        scores = pd.Series(0.0, index=day_data['symbol'].values)
        total_w = 0

        for f in factor_names:
            if f not in day_data.columns or f not in weights:
                continue
            vals = day_data.set_index('symbol')[f].astype(float)
            mean, std = vals.mean(), vals.std()
            if std == 0 or pd.isna(std):
                continue
            z = (vals - mean) / std
            w = weights.get(f, 0)
            scores = scores.add(z * w, fill_value=0)
            total_w += abs(w)

        if total_w > 0:
            scores = scores / total_w

        return scores.dropna()


def run_multi_scheme_backtest(
    schemes: List[StrategyScheme],
    factor_df: pd.DataFrame,
    price_df: pd.DataFrame,
    factor_names: List[str],
    **kwargs,
) -> List[SchemeBacktestResult]:
    """批量回测多个方案"""
    backtester = SchemeBacktester()
    results = []
    for scheme in schemes:
        try:
            result = backtester.run(scheme, factor_df, price_df, factor_names, **kwargs)
            results.append(result)
        except Exception as e:
            logger.error(f"[MultiBacktest] {scheme.name} 失败: {e}")
    return sorted(results, key=lambda r: r.total_return, reverse=True)
