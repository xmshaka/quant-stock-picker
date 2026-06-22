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
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

from strategy.schemes import BUILTIN_SCHEMES, StrategyScheme
from signals.layers import evaluate_layered
from market.timing import MarketTimingModel
from signals.rules import TradePoint, confidence_audit, evaluate_all_rules
from signals.engine import SignalEngine
from backtest.engine import BacktestEngine, BacktestParams, estimate_turnover_amount, get_liquidity_slippage_rate
from backtest.records import (
    BacktestRunConfig, make_run_id, trade_points_to_frame,
    trade_details_to_frame, skipped_signals_to_frame, equity_curve_to_frame, persist_backtest_run,
)
from data.bars_normalizer import assert_raw_for_execution, normalize_daily_bars
from config.settings import settings
import copy


def _tp_entry_model(tp: Optional[TradePoint], fallback: str = "") -> str:
    """读取信号买点模型；缺失时使用回退值，但不伪造为已验证因子。"""
    model = str(getattr(tp, "entry_model", "") or "").strip() if tp is not None else ""
    return model or str(fallback or "").strip()


def evaluate_add_position_contract(
    pos: Dict,
    source_tp: Optional[TradePoint],
    *,
    current_price: float,
    projected_pnl: float,
    projected_pnl_pct: float,
    max_alloc: float,
    target_position_value: Optional[float] = None,
) -> Tuple[bool, str]:
    """P5 加仓执行契约：只判断是否允许加仓，不改变原持仓。

    合同口径：扣成本后盈利、原始买点结构未破、同 entry_model、新信号
    confidence 严格更高、加仓后单票不超过 20% 上限。
    """
    if projected_pnl <= 0:
        return False, f"加仓拒绝: 扣成本后未盈利(projected_pnl={projected_pnl:.2f}, pct={projected_pnl_pct:.2%})"

    old_model = str(pos.get("entry_model") or pos.get("strategy_id") or "").strip()
    new_model = _tp_entry_model(source_tp, fallback=old_model)
    if old_model and new_model and old_model != new_model:
        return False, f"加仓拒绝: entry_model不一致({old_model}!={new_model})"

    old_conf = float(pos.get("entry_confidence", 0.0) or 0.0)
    new_conf = float(getattr(source_tp, "confidence", 0.0) or 0.0) if source_tp is not None else 0.0
    if new_conf <= old_conf:
        return False, f"加仓拒绝: confidence未增强({new_conf:.4f}<={old_conf:.4f})"

    stop_loss = float(pos.get("stop_loss", 0.0) or 0.0)
    platform_low = float(pos.get("platform_low", 0.0) or 0.0)
    if stop_loss > 0 and current_price <= stop_loss:
        return False, f"加仓拒绝: 原始结构破坏(价格{current_price:.2f}<=止损{stop_loss:.2f})"
    if platform_low > 0 and current_price < platform_low:
        return False, f"加仓拒绝: 原始结构破坏(价格{current_price:.2f}<平台低点{platform_low:.2f})"

    if target_position_value is not None and target_position_value > max_alloc + 1e-6:
        return False, f"加仓拒绝: 单票仓位超20%({target_position_value:.2f}>{max_alloc:.2f})"

    return True, "加仓通过: 盈利/同模型/confidence增强/结构未破/单票上限"


def evaluate_entry_confidence_contract(confidence: float) -> Tuple[bool, float, Dict[str, object], str]:
    """开仓 confidence 执行契约。

    低置信度不再与高置信度同等开仓：
    - observe_only(weight=0): 只保留 raw signal，不进入 executed/trades。
    - reduced_or_pending(weight=0.5): 降仓执行，仍受成本、100股、单票20%约束。
    - standard/strong(weight=1): 标准执行。

    返回: (允许执行, 仓位权重, confidence审计字段, 执行说明)
    """
    audit = confidence_audit(confidence, "BUY")
    weight = float(audit.get("confidence_weight", 0.0) or 0.0)
    action = str(audit.get("confidence_action", "") or "")
    if weight <= 0 or action == "observe_only":
        return False, 0.0, audit, f"开仓观察: confidence={float(confidence or 0.0):.4f}, action={action}"
    weight = max(0.0, min(1.0, weight))
    if weight < 1.0:
        return True, weight, audit, f"开仓降仓: confidence={float(confidence or 0.0):.4f}, weight={weight:.2f}"
    return True, 1.0, audit, f"开仓标准执行: confidence={float(confidence or 0.0):.4f}"


def _fetch_ohlcv(
    symbols: list,
    lookback_days: int,
    adjust: str = "",
    start_date=None,
    end_date=None,
    warmup_days: int = 120,
) -> pd.DataFrame:
    """从腾讯数据源拉取 OHLCV，补齐快照缺失的 open/high/low/volume。

    P0 数据口径：撮合 / 默认K线买卖点必须使用不复权价格。
    - adjust=""   : 不复权，默认用于回测撮合和默认K线
    - adjust="qfq": 前复权，仅允许用于趋势展示叠加，不参与成交统计
    """
    try:
        from data.fetchers.tencent_fetcher import TencentFetcher
        from datetime import datetime, timedelta
        fetcher = TencentFetcher()
        if end_date is not None:
            end_dt = pd.Timestamp(end_date).to_pydatetime()
        else:
            end_dt = datetime.now()
        if start_date is not None:
            start_dt = pd.Timestamp(start_date).to_pydatetime() - timedelta(days=max(int(warmup_days or 0), 0))
        else:
            start_dt = end_dt - timedelta(days=lookback_days + max(int(warmup_days or 0), 60))
        end = end_dt.strftime('%Y%m%d')
        start = start_dt.strftime('%Y%m%d')
        frames = []
        for sym in symbols:
            try:
                # 确保adjust参数正确传递：空字符串表示raw
                effective_adjust = "raw" if adjust == "" else adjust
                df = fetcher.get_daily_bars(sym, start_date=start, end_date=end, adjust=effective_adjust)
                if df is not None and not df.empty:
                    # 确保数据包含source和adjust字段
                    if not {"source", "adjust"}.issubset(df.columns):
                        df = normalize_daily_bars(df, source="tencent", symbol=sym, adjust=effective_adjust)
                    # 验证回测使用raw数据
                    if effective_adjust == "raw":
                        assert_raw_for_execution(df)
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


def _fetch_ohlcv_for_backtest(symbols: list, lookback_days: int, start_date=None, end_date=None, adjust: str = "") -> pd.DataFrame:
    """按回测区间拉取含 warmup 的 OHLCV，并兼容旧测试 monkeypatch。

    历史问题：直接按“当前日期 - lookback_days”拉取，会让早期回测日期缺少
    MA40/ADX/MACD 等前置K线，导致类似 2026-03-20/04-03 的候选被 L1
    `数据不足` 误挡。这里显式使用回测 start/end。
    """
    try:
        return _fetch_ohlcv(
            symbols,
            lookback_days,
            adjust=adjust,
            start_date=start_date,
            end_date=end_date,
            warmup_days=max(120, int(lookback_days or 0)),
        )
    except TypeError:
        # tests 中大量 monkeypatch 仍使用 lambda symbols, lookback_days: ...
        return _fetch_ohlcv(symbols, lookback_days)


def _prepare_execution_bars(df: pd.DataFrame, *, fallback_source: str = "snapshot") -> pd.DataFrame:
    """回测撮合前K线硬校验与单位兜底标准化。

    回测入口只能接受 adjust=raw 的不复权K线。若输入来自旧快照且缺 adjust/source，
    为兼容历史数据默认按 raw+fallback_source 标准化；若显式标记 qfq/hfq，直接拒绝。
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    # 新版 fetcher 已经输出标准化单位；只做类型整理，严禁二次乘以100/1000。
    if {"source", "adjust"}.issubset(out.columns):
        out["trade_date"] = pd.to_datetime(out["trade_date"])
        for col in ("open", "high", "low", "close", "volume", "amount", "pct_change", "change", "turnover"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.sort_values("trade_date").reset_index(drop=True)
    else:
        if "source" not in out.columns:
            out["source"] = fallback_source
        if "adjust" not in out.columns:
            out["adjust"] = "raw"
        out = normalize_daily_bars(
            out,
            source=str(out["source"].iloc[0]),
            symbol=str(out["symbol"].iloc[0]) if "symbol" in out.columns and not out.empty else None,
            adjust=str(out["adjust"].iloc[0]),
        )
    assert_raw_for_execution(out)
    return out


EXIT_REASON_MAP = {
    'ATR止损': ('stop_loss', 'atr_hard_stop'),
    'ATR跟踪止盈': ('take_profit', 'atr_trailing_profit'),
    'ATR跟踪回撤止损': ('stop_loss', 'atr_trailing_profit_failed'),
    'ATR止盈': ('take_profit', 'atr_fixed_profit'),
    'ATR回撤止损': ('stop_loss', 'atr_profit_failed'),
    '时间止损': ('stop_loss', 'time_stop'),
    '最长持仓退出': ('time_exit', 'max_holding_days'),
    '动量失效退出': ('strategy_failure', 'trend_momentum_failed'),
    '回调破位退出': ('strategy_failure', 'pullback_breakdown'),
    '突破失败退出': ('strategy_failure', 'breakout_failed'),
    '大盘防御减仓': ('market_exit', 'market_defense'),
    'L3共振卖出': ('signal_exit', 'rule_signal'),
    '信号卖出': ('signal_exit', 'rule_signal'),
    '末日清仓': ('final_liquidation', 'end_of_backtest'),
}


def classify_exit_reason(rule_name: str = "", reason: str = "") -> tuple[str, str]:
    """将自由文本退出原因归一到审计字段。

    P0 要求回测记录里有可机器检索的 exit_type / exit_subtype；UI 仍保留
    reason/rule_name 供人工阅读。
    """
    rn = str(rule_name or "")
    rs = str(reason or "")
    for key, value in EXIT_REASON_MAP.items():
        if key in rn or key in rs:
            return value
    if '止损' in rn or '止损' in rs:
        return 'stop_loss', 'generic_stop_loss'
    if '止盈' in rn or '止盈' in rs:
        return 'take_profit', 'generic_take_profit'
    if '信号' in rn or '信号' in rs:
        return 'signal_exit', 'generic_signal'
    return 'other_exit', 'manual_or_unknown'


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
    
    # 数据来源追踪
    data_source: str = ""
    data_adjust: str = "raw"
    data_version: str = ""
    
    run_id: str = ""
    # 事件源（P0：K线默认展示 signals_executed）
    signals_executed: Dict[str, List[TradePoint]] = field(default_factory=dict)
    signals_raw: Dict[str, List[TradePoint]] = field(default_factory=dict)
    # 详情
    equity_curve: Dict[str, float] = field(default_factory=dict)
    trade_details: List[Dict] = field(default_factory=list)
    skipped_signals: List[Dict] = field(default_factory=list)
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
            f"数据源: {self.data_source}/{self.data_adjust}",
            f"数据版本: {self.data_version}",
        ]
        return "\n".join(lines)

    def persist(self, config=None, **overrides) -> Path:
        """持久化回测结果到 parquet。"""
        from backtest.records import (
            BacktestRunConfig, trade_points_to_frame,
            trade_details_to_frame, equity_curve_to_frame, persist_backtest_run, scheme_audit_snapshot,
        )
        if config is None:
            scheme_snapshot = scheme_audit_snapshot(BUILTIN_SCHEMES.get(self.scheme_id))
            config = BacktestRunConfig(
                run_id=self.run_id,
                scheme_id=self.scheme_id,
                scheme_name=self.scheme_name,
                start_date=self.start_date,
                end_date=self.end_date,
                lookback_days=0,
                top_n=0,
                initial_capital=1_000_000.0,
                scheme_config=scheme_snapshot["scheme_config"],
                resonance_config=scheme_snapshot["resonance_config"],
            )
        # P0: trades.parquet 以真实成交明细为准，避免丢失成本/PnL字段
        trades_df = trade_details_to_frame(self.trade_details, run_id=config.run_id, source="executed")
        raw_df = trade_points_to_frame(self.signals_raw, source="raw_rule")
        executed_df = trade_points_to_frame(self.signals_executed, source="executed")
        skipped_df = skipped_signals_to_frame(self.skipped_signals, run_id=config.run_id)
        equity_df = equity_curve_to_frame(self.equity_curve, config.run_id)
        return persist_backtest_run(
            result=self, config=config,
            trades=trades_df, signals_raw=raw_df, signals_executed=executed_df,
            skipped_signals=skipped_df,
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
        enable_entry_confidence_contract: bool = True,
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
        enable_market_timing = scheme.enable_market_timing if scheme else True
        exit_cfg = getattr(scheme, 'exit_config', None)
        max_holding_days = int(getattr(exit_cfg, 'max_holding_days', 20) or 20)
        time_stop_days = int(getattr(exit_cfg, 'time_stop_days', 7) or 7)
        time_stop_min_profit_pct = float(getattr(exit_cfg, 'time_stop_min_profit_pct', 0.0) or 0.0)
        failure_window_days = int(getattr(exit_cfg, 'failure_window_days', 3) or 3)
        market_defense_score = float(getattr(exit_cfg, 'market_defense_score', 20.0) or 20.0)
        enable_market_defense_exit = bool(getattr(exit_cfg, 'enable_market_defense_exit', True))
        enable_strategy_failure_exit = bool(getattr(exit_cfg, 'enable_strategy_failure_exit', True))
        enable_trailing_exit = bool(getattr(exit_cfg, 'enable_trailing_exit', True))
        enable_time_stop = bool(getattr(exit_cfg, 'enable_time_stop', True))
        enable_max_holding_exit = bool(getattr(exit_cfg, 'enable_max_holding_exit', True))
        trailing_activation_pct = float(getattr(exit_cfg, 'trailing_activation_pct', 0.05) or 0.0)
        trailing_activation_atr_mult = float(getattr(exit_cfg, 'trailing_activation_atr_mult', 1.0) or 0.0)

        # ── 大盘择时 ──
        market_timing = None
        if enable_market_timing:
            try:
                start_str = start_date.strftime('%Y%m%d') if hasattr(start_date, 'strftime') else str(start_date).replace('-', '')
                end_str = end_date.strftime('%Y%m%d') if hasattr(end_date, 'strftime') else str(end_date).replace('-', '')
                market_timing = MarketTimingModel()
                market_timing.fetch_all(start_str, end_str)
                logger.info(f"[MarketTiming] 大盘择时已启用")
            except Exception as e:
                logger.warning(f"[MarketTiming] 初始化失败，回退到固定仓位: {e}")
                market_timing = None

        # 拉取 OHLCV
        ohlcv_df = _fetch_ohlcv_for_backtest(symbols, lookback_days, start_date=start_date, end_date=end_date)
        have_ohlcv = not ohlcv_df.empty
        if have_ohlcv:
            ohlcv_df['trade_date'] = pd.to_datetime(ohlcv_df['trade_date'])
        
        # ========== 关键修复：确保price_df包含source和adjust字段 ==========
        if not price_df.empty:
            # 如果price_df没有source字段，添加默认值
            if 'source' not in price_df.columns:
                logger.warning(f"[SingleStockFix] price_df缺少source字段，添加默认值'tencent'")
                price_df['source'] = 'tencent'
            # 如果source字段存在但全部为空，填充默认值
            elif price_df['source'].isna().all() or (price_df['source'] == '').all():
                logger.warning(f"[SingleStockFix] price_df.source字段全部为空，填充为'tencent'")
                price_df['source'] = 'tencent'
            
            # 如果price_df没有adjust字段，添加默认值
            if 'adjust' not in price_df.columns:
                logger.warning(f"[SingleStockFix] price_df缺少adjust字段，添加默认值'raw'")
                price_df['adjust'] = 'raw'
            # 如果adjust字段存在但全部为空，填充默认值
            elif price_df['adjust'].isna().all() or (price_df['adjust'] == '').all():
                logger.warning(f"[SingleStockFix] price_df.adjust字段全部为空，填充为'raw'")
                price_df['adjust'] = 'raw'
        # ===============================================================

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
            sym_bars = _prepare_execution_bars(sym_bars, fallback_source="scheme_backtest")
            sym_bars_full = _prepare_execution_bars(sym_bars_full, fallback_source="scheme_backtest")
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

            signal_mode = scheme.signal_mode if scheme else "layered"
            scheme_type = scheme.scheme_id if scheme else "balanced"
            if signal_mode == "layered":
                # FIX: 必须用 sym_bars_full（含前置60天数据）否则 TrendFilter(ma_long=40) 永远失败
                points = evaluate_layered(sym_bars_full, strategy_type=scheme_type)
                # 过滤到回测区间内
                start_ts = pd.Timestamp(start_date)
                end_ts = pd.Timestamp(end_date)
                points = [p for p in points
                          if start_ts <= pd.Timestamp(p.date) <= end_ts]
            else:
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
        skipped_signals: List[Dict] = [] # raw有信号但未执行的审计明细

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
        trading_date_keys = [dt.date() if hasattr(dt, 'date') else dt for dt in date_list]
        trading_day_index = {dt_key: idx for idx, dt_key in enumerate(trading_date_keys)}
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

        def _project_sell_pnl(sym, dt_key, price, pos):
            """按当前撮合价预估卖出净盈亏，含滑点/佣金/印花税/过户费。

            用于止盈类规则触发前的业务条件校验：只有真实成交后仍能覆盖成本，
            才允许归因为“止盈/跟踪止盈”。
            """
            slippage_rate, _, _ = _slippage_info(sym, dt_key)
            exec_price = price * (1 - slippage_rate)
            revenue = pos['shares'] * exec_price
            commission = max(revenue * COMMISSION, MIN_COMMISSION)
            stamp = revenue * STAMP_DUTY
            transfer_fee = revenue * TRANSFER_FEE
            net = revenue - commission - stamp - transfer_fee
            pnl = net - pos['shares'] * pos['avg_cost']
            pnl_pct = pnl / (pos['shares'] * pos['avg_cost']) if pos['avg_cost'] > 0 else 0
            return pnl, pnl_pct, exec_price

        def _breakeven_sell_price(sym, dt_key, pos):
            """卖出侧盈亏平衡价近似值，含卖出佣金下限、印花税、过户费、默认滑点。

            这里用于判断 trailing_stop 是否已经抬升到“盈利保护区”。实际成交前仍会
            用 _project_sell_pnl 做最终净收益校验，避免把亏损交易归因为止盈。
            """
            shares = max(int(pos.get('shares', 0) or 0), 0)
            if shares <= 0:
                return float('inf')
            cost_value = shares * float(pos.get('avg_cost', 0.0) or 0.0)
            slippage_rate, _, _ = _slippage_info(sym, dt_key)
            fixed_commission_per_share = MIN_COMMISSION / shares
            variable_sell_cost = STAMP_DUTY + TRANSFER_FEE + slippage_rate
            denom = max(1.0 - variable_sell_cost, 1e-9)
            return (cost_value / shares + fixed_commission_per_share) / denom

        def _holding_days(dt_key, pos):
            """持仓天数按交易日计数，不按自然日。

            买入执行日记为第0个持仓交易日；后续每过一个回测交易日+1。
            用于时间止损、最长持仓退出、策略失败窗口以及交易明细审计。
            """
            entry_dt = pos.get('entry_dt')
            if entry_dt is None:
                return 0
            if dt_key in trading_day_index and entry_dt in trading_day_index:
                return max(trading_day_index[dt_key] - trading_day_index[entry_dt], 0)
            return (dt_key - entry_dt).days if hasattr(dt_key - entry_dt, 'days') else 0

        def _unrealized_pnl_pct(sym, dt_key, price, pos):
            pnl, pnl_pct, _ = _project_sell_pnl(sym, dt_key, price, pos)
            return pnl, pnl_pct

        def _market_defense_exit(dt_key):
            if not enable_market_defense_exit:
                return False
            if market_timing is None:
                return False
            try:
                return float(market_timing.score_on(dt_key)) < market_defense_score
            except Exception:
                return False

        def _strategy_failure_exit(sym, dt_key, open_price, pos):
            """P2: 策略专属失败退出，全部只用截至当日开盘前已知/当日开盘价信息。"""
            if not enable_strategy_failure_exit:
                return None
            holding_days = _holding_days(dt_key, pos)
            if holding_days > failure_window_days:
                return None
            bars = sym_bars_map.get(sym, pd.DataFrame())
            if bars.empty:
                return None
            hist = bars[pd.to_datetime(bars['trade_date']).dt.date < dt_key].copy()
            if len(hist) < 20:
                return None
            close = hist['close'].astype(float)
            ma20 = float(close.rolling(20).mean().iloc[-1])
            low20 = float(close.iloc[-20:].min())
            if scheme_type == 'trend_momentum' and open_price < ma20:
                return f'动量失效退出(跌破MA20 {ma20:.2f})', '动量失效退出', ma20
            if scheme_type == 'pullback' and open_price < low20:
                return f'回调破位退出(跌破20日低点 {low20:.2f})', '回调破位退出', low20
            if scheme_type == 'breakout':
                platform_high = pos.get('platform_high', 0.0)
                if platform_high > 0 and open_price < platform_high:
                    return f'突破失败退出(跌回平台 {platform_high:.2f})', '突破失败退出', platform_high
            return None

        def _time_exit_decision(sym, dt_key, open_price, pos):
            holding_days = _holding_days(dt_key, pos)
            projected_pnl, projected_pct = _unrealized_pnl_pct(sym, dt_key, open_price, pos)
            if enable_max_holding_exit and holding_days >= max_holding_days:
                return '最长持仓退出', '最长持仓退出', open_price, projected_pnl
            if enable_time_stop and holding_days >= time_stop_days and projected_pct < time_stop_min_profit_pct:
                return (
                    f'时间止损({holding_days}日收益{projected_pct:.2%}未达{time_stop_min_profit_pct:.2%})',
                    '时间止损', open_price, projected_pnl,
                )
            return None

        def _trailing_activated(pos):
            """跟踪止盈激活区间：最高浮盈达标后才允许检查跟踪退出。"""
            if not enable_trailing_exit:
                return False
            avg_cost = float(pos.get('avg_cost', 0.0) or 0.0)
            highest = float(pos.get('highest', 0.0) or 0.0)
            atr_val = float(pos.get('atr', 0.0) or 0.0)
            if avg_cost <= 0 or highest <= 0:
                return False
            pct_ok = trailing_activation_pct <= 0 or highest >= avg_cost * (1 + trailing_activation_pct)
            atr_ok = atr_val > 0 and (trailing_activation_atr_mult <= 0 or highest >= avg_cost + trailing_activation_atr_mult * atr_val)
            return pct_ok or atr_ok

        def _sell(sym, dt_key, price, reason, rule_name, signal_date=None, trigger_price=None, projected_pnl=None):
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
            # FIX: 止盈类规则必须以扣除完整交易成本后的净收益为准。
            # 若跳空/滑点/费用导致实际亏损，不能把交易归因为“止盈”。
            if pnl <= 0 and ('止盈' in str(reason) or '止盈' in str(rule_name)):
                if '跟踪' in str(reason) or '跟踪' in str(rule_name):
                    reason = f'跟踪止盈失效-回撤止损(最高{pos.get("highest", price):.2f})'
                    rule_name = 'ATR跟踪回撤止损'
                else:
                    reason = f'止盈触发失败-回撤止损({price:.2f})'
                    rule_name = 'ATR回撤止损'
            exit_type, exit_subtype = classify_exit_reason(rule_name, reason)
            trigger_price = price if trigger_price is None else trigger_price
            projected_pnl = pnl if projected_pnl is None else projected_pnl
            conf_audit = confidence_audit(1.0, "SELL")
            trades_pnl.append(pnl)
            cash += net
            sell_count += 1
            holding_days = _holding_days(dt_key, pos)
            tp_out = TradePoint(
                date=dt_key, action='SELL', reason=reason,
                confidence=1.0, price=exec_price, rule_name=rule_name,
                exec_price=exec_price, shares=pos['shares'],
                cash_after=cash, position_shares=0,
                avg_cost=pos['avg_cost'],
                pnl=pnl, pnl_pct=pnl_pct, holding_days=holding_days,
                signal_date=signal_date or '',
                exec_date=dt_key,
                exit_type=exit_type,
                exit_subtype=exit_subtype,
                trigger_price=trigger_price,
                projected_pnl=projected_pnl,
                confidence_bucket=str(conf_audit['confidence_bucket']),
                confidence_action=str(conf_audit['confidence_action']),
                confidence_weight=float(conf_audit['confidence_weight']),
                confidence_note=str(conf_audit['confidence_note']),
            )
            actual_trades.append((sym, tp_out))
            trade_details.append({
                'symbol': sym, 'date': dt_key, 'action': 'SELL',
                'exec_date': dt_key, 'signal_date': signal_date or '',
                'price': exec_price, 'exec_price': exec_price, 'shares': pos['shares'],
                'cost': pos['avg_cost'], 'pnl': pnl, 'pnl_pct': pnl_pct,
                'commission': commission, 'stamp_duty': stamp,
                'transfer_fee': transfer_fee, 'slippage': slippage_cost,
                'slippage_rate': slippage_rate,
                'liquidity_bucket': liquidity_bucket,
                'turnover_amount': turnover_amount,
                'cash_after': cash, 'reason': reason,
                'rule_name': rule_name,
                'exit_type': exit_type,
                'exit_subtype': exit_subtype,
                'trigger_price': trigger_price,
                'projected_pnl': projected_pnl,
                'holding_days': holding_days,
                'entries': list(pos.get('entries', [])),
                'confidence': 1.0,
                'confidence_bucket': conf_audit['confidence_bucket'],
                'confidence_action': conf_audit['confidence_action'],
                'confidence_weight': conf_audit['confidence_weight'],
                'confidence_note': conf_audit['confidence_note'],
            })
            if verbose:
                logger.info(f"  [{dt_key}] 卖出 {sym} {pos['shares']}股 @ {price:.2f} 盈亏={pnl:+.0f} ({reason})")
            return tp_out

        def _entry_audit_from_source(source_tp=None):
            fields = [
                'entry_model', 'main_trigger', 'confirmations', 'factor_evidence', 'market_context',
                'fund_flow_context', 'technical_confirmations', 'veto_checks', 'risk_tags', 'missing_fields',
            ]
            return {f: str(getattr(source_tp, f, '') or '') for f in fields}

        def _record_skipped_signal(sym, dt_key, tp, stage, note):
            """记录 raw 信号未进入执行层的原因，不改变交易结果。"""
            conf_audit = confidence_audit(float(getattr(tp, 'confidence', 0.0) or 0.0), getattr(tp, 'action', 'BUY'))
            entry_audit = _entry_audit_from_source(tp)
            skipped_signals.append({
                'symbol': sym,
                'signal_date': getattr(tp, 'date', '') or '',
                'exec_date': dt_key,
                'action': getattr(tp, 'action', ''),
                'skip_stage': stage,
                'skip_reason': note,
                'reason': getattr(tp, 'reason', '') or '',
                'rule_name': getattr(tp, 'rule_name', '') or '',
                'signal_price': float(getattr(tp, 'price', 0.0) or 0.0),
                'confidence': float(getattr(tp, 'confidence', 0.0) or 0.0),
                'confidence_bucket': conf_audit['confidence_bucket'],
                'confidence_action': conf_audit['confidence_action'],
                'confidence_weight': conf_audit['confidence_weight'],
                'confidence_note': conf_audit['confidence_note'],
                **entry_audit,
            })

        def _buy(sym, dt_key, price, reason, rule_name, confidence=1.0, is_add=False, signal_date=None, source_tp=None, confidence_weight=None):
            """统一买入/加仓逻辑，含资金校验"""
            nonlocal cash, buy_count
            conf_audit = confidence_audit(confidence, "ADD" if is_add else "BUY")
            entry_audit = _entry_audit_from_source(source_tp)
            total_value = cash + sum(
                p['shares'] * close_lookup.get(s, {}).get(dt_key, p['avg_cost'])
                for s, p in positions.items()
            )
            max_alloc = total_value * max_single_pct

            if is_add:
                # 加仓：受单股上限约束，同样受大盘择时调制
                existing = positions[sym]
                current_val = existing['shares'] * price
                remaining = max_alloc - current_val
                market_mult = market_timing.position_multiplier_on(dt_key) if market_timing else 0.78
                alloc = min(cash * pos_pct_per_entry * market_mult, remaining)
            else:
                # 建仓
                # ── 大盘择时仓位调制 ──
                market_mult = market_timing.position_multiplier_on(dt_key) if market_timing else 0.78
                weight = float(confidence_weight if confidence_weight is not None else conf_audit.get('confidence_weight', 1.0) or 0.0)
                weight = max(0.0, min(1.0, weight))
                alloc = min(cash * pos_pct_per_entry * market_mult * weight, max_alloc)

            shares = int(alloc / price / 100) * 100
            if shares <= 0:
                if verbose:
                    logger.info(f"  [{dt_key}] 跳过 {sym} 余额不足 (cash={cash:.0f}, alloc={alloc:.0f})")
                return None

            slippage_rate, liquidity_bucket, turnover_amount = _slippage_info(sym, dt_key)
            exec_price = price * (1 + slippage_rate)  # 买入滑点：成交价略高
            if is_add:
                existing = positions[sym]
                strict_remaining = max_alloc - existing['shares'] * exec_price
                max_add_shares = int(strict_remaining / exec_price / 100) * 100
                if max_add_shares <= 0:
                    if verbose:
                        logger.info(f"  [{dt_key}] 跳过 {sym} 加仓后将超过单票20%上限")
                    return None
                if shares > max_add_shares:
                    shares = max_add_shares
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
                existing['entry_confidence'] = float(confidence)
                existing['last_add_confidence'] = float(confidence)
                existing['entries'].append({
                    'date': dt_key, 'price': exec_price, 'shares': shares, 'reason': reason,
                    'confidence': float(confidence),
                    'confidence_bucket': conf_audit['confidence_bucket'],
                    'confidence_action': conf_audit['confidence_action'],
                    'confidence_note': conf_audit['confidence_note'],
                })
                # 止盈止损用加权ATR
                old_atr = existing.get('atr', atr_val)
                old_shares = max(new_shares - shares, 0)
                existing['atr'] = (old_atr * old_shares + atr_val * shares) / new_shares if new_shares > 0 else atr_val
                existing['stop_loss'] = new_avg - sl_mult * existing['atr']
                existing['take_profit'] = new_avg + tp_mult * existing['atr']
            else:
                stop_loss = price - sl_mult * atr_val if atr_val > 0 else price * 0.92
                take_profit = price + tp_mult * atr_val if atr_val > 0 else price * 1.15
                hist = sym_bars_map.get(sym, pd.DataFrame())
                platform_high = 0.0
                platform_low = 0.0
                if not hist.empty:
                    pre = hist[pd.to_datetime(hist['trade_date']).dt.date < dt_key].tail(15)
                    if not pre.empty:
                        platform_high = float(pre['close'].astype(float).max())
                        platform_low = float(pre['close'].astype(float).min())
                positions[sym] = {
                    'shares': shares, 'avg_cost': exec_price, 'entry_dt': dt_key,
                    'highest': price, 'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'trailing_stop': stop_loss,
                    'atr': atr_val, 'add_count': 0,
                    'entry_confidence': float(confidence),
                    'entry_confidence_bucket': conf_audit['confidence_bucket'],
                    'entry_confidence_action': conf_audit['confidence_action'],
                    'entry_model': entry_audit.get('entry_model') or scheme_type,
                    'strategy_id': scheme_type,
                    'platform_high': platform_high,
                    'platform_low': platform_low,
                    'entries': [{
                        'date': dt_key, 'price': exec_price, 'shares': shares, 'reason': reason,
                        'confidence': float(confidence),
                        'confidence_bucket': conf_audit['confidence_bucket'],
                        'confidence_action': conf_audit['confidence_action'],
                        'confidence_note': conf_audit['confidence_note'],
                    }],
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
                trailing_stop=pos['trailing_stop'],
                signal_date=signal_date or '',
                exec_date=dt_key,
                confidence_bucket=str(conf_audit['confidence_bucket']),
                confidence_action=str(conf_audit['confidence_action']),
                confidence_weight=float(conf_audit['confidence_weight']),
                confidence_note=str(conf_audit['confidence_note']),
                **entry_audit,
            )
            actual_trades.append((sym, tp_out))
            trade_details.append({
                # FIX:P0: 加仓也属于 BUY 执行事件，K线/明细/统计必须一致
                'symbol': sym, 'date': dt_key, 'action': 'BUY', 'event_type': 'ADD' if is_add else 'BUY',
                'exec_date': dt_key, 'signal_date': signal_date or '',
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
                'trailing_stop': pos['trailing_stop'],
                'confidence': float(confidence),
                'confidence_bucket': conf_audit['confidence_bucket'],
                'confidence_action': conf_audit['confidence_action'],
                'confidence_weight': conf_audit['confidence_weight'],
                'confidence_note': conf_audit['confidence_note'],
                **entry_audit,
            })
            if verbose:
                tag = '加仓' if is_add else '买入'
                logger.info(f"  [{dt_key}] {tag} {sym} {shares}股 @ {price:.2f} "
                           f"持仓={pos['shares']} 均价={pos['avg_cost']:.2f} "
                           f"止损={pos['stop_loss']:.2f} 跟止={pos['trailing_stop']:.2f} 止盈={pos['take_profit']:.2f}")
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
                    projected_pnl, _, _ = _project_sell_pnl(sym, dt_key, open_price, pos)
                    _sell(sym, dt_key, open_price, f'止损({pos["stop_loss"]:.2f})', 'ATR止损',
                          trigger_price=pos['stop_loss'], projected_pnl=projected_pnl)
                    continue
                # 大盘防御减仓：低回撤优先，进入防御档后退出持仓并保留审计字段。
                if _market_defense_exit(dt_key):
                    projected_pnl, _, _ = _project_sell_pnl(sym, dt_key, open_price, pos)
                    _sell(sym, dt_key, open_price, f'大盘防御减仓(评分<{market_defense_score:.0f})', '大盘防御减仓',
                          trigger_price=open_price, projected_pnl=projected_pnl)
                    continue
                # 策略失败退出：动量跌破、回调破位、突破失败等短线失效信号。
                failure = _strategy_failure_exit(sym, dt_key, open_price, pos)
                if failure:
                    reason, rule_name, trigger = failure
                    projected_pnl, _, _ = _project_sell_pnl(sym, dt_key, open_price, pos)
                    _sell(sym, dt_key, open_price, reason, rule_name,
                          trigger_price=trigger, projected_pnl=projected_pnl)
                    continue
                # 跟踪止盈：只有 trailing_stop 已进入扣成本后的盈利保护区，且当日实际
                # 开盘撮合价预估净收益仍为正，才归因为“止盈”。
                # 若跳空跌破跟踪线导致扣成本后亏损，仍应退出，但归因为“ATR跟踪回撤止损”，
                # 不能被后面的时间止损掩盖真实价格触发原因。
                if _trailing_activated(pos) and open_price <= pos['trailing_stop'] and pos['highest'] > pos['avg_cost']:
                    projected_pnl, _, _ = _project_sell_pnl(sym, dt_key, open_price, pos)
                    if pos['trailing_stop'] > _breakeven_sell_price(sym, dt_key, pos):
                        _sell(sym, dt_key, open_price, f'跟踪止盈(最高{pos["highest"]:.2f})', 'ATR跟踪止盈',
                              trigger_price=pos['trailing_stop'], projected_pnl=projected_pnl)
                    continue
                # 固定止盈
                if open_price >= pos['take_profit']:
                    projected_pnl, _, _ = _project_sell_pnl(sym, dt_key, open_price, pos)
                    if projected_pnl > 0:
                        _sell(sym, dt_key, open_price, f'止盈({pos["take_profit"]:.2f})', 'ATR止盈',
                              trigger_price=pos['take_profit'], projected_pnl=projected_pnl)
                    continue
                # 时间止损 / 最长持仓退出：短线系统不长期占用资金。
                # 放在价格类止盈/止损后，避免掩盖更具体的触发原因。
                time_exit = _time_exit_decision(sym, dt_key, open_price, pos)
                if time_exit:
                    reason, rule_name, trigger, projected_pnl = time_exit
                    _sell(sym, dt_key, open_price, reason, rule_name,
                          trigger_price=trigger, projected_pnl=projected_pnl)
                    continue

            # 2. 执行今日的待处理信号
            actions = pending_actions.get(dt_key, {})
            for sym, tp in actions.items():
                # FIX: 信号在 T 日收盘后产生，只能在 T+1 开盘/撮合价执行；
                # 禁止使用 T+1 收盘价回看成交，避免未来函数和成交价失真。
                price = open_lookup.get(sym, {}).get(dt_key, 0)
                if price <= 0:
                    continue

                if tp.action == 'SELL' and sym in positions:
                    projected_pnl, _, _ = _project_sell_pnl(sym, dt_key, price, positions[sym])
                    _sell(sym, dt_key, price, tp.reason or '信号卖出', tp.rule_name or '信号卖出',
                          signal_date=getattr(tp, 'date', None), trigger_price=getattr(tp, 'price', price), projected_pnl=projected_pnl)

                elif tp.action == 'BUY':
                    if sym in positions:
                        # 已持仓 → 加仓（检查次数上限）
                        if positions[sym].get('add_count', 0) < max_add_times:
                            projected_pnl, projected_pnl_pct, _ = _project_sell_pnl(sym, dt_key, price, positions[sym])
                            total_value = cash + sum(
                                p['shares'] * close_lookup.get(s, {}).get(dt_key, p['avg_cost'])
                                for s, p in positions.items()
                            )
                            max_alloc = total_value * max_single_pct
                            current_value = positions[sym]['shares'] * price
                            add_ok, add_note = evaluate_add_position_contract(
                                positions[sym], tp,
                                current_price=price,
                                projected_pnl=projected_pnl,
                                projected_pnl_pct=projected_pnl_pct,
                                max_alloc=max_alloc,
                                target_position_value=current_value,
                            )
                            if not add_ok:
                                if verbose:
                                    logger.info(f"  [{dt_key}] {sym} {add_note}")
                                continue
                            result_tp = _buy(sym, dt_key, price, tp.reason or '加仓', tp.rule_name or '加仓',
                                            confidence=tp.confidence, is_add=True, signal_date=getattr(tp, 'date', None), source_tp=tp)
                            if result_tp:
                                positions[sym]['add_count'] = positions[sym].get('add_count', 0) + 1
                        elif verbose:
                            logger.info(f"  [{dt_key}] {sym} 已达最大加仓次数({max_add_times})")
                    else:
                        # 新建仓
                        if enable_entry_confidence_contract:
                            entry_ok, entry_weight, _, entry_note = evaluate_entry_confidence_contract(tp.confidence)
                            if not entry_ok:
                                _record_skipped_signal(sym, dt_key, tp, 'entry_confidence_contract', entry_note)
                                if verbose:
                                    logger.info(f"  [{dt_key}] {sym} {entry_note}")
                                continue
                            if verbose and entry_weight < 1.0:
                                logger.info(f"  [{dt_key}] {sym} {entry_note}")
                        else:
                            entry_weight = 1.0
                        _buy(sym, dt_key, price, tp.reason or '信号买入', tp.rule_name or '信号买入',
                             confidence=tp.confidence, signal_date=getattr(tp, 'date', None), source_tp=tp,
                             confidence_weight=entry_weight)

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
            projected_pnl, _, _ = _project_sell_pnl(sym, final_dt, price, positions[sym])
            _sell(sym, final_dt, price, '末日清仓', '末日清仓', trigger_price=price, projected_pnl=projected_pnl)

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
        
        # 提取数据源信息（单股模式）
        data_source = ""
        data_adjust = "raw"
        data_version = "single_stock_mode"
        
        # ========== 调试日志：追踪数据源 ==========
        logger.info(f"[SingleStockDebug] 单股模式数据源追踪:")
        logger.info(f"  have_ohlcv: {have_ohlcv}")
        logger.info(f"  price_df空: {price_df.empty}")
        if not price_df.empty:
            logger.info(f"  price_df字段: {list(price_df.columns)}")
            if 'source' in price_df.columns:
                source_values = price_df['source'].unique()
                logger.info(f"  price_df.source值: {source_values[:5] if len(source_values) > 0 else '空'}")
                logger.info(f"  price_df.source非空: {not price_df['source'].isna().all()}")
            else:
                logger.warning(f"  price_df没有source字段")
            
            if 'adjust' in price_df.columns:
                adjust_values = price_df['adjust'].unique()
                logger.info(f"  price_df.adjust值: {adjust_values[:5] if len(adjust_values) > 0 else '空'}")
                logger.info(f"  price_df.adjust非空: {not price_df['adjust'].isna().all()}")
            else:
                logger.warning(f"  price_df没有adjust字段")
        
        if have_ohlcv:
            logger.info(f"  ohlcv_df空: {ohlcv_df.empty}")
            if not ohlcv_df.empty:
                logger.info(f"  ohlcv_df字段: {list(ohlcv_df.columns)}")
                if 'source' in ohlcv_df.columns:
                    source_values = ohlcv_df['source'].unique()
                    logger.info(f"  ohlcv_df.source值: {source_values[:5] if len(source_values) > 0 else '空'}")
                    logger.info(f"  ohlcv_df.source非空: {not ohlcv_df['source'].isna().all()}")
                else:
                    logger.warning(f"  ohlcv_df没有source字段")
        # =======================================
        
        # 尝试从price_df提取数据源
        if not price_df.empty:
            if 'source' in price_df.columns and not price_df['source'].isna().all():
                data_source = str(price_df.iloc[0]['source'])
                logger.info(f"[SingleStockDebug] 从price_df提取source: {data_source}")
            else:
                logger.warning(f"[SingleStockDebug] price_df没有source字段或全部为空")
                
            if 'adjust' in price_df.columns and not price_df['adjust'].isna().all():
                data_adjust = str(price_df.iloc[0]['adjust'])
                logger.info(f"[SingleStockDebug] 从price_df提取adjust: {data_adjust}")
            else:
                logger.warning(f"[SingleStockDebug] price_df没有adjust字段或全部为空")
                
            data_version = f"source={data_source}, adjust={data_adjust}, single_stock_mode"
            logger.info(f"[SingleStockDebug] 最终data_version: {data_version}")
        
        # 如果使用了ohlcv_df，也从中提取
        elif have_ohlcv and not ohlcv_df.empty:
            if 'source' in ohlcv_df.columns and not ohlcv_df['source'].isna().all():
                data_source = str(ohlcv_df.iloc[0]['source'])
                logger.info(f"[SingleStockDebug] 从ohlcv_df提取source: {data_source}")
            else:
                logger.warning(f"[SingleStockDebug] ohlcv_df没有source字段或全部为空")
                
            if 'adjust' in ohlcv_df.columns and not ohlcv_df['adjust'].isna().all():
                data_adjust = str(ohlcv_df.iloc[0]['adjust'])
                logger.info(f"[SingleStockDebug] 从ohlcv_df提取adjust: {data_adjust}")
            else:
                logger.warning(f"[SingleStockDebug] ohlcv_df没有adjust字段或全部为空")
                
            data_version = f"source={data_source}, adjust={data_adjust}, single_stock_mode"
            logger.info(f"[SingleStockDebug] 最终data_version: {data_version}")

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
            skipped_signals=skipped_signals,
            data_source=data_source,
            data_adjust=data_adjust,
            data_version=data_version,
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
        enable_entry_confidence_contract: bool = True,
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
            enable_entry_confidence_contract: 是否启用开仓 confidence 执行契约；关闭仅用于 A/B 对比旧口径
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
                enable_entry_confidence_contract=enable_entry_confidence_contract,
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
                data_source="", data_adjust="raw", data_version="no_signals",
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

            ohlcv_df = _fetch_ohlcv_for_backtest(list(all_selected), lookback_days, start_date=start_date, end_date=end_date)
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
                # 保留完整数据用于信号生成（TrendFilter 需要 ≥60 根 bar）
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
                    if col not in sym_bars_full.columns:
                        sym_bars_full[col] = sym_bars_full['close']
                sym_bars = _prepare_execution_bars(sym_bars, fallback_source="scheme_backtest")
                sym_bars_full = _prepare_execution_bars(sym_bars_full, fallback_source="scheme_backtest")
                signal_mode = getattr(scheme, 'signal_mode', 'layered')
                if signal_mode == "layered":
                    # FIX: 用完整数据生成信号（前置60天），然后过滤到回测区间
                    points = evaluate_layered(sym_bars_full, strategy_type=scheme.scheme_id)
                    start_ts = pd.Timestamp(start_date)
                    end_ts = pd.Timestamp(end_date)
                    points = [p for p in points
                              if start_ts <= pd.Timestamp(p.date) <= end_ts]
                else:
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
            sym_data = _prepare_execution_bars(sym_data, fallback_source="scheme_backtest")
            sym_data = sym_data.set_index('trade_date')[['open', 'high', 'low', 'close', 'volume', 'amount']]
            sym_data.index = pd.to_datetime(sym_data.index)
            engine.add_data(sym, sym_data)
            added_symbols.add(sym)

        if not added_symbols:
            logger.warning("[SchemeBacktest] 无可用股票数据")
            return SchemeBacktestResult(
                scheme_id=scheme.scheme_id, scheme_name=scheme.name,
                start_date=str(start_date), end_date=str(end_date),
                data_source="", data_adjust="raw", data_version="no_data",
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
                data_source="", data_adjust="raw", data_version="backtrader_failed",
            )

        # 5. 汇总结果
        
        # 提取数据源信息
        data_source = ""
        data_adjust = "raw"
        data_version = ""
        
        # 尝试从ohlcv_df提取数据源
        if have_ohlcv and not ohlcv_df.empty:
            # 从第一个有数据的记录中提取
            first_row = ohlcv_df.iloc[0]
            if 'source' in ohlcv_df.columns and not ohlcv_df['source'].isna().all():
                data_source = str(first_row['source'])
            if 'adjust' in ohlcv_df.columns and not ohlcv_df['adjust'].isna().all():
                data_adjust = str(first_row['adjust'])
            data_version = f"source={data_source}, adjust={data_adjust}"
        elif not price_df.empty:
            # 单股模式从price_df提取
            if 'source' in price_df.columns and not price_df['source'].isna().all():
                data_source = str(price_df.iloc[0]['source'])
            if 'adjust' in price_df.columns and not price_df['adjust'].isna().all():
                data_adjust = str(price_df.iloc[0]['adjust'])
            data_version = f"source={data_source}, adjust={data_adjust}"
        else:
            # 没有数据，记录异常情况
            data_version = f"empty_factor_panel"
        
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
            data_source=data_source,
            data_adjust=data_adjust,
            data_version=data_version,
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
