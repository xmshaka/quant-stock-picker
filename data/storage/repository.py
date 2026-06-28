"""数据仓库 - 封装数据库操作"""
from typing import List, Optional, Tuple
from datetime import date, timedelta
from contextlib import contextmanager

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc
from sqlalchemy.dialects.postgresql import insert as pg_insert
from loguru import logger

from .models import (
    get_session_factory, StockInfo, StockBar, FactorValue,
    SectorData, BacktestResult, FactorIC
)


class BaseRepository:
    """基础仓库类"""
    
    def __init__(self):
        self.session_factory = get_session_factory()
    
    @contextmanager
    def session(self):
        """会话上下文管理器"""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"数据库操作失败: {e}")
            raise
        finally:
            session.close()


class StockRepository(BaseRepository):
    """股票数据仓库"""
    
    def save_bars(self, df: pd.DataFrame, chunk_size: int = 2000) -> int:
        """批量保存行情数据 (PostgreSQL ON CONFLICT 走 UPSERT)

        - 新表结构：唯一键 (symbol, trade_date, source, adjust)
        - 如果 source/adjust 缺失，默认 source='', adjust='raw'
        - 按 chunk_size 分批避免单条 SQL 过大

        Returns:
            实际写入行数
        """
        if df is None or df.empty:
            return 0

        # 保留表中字段
        table_cols = {c.name for c in StockBar.__table__.columns} - {"id", "created_at"}
        cols = [c for c in df.columns if c in table_cols]
        if "symbol" not in cols or "trade_date" not in cols:
            logger.warning("[Repo] save_bars: 缺少 symbol/trade_date, 跳过")
            return 0

        # 填充默认 source/adjust
        df_clean = df.copy()
        if "source" not in df_clean.columns:
            df_clean["source"] = ""
        if "adjust" not in df_clean.columns:
            df_clean["adjust"] = "raw"

        # 去重新唯一键 + NaN -> None（用 df_clean 而非 df，确保 source/adjust 存在）
        cols_clean = [c for c in df_clean.columns if c in table_cols]
        df_clean = (df_clean[cols_clean]
                    .drop_duplicates(subset=["symbol", "trade_date", "source", "adjust"], keep="last")
                    .where(pd.notna(df_clean[cols_clean]), None))
        # 日期转成 python date
        df_clean = df_clean.copy()
        df_clean["trade_date"] = pd.to_datetime(df_clean["trade_date"]).dt.date

        records = df_clean.to_dict(orient="records")
        total = 0
        update_cols = [c for c in cols if c not in ("symbol", "trade_date", "source", "adjust")]

        with self.session() as s:
            for i in range(0, len(records), chunk_size):
                chunk = records[i:i + chunk_size]
                stmt = pg_insert(StockBar.__table__).values(chunk)
                if update_cols:
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["symbol", "trade_date", "source", "adjust"],
                        set_={c: stmt.excluded[c] for c in update_cols},
                    )
                else:
                    stmt = stmt.on_conflict_do_nothing(
                        index_elements=["symbol", "trade_date", "source", "adjust"]
                    )
                s.execute(stmt)
                total += len(chunk)

        logger.info(f"[Repo] upsert 行情 {total} 条 (chunk={chunk_size})")
        return total

    def get_latest_dates_bulk(self, symbols: List[str]) -> dict:
        """批量获取多只股票的本地最新交易日

        Returns: {symbol: date or None}
        """
        if not symbols:
            return {}
        with self.session() as s:

            sub = (
                s.query(StockBar.symbol, func.max(StockBar.trade_date).label("md"))
                .filter(StockBar.symbol.in_(symbols))
                .group_by(StockBar.symbol)
                .subquery()
            )
            # 联表取 created_at（取最新日期对应记录中 created_at 最大的那条，防止重复）
            rows = (
                s.query(StockBar.symbol, StockBar.trade_date, func.max(StockBar.created_at))
                .join(sub, and_(StockBar.symbol == sub.c.symbol, StockBar.trade_date == sub.c.md))
                .group_by(StockBar.symbol, StockBar.trade_date)
                .all()
            )
        out = {sym: (None, None) for sym in symbols}
        for sym, dt, ca in rows:
            out[sym] = (dt, ca)
        return out

    def get_latest_dates_with_created_at(self, symbols: List[str]) -> dict:
        """兼容增量扫描器的批量最新日期接口。

        增量扫描需要同时知道每只股票的最新 ``trade_date`` 和该记录的
        ``created_at``，用于判断收盘后是否要重刷盘中临时数据。
        旧实现只有 ``get_latest_dates_bulk``，导致每日扫描进入异常兜底路径，
        容易把全池都当作待处理任务，进而造成扫描历史口径失真。

        Returns:
            {symbol: (trade_date or None, created_at or None)}
        """
        return self.get_latest_dates_bulk(symbols)

    def count_bars(self, symbol: Optional[str] = None) -> int:
        with self.session() as s:
            q = s.query(func.count(StockBar.id))
            if symbol:
                q = q.filter(StockBar.symbol == symbol)
            # 不按 source/adjust 过滤
            return q.scalar() or 0
    
    def count_bars_by_source_adjust(self) -> pd.DataFrame:
        """统计 PG 中不同 source/adjust 的数据量"""
        with self.session() as s:
            rows = s.query(
                StockBar.source,
                StockBar.adjust,
                func.count(StockBar.id).label("count"),
                func.min(StockBar.trade_date).label("min_date"),
                func.max(StockBar.trade_date).label("max_date")
            ).group_by(StockBar.source, StockBar.adjust).order_by(StockBar.source, StockBar.adjust).all()
            data = [
                {
                    "source": source,
                    "adjust": adjust,
                    "count": count,
                    "min_date": min_date.strftime("%Y-%m-%d") if min_date else "",
                    "max_date": max_date.strftime("%Y-%m-%d") if max_date else "",
                }
                for source, adjust, count, min_date, max_date in rows
            ]
            return pd.DataFrame(data)
    
    def get_bars(
        self,
        symbol: str,
        source: str = "",
        adjust: str = "raw",
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """获取个股历史行情（按 source/adjust 过滤）"""
        with self.session() as s:
            query = s.query(StockBar).filter(StockBar.symbol == symbol).filter(StockBar.source == source).filter(StockBar.adjust == adjust)
            
            if start_date:
                query = query.filter(StockBar.trade_date >= start_date)
            if end_date:
                query = query.filter(StockBar.trade_date <= end_date)
            
            query = query.order_by(StockBar.trade_date)
            
            if limit:
                query = query.limit(limit)
            
            results = query.all()
            
            if not results:
                return pd.DataFrame()
            
            data = []
            for r in results:
                data.append({
                    "symbol": r.symbol,
                    "trade_date": r.trade_date,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "pre_close": r.pre_close,
                    "change": r.change,
                    "pct_change": r.pct_change,
                    "volume": r.volume,
                    "amount": r.amount,
                    "turnover": r.turnover,
                    "amplitude": r.amplitude,
                    "source": r.source,
                    "adjust": r.adjust,
                })
        
        return pd.DataFrame(data)
    
    def get_bars_legacy(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """兼容旧接口，默认 source='' & adjust='raw'"""
        return self.get_bars(symbol, source="", adjust="raw", start_date=start_date, end_date=end_date, limit=limit)
    
    def get_bars_multi(
        self,
        symbols: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """获取多只股票行情"""
        dfs = []
        for symbol in symbols:
            df = self.get_bars_legacy(symbol, start_date, end_date)
            if not df.empty:
                dfs.append(df)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()
    
    def get_trade_dates(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[date]:
        """获取交易日期列表"""
        with self.session() as s:
            query = s.query(StockBar.trade_date).distinct()
            if start_date:
                query = query.filter(StockBar.trade_date >= start_date)
            if end_date:
                query = query.filter(StockBar.trade_date <= end_date)
            dates = [r[0] for r in query.order_by(StockBar.trade_date).all()]
        return dates
    
    def get_latest_date(self, symbol: Optional[str] = None) -> Optional[date]:
        """获取最新数据日期"""
        with self.session() as s:
            query = s.query(func.max(StockBar.trade_date))
            if symbol:
                query = query.filter(StockBar.symbol == symbol)
            # 不按 source/adjust 过滤
            result = query.scalar()
        return result
    
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        with self.session() as s:
            results = s.query(StockInfo).all()
        
        data = []
        for r in results:
            data.append({
                "symbol": r.symbol,
                "name": r.name,
                "industry": r.industry,
                "area": r.area,
                "list_date": r.list_date,
                "total_mv": r.total_mv,
                "float_mv": r.float_mv,
            })
        return pd.DataFrame(data)
    
    def save_stock_list(self, df: pd.DataFrame) -> int:
        """保存股票列表"""
        with self.session() as s:
            count = 0
            for _, row in df.iterrows():
                info = s.query(StockInfo).filter_by(symbol=row.get("symbol")).first()
                if info is None:
                    info = StockInfo()
                
                for col in ["symbol", "name", "exchange", "industry", 
                           "area", "list_date", "total_mv", "float_mv"]:
                    if col in row and pd.notna(row[col]):
                        setattr(info, col, row[col])
                
                s.add(info)
                count += 1
            return count


class FactorRepository(BaseRepository):
    """因子数据仓库"""
    
    def save_factors(self, df: pd.DataFrame) -> int:
        """批量保存因子值"""
        if df.empty:
            return 0
        
        with self.session() as s:
            count = 0
            for _, row in df.iterrows():
                fv = s.query(FactorValue).filter_by(
                    symbol=row.get("symbol"),
                    trade_date=row.get("trade_date")
                ).first()
                
                if fv is None:
                    fv = FactorValue()
                
                for col in df.columns:
                    if col not in ["id", "created_at"] and pd.notna(row[col]):
                        setattr(fv, col, row[col])
                
                s.add(fv)
                count += 1
            
            logger.info(f"保存 {count} 条因子数据")
            return count
    
    def get_factors(
        self,
        trade_date: Optional[date] = None,
        symbol: Optional[str] = None,
        factor_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """获取因子值"""
        with self.session() as s:
            query = s.query(FactorValue)
            
            if trade_date:
                query = query.filter(FactorValue.trade_date == trade_date)
            if symbol:
                query = query.filter(FactorValue.symbol == symbol)
            
            results = query.all()
        
        if not results:
            return pd.DataFrame()
        
        data = []
        for r in results:
            item = {
                "symbol": r.symbol,
                "trade_date": r.trade_date,
                "pe_ttm": r.pe_ttm,
                "pb": r.pb,
                "ps": r.ps,
                "peg": r.peg,
                "roe": r.roe,
                "roa": r.roa,
                "gross_margin": r.gross_margin,
                "net_margin": r.net_margin,
                "revenue_growth": r.revenue_growth,
                "profit_growth": r.profit_growth,
                "momentum_20d": r.momentum_20d,
                "momentum_60d": r.momentum_60d,
                "momentum_120d": r.momentum_120d,
                "high_52w_ratio": r.high_52w_ratio,
                "volatility_20d": r.volatility_20d,
                "beta": r.beta,
                "max_drawdown_60d": r.max_drawdown_60d,
                "turnover_20d": r.turnover_20d,
                "amt_per_cap": r.amt_per_cap,
                "rsi_14": r.rsi_14,
                "macd_dif": r.macd_dif,
                "macd_dea": r.macd_dea,
                "macd_hist": r.macd_hist,
                "bband_width": r.bband_width,
                "bband_position": r.bband_position,
                "north_pct": r.north_pct,
                "margin_balance": r.margin_balance,
                "total_score": r.total_score,
            }
            data.append(item)
        
        df = pd.DataFrame(data)
        
        if factor_names:
            cols = ["symbol", "trade_date"] + [c for c in factor_names if c in df.columns]
            df = df[[c for c in cols if c in df.columns]]
        
        return df
    
    def get_top_stocks(
        self,
        trade_date: date,
        n: int = 20,
        ascending: bool = False
    ) -> pd.DataFrame:
        """获取某日期得分最高的股票"""
        with self.session() as s:
            results = s.query(FactorValue).filter(
                FactorValue.trade_date == trade_date
            ).order_by(
                desc(FactorValue.total_score) if not ascending else FactorValue.total_score
            ).limit(n).all()
        
        data = []
        for r in results:
            data.append({
                "symbol": r.symbol,
                "total_score": r.total_score,
                "pe_ttm": r.pe_ttm,
                "roe": r.roe,
                "momentum_20d": r.momentum_20d,
            })
        return pd.DataFrame(data)
    
    def get_ic_series(self, factor_name: str, start_date: Optional[date] = None) -> pd.DataFrame:
        """获取因子的 IC 序列"""
        with self.session() as s:
            query = s.query(FactorIC).filter(FactorIC.factor_name == factor_name)
            if start_date:
                query = query.filter(FactorIC.trade_date >= start_date)
            results = query.order_by(FactorIC.trade_date).all()
        
        data = []
        for r in results:
            data.append({
                "trade_date": r.trade_date,
                "ic": r.ic,
                "p_value": r.p_value,
                "rank_ic": r.rank_ic,
                "rank_p_value": r.rank_p_value,
            })
        return pd.DataFrame(data)
    
    def save_ic(self, df: pd.DataFrame) -> int:
        """保存 IC 计算结果"""
        if df.empty:
            return 0
        
        with self.session() as s:
            count = 0
            for _, row in df.iterrows():
                ic = s.query(FactorIC).filter_by(
                    factor_name=row.get("factor_name"),
                    trade_date=row.get("trade_date")
                ).first()
                
                if ic is None:
                    ic = FactorIC()
                
                for col in df.columns:
                    if col not in ["id", "created_at"] and pd.notna(row[col]):
                        setattr(ic, col, row[col])
                
                s.add(ic)
                count += 1
            
            logger.info(f"保存 {count} 条 IC 数据")
            return count


class BacktestRepository(BaseRepository):
    """回测结果仓库"""
    
    def save_result(self, df: pd.DataFrame) -> int:
        """保存回测结果"""
        if df.empty:
            return 0
        
        with self.session() as s:
            count = 0
            for _, row in df.iterrows():
                result = s.query(BacktestResult).filter_by(
                    strategy_name=row.get("strategy_name"),
                    start_date=row.get("start_date"),
                    end_date=row.get("end_date")
                ).first()
                
                if result is None:
                    result = BacktestResult()
                
                for col in df.columns:
                    if col not in ["id", "created_at"] and pd.notna(row[col]):
                        setattr(result, col, row[col])
                
                s.add(result)
                count += 1
            
            logger.info(f"保存 {count} 条回测结果")
            return count
    
    def list_results(self, strategy_name: Optional[str] = None) -> pd.DataFrame:
        """列出回测结果"""
        with self.session() as s:
            query = s.query(BacktestResult)
            if strategy_name:
                query = query.filter(BacktestResult.strategy_name == strategy_name)
            results = query.order_by(BacktestResult.start_date.desc()).all()
        
        data = []
        for r in results:
            data.append({
                "strategy_name": r.strategy_name,
                "start_date": r.start_date,
                "end_date": r.end_date,
                "total_return": r.total_return,
                "annual_return": r.annual_return,
                "benchmark_return": r.benchmark_return,
                "excess_return": r.excess_return,
                "sharpe_ratio": r.sharpe_ratio,
                "max_drawdown": r.max_drawdown,
                "win_rate": r.win_rate,
                "profit_loss_ratio": r.profit_loss_ratio,
                "trade_count": r.trade_count,
            })
        return pd.DataFrame(data)


# ── 全局单例 ──
_stock_repo: Optional[StockRepository] = None
_factor_repo: Optional[FactorRepository] = None
_backtest_repo: Optional[BacktestRepository] = None


def stock_repo() -> StockRepository:
    global _stock_repo
    if _stock_repo is None:
        _stock_repo = StockRepository()
    return _stock_repo


def factor_repo() -> FactorRepository:
    global _factor_repo
    if _factor_repo is None:
        _factor_repo = FactorRepository()
    return _factor_repo


def backtest_repo() -> BacktestRepository:
    global _backtest_repo
    if _backtest_repo is None:
        _backtest_repo = BacktestRepository()
    return _backtest_repo


# ── MoneyFlow 相关方法（添加到StockRepository） ──
def add_moneyflow_methods_to_stock_repository():
    """动态添加资金流数据方法到StockRepository"""
    from .models import MoneyFlow
    
    def save_moneyflow(self, df: pd.DataFrame, source: str = "tushare", chunk_size: int = 2000) -> int:
        """批量保存资金流数据（简化版本）
        
        Args:
            df: 包含资金流数据的DataFrame
            source: 数据来源
            chunk_size: 分批大小
        
        Returns:
            实际写入行数
        """
        if df is None or df.empty:
            return 0
        
        # 准备数据
        df_clean = df.copy()
        
        # 确保有symbol字段（从ts_code提取）
        if "symbol" not in df_clean.columns and "ts_code" in df_clean.columns:
            df_clean["symbol"] = df_clean["ts_code"].str.slice(0, 6)
        
        # 确保有source字段
        df_clean["source"] = source
        
        # 选择需要的字段
        table_cols = {c.name for c in MoneyFlow.__table__.columns} - {"id", "created_at", "updated_at"}
        cols = [c for c in df_clean.columns if c in table_cols]
        
        if "symbol" not in cols or "trade_date" not in cols:
            logger.warning("[Repo] save_moneyflow: 缺少 symbol/trade_date, 跳过")
            return 0
        
        # 去重
        df_clean = df_clean[cols].drop_duplicates(
            subset=["symbol", "trade_date", "source"], 
            keep="last"
        ).where(pd.notna(df_clean[cols]), None)
        
        total = 0
        
        with self.session() as s:
            # 分批插入
            for i in range(0, len(df_clean), chunk_size):
                chunk_df = df_clean.iloc[i:i+chunk_size]
                
                # 转换为字典列表
                records = chunk_df.to_dict('records')
                
                # 批量插入
                for record in records:
                    try:
                        # 检查是否已存在
                        existing = s.query(MoneyFlow).filter(
                            MoneyFlow.symbol == record.get('symbol'),
                            MoneyFlow.trade_date == record.get('trade_date'),
                            MoneyFlow.source == record.get('source', source)
                        ).first()
                        
                        if existing:
                            # 更新现有记录
                            for key, value in record.items():
                                if key not in ['symbol', 'trade_date', 'source'] and value is not None:
                                    setattr(existing, key, value)
                        else:
                            # 创建新记录
                            moneyflow = MoneyFlow(**record)
                            s.add(moneyflow)
                        
                        total += 1
                        
                        # 每500条提交一次
                        if total % 500 == 0:
                            s.commit()
                    
                    except Exception as e:
                        logger.warning(f"保存资金流记录失败: {e}")
                        s.rollback()
                
                # 提交剩余记录
                s.commit()
        
        logger.info(f"[Repo] 保存资金流 {total} 条 (source={source})")
        return total
    
    def get_moneyflow_by_dates(self, trade_dates: List[str], symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """获取指定日期范围的资金流数据
        
        Args:
            trade_dates: 交易日期列表 (YYYYMMDD格式)
            symbols: 股票代码列表，None表示所有股票
        
        Returns:
            资金流数据DataFrame
        """
        if not trade_dates:
            return pd.DataFrame()
        
        with self.session() as s:
            query = s.query(MoneyFlow).filter(MoneyFlow.trade_date.in_(trade_dates))
            
            if symbols:
                query = query.filter(MoneyFlow.symbol.in_(symbols))
            
            results = query.all()
            
            if not results:
                return pd.DataFrame()
            
            data = []
            for r in results:
                data.append({
                    "ts_code": r.ts_code,
                    "symbol": r.symbol,
                    "trade_date": r.trade_date,
                    "buy_sm_vol": r.buy_sm_vol,
                    "buy_sm_amount": r.buy_sm_amount,
                    "sell_sm_vol": r.sell_sm_vol,
                    "sell_sm_amount": r.sell_sm_amount,
                    "buy_md_vol": r.buy_md_vol,
                    "buy_md_amount": r.buy_md_amount,
                    "sell_md_vol": r.sell_md_vol,
                    "sell_md_amount": r.sell_md_amount,
                    "buy_lg_vol": r.buy_lg_vol,
                    "buy_lg_amount": r.buy_lg_amount,
                    "sell_lg_vol": r.sell_lg_vol,
                    "sell_lg_amount": r.sell_lg_amount,
                    "buy_elg_vol": r.buy_elg_vol,
                    "buy_elg_amount": r.buy_elg_amount,
                    "sell_elg_vol": r.sell_elg_vol,
                    "sell_elg_amount": r.sell_elg_amount,
                    "net_mf_vol": r.net_mf_vol,
                    "net_mf_amount": r.net_mf_amount,
                    "source": r.source,
                })
            
            return pd.DataFrame(data)
    
    def get_moneyflow_by_symbol_date(self, symbol: str, trade_date: str) -> Optional[dict]:
        """获取单只股票单日资金流数据"""
        with self.session() as s:
            result = s.query(MoneyFlow).filter(
                MoneyFlow.symbol == symbol,
                MoneyFlow.trade_date == trade_date
            ).first()
            
            if result is None:
                return None
            
            return {
                "ts_code": result.ts_code,
                "symbol": result.symbol,
                "trade_date": result.trade_date,
                "buy_sm_vol": result.buy_sm_vol,
                "buy_sm_amount": result.buy_sm_amount,
                "sell_sm_vol": result.sell_sm_vol,
                "sell_sm_amount": result.sell_sm_amount,
                "buy_md_vol": result.buy_md_vol,
                "buy_md_amount": result.buy_md_amount,
                "sell_md_vol": result.sell_md_vol,
                "sell_md_amount": result.sell_md_amount,
                "buy_lg_vol": result.buy_lg_vol,
                "buy_lg_amount": result.buy_lg_amount,
                "sell_lg_vol": result.sell_lg_vol,
                "sell_lg_amount": result.sell_lg_amount,
                "buy_elg_vol": result.buy_elg_vol,
                "buy_elg_amount": result.buy_elg_amount,
                "sell_elg_vol": result.sell_elg_vol,
                "sell_elg_amount": result.sell_elg_amount,
                "net_mf_vol": result.net_mf_vol,
                "net_mf_amount": result.net_mf_amount,
                "source": result.source,
            }
    
    def count_moneyflow_by_date(self, trade_date: str) -> int:
        """统计指定日期的资金流数据条数"""
        with self.session() as s:
            count = s.query(func.count(MoneyFlow.id)).filter(
                MoneyFlow.trade_date == trade_date
            ).scalar()
            return count or 0
    
    # 动态添加到StockRepository类
    StockRepository.save_moneyflow = save_moneyflow
    StockRepository.get_moneyflow_by_dates = get_moneyflow_by_dates
    StockRepository.get_moneyflow_by_symbol_date = get_moneyflow_by_symbol_date
    StockRepository.count_moneyflow_by_date = count_moneyflow_by_date


# 初始化时添加资金流方法
add_moneyflow_methods_to_stock_repository()
