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

        - 使用 pg_insert + on_conflict_do_update, 一条 SQL 处理 N 行
        - 按 chunk_size 分批, 避免单次 SQL 太大
        - 只写 ORM 定义中有的列

        Returns:
            实际写入行数
        """
        if df is None or df.empty:
            return 0

        # 仅保留表中存在的字段
        table_cols = {c.name for c in StockBar.__table__.columns} - {"id", "created_at"}
        cols = [c for c in df.columns if c in table_cols]
        if "symbol" not in cols or "trade_date" not in cols:
            logger.warning("[Repo] save_bars: 缺少 symbol/trade_date, 跳过")
            return 0

        # 清洗: 去重 (symbol, trade_date) + NaN -> None
        df_clean = (df[cols]
                    .drop_duplicates(subset=["symbol", "trade_date"], keep="last")
                    .where(pd.notna(df[cols]), None))
        # 日期转成 python date
        df_clean = df_clean.copy()
        df_clean["trade_date"] = pd.to_datetime(df_clean["trade_date"]).dt.date

        records = df_clean.to_dict(orient="records")
        total = 0
        update_cols = [c for c in cols if c not in ("symbol", "trade_date")]

        with self.session() as s:
            for i in range(0, len(records), chunk_size):
                chunk = records[i:i + chunk_size]
                stmt = pg_insert(StockBar.__table__).values(chunk)
                if update_cols:
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["symbol", "trade_date"],
                        set_={c: stmt.excluded[c] for c in update_cols},
                    )
                else:
                    stmt = stmt.on_conflict_do_nothing(
                        index_elements=["symbol", "trade_date"]
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
            rows = (s.query(StockBar.symbol, func.max(StockBar.trade_date))
                    .filter(StockBar.symbol.in_(symbols))
                    .group_by(StockBar.symbol)
                    .all())
        out = {sym: None for sym in symbols}
        for sym, dt in rows:
            out[sym] = dt
        return out

    def get_latest_dates_with_created_at(self, symbols: List[str]) -> dict:
        """批量获取多只股票的最新交易日及该记录的创建时间（用于判断盘中临时数据）

        Returns: {symbol: (trade_date, created_at) or (None, None)}
        """
        if not symbols:
            return {}
        with self.session() as s:
            # 子查询：每只股票的最新 trade_date
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

    def count_bars(self, symbol: Optional[str] = None) -> int:
        with self.session() as s:
            q = s.query(func.count(StockBar.id))
            if symbol:
                q = q.filter(StockBar.symbol == symbol)
            return q.scalar() or 0
    
    def get_bars(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """获取个股历史行情"""
        with self.session() as s:
            query = s.query(StockBar).filter(StockBar.symbol == symbol)
            
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
                })
        
        return pd.DataFrame(data)
    
    def get_bars_multi(
        self,
        symbols: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """获取多只股票行情"""
        dfs = []
        for symbol in symbols:
            df = self.get_bars(symbol, start_date, end_date)
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
                "trade_date": r.trade_date,
                "total_score": r.total_score,
                "pe_ttm": r.pe_ttm,
                "pb": r.pb,
                "roe": r.roe,
                "momentum_20d": r.momentum_20d,
                "rsi_14": r.rsi_14,
            })
        return pd.DataFrame(data)
    
    def save_ic(self, df: pd.DataFrame) -> int:
        """保存IC分析结果"""
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
                
                for col in ["factor_name", "trade_date", "ic", "ic_rank", 
                           "ic_decay_5d", "ic_decay_10d", "ir"]:
                    if col in row and pd.notna(row[col]):
                        setattr(ic, col, row[col])
                
                s.add(ic)
                count += 1
            return count
    
    def get_ic_history(
        self,
        factor_name: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """获取因子IC历史"""
        with self.session() as s:
            query = s.query(FactorIC).filter(FactorIC.factor_name == factor_name)
            if start_date:
                query = query.filter(FactorIC.trade_date >= start_date)
            if end_date:
                query = query.filter(FactorIC.trade_date <= end_date)
            results = query.order_by(FactorIC.trade_date).all()
        
        data = []
        for r in results:
            data.append({
                "factor_name": r.factor_name,
                "trade_date": r.trade_date,
                "ic": r.ic,
                "ic_rank": r.ic_rank,
                "ic_decay_5d": r.ic_decay_5d,
                "ic_decay_10d": r.ic_decay_10d,
                "ir": r.ir,
            })
        return pd.DataFrame(data)


class SectorRepository(BaseRepository):
    """板块数据仓库"""
    
    def save_sectors(self, df: pd.DataFrame) -> int:
        """保存板块数据"""
        if df.empty:
            return 0
        
        with self.session() as s:
            count = 0
            for _, row in df.iterrows():
                sd = s.query(SectorData).filter_by(
                    sector_code=row.get("sector_code"),
                    trade_date=row.get("trade_date")
                ).first()
                
                if sd is None:
                    sd = SectorData()
                
                for col in ["sector_code", "sector_name", "trade_date", "close",
                           "pct_change", "total_mv", "turnover", "up_count",
                           "down_count", "leading_stock", "leading_pct"]:
                    if col in row and pd.notna(row[col]):
                        setattr(sd, col, row[col])
                
                s.add(sd)
                count += 1
            return count
    
    def get_sector_ranking(
        self,
        trade_date: date,
        top_n: int = 10
    ) -> pd.DataFrame:
        """获取板块涨幅排行"""
        with self.session() as s:
            results = s.query(SectorData).filter(
                SectorData.trade_date == trade_date
            ).order_by(desc(SectorData.pct_change)).limit(top_n).all()
        
        data = []
        for r in results:
            data.append({
                "sector_code": r.sector_code,
                "sector_name": r.sector_name,
                "pct_change": r.pct_change,
                "turnover": r.turnover,
                "up_count": r.up_count,
                "down_count": r.down_count,
                "leading_stock": r.leading_stock,
            })
        return pd.DataFrame(data)
    
    def get_sector_momentum(
        self,
        sector_code: str,
        window: int = 20
    ) -> pd.DataFrame:
        """获取板块动量"""
        with self.session() as s:
            results = s.query(SectorData).filter(
                SectorData.sector_code == sector_code
            ).order_by(desc(SectorData.trade_date)).limit(window).all()
        
        data = []
        for r in reversed(results):
            data.append({
                "trade_date": r.trade_date,
                "close": r.close,
                "pct_change": r.pct_change,
                "turnover": r.turnover,
                "up_count": r.up_count,
            })
        return pd.DataFrame(data)
