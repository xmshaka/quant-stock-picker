"""数据库模型定义 - SQLAlchemy ORM"""
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, String, Float, DateTime, Date, Integer, BigInteger,
    Index, UniqueConstraint, create_engine, text
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import JSONB

from config.settings import settings

Base = declarative_base()


class StockInfo(Base):
    """股票基本信息表"""
    __tablename__ = "stock_info"
    
    symbol = Column(String(10), primary_key=True, comment="股票代码")
    name = Column(String(50), comment="股票名称")
    exchange = Column(String(10), comment="交易所 SH/SZ/BJ")
    industry = Column(String(50), comment="所属行业")
    area = Column(String(30), comment="地区")
    list_date = Column(Date, comment="上市日期")
    total_mv = Column(BigInteger, comment="总市值")
    float_mv = Column(BigInteger, comment="流通市值")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    __table_args__ = (
        Index("idx_industry", "industry"),
        Index("idx_list_date", "list_date"),
    )


class StockBar(Base):
    """日线行情数据表（时序数据，建议用TimescaleDB hypertable）"""
    __tablename__ = "stock_bars"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, comment="股票代码")
    trade_date = Column(Date, nullable=False, comment="交易日期")
    open = Column(Float, comment="开盘价")
    high = Column(Float, comment="最高价")
    low = Column(Float, comment="最低价")
    close = Column(Float, comment="收盘价")
    pre_close = Column(Float, comment="昨收")
    change = Column(Float, comment="涨跌额")
    pct_change = Column(Float, comment="涨跌幅%")
    volume = Column(BigInteger, comment="成交量")
    amount = Column(BigInteger, comment="成交额")
    turnover = Column(Float, comment="换手率")
    amplitude = Column(Float, comment="振幅")
    source = Column(String(20), comment="数据来源", default="", server_default="")
    adjust = Column(String(10), comment="复权口径", default="raw", server_default="raw")
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        UniqueConstraint("symbol", "trade_date", "source", "adjust", name="uix_symbol_date_source_adjust"),
        Index("idx_symbol_date_src_adj", "symbol", "trade_date", "source", "adjust"),
        Index("idx_trade_date", "trade_date"),
        Index("idx_source_adjust", "source", "adjust"),
    )


class FactorValue(Base):
    """因子值表 - 每日每只股票的各因子得分"""
    __tablename__ = "factor_values"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, comment="股票代码")
    trade_date = Column(Date, nullable=False, comment="交易日期")
    
    # 估值因子
    pe_ttm = Column(Float, comment="市盈率TTM")
    pb = Column(Float, comment="市净率")
    ps = Column(Float, comment="市销率")
    peg = Column(Float, comment="PEG")
    
    # 质量因子
    roe = Column(Float, comment="净资产收益率")
    roa = Column(Float, comment="总资产收益率")
    gross_margin = Column(Float, comment="毛利率")
    net_margin = Column(Float, comment="净利率")
    revenue_growth = Column(Float, comment="营收增长率")
    profit_growth = Column(Float, comment="净利润增长率")
    
    # 动量因子
    momentum_20d = Column(Float, comment="20日动量")
    momentum_60d = Column(Float, comment="60日动量")
    momentum_120d = Column(Float, comment="120日动量")
    high_52w_ratio = Column(Float, comment="52周新高距离")
    
    # 波动因子
    volatility_20d = Column(Float, comment="20日波动率")
    beta = Column(Float, comment="Beta值")
    max_drawdown_60d = Column(Float, comment="60日最大回撤")
    
    # 资金流因子（新增）
    main_net_mf_amount = Column(Float, comment="主力净流入金额(万元)")
    large_net_mf_amount = Column(Float, comment="大单净流入金额(万元)")
    elg_net_mf_amount = Column(Float, comment="超大单净流入金额(万元)")
    large_elg_net_mf_amount = Column(Float, comment="大单+超大单净流入(万元)")
    main_net_mf_pct_amount = Column(Float, comment="主力净流入占成交额比例")
    large_elg_net_mf_pct_amount = Column(Float, comment="大单+超大单净流入占成交额比例")
    main_net_mf_rank = Column(Float, comment="主力净流入比例排名(0-1)")
    large_elg_net_mf_rank = Column(Float, comment="大单+超大单净流入比例排名(0-1)")
    
    # 相对换手率因子（新增）
    relative_turnover_5d = Column(Float, comment="5日相对换手率")
    relative_turnover_20d = Column(Float, comment="20日相对换手率")
    turnover_percentile_60d = Column(Float, comment="60日换手率分位")
    amount_percentile_60d = Column(Float, comment="60日成交额分位")
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    __table_args__ = (
        UniqueConstraint("symbol", "trade_date", name="uix_symbol_date_factor"),
        Index("idx_symbol_date", "symbol", "trade_date"),
        Index("idx_trade_date", "trade_date"),
    )


class MoneyFlow(Base):
    """资金流原始数据表 - Tushare moneyflow数据"""
    __tablename__ = "moneyflow"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ts_code = Column(String(20), nullable=False, comment="TS代码")
    symbol = Column(String(10), nullable=False, comment="股票代码(6位)")
    trade_date = Column(String(8), nullable=False, comment="交易日期YYYYMMDD")
    
    # 小单
    buy_sm_vol = Column(BigInteger, comment="小单买入成交量(手)")
    buy_sm_amount = Column(Float, comment="小单买入成交额(万元)")
    sell_sm_vol = Column(BigInteger, comment="小单卖出成交量(手)")
    sell_sm_amount = Column(Float, comment="小单卖出成交额(万元)")
    
    # 中单
    buy_md_vol = Column(BigInteger, comment="中单买入成交量(手)")
    buy_md_amount = Column(Float, comment="中单买入成交额(万元)")
    sell_md_vol = Column(BigInteger, comment="中单卖出成交量(手)")
    sell_md_amount = Column(Float, comment="中单卖出成交额(万元)")
    
    # 大单
    buy_lg_vol = Column(BigInteger, comment="大单买入成交量(手)")
    buy_lg_amount = Column(Float, comment="大单买入成交额(万元)")
    sell_lg_vol = Column(BigInteger, comment="大单卖出成交量(手)")
    sell_lg_amount = Column(Float, comment="大单卖出成交额(万元)")
    
    # 超大单
    buy_elg_vol = Column(BigInteger, comment="超大单买入成交量(手)")
    buy_elg_amount = Column(Float, comment="超大单买入成交额(万元)")
    sell_elg_vol = Column(BigInteger, comment="超大单卖出成交量(手)")
    sell_elg_amount = Column(Float, comment="超大单卖出成交额(万元)")
    
    # 合计
    net_mf_vol = Column(BigInteger, comment="净流入成交量(手)")
    net_mf_amount = Column(Float, comment="净流入成交额(万元)")
    
    source = Column(String(20), comment="数据来源", default="tushare")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    __table_args__ = (
        UniqueConstraint("symbol", "trade_date", "source", name="uix_moneyflow_symbol_date_source"),
        Index("idx_moneyflow_symbol_date", "symbol", "trade_date"),
        Index("idx_moneyflow_date", "trade_date"),
        Index("idx_moneyflow_source", "source"),
    )

    # 流动性因子
    turnover_20d = Column(Float, comment="20日均换手")
    amt_per_cap = Column(Float, comment="流通市值换手率")
    
    # 技术因子
    rsi_14 = Column(Float, comment="RSI14")
    macd_dif = Column(Float, comment="MACD DIF")
    macd_dea = Column(Float, comment="MACD DEA")
    macd_hist = Column(Float, comment="MACD柱")
    bband_width = Column(Float, comment="布林带宽")
    bband_position = Column(Float, comment="布林带位置")
    
    # 情绪因子
    north_pct = Column(Float, comment="北向持股比例")
    margin_balance = Column(Float, comment="融资余额")
    
    # 综合得分
    total_score = Column(Float, comment="综合因子得分")
    
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        UniqueConstraint("symbol", "trade_date", name="uix_factor_symbol_date"),
        Index("idx_fv_symbol_date", "symbol", "trade_date"),
        Index("idx_fv_date_score", "trade_date", "total_score"),
    )


class SectorData(Base):
    """板块/行业数据表"""
    __tablename__ = "sector_data"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    sector_code = Column(String(20), nullable=False, comment="板块代码")
    sector_name = Column(String(50), nullable=False, comment="板块名称")
    trade_date = Column(Date, nullable=False, comment="交易日期")
    close = Column(Float, comment="板块指数收盘价")
    pct_change = Column(Float, comment="涨跌幅")
    total_mv = Column(BigInteger, comment="板块总市值")
    turnover = Column(Float, comment="换手率")
    up_count = Column(Integer, comment="上涨家数")
    down_count = Column(Integer, comment="下跌家数")
    leading_stock = Column(String(10), comment="领涨股")
    leading_pct = Column(Float, comment="领涨股涨幅")
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        UniqueConstraint("sector_code", "trade_date", name="uix_sector_date"),
        Index("idx_sector_date", "sector_code", "trade_date"),
        Index("idx_sector_chg", "trade_date", "pct_change"),
    )


class HotspotRecord(Base):
    """热点记录表"""
    __tablename__ = "hotspot_records"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    trade_date = Column(Date, nullable=False, comment="日期")
    hotspot_name = Column(String(100), nullable=False, comment="热点名称")
    source = Column(String(50), comment="来源")
    sentiment_score = Column(Float, comment="情感得分")
    related_symbols = Column(JSONB, comment="相关股票列表")
    heat_score = Column(Float, comment="热度得分")
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        Index("idx_hotspot_date", "trade_date"),
    )


class BacktestResult(Base):
    """回测结果表"""
    __tablename__ = "backtest_results"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    strategy_name = Column(String(100), nullable=False, comment="策略名称")
    params = Column(JSONB, comment="策略参数")
    
    # 时间范围
    start_date = Column(Date, comment="回测开始")
    end_date = Column(Date, comment="回测结束")
    
    # 收益指标
    total_return = Column(Float, comment="总收益率")
    annual_return = Column(Float, comment="年化收益率")
    benchmark_return = Column(Float, comment="基准收益率")
    excess_return = Column(Float, comment="超额收益")
    
    # 风险指标
    sharpe_ratio = Column(Float, comment="夏普比率")
    sortino_ratio = Column(Float, comment="索提诺比率")
    max_drawdown = Column(Float, comment="最大回撤")
    max_drawdown_duration = Column(Integer, comment="最大回撤天数")
    calmar_ratio = Column(Float, comment="卡玛比率")
    
    # 交易统计
    win_rate = Column(Float, comment="胜率")
    profit_loss_ratio = Column(Float, comment="盈亏比")
    avg_holding_days = Column(Float, comment="平均持仓天数")
    trade_count = Column(Integer, comment="交易次数")
    turnover_rate = Column(Float, comment="换手率")
    
    # 曲线数据
    equity_curve = Column(JSONB, comment="权益曲线")
    
    # 数据来源追踪
    data_source = Column(String(20), comment="数据来源", default="", server_default="")
    data_adjust = Column(String(10), comment="复权口径", default="raw", server_default="raw")
    data_version = Column(String(40), comment="数据版本", default="", server_default="")
    
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        Index("idx_bt_strategy", "strategy_name"),
        Index("idx_bt_date", "start_date", "end_date"),
        Index("idx_bt_data_source", "data_source", "data_adjust"),
    )


class FactorIC(Base):
    """因子IC分析结果表"""
    __tablename__ = "factor_ic"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    factor_name = Column(String(50), nullable=False, comment="因子名")
    trade_date = Column(Date, nullable=False, comment="日期")
    ic = Column(Float, comment="IC值")
    ic_rank = Column(Float, comment="Rank IC")
    ic_decay_5d = Column(Float, comment="5日IC衰减")
    ic_decay_10d = Column(Float, comment="10日IC衰减")
    ir = Column(Float, comment="信息比率IR")
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        UniqueConstraint("factor_name", "trade_date", name="uix_ic_factor_date"),
        Index("idx_ic_factor", "factor_name", "trade_date"),
    )


# 数据库引擎和会话
def get_engine():
    return create_engine(settings.database_url, pool_pre_ping=True, echo=False)


def get_session_factory():
    engine = get_engine()
    return sessionmaker(bind=engine)


def init_db():
    """初始化数据库，创建所有表"""
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("✅ 数据库表创建完成")


def init_timescaledb():
    """
    如果使用TimescaleDB，将stock_bars转为hypertable
    需要在init_db()之后调用
    """
    engine = get_engine()
    with engine.connect() as conn:
        # 检查是否已转换
        result = conn.execute(text("""
            SELECT * FROM _timescaledb_catalog.hypertable 
            WHERE table_name = 'stock_bars'
        """))
        if result.rowcount == 0:
            conn.execute(text("""
                SELECT create_hypertable('stock_bars', 'trade_date', 
                if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 month')
            """))
            print("✅ stock_bars 已转换为 TimescaleDB hypertable")
        conn.commit()


if __name__ == "__main__":
    init_db()
