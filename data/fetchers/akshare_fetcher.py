"""AKShare 数据获取器 - A股免费数据源"""
import akshare as ak
import pandas as pd
from typing import Optional
from datetime import datetime
from .base import BaseFetcher
from data.bars_normalizer import normalize_daily_bars


class AKShareFetcher(BaseFetcher):
    """AKShare数据获取器

    AKShare是Python开源财经数据接口库，免费提供A股各类数据
    文档: https://www.akshare.xyz/
    """

    def __init__(self):
        super().__init__()

    def get_stock_list(self) -> pd.DataFrame:
        """获取A股所有股票列表"""
        def _fetch():
            df = ak.stock_zh_a_spot_em()
            # 标准化列名
            df = df.rename(columns={
                "代码": "symbol",
                "名称": "name",
                "最新价": "close",
                "涨跌幅": "pct_change",
                "换手率": "turnover",
                "市盈率-动态": "pe_ttm",
                "市净率": "pb",
                "总市值": "total_mv",
                "流通市值": "float_mv",
                "所属行业": "industry",
            })
            cols = ["symbol", "name", "industry", "close", "pct_change",
                    "turnover", "pe_ttm", "pb", "total_mv", "float_mv"]
            return df[[c for c in cols if c in df.columns]]

        result = self._safe_fetch(_fetch)
        return result if result is not None else pd.DataFrame()

    def get_daily_bars(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = "qfq"  # 前复权
    ) -> pd.DataFrame:
        """获取个股日线数据

        Args:
            symbol: 股票代码，如 "000001"
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            adjust: qfq=前复权, hfq=后复权, 空=不复权
        """
        def _fetch():
            if adjust == "qfq":
                df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                         start_date=start_date or "19900101",
                                         end_date=end_date or datetime.now().strftime("%Y%m%d"),
                                         adjust="qfq")
            else:
                df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                         start_date=start_date or "19900101",
                                         end_date=end_date or datetime.now().strftime("%Y%m%d"))

            # 标准化
            df = df.rename(columns={
                "日期": "trade_date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "涨跌幅": "pct_change",
                "涨跌额": "change",
                "换手率": "turnover",
            })
            df["symbol"] = symbol
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = normalize_daily_bars(df, source="akshare", symbol=symbol, adjust=adjust)
            return df

        result = self._safe_fetch(_fetch)
        return result if result is not None else pd.DataFrame()

    def get_sector_list(self) -> pd.DataFrame:
        """获取行业板块列表"""
        def _fetch():
            df = ak.stock_board_industry_name_em()
            df = df.rename(columns={
                "板块名称": "sector_name",
                "板块代码": "sector_code",
                "最新价": "close",
                "涨跌额": "change",
                "涨跌幅": "pct_change",
                "总市值": "total_mv",
                "换手率": "turnover",
                "上涨家数": "up_count",
                "下跌家数": "down_count",
                "领涨股票": "leading_stock",
                "领涨股票-涨跌幅": "leading_pct",
            })
            return df

        result = self._safe_fetch(_fetch)
        return result if result is not None else pd.DataFrame()

    def get_sector_stocks(self, sector_name: str) -> pd.DataFrame:
        """获取某板块内的所有股票"""
        def _fetch():
            df = ak.stock_board_industry_cons_em(symbol=sector_name)
            df = df.rename(columns={
                "序号": "no",
                "代码": "symbol",
                "名称": "name",
                "最新价": "close",
                "涨跌幅": "pct_change",
                "涨跌额": "change",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "最高": "high",
                "最低": "low",
                "今开": "open",
                "昨收": "pre_close",
                "换手率": "turnover",
                "市盈率-动态": "pe_ttm",
                "市净率": "pb",
            })
            return df

        result = self._safe_fetch(_fetch)
        return result if result is not None else pd.DataFrame()

    def get_north_flow(self, start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """获取北向资金（沪深港通）流向"""
        def _fetch():
            df = ak.stock_hsgt_hist_em(symbol="港股通沪")
            df = df.rename(columns={
                "日期": "trade_date",
                "当日资金流入": "inflow",
                "当日余额": "balance",
                "历史累计流入": "cumulative_inflow",
                "当日成交净买额": "net_buy",
                "买入成交额": "buy_amount",
                "卖出成交额": "sell_amount",
                "领涨股": "leading_stock",
                "领涨股-涨跌幅": "leading_pct",
                "上证指数": "sh_index",
                "上证指数-涨跌幅": "sh_pct",
            })
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            if start_date:
                df = df[df["trade_date"] >= start_date]
            if end_date:
                df = df[df["trade_date"] <= end_date]
            return df

        result = self._safe_fetch(_fetch)
        return result if result is not None else pd.DataFrame()

    def get_limit_up_stocks(self, trade_date: Optional[str] = None) -> pd.DataFrame:
        """获取涨停股列表"""
        _date = trade_date or datetime.now().strftime("%Y%m%d")
        def _fetch():
            df = ak.stock_zt_pool_em(date=_date)
            df = df.rename(columns={
                "序号": "no",
                "代码": "symbol",
                "名称": "name",
                "涨跌幅": "pct_change",
                "最新价": "close",
                "成交额": "amount",
                "流通市值": "float_mv",
                "总市值": "total_mv",
                "封板资金": "seal_amount",
                "首次封板时间": "first_seal_time",
                "最后封板时间": "last_seal_time",
                "炸板次数": "open_count",
                "涨停统计": "zt_stats",
                "连板数": "consecutive_boards",
                "所属行业": "industry",
            })
            return df

        result = self._safe_fetch(_fetch)
        return result if result is not None else pd.DataFrame()

    def get_index_components(self, index_code: str = "000300") -> pd.DataFrame:
        """获取指数成分股列表

        Args:
            index_code: 指数代码, 000300=沪深300, 000905=中证500, 000001=上证指数
        """
        def _fetch():
            df = ak.index_stock_cons_weight_csindex(symbol=index_code)
            df = df.rename(columns={
                "成分券代码": "symbol",
                "成分券名称": "name",
                "交易所": "exchange",
                "权重": "weight",
            })
            df["symbol"] = df["symbol"].astype(str).str.zfill(6)
            return df

        result = self._safe_fetch(_fetch)
        return result if result is not None else pd.DataFrame()

    def get_index_daily(self, index_code: str = "000300",
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """获取指数日线"""
        def _fetch():
            df = ak.index_zh_a_hist(symbol=index_code, period="daily",
                                    start_date=start_date or "19900101",
                                    end_date=end_date or datetime.now().strftime("%Y%m%d"))
            df = df.rename(columns={
                "日期": "trade_date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "涨跌幅": "pct_change",
                "涨跌额": "change",
            })
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df["symbol"] = index_code
            return df

        result = self._safe_fetch(_fetch)
        return result if result is not None else pd.DataFrame()

    def get_fund_flow_industry(self) -> pd.DataFrame:
        """获取行业资金流向"""
        def _fetch():
            df = ak.stock_sector_fund_flow_rank(indicator="今日")
            df = df.rename(columns={
                "序号": "rank",
                "名称": "sector_name",
                "最新价": "close",
                "涨跌幅": "pct_change",
                "主力净流入-净额": "main_inflow",
                "主力净流入-净占比": "main_inflow_pct",
                "超大单净流入-净额": "super_inflow",
                "超大单净流入-净占比": "super_inflow_pct",
                "大单净流入-净额": "big_inflow",
                "大单净流入-净占比": "big_inflow_pct",
                "中单净流入-净额": "mid_inflow",
                "中单净流入-净占比": "mid_inflow_pct",
                "小单净流入-净额": "small_inflow",
                "小单净流入-净占比": "small_inflow_pct",
            })
            return df

        result = self._safe_fetch(_fetch)
        return result if result is not None else pd.DataFrame()

    def get_stock_news(self, symbol: str) -> pd.DataFrame:
        """获取个股新闻"""
        def _fetch():
            df = ak.stock_news_em(symbol=symbol)
            df = df.rename(columns={
                "关键词": "keyword",
                "新闻标题": "title",
                "新闻摘要": "summary",
                "发布时间": "publish_time",
            })
            return df

        result = self._safe_fetch(_fetch)
        return result if result is not None else pd.DataFrame()
