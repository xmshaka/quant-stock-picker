"""Tushare 数据获取器 - 更专业的财经数据

安装依赖: pip install tushare
配置Token: .env 中设置 TUSHARE_TOKEN=你的token
"""
try:
    import tushare as ts
except ImportError:
    ts = None

import pandas as pd
from typing import Optional
from datetime import datetime
from loguru import logger

from .base import BaseFetcher


def _get_tushare_token() -> Optional[str]:
    """尝试从环境或配置文件获取 token"""
    import os
    token = os.environ.get("TUSHARE_TOKEN", "")
    if token:
        return token
    try:
        from config.settings import settings
        token = getattr(settings, "tushare_token", "")
        if token:
            return token
    except Exception:
        pass
    return None


class TushareFetcher(BaseFetcher):
    """Tushare数据获取器

    Tushare提供更专业的财务数据，需要注册获取Token
    官网: https://tushare.pro/

    无Token或未安装tushare时会优雅降级（返回空DataFrame并提示）
    """

    def __init__(self):
        super().__init__()
        if ts is None:
            logger.warning("[Tushare] tushare 未安装，运行: pip install tushare")
            self._token = None
            self.pro = None
            return
        self._token = _get_tushare_token()
        self.pro = ts.pro_api(self._token) if self._token else None
        if not self.pro:
            logger.warning("[Tushare] Token未配置，请在.env中设置 TUSHARE_TOKEN=你的token")

    def _has_token(self) -> bool:
        return self.pro is not None

    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        if not self._has_token():
            logger.warning("[Tushare] 未配置Token，无法获取股票列表")
            return pd.DataFrame()

        def _fetch():
            df = self.pro.stock_basic(exchange='', list_status='L',
                                       fields='ts_code,symbol,name,area,industry,list_date')
            df = df.rename(columns={
                "ts_code": "ts_code",
                "symbol": "symbol",
                "name": "name",
                "area": "area",
                "industry": "industry",
                "list_date": "list_date",
            })
            return df

        result = self._safe_fetch(_fetch)
        return result if result is not None else pd.DataFrame()

    def get_daily_bars(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """获取日线数据"""
        if not self._has_token():
            logger.warning("[Tushare] 未配置Token，无法获取日线")
            return pd.DataFrame()

        ts_code = self._format_symbol(symbol)
        sd = start_date or "19900101"
        ed = end_date or datetime.now().strftime("%Y%m%d")

        def _fetch():
            df = self.pro.daily(ts_code=ts_code, start_date=sd, end_date=ed)
            df = df.rename(columns={
                "trade_date": "trade_date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "pre_close": "pre_close",
                "change": "change",
                "pct_chg": "pct_change",
                "vol": "volume",
                "amount": "amount",
            })
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df["symbol"] = symbol
            return df.sort_values("trade_date").reset_index(drop=True)

        result = self._safe_fetch(_fetch)
        return result if result is not None else pd.DataFrame()

    def get_sector_list(self) -> pd.DataFrame:
        """获取行业板块列表"""
        if not self._has_token():
            logger.warning("[Tushare] 未配置Token，无法获取板块列表")
            return pd.DataFrame()

        def _fetch():
            # Tushare用 concept 或 industry 接口
            df = self.pro.ths_index()
            df = df.rename(columns={
                "ts_code": "sector_code",
                "name": "sector_name",
            })
            return df

        result = self._safe_fetch(_fetch)
        return result if result is not None else pd.DataFrame()

    def get_daily_basic(self, trade_date: Optional[str] = None) -> pd.DataFrame:
        """获取每日指标（估值等）"""
        if not self._has_token():
            return pd.DataFrame()
        td = trade_date or datetime.now().strftime("%Y%m%d")

        def _fetch():
            df = self.pro.daily_basic(trade_date=td)
            return df

        result = self._safe_fetch(_fetch)
        return result if result is not None else pd.DataFrame()

    def get_financial_indicator(self, symbol: str) -> pd.DataFrame:
        """获取财务指标"""
        if not self._has_token():
            return pd.DataFrame()
        ts_code = self._format_symbol(symbol)

        def _fetch():
            df = self.pro.fina_indicator(ts_code=ts_code)
            return df

        result = self._safe_fetch(_fetch)
        return result if result is not None else pd.DataFrame()

    def get_money_flow(self, trade_date: Optional[str] = None) -> pd.DataFrame:
        """获取个股资金流向"""
        if not self._has_token():
            return pd.DataFrame()
        td = trade_date or datetime.now().strftime("%Y%m%d")

        def _fetch():
            df = self.pro.moneyflow(trade_date=td)
            return df

        result = self._safe_fetch(_fetch)
        return result if result is not None else pd.DataFrame()

    def get_north_flow(self, start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """获取北向资金 (沪深港通)"""
        if not self._has_token():
            return pd.DataFrame()
        sd = start_date or "20200101"
        ed = end_date or datetime.now().strftime("%Y%m%d")

        def _fetch():
            df = self.pro.hk_hold(start_date=sd, end_date=ed)
            return df

        result = self._safe_fetch(_fetch)
        return result if result is not None else pd.DataFrame()

    def get_limit_up_stocks(self, trade_date: Optional[str] = None) -> pd.DataFrame:
        """获取涨停股列表"""
        if not self._has_token():
            return pd.DataFrame()
        td = trade_date or datetime.now().strftime("%Y%m%d")

        def _fetch():
            df = self.pro.limit_list(trade_date=td)
            df = df.rename(columns={
                "ts_code": "ts_code",
                "name": "name",
                "close": "close",
                "pct_chg": "pct_change",
            })
            return df

        result = self._safe_fetch(_fetch)
        return result if result is not None else pd.DataFrame()

    def get_index_daily(self, index_code: str = "000300",
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """获取指数日线"""
        if not self._has_token():
            return pd.DataFrame()
        ts_code = self._format_symbol(index_code)
        sd = start_date or "20200101"
        ed = end_date or datetime.now().strftime("%Y%m%d")

        def _fetch():
            df = self.pro.index_daily(ts_code=ts_code, start_date=sd, end_date=ed)
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df["symbol"] = index_code
            return df

        result = self._safe_fetch(_fetch)
        return result if result is not None else pd.DataFrame()

    @staticmethod
    def _format_symbol(symbol: str) -> str:
        """将纯数字代码格式化为Tushare格式"""
        if symbol.endswith(('.SH', '.SZ', '.BJ')):
            return symbol
        if symbol.startswith('6'):
            return f"{symbol}.SH"
        elif symbol.startswith(('0', '3', '4', '8', '9')):
            return f"{symbol}.SZ"
        return f"{symbol}.SZ"
