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
from datetime import datetime, timedelta
from loguru import logger

from .base import BaseFetcher
from data.cache_manager import CacheManager
from data.bars_normalizer import normalize_daily_bars


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
            if df is None or df.empty:
                logger.debug(f"[Tushare] daily {ts_code} {sd}-{ed} 返回空")
                return pd.DataFrame()
            if "trade_date" not in df.columns:
                logger.warning(f"[Tushare] daily {ts_code} 缺少 trade_date 列: {list(df.columns)}")
                return pd.DataFrame()
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
            # Tushare daily 为不复权；volume=手，amount=千元，统一交给标准化层转为股/元。
            df = normalize_daily_bars(df, source="tushare", symbol=symbol, adjust="raw")
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
        """获取每日指标（估值等），带本地 Parquet 缓存。

        - 显式 trade_date：历史日数据近似不可变，优先读本地缓存，避免重复消耗 Tushare 积分。
        - 默认 trade_date=None：从今天向前回看最近 10 天，逐日先查缓存，再请求接口。
        - 返回值 attrs["trade_date"] / attrs["source"] 用于前台暴露真实数据来源。
        """
        if not self._has_token():
            return pd.DataFrame()
        if trade_date:
            dates = [trade_date]
        else:
            today = datetime.now()
            dates = [(today - timedelta(days=i)).strftime("%Y%m%d") for i in range(10)]

        cache = CacheManager.get()
        ttl_seconds = 86400 * 30 if trade_date else 86400 * 7

        for td in dates:
            cache_key = f"tushare_daily_basic_{td}"
            cached = cache.l2.get_snapshot(cache_key, ttl_seconds=ttl_seconds)
            if cached is not None and not cached.empty:
                cached = cached.copy()
                cached.attrs["trade_date"] = td
                cached.attrs["source"] = "tushare_daily_basic_cache"
                logger.info(f"[Tushare] daily_basic 命中本地缓存 {td}: {len(cached)} 条")
                return cached

            result = self._fetch_daily_basic_once(td)
            if result is not None and not result.empty:
                result = result.copy()
                result.attrs["trade_date"] = td
                result.attrs["source"] = "tushare_daily_basic_api"
                cache.l2.set_snapshot(cache_key, result)
                logger.info(f"[Tushare] daily_basic 已缓存 {td}: {len(result)} 条")
                return result
            if trade_date:
                logger.debug(f"[Tushare] daily_basic {td} 为空")
            else:
                logger.debug(f"[Tushare] daily_basic {td} 为空，尝试前一日")

        return pd.DataFrame()

    def _fetch_daily_basic_once(self, trade_date: str) -> pd.DataFrame:
        """拉取单日 daily_basic。

        Tushare 对尚未发布/非交易日的 daily_basic 会正常返回空表，
        这不是接口异常，不能走 BaseFetcher._safe_fetch，否则会被误记为 ERROR。
        """
        max_retries = 3
        for i in range(max_retries):
            try:
                df = self.pro.daily_basic(trade_date=trade_date)
                if df is None or df.empty:
                    logger.debug(f"[Tushare] daily_basic {trade_date} 返回空")
                    return pd.DataFrame()
                if "trade_date" not in df.columns:
                    # 部分 Tushare/Mock 返回不带 trade_date；daily_basic 的日期由请求参数唯一确定。
                    logger.debug(f"[Tushare] daily_basic {trade_date} 缺少 trade_date 列，使用请求日期补齐")
                    df = df.copy()
                    df["trade_date"] = trade_date
                return df
            except Exception as e:
                logger.warning(f"[Tushare] daily_basic {trade_date} 第{i + 1}次获取异常: {e}")
        logger.error(f"[Tushare] daily_basic {trade_date} 获取异常，已达最大重试次数")
        return pd.DataFrame()

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
            if df is None or df.empty:
                logger.debug(f"[Tushare] index_daily {ts_code} {sd}-{ed} 返回空")
                return pd.DataFrame()
            if "trade_date" not in df.columns:
                logger.warning(f"[Tushare] index_daily {ts_code} 缺少 trade_date 列: {list(df.columns)}")
                return pd.DataFrame()
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
