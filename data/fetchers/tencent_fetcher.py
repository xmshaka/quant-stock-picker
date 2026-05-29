"""腾讯股票API获取器 - 免费、速度快、无需注册

腾讯股票API特点：
- 实时行情：qt.gtimg.cn，返回文本，解析简单
- 历史K线：web.ifzq.gtimg.cn，支持前复权
- 港股、美股也可用同接口
- 无需Token，无需注册

实时行情字段索引 (按返回顺序):
  0: market, 1: name, 2: code, 3: close, 4: pre_close, 5: open,
  6: volume(手), 7: 外盘, 8: 内盘,
  9-18: 买1~买5 (价量交替),
  19-28: 卖1~卖5 (价量交替),
  29: (空), 30: datetime, 31: change, 32: pct_change,
  33: high, 34: low, 35: 现价/量/额, 36: volume, 37: amount(元),
  38: turnover%, 39: pe_ttm, 40: (空), 41: high, 42: low,
  43: amplitude%, 44: float_mv(万), 45: total_mv(万), 46: pb,
  47: high_limit, 48: low_limit, 49: volume_ratio, ...
"""
import re
import time
from typing import Optional, List
from datetime import datetime

import requests
import pandas as pd
from loguru import logger

from .base import BaseFetcher
from data.ratelimit import SourceGateway, retry_with_backoff, RateLimitError
from data.cache_manager import CacheManager
from data.validator import Validator
from config.settings import settings


class TencentFetcher(BaseFetcher):
    """腾讯财经数据获取器"""

    BASE_QT_URL = "http://qt.gtimg.cn/q="
    BASE_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

    def __init__(self, timeout: int = 15, delay: float = 0.3):
        super().__init__()
        self.timeout = timeout
        self.delay = delay  # 保留参数兼容, 实际由 RateLimiter 控制
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://stock.finance.qq.com/"
        })
        # 治理层接入
        self.gateway = SourceGateway.get()
        self.cache = CacheManager.get()
        self.validator = Validator()

    # ── 内部工具 ──
    def _symbol_to_tencent(self, symbol: str) -> str:
        """将纯数字代码转为腾讯格式 (sh/sz前缀)
        如 600519 -> sh600519, 000858 -> sz000858
        港股 00700 -> hk00700
        """
        s = symbol.strip().lower()
        if s.startswith(("sh", "sz", "hk", "us")):
            return s
        if s.startswith(("60", "68", "51", "52")):
            return f"sh{s}"
        return f"sz{s}"

    def _safe_get(
        self,
        url: str,
        params: Optional[dict] = None,
        source: str = "tencent_qt",
    ) -> Optional[requests.Response]:
        """带限流/熔断/退避的 GET

        source: 用于选择不同的限流通道 (tencent_qt | tencent_kline)
        """
        def _do_request():
            with self.gateway.guard(source, timeout=30.0):
                resp = self.session.get(url, params=params, timeout=self.timeout)
                # 429/403/418 = 被限流, 当作错误触发熔断
                if resp.status_code in (403, 418, 429, 503):
                    raise requests.HTTPError(
                        f"HTTP {resp.status_code} (疑似限流)", response=resp
                    )
                resp.raise_for_status()
                # 腾讯返回空文本也算异常
                if not resp.text or len(resp.text) < 10:
                    raise ValueError(f"响应过短: {len(resp.text)} bytes")
                return resp

        try:
            return retry_with_backoff(
                _do_request,
                max_attempts=settings.retry_max_attempts,
                base_delay=settings.retry_base_delay,
                max_delay=settings.retry_max_delay,
                retry_on=(requests.RequestException, ValueError),
                no_retry_on=(RateLimitError,),
                name=f"tencent_{source}",
            )
        except RateLimitError as e:
            logger.error(f"[Tencent] 被熔断: {e}")
            return None
        except Exception as e:
            logger.error(f"[Tencent] 请求最终失败 {url}: {e}")
            return None

    def _parse_qt_items(self, text: str) -> List[dict]:
        """解析腾讯实时行情文本，按索引直接取字段"""
        items = []
        # 匹配 v_sh600519="...";
        pattern = r'v_([a-z]{2}\d{6,})="([^"]*)";'
        for match in re.finditer(pattern, text):
            tcode, raw = match.group(1), match.group(2)
            parts = raw.split("~")
            if len(parts) < 35:
                continue

            def _f(idx: int, default=""):
                return parts[idx] if idx < len(parts) else default

            def _flt(idx: int, default=0.0):
                try:
                    v = _f(idx)
                    return float(v) if v else default
                except (ValueError, TypeError):
                    return default

            items.append({
                "_tencent_code": tcode,
                "market": _f(0),
                "name": _f(1),
                "code": _f(2),
                "close": _flt(3),
                "pre_close": _flt(4),
                "open": _flt(5),
                "volume": _flt(6),          # 手
                "outer_vol": _flt(7),       # 外盘
                "inner_vol": _flt(8),       # 内盘
                "datetime": _f(30),
                "change": _flt(31),
                "pct_change": _flt(32),
                "high": _flt(33),
                "low": _flt(34),
                "amount": _flt(37),         # 元
                "turnover": _flt(38),
                "pe_ttm": _flt(39) if _f(39) else None,
                "amplitude": _flt(43),
                # 腾讯接口 idx44/45 单位是**亿元**, 转化为元 (与东财一致)
                "float_mv": _flt(44) * 1e8 if _f(44) else None,   # 亿->元
                "total_mv": _flt(45) * 1e8 if _f(45) else None,   # 亿->元
                "pb": _flt(46) if _f(46) else None,
                "high_limit": _flt(47),
                "low_limit": _flt(48),
                "volume_ratio": _flt(49),
                "pe_static": _flt(52) if _f(52) else None,
                "pe_lyr": _flt(53) if _f(53) else None,
            })
        return items

    # ── BaseFetcher 接口 ──
    def get_stock_list(self, universe: Optional[str] = None) -> pd.DataFrame:
        """获取股票列表 (带缓存)

        universe: all_a | hs300 | zz500 | zz1000 (默认走 settings)

        - all_a: 走 Universe.load(), 东财快照 + 基本面过滤 (推荐)
        - 其它: 走腾讯批量查 (代码走 _fetch_universe_symbols)
        """
        universe = (universe or settings.universe_source).lower()

        # 全 A 股: 直接用 Universe 模块 (一次拉全市场快照, 极快)
        if universe == "all_a":
            try:
                from data.universe import Universe
                df = Universe.load()
                # 补齐 BaseFetcher 调用方需要的字段
                expected = ["symbol", "name", "close", "pct_change",
                            "pe_ttm", "pb", "total_mv", "float_mv", "turnover"]
                for col in expected:
                    if col not in df.columns:
                        df[col] = None
                return df[expected + [c for c in df.columns if c not in expected]]
            except Exception as e:
                logger.error(f"[Tencent] Universe 模块加载失败, 降级到 hs300: {e}")
                universe = "hs300"

        # 指数成分股: 走腾讯批量查询 (保留原有逻辑)
        cache_key = f"tencent_stock_list_{universe}"

        def _fetch():
            logger.info(f"[Tencent] 拉取股票列表 universe={universe}")
            symbols = self._fetch_universe_symbols(universe)
            if not symbols:
                return pd.DataFrame()

            batch_size = 60
            all_data = []
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                tencent_codes = ",".join([self._symbol_to_tencent(s) for s in batch])
                url = f"{self.BASE_QT_URL}{tencent_codes}"
                resp = self._safe_get(url, source="tencent_qt")
                if not resp:
                    continue
                items = self._parse_qt_items(resp.text)
                for item in items:
                    all_data.append({
                        "symbol": item.get("code", ""),
                        "name": item.get("name", ""),
                        "close": item.get("close", 0),
                        "pct_change": item.get("pct_change", 0),
                        "pe_ttm": item.get("pe_ttm"),
                        "pb": item.get("pb"),
                        "total_mv": item.get("total_mv"),
                        "float_mv": item.get("float_mv"),
                        "turnover": item.get("turnover"),
                    })
                # 不需要 time.sleep, gateway 已经限流

            if not all_data:
                return pd.DataFrame(columns=[
                    "symbol", "name", "close", "pct_change",
                    "pe_ttm", "pb", "total_mv", "float_mv", "turnover"
                ])
            df = pd.DataFrame(all_data)
            df = self.validator.validate_stock_list(df)
            return df

        # 股票列表 24 小时 TTL, 盘中变化的部分用 get_realtime_quotes 单独刷
        return self.cache.get_or_fetch_snapshot(
            key=cache_key,
            fetch_fn=_fetch,
            l2_ttl_seconds=86400,
            l1_ttl_seconds=3600,
        )

    def _fetch_universe_symbols(self, universe: str) -> List[str]:
        """按 universe 类型拉取股票代码列表 (走 AKShare)"""
        try:
            import akshare as ak
        except ImportError:
            logger.error("[Tencent] 需要安装 akshare 才能取股票池")
            return []

        universe = (universe or "hs300").lower()
        try:
            if universe == "hs300":
                df = ak.index_stock_cons_weight_csindex(symbol="000300")
                return df["成分券代码"].astype(str).str.zfill(6).tolist()
            if universe == "zz500":
                df = ak.index_stock_cons_weight_csindex(symbol="000905")
                return df["成分券代码"].astype(str).str.zfill(6).tolist()
            if universe == "zz1000":
                df = ak.index_stock_cons_weight_csindex(symbol="000852")
                return df["成分券代码"].astype(str).str.zfill(6).tolist()
            # 全 A 股 (默认)
            with self.gateway.guard("akshare", timeout=30.0):
                df = ak.stock_info_a_code_name()
            symbols = df["code"].astype(str).str.zfill(6).tolist()
            if settings.exclude_bj:
                symbols = [s for s in symbols if not s.startswith(("43", "83", "87", "88", "92"))]
            return symbols
        except Exception as e:
            logger.error(f"[Tencent] 取 universe={universe} 失败: {e}, 降级 hs300")
            try:
                df = ak.index_stock_cons_weight_csindex(symbol="000300")
                return df["成分券代码"].astype(str).str.zfill(6).tolist()
            except Exception:
                return []

    def get_daily_bars(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = "qfq"
    ) -> pd.DataFrame:
        """获取个股日K线数据"""
        tcode = self._symbol_to_tencent(symbol)
        code = tcode[2:] if tcode[:2] in ("sh", "sz") else tcode
        market = tcode[:2]

        adj_map = {"qfq": "qfq", "hfq": "hfq", "": ""}
        adj = adj_map.get(adjust, "qfq")

        if start_date and end_date:
            sd = datetime.strptime(start_date, "%Y%m%d")
            ed = datetime.strptime(end_date, "%Y%m%d")
            ndays = max((ed - sd).days + 100, 500)
        else:
            ndays = 500

        params = {"param": f"{tcode},day,,,{ndays},{adj}"}
        resp = self._safe_get(self.BASE_KLINE_URL, params=params, source="tencent_kline")
        if not resp:
            return pd.DataFrame()

        try:
            data = resp.json()
            key = f"{market}{code}"
            raw_list = []
            if key in data.get("data", {}):
                stock_data = data["data"][key]
                if adj and f"{adj}day" in stock_data:
                    raw_list = stock_data[f"{adj}day"]
                elif "day" in stock_data:
                    raw_list = stock_data["day"]
                elif "data" in stock_data:
                    raw_list = stock_data["data"]

            if not raw_list:
                logger.warning(f"[Tencent] 无K线数据: {symbol}")
                return pd.DataFrame()

            rows = []
            for item in raw_list:
                if isinstance(item, list) and len(item) >= 6:
                    # 腾讯 K 线字段顺序: 日期, 开盘, 收盘, 最高, 最低, 成交量...
                    rows.append({
                        "trade_date": item[0],
                        "open": float(item[1]),
                        "close": float(item[2]),
                        "high": float(item[3]),
                        "low": float(item[4]),
                        "volume": float(item[5]) if len(item) > 5 else 0,
                    })

            df = pd.DataFrame(rows)
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df["symbol"] = code

            if start_date:
                df = df[df["trade_date"] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df["trade_date"] <= pd.to_datetime(end_date)]

            df["amount"] = 0.0
            df["pct_change"] = df["close"].pct_change() * 100
            df["change"] = df["close"].diff()

            df = df[["symbol", "trade_date", "open", "high", "low", "close",
                     "volume", "amount", "pct_change", "change"]].sort_values("trade_date").reset_index(drop=True)
            # 写入前过校验
            df = self.validator.validate_bars_df(df)
            return df

        except Exception as e:
            logger.error(f"[Tencent] 解析K线数据失败 {symbol}: {e}")
            return pd.DataFrame()

    # ── 缓存版 K 线接口 (推荐使用) ──
    def get_daily_bars_cached(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """带 L2 增量缓存的日K接口

        - 本地有就用本地, 缺什么补什么
        - 自动写入 parquet 缓存 (按 symbol 单文件)
        - 复权基准漂移会自动检测并整体重写
        """
        if not settings.cache_l2_kline_use_increment:
            return self.get_daily_bars(symbol, start_date, end_date, adjust=adjust)

        def _fetch(s: str, e: str) -> pd.DataFrame:
            return self.get_daily_bars(symbol, s, e, adjust=adjust)

        return self.cache.get_or_fetch_bars(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            fetch_fn=_fetch,
        )

    def get_sector_list(self) -> pd.DataFrame:
        """获取行业板块列表 - 用AKShare兜底"""
        logger.info("[Tencent] 板块列表使用AKShare兜底")
        try:
            import akshare as ak
            df = ak.stock_board_industry_name_em()
            df = df.rename(columns={
                "板块名称": "sector_name",
                "板块代码": "sector_code",
                "最新价": "close",
                "涨跌额": "change",
                "涨跌幅": "pct_change",
            })
            return df[["sector_name", "sector_code", "close", "pct_change"]]
        except Exception as e:
            logger.error(f"[Tencent] 获取板块列表失败: {e}")
            return pd.DataFrame()

    # ── 腾讯特有接口 ──
    def get_realtime_quotes(self, symbols: List[str]) -> pd.DataFrame:
        """批量获取实时行情 (腾讯一次支持60-80只，内部自动分批)"""
        if not symbols:
            return pd.DataFrame()

        batch_size = 60
        all_rows = []
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            tencent_codes = ",".join([self._symbol_to_tencent(s) for s in batch])
            url = f"{self.BASE_QT_URL}{tencent_codes}"
            resp = self._safe_get(url, source="tencent_qt")
            if not resp:
                continue

            items = self._parse_qt_items(resp.text)
            for item in items:
                all_rows.append({
                    "symbol": item.get("code", ""),
                    "name": item.get("name", ""),
                    "close": item.get("close", 0),
                    "open": item.get("open", 0),
                    "high": item.get("high", 0),
                    "low": item.get("low", 0),
                    "pre_close": item.get("pre_close", 0),
                    "change": item.get("change", 0),
                    "pct_change": item.get("pct_change", 0),
                    "volume": item.get("volume", 0),
                    "amount": item.get("amount", 0),
                    "pe_ttm": item.get("pe_ttm"),
                    "pb": item.get("pb"),
                    "turnover": item.get("turnover"),
                    "total_mv": item.get("total_mv"),
                    "float_mv": item.get("float_mv"),
                    "high_limit": item.get("high_limit"),
                    "low_limit": item.get("low_limit"),
                    "amplitude": item.get("amplitude"),
                    "volume_ratio": item.get("volume_ratio"),
                })
            # 限流已由 gateway 处理

        return pd.DataFrame(all_rows)

    def get_hk_stock_list(self, top_n: int = 100) -> pd.DataFrame:
        """获取港股列表 + 实时行情"""
        try:
            import akshare as ak
            df = ak.stock_hk_ggt_components_em()
            symbols = df["代码"].astype(str).str.zfill(5).tolist()[:top_n]

            batch_size = 60
            all_data = []
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                t_codes = ",".join([f"hk{s}" for s in batch])
                url = f"{self.BASE_QT_URL}{t_codes}"
                resp = self._safe_get(url, source="tencent_qt")
                if not resp:
                    continue
                items = self._parse_qt_items(resp.text)
                for item in items:
                    all_data.append({
                        "symbol": item.get("code", ""),
                        "name": item.get("name", ""),
                        "close": item.get("close", 0),
                        "pct_change": item.get("pct_change", 0),
                        "pe_ttm": item.get("pe_ttm"),
                        "pb": item.get("pb"),
                    })
                # 限流已由 gateway 处理

            return pd.DataFrame(all_data)
        except Exception as e:
            logger.error(f"[Tencent] 获取港股列表失败: {e}")
            return pd.DataFrame()
