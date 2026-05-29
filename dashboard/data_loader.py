"""数据加载工具 - 支持模拟数据和多真实数据源"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import os
import pickle

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Optional, Dict, Any

from loguru import logger

# ── 本地文件缓存配置 ──
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_key(source: str, n_stocks: int, n_days: int, include_symbols: Optional[List[str]] = None) -> str:
    today = datetime.now().strftime("%Y%m%d")
    if include_symbols:
        h = hash(tuple(sorted(include_symbols))) % 10000
        return f"{source}_{n_stocks}_{n_days}_inc{h}_{today}"
    return f"{source}_{n_stocks}_{n_days}_{today}"


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.pkl")


def _load_cache(key: str, ttl_hours: int = 6) -> Optional[dict]:
    """加载本地缓存，超出 TTL 返回 None"""
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    age_hours = (datetime.now().timestamp() - os.path.getmtime(path)) / 3600
    if age_hours > ttl_hours:
        logger.info(f"[Cache] 缓存过期 ({age_hours:.1f}h > {ttl_hours}h): {key}")
        return None
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"[Cache] 命中缓存: {key} (age={age_hours:.1f}h)")
        return data
    except Exception as e:
        logger.warning(f"[Cache] 读取缓存失败: {e}")
        return None


def _save_cache(key: str, data: dict) -> None:
    """保存数据到本地缓存"""
    path = _cache_path(key)
    try:
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"[Cache] 已保存缓存: {key}")
    except Exception as e:
        logger.warning(f"[Cache] 保存缓存失败: {e}")


# ── 名称映射缓存 ──
_NAME_MAP_CACHE_PATH = os.path.join(CACHE_DIR, "name_map_cache.pkl")
_NAME_MAP_CACHE_TTL_HOURS = 24


def _load_name_map_cache() -> Optional[Dict[str, str]]:
    """加载名称映射本地缓存"""
    if not os.path.exists(_NAME_MAP_CACHE_PATH):
        return None
    age_hours = (datetime.now().timestamp() - os.path.getmtime(_NAME_MAP_CACHE_PATH)) / 3600
    if age_hours > _NAME_MAP_CACHE_TTL_HOURS:
        return None
    try:
        with open(_NAME_MAP_CACHE_PATH, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"[Cache] 命中名称映射缓存 ({len(data)} 条, age={age_hours:.1f}h)")
        return data
    except Exception:
        return None


def _save_name_map_cache(name_map: Dict[str, str]) -> None:
    """保存名称映射到本地缓存"""
    try:
        with open(_NAME_MAP_CACHE_PATH, 'wb') as f:
            pickle.dump(name_map, f)
        logger.info(f"[Cache] 已保存名称映射缓存: {len(name_map)} 条")
    except Exception:
        pass

# ── 全局股票名称映射（由 load_data 自动填充） ──
NAME_MAP: Dict[str, str] = {}

# 因子中文名映射
FACTOR_NAME_MAP = {
    # 技术因子
    'rsi14': 'RSI14',
    'macd_hist': 'MACD柱状线',
    'boll_position': '布林带位置',
    'volatility_20d': '20日波动率',
    'max_dd_60d': '60日最大回撤',
    # 情绪因子
    'north_hold_change': '北向资金20日变化',
    'margin_change': '融资融券20日变化',
    'turnover_ratio': '换手率比率',
    'volume_ratio': '量比',
    # 估值因子
    'pe_ttm': '市盈率TTM',
    'pb': '市净率',
    'ep': '盈利收益率',
    # 质量因子
    'roe': '净资产收益率',
    'gross_margin': '毛利率',
    'revenue_growth': '营收增长率',
    'profit_growth': '利润增长率',
    # 动量/流动性
    'momentum_5d': '5日动量',
    'momentum_20d': '20日动量',
    'momentum_60d': '60日动量',
    'liquidity': '流动性综合',
    'reversal': '反转因子',
}


# ──────────────────────────────────────────
# 模拟数据生成器
# ──────────────────────────────────────────
def generate_mock_data(
    n_stocks: int = 50,
    n_days: int = 120,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    生成模拟数据用于演示 - 包含多类因子

    Returns:
        factor_df: DataFrame [symbol, trade_date, factor1, factor2, ...]
        price_df: DataFrame [symbol, trade_date, close]
        factor_names: list of factor names
    """
    np.random.seed(seed)

    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')
    symbols = [f"STK{i:03d}" for i in range(n_stocks)]

    # ==================== 生成价格数据 ====================
    price_data = []
    all_prices = {}

    for sym in symbols:
        base = np.random.uniform(20, 200)
        returns = np.random.normal(0.0005, 0.025, len(dates))
        for i in range(5, len(returns)):
            returns[i] += -0.1 * returns[i-1] + 0.05 * np.random.randn()
        prices = base * np.exp(np.cumsum(returns))
        all_prices[sym] = prices
        for i, d in enumerate(dates):
            price_data.append({
                'symbol': sym,
                'trade_date': d.date(),
                'close': round(prices[i], 2)
            })
    price_df = pd.DataFrame(price_data)

    # ==================== 生成因子数据 ====================
    factor_data = []

    for sym in symbols:
        prices = all_prices[sym]
        for i, d in enumerate(dates):
            row = {'symbol': sym, 'trade_date': d.date()}

            p_window = prices[max(0, i-60):i+1]
            if len(p_window) < 20:
                continue

            # 技术因子
            deltas = np.diff(p_window[-15:])
            gains = np.sum(deltas[deltas > 0]) if len(deltas) > 0 else 0
            losses = abs(np.sum(deltas[deltas < 0])) if len(deltas) > 0 else 1
            rsi = 100 - 100 / (1 + gains / max(losses, 1e-6))
            row['rsi14'] = rsi + np.random.randn() * 2

            if len(p_window) >= 26:
                ema12 = pd.Series(p_window).ewm(span=12).mean().iloc[-1]
                ema26 = pd.Series(p_window).ewm(span=26).mean().iloc[-1]
                dif = ema12 - ema26
                dea = pd.Series(p_window).ewm(span=9).mean().iloc[-1]
                row['macd_hist'] = (dif - dea) / prices[i] * 100 + np.random.randn() * 0.5
            else:
                row['macd_hist'] = np.random.randn() * 0.5

            if len(p_window) >= 20:
                ma20 = np.mean(p_window[-20:])
                std20 = np.std(p_window[-20:])
                row['boll_position'] = (prices[i] - (ma20 - 2*std20)) / (4*std20 + 1e-6) * 100 + np.random.randn() * 5
            else:
                row['boll_position'] = 50 + np.random.randn() * 10

            rets = np.diff(p_window[-20:]) / p_window[-20:-1]
            row['volatility_20d'] = np.std(rets) * np.sqrt(252) * 100

            if len(p_window) >= 60:
                peak = np.maximum.accumulate(p_window)
                dd = (p_window - peak) / peak
                row['max_dd_60d'] = np.min(dd) * 100
            else:
                row['max_dd_60d'] = np.random.uniform(-30, 0)

            # 情绪因子
            row['north_hold_change'] = np.random.randn() * 2
            row['margin_change'] = np.random.randn() * 1.5
            row['turnover_ratio'] = np.random.uniform(0.5, 5)

            # 估值因子
            row['pe_ttm'] = np.random.uniform(10, 80) + np.random.randn() * 5
            row['pb'] = np.random.uniform(0.8, 5) + np.random.randn() * 0.3
            row['ep'] = 1 / max(row['pe_ttm'], 1) * 100

            # 质量因子
            row['roe'] = np.random.uniform(5, 25) + np.random.randn() * 2
            row['gross_margin'] = np.random.uniform(10, 60) + np.random.randn() * 3
            row['revenue_growth'] = np.random.uniform(-20, 50) + np.random.randn() * 5
            row['profit_growth'] = np.random.uniform(-30, 60) + np.random.randn() * 8

            # 动量/流动性
            row['momentum_5d'] = (prices[i] / prices[max(0, i-5)] - 1) * 100 + np.random.randn() * 0.5 if i >= 5 else np.random.randn() * 2
            if len(p_window) >= 20:
                row['momentum_20d'] = (prices[i] / p_window[-20] - 1) * 100
            else:
                row['momentum_20d'] = np.random.randn() * 5

            if len(p_window) >= 60:
                row['momentum_60d'] = (prices[i] / p_window[-60] - 1) * 100
            else:
                row['momentum_60d'] = np.random.randn() * 10

            row['liquidity'] = np.random.uniform(0.3, 3)
            row['reversal'] = -(prices[i] / prices[max(0, i-5)] - 1) * 100 + np.random.randn() if i >= 5 else np.random.randn() * 2

            factor_data.append(row)

    factor_df = pd.DataFrame(factor_data)

    factor_names = [
        'rsi14', 'macd_hist', 'boll_position', 'volatility_20d', 'max_dd_60d',
        'north_hold_change', 'margin_change', 'turnover_ratio',
        'pe_ttm', 'pb', 'ep',
        'roe', 'gross_margin', 'revenue_growth', 'profit_growth',
        'momentum_5d', 'momentum_20d', 'momentum_60d', 'liquidity', 'reversal'
    ]
    factor_names = [f for f in factor_names if f in factor_df.columns]

    return factor_df, price_df, factor_names


# ──────────────────────────────────────────
# 多数据源加载器
# ──────────────────────────────────────────
class DataLoader:
    """多数据源加载器 - 支持优先级回退

    Usage:
        loader = DataLoader(preferred="tencent")
        factor_df, price_df, names = loader.load(n_stocks=100, n_days=60)

        # 或指定多个源
        loader = DataLoader(sources=["tencent", "akshare", "tushare", "mock"])
    """

    def __init__(self, sources: Optional[List[str]] = None, preferred: Optional[str] = None, include_symbols: Optional[List[str]] = None):
        """
        Args:
            sources: 数据源优先级列表，如 ["tencent", "akshare", "tushare", "mock"]
            preferred: 优先使用的单个数据源（简写，自动补全回退链）
            include_symbols: 必须包含的股票代码列表（优先加载）
        """
        if preferred:
            chain = {
                "tencent": ["tencent", "akshare", "tushare", "mock"],
                "akshare": ["akshare", "tencent", "tushare", "mock"],
                "tushare": ["tushare", "tencent", "akshare", "mock"],
                "mock": ["mock"],
            }
            self.sources = chain.get(preferred, [preferred, "mock"])
        elif sources:
            self.sources = sources
        else:
            self.sources = ["tencent", "akshare", "tushare", "mock"]

        self.include_symbols: List[str] = list(include_symbols) if include_symbols else []
        self._fetchers: Dict[str, Any] = {}
        self._init_fetchers()

    def _init_fetchers(self):
        """延迟初始化 fetcher"""
        try:
            from data.fetchers import TencentFetcher, AKShareFetcher, TushareFetcher
        except ImportError:
            try:
                sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")
                from data.fetchers import TencentFetcher, AKShareFetcher, TushareFetcher
            except ImportError as e:
                logger.warning(f"无法导入Fetcher: {e}")
                return

        for src in self.sources:
            if src in self._fetchers:
                continue
            try:
                if src == "tencent":
                    self._fetchers["tencent"] = TencentFetcher()
                elif src == "akshare":
                    self._fetchers["akshare"] = AKShareFetcher()
                elif src == "tushare":
                    self._fetchers["tushare"] = TushareFetcher()
            except Exception as e:
                logger.warning(f"初始化 {src} 失败: {e}")

    def _try_source(self, source: str, n_stocks: int, n_days: int) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, List[str]]]:
        """尝试从某个数据源获取数据"""
        if source == "mock":
            logger.info("[DataLoader] 使用模拟数据")
            return generate_mock_data(n_stocks=n_stocks, n_days=n_days)

        fetcher = self._fetchers.get(source)
        if fetcher is None:
            logger.warning(f"[DataLoader] 数据源 {source} 未初始化")
            return None

        logger.info(f"[DataLoader] 尝试从 {source} 获取数据")
        try:
            if source == "tencent":
                return self._load_from_tencent(fetcher, n_stocks, n_days)
            elif source == "akshare":
                return self._load_from_akshare(fetcher, n_stocks, n_days)
            elif source == "tushare":
                return self._load_from_tushare(fetcher, n_stocks, n_days)
        except Exception as e:
            logger.warning(f"[DataLoader] {source} 获取失败: {e}")
            return None

        return None

    def _compute_factors(self, df: pd.DataFrame, symbol: str, extra: Optional[dict] = None) -> List[dict]:
        """从价格DataFrame计算因子
        
        Args:
            extra: 额外静态数据，如 {"pe_ttm": 10.5, "pb": 1.2, "turnover": 2.5}
        """
        df = df.sort_values("trade_date").reset_index(drop=True)
        prices = df["close_num"].values
        volumes = df.get("volume", pd.Series(np.zeros(len(df)))).values
        all_factor = []

        for i, row in df.iterrows():
            if i < 20:
                continue
            p_window = prices[max(0, i-60):i+1]
            v_window = volumes[max(0, i-20):i+1]
            if len(p_window) < 20:
                continue

            frow = {
                "symbol": symbol,
                "trade_date": row["trade_date"],
                "momentum_5d": (prices[i] / prices[max(0, i-5)] - 1) * 100,
                "momentum_20d": (prices[i] / prices[max(0, i-20)] - 1) * 100,
                "momentum_60d": (prices[i] / prices[max(0, i-60)] - 1) * 100 if len(p_window) >= 60 else np.nan,
                "reversal": -(prices[i] / prices[max(0, i-5)] - 1) * 100,
                "volatility_20d": np.std(np.diff(p_window[-20:]) / p_window[-20:-1]) * np.sqrt(252) * 100 if len(p_window) >= 20 else np.nan,
            }

            # RSI
            deltas = np.diff(p_window[-15:])
            gains = np.sum(deltas[deltas > 0])
            losses = abs(np.sum(deltas[deltas < 0]))
            frow["rsi14"] = 100 - 100 / (1 + gains / max(losses, 1e-6))

            # 60日最大回撤
            if len(p_window) >= 60:
                peak = np.maximum.accumulate(p_window)
                dd = (p_window - peak) / peak
                frow["max_dd_60d"] = np.min(dd) * 100

            # 布林带
            ma20 = np.mean(p_window[-20:])
            std20 = np.std(p_window[-20:])
            frow["boll_position"] = (prices[i] - (ma20 - 2*std20)) / (4*std20 + 1e-6) * 100

            # ── 估值因子 (从 extra 传入) ──
            if extra:
                pe = extra.get("pe_ttm")
                pb = extra.get("pb")
                if pe is not None and pe > 0:
                    frow["pe_ttm"] = pe
                    frow["ep"] = 1.0 / pe  # 盈利收益率
                if pb is not None and pb > 0:
                    frow["pb"] = pb

            # ── 情绪因子 (从价格和成交量计算) ──
            # 换手率比率 (近5日平均 / 近20日平均)
            if len(v_window) >= 20 and volumes[i] > 0:
                avg_vol_5 = np.mean(volumes[max(0, i-4):i+1])
                avg_vol_20 = np.mean(volumes[max(0, i-19):i+1])
                if avg_vol_20 > 0:
                    frow["turnover_ratio"] = avg_vol_5 / avg_vol_20

            # 量比 (从 extra 传入的实时量比)
            if extra and "volume_ratio" in extra and extra["volume_ratio"] is not None:
                frow["volume_ratio"] = extra["volume_ratio"]

            # 北向资金变化 / 融资融券变化: 腾讯API无此数据
            # 如后续接入akshare北向数据，可在这里补充

            all_factor.append(frow)

        return all_factor

    def _load_from_tencent(self, fetcher, n_stocks: int, n_days: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """从腾讯数据源加载"""
        stock_list = fetcher.get_stock_list()
        if stock_list.empty or len(stock_list) < 10:
            raise ValueError("腾讯数据源返回股票列表为空或过少")

        symbols = stock_list["symbol"].head(n_stocks).tolist()
        if self.include_symbols:
            extra = [s for s in self.include_symbols if s in stock_list["symbol"].values]
            base = [s for s in symbols if s not in extra]
            symbols = extra + base

        end_date = datetime.now()
        start_date = end_date - timedelta(days=n_days + 30)
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        all_price = []
        all_factor = []

        # 批量获取实时行情（补充估值数据）
        logger.info(f"[Tencent] 获取 {len(symbols)} 只股票的实时行情补充估值数据")
        realtime_df = fetcher.get_realtime_quotes(symbols)
        extra_map = {}
        if not realtime_df.empty:
            for _, row in realtime_df.iterrows():
                sym = row.get("symbol", "")
                if sym:
                    extra_map[sym] = {
                        "pe_ttm": row.get("pe_ttm"),
                        "pb": row.get("pb"),
                        "turnover": row.get("turnover"),
                        "volume_ratio": row.get("volume_ratio"),
                    }

        for symbol in symbols:
            try:
                df = fetcher.get_daily_bars_cached(symbol, start_date=start_str, end_date=end_str) if hasattr(fetcher, "get_daily_bars_cached") else fetcher.get_daily_bars(symbol, start_date=start_str, end_date=end_str)
                if df is None or df.empty or len(df) < 20:
                    continue

                all_price.append(df[["symbol", "trade_date", "close"]].copy())

                df["close_num"] = pd.to_numeric(df["close"], errors="coerce")
                extra = extra_map.get(symbol)
                all_factor.extend(self._compute_factors(df, symbol, extra=extra))

            except Exception as e:
                logger.debug(f"[Tencent] 处理 {symbol} 失败: {e}")
                continue

        if not all_price:
            raise ValueError("腾讯数据源未获取到任何价格数据")

        price_df = pd.concat(all_price, ignore_index=True)
        factor_df = pd.DataFrame(all_factor)
        factor_df = factor_df.dropna(subset=["momentum_20d", "reversal"])

        # ── 从Tushare补充情绪数据 ──
        factor_df = self._load_sentiment_from_tushare(factor_df, start_str, end_str)

        factor_names = [c for c in factor_df.columns if c not in ("symbol", "trade_date")]
        logger.info(f"[Tencent] 因子列表: {factor_names}")
        return factor_df, price_df, factor_names

    def _load_sentiment_from_tushare(self, factor_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """从Tushare获取北向资金和融资融券数据，合并到factor_df"""
        try:
            import tushare as ts
            pro = ts.pro_api()
        except Exception:
            logger.debug("[Tushare] 未安装或未配置Token")
            return factor_df

        factor_df = factor_df.copy()

        # 1. 北向资金整体变化率 (作为全局情绪指标)
        try:
            north_df = pro.moneyflow_hsgt(start_date=start_date, end_date=end_date)
            if not north_df.empty and len(north_df) >= 2:
                north_df["trade_date"] = pd.to_datetime(north_df["trade_date"])
                north_df = north_df.sort_values("trade_date")
                north_df["north_money"] = pd.to_numeric(north_df["north_money"], errors="coerce")
                north_df = north_df.dropna(subset=["north_money"])
                if len(north_df) < 2:
                    raise ValueError("北向资金数据不足")
                # 最近1日 vs 20日前 的变化率
                recent = north_df["north_money"].iloc[-1]
                past = north_df["north_money"].iloc[-min(20, len(north_df))]
                if past != 0:
                    north_change = (recent - past) / abs(past) * 100
                else:
                    north_change = 0.0
                factor_df["north_hold_change"] = north_change
                logger.info(f"[Tushare] 北向资金20日变化率: {north_change:+.2f}%")
        except Exception as e:
            logger.debug(f"[Tushare] 北向资金获取失败: {e}")

        # 2. 个股融资融券变化率 (限制前50只，控制API调用)
        symbols = factor_df["symbol"].unique()[:50]
        margin_records = []
        for symbol in symbols:
            try:
                ts_code = self._to_tushare_code(symbol)
                df = pro.margin_detail(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if df.empty or len(df) < 2:
                    continue
                df = df.sort_values("trade_date")
                latest = float(df["rzrqye"].iloc[-1])
                past = float(df["rzrqye"].iloc[-min(20, len(df))])
                if past > 0:
                    change = (latest - past) / past * 100
                    margin_records.append({"symbol": symbol, "margin_change": change})
            except Exception:
                continue

        if margin_records:
            margin_df = pd.DataFrame(margin_records)
            factor_df = factor_df.merge(margin_df, on="symbol", how="left")
            logger.info(f"[Tushare] 融资融券数据已合并: {len(margin_records)} 只")

        return factor_df

    @staticmethod
    def _to_tushare_code(symbol: str) -> str:
        """纯数字代码转Tushare格式"""
        if symbol.startswith(("6", "68")):
            return f"{symbol}.SH"
        return f"{symbol}.SZ"

    def _load_from_akshare(self, fetcher, n_stocks: int, n_days: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """从AKShare数据源加载"""
        hs300 = fetcher.get_index_components("000300")
        if hs300.empty:
            raise ValueError("AKShare数据源返回指数成分为空")

        symbols = hs300["symbol"].head(n_stocks).tolist()
        if self.include_symbols:
            extra = [s for s in self.include_symbols if s in hs300["symbol"].values]
            base = [s for s in symbols if s not in extra]
            symbols = extra + base

        end_date = datetime.now()
        start_date = end_date - timedelta(days=n_days + 30)
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        all_price = []
        all_factor = []

        for symbol in symbols:
            try:
                df = fetcher.get_daily_bars_cached(symbol, start_date=start_str, end_date=end_str) if hasattr(fetcher, "get_daily_bars_cached") else fetcher.get_daily_bars(symbol, start_date=start_str, end_date=end_str)
                if df is None or df.empty or len(df) < 20:
                    continue

                all_price.append(df[["symbol", "trade_date", "close"]].copy())

                df["close_num"] = pd.to_numeric(df["close"], errors="coerce")
                all_factor.extend(self._compute_factors(df, symbol))

            except Exception:
                continue

        if not all_price:
            raise ValueError("AKShare数据源未获取到任何价格数据")

        price_df = pd.concat(all_price, ignore_index=True)
        factor_df = pd.DataFrame(all_factor)
        factor_df = factor_df.dropna(subset=["momentum_20d", "reversal"])

        factor_names = [c for c in factor_df.columns if c not in ("symbol", "trade_date")]
        return factor_df, price_df, factor_names

    def _load_from_tushare(self, fetcher, n_stocks: int, n_days: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """从Tushare数据源加载"""
        if not getattr(fetcher, "_has_token", lambda: False)():
            raise ValueError("Tushare未配置Token")

        stock_list = fetcher.get_stock_list()
        if stock_list.empty or len(stock_list) < 10:
            raise ValueError("Tushare返回股票列表为空或过少")

        symbols = stock_list["symbol"].head(n_stocks).tolist()
        if self.include_symbols:
            extra = [s for s in self.include_symbols if s in stock_list["symbol"].values]
            base = [s for s in symbols if s not in extra]
            symbols = extra + base

        end_date = datetime.now()
        start_date = end_date - timedelta(days=n_days + 30)
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        all_price = []
        all_factor = []

        for symbol in symbols:
            try:
                df = fetcher.get_daily_bars_cached(symbol, start_date=start_str, end_date=end_str) if hasattr(fetcher, "get_daily_bars_cached") else fetcher.get_daily_bars(symbol, start_date=start_str, end_date=end_str)
                if df is None or df.empty or len(df) < 20:
                    continue

                all_price.append(df[["symbol", "trade_date", "close"]].copy())

                df["close_num"] = pd.to_numeric(df["close"], errors="coerce")
                all_factor.extend(self._compute_factors(df, symbol))

            except Exception:
                continue

        if not all_price:
            raise ValueError("Tushare数据源未获取到任何价格数据")

        price_df = pd.concat(all_price, ignore_index=True)
        factor_df = pd.DataFrame(all_factor)
        factor_df = factor_df.dropna(subset=["momentum_20d", "reversal"])

        factor_names = [c for c in factor_df.columns if c not in ("symbol", "trade_date")]
        return factor_df, price_df, factor_names

    def load(self, n_stocks: int = 100, n_days: int = 60,
             use_cache: bool = True, cache_ttl_hours: int = 6,
             include_symbols: Optional[List[str]] = None
             ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """按优先级加载数据，自动回退，支持本地文件缓存"""
        global NAME_MAP

        if include_symbols is not None:
            self.include_symbols = list(include_symbols)

        # ── 尝试读取本地缓存 ──
        if use_cache and self.sources[0] != "mock":
            cache_key = _cache_key(self.sources[0], n_stocks, n_days, self.include_symbols)
            cached = _load_cache(cache_key, ttl_hours=cache_ttl_hours)
            if cached is not None:
                factor_df = cached["factor_df"]
                price_df = cached["price_df"]
                factor_names = cached["factor_names"]
                if cached.get("name_map"):
                    NAME_MAP.update(cached["name_map"])
                    logger.info(f"[Cache] 已恢复 {len(NAME_MAP)} 只股票名称")
                logger.info(f"[DataLoader] 从缓存加载: {len(factor_df)} 条因子, {len(price_df)} 条价格")
                return factor_df, price_df, factor_names

        # ── 从数据源加载 ──
        for source in self.sources:
            result = self._try_source(source, n_stocks, n_days)
            if result is not None and result[0] is not None and not result[0].empty:
                logger.info(f"[DataLoader] 成功从 {source} 加载数据: "
                            f"{len(result[0])} 条因子, {len(result[1])} 条价格")
                # 填充 NAME_MAP
                try:
                    sl = self.get_stock_list(source)
                    if not sl.empty and "name" in sl.columns:
                        NAME_MAP.update({
                            str(row["symbol"]): str(row["name"])
                            for _, row in sl.iterrows()
                            if pd.notna(row.get("name"))
                        })
                        logger.info(f"[DataLoader] 已加载 {len(NAME_MAP)} 只股票名称")
                except Exception:
                    pass
                # 保存缓存
                if use_cache and source != "mock":
                    cache_key = _cache_key(source, n_stocks, n_days, self.include_symbols)
                    _save_cache(cache_key, {
                        "factor_df": result[0],
                        "price_df": result[1],
                        "factor_names": result[2],
                        "name_map": dict(NAME_MAP),
                    })
                return result

        # 所有源都失败，返回mock
        logger.warning("[DataLoader] 所有数据源均失败，回退到模拟数据")
        return generate_mock_data(n_stocks=n_stocks, n_days=n_days)

    def get_stock_list(self, source: Optional[str] = None) -> pd.DataFrame:
        """获取股票列表"""
        src = source or self.sources[0]
        if src == "mock":
            return pd.DataFrame()
        fetcher = self._fetchers.get(src)
        if fetcher:
            return fetcher.get_stock_list()
        return pd.DataFrame()


# ──────────────────────────────────────────
# 兼容旧接口
# ──────────────────────────────────────────
def load_data(data_source: str = "real", prefer_snapshot: bool = True, **kwargs):
    """统一数据加载接口 (兼容旧版)

    Args:
        data_source: "mock" | "real" | "tencent" | "akshare" | "tushare"
        prefer_snapshot: True 时，若当日全池快照存在则优先读取（秒开）
    """
    # ── 优先读取每日全池因子快照 ──
    if data_source != "mock" and prefer_snapshot:
        try:
            from data.daily_factors import has_daily_factors, load_daily_factors, latest_snapshot_date
            snap_date = latest_snapshot_date()
            if snap_date and has_daily_factors(snap_date):
                factor_df, price_df, factor_names = load_daily_factors(snap_date)
                logger.info(f"[DataLoader] 命中每日全池快照 {snap_date}: {len(factor_df)} 只")
                # 补名称映射（快照无 name 列）— 优先读本地缓存，不足再从 Universe 加载
                if not NAME_MAP:
                    cached_map = _load_name_map_cache()
                    if cached_map:
                        NAME_MAP.update(cached_map)
                if len(NAME_MAP) < 4000:
                    try:
                        from data.universe import Universe
                        uni = Universe().load(use_cache=True)
                        if not uni.empty and "name" in uni.columns:
                            NAME_MAP.update({
                                str(row["symbol"]): str(row["name"])
                                for _, row in uni.iterrows()
                                if pd.notna(row.get("name"))
                            })
                            logger.info(f"[DataLoader] 已加载 {len(NAME_MAP)} 只股票名称")
                            _save_name_map_cache(dict(NAME_MAP))
                    except Exception:
                        pass
                factor_df['trade_date'] = pd.to_datetime(factor_df['trade_date'])
                price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
                return factor_df, price_df, factor_names
        except Exception as e:
            logger.warning(f"[DataLoader] 读取每日快照失败，回退实时计算: {e}")

    if data_source == "mock":
        factor_df, price_df, factor_names = generate_mock_data(**kwargs)
    else:
        preferred = data_source if data_source in ("tencent", "akshare", "tushare") else "tencent"
        loader = DataLoader(preferred=preferred)
        factor_df, price_df, factor_names = loader.load(**kwargs)

    # 统一 trade_date 为 datetime64，避免页面代码与底层模块的类型不匹配
    factor_df['trade_date'] = pd.to_datetime(factor_df['trade_date'])
    price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
    return factor_df, price_df, factor_names
