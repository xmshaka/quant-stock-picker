"""股票池(Universe)管理 - 全 A 加载 + 基本面过滤

数据源策略:
- 主表: Tushare stock_basic + daily_basic (稳定股票基础信息/上市日期/估值)
- 实时行情: 腾讯批量行情补 close/amount/turnover
- 兜底: AKShare 东财快照 / AKShare 代码列表 + 腾讯行情
- 北交所: 自动识别 (代码以 43/83/87/88/92 开头)
- ST 识别: name 字段含 'ST' (含 *ST/SST/S/PT)
- 退市: name 含 '退' 或东财数据查不到

过滤维度 (全部可在 settings 配置):
- exclude_st              -> 排除 ST/*ST/SST
- exclude_delisting       -> 排除退市
- exclude_suspended       -> 排除停牌 (close=0 或成交额=0)
- exclude_new_stock_days  -> 排除上市不足 N 天的次新
- exclude_bj              -> 排除北交所
- min/max_float_mv_yi     -> 流通市值区间 (亿元)
- min_avg_turnover        -> 最小换手率 %
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
from loguru import logger

from config.settings import settings
from data.ratelimit import SourceGateway, retry_with_backoff, RateLimitError
from data.cache_manager import CacheManager


# ════════════════════════════════════════════════════════════
# 加载器
# ════════════════════════════════════════════════════════════
def _load_spot_em() -> pd.DataFrame:
    """股票池主表。优先 Tushare，AKShare 仅最后兜底。"""
    df = _load_via_tushare()
    if df is not None and not df.empty:
        return df

    logger.warning("[Universe] Tushare 股票池失败，兜底 AKShare 东财快照")
    return _load_spot_em_akshare()


def _load_via_tushare() -> pd.DataFrame:
    """Tushare 股票基础信息 + daily_basic + 腾讯实时行情。"""
    try:
        from data.fetchers.tushare_fetcher import TushareFetcher
        fetcher = TushareFetcher()
        if not getattr(fetcher, "_has_token", lambda: False)():
            logger.warning("[Universe] Tushare token 不可用，跳过 Tushare 股票池")
            return pd.DataFrame()

        base = fetcher.get_stock_list()
        if base is None or base.empty:
            logger.warning("[Universe] Tushare stock_basic 为空")
            return pd.DataFrame()
        base = base.copy()
        base["symbol"] = base["symbol"].astype(str).str.zfill(6)
        base["list_date"] = pd.to_datetime(base.get("list_date"), errors="coerce")

        # daily_basic 使用最近一个可用交易日；如果当日非交易日/未收盘，Tushare 可能返回空，不阻断股票池。
        basic = fetcher.get_daily_basic()
        if basic is not None and not basic.empty:
            basic = basic.copy()
            basic["symbol"] = basic["ts_code"].astype(str).str.slice(0, 6)
            rename = {
                "turnover_rate": "turnover",
                "pe_ttm": "pe_ttm",
                "pb": "pb",
                "total_mv": "total_mv",
                "circ_mv": "float_mv",
            }
            cols = ["symbol"] + [c for c in rename if c in basic.columns]
            basic = basic[cols].rename(columns=rename)
            # Tushare 市值单位为万元，统一转元，匹配现有过滤逻辑。
            for col in ["total_mv", "float_mv"]:
                if col in basic.columns:
                    basic[col] = pd.to_numeric(basic[col], errors="coerce") * 10000.0
            base = base.merge(basic, on="symbol", how="left")
        else:
            logger.warning("[Universe] Tushare daily_basic 为空，仅使用基础信息 + 腾讯行情")

        # 用腾讯补 close/pct_change/amount/volume；失败不阻断，至少保留 Tushare 股票基础信息。
        try:
            from data.fetchers.tencent_fetcher import TencentFetcher
            tf = TencentFetcher()
            symbols = base["symbol"].tolist()
            all_quotes = []
            batch_size = 60
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                rt = tf.get_realtime_quotes(batch)
                if rt is not None and not rt.empty:
                    all_quotes.append(rt)
                if i // batch_size % 10 == 0 and i > 0:
                    logger.info(f"[Universe] 腾讯行情补充进度 {i}/{len(symbols)}")
            if all_quotes:
                quotes = pd.concat(all_quotes, ignore_index=True).drop_duplicates(subset=["symbol"])
                quotes["symbol"] = quotes["symbol"].astype(str).str.zfill(6)
                quote_cols = [
                    "symbol", "close", "pct_change", "amount", "volume",
                    "turnover", "pe_ttm", "pb", "total_mv", "float_mv",
                ]
                quotes = quotes[[c for c in quote_cols if c in quotes.columns]]
                base = base.merge(quotes, on="symbol", how="left", suffixes=("", "_q"))
                for col in ["close", "pct_change", "amount", "volume", "turnover", "pe_ttm", "pb", "total_mv", "float_mv"]:
                    qcol = f"{col}_q"
                    if qcol in base.columns:
                        if col in base.columns:
                            base[col] = base[col].where(base[col].notna(), base[qcol])
                        else:
                            base[col] = base[qcol]
                        base = base.drop(columns=[qcol])
        except Exception as e:
            logger.warning(f"[Universe] 腾讯行情补充失败: {e}")

        defaults = {
            "close": 0, "pct_change": 0, "volume": 0, "amount": 0,
            "turnover": 0, "pe_ttm": None, "pb": None,
            "total_mv": None, "float_mv": None,
        }
        for col, default in defaults.items():
            if col not in base.columns:
                base[col] = default

        keep = [
            "symbol", "name", "close", "pct_change", "volume", "amount",
            "turnover", "pe_ttm", "pb", "total_mv", "float_mv", "list_date",
        ]
        out = base[[c for c in keep if c in base.columns]].copy()
        logger.success(f"[Universe] Tushare 股票池加载成功: {len(out)} 只")
        return out
    except Exception as e:
        logger.warning(f"[Universe] Tushare 股票池失败: {e}")
        return pd.DataFrame()


def _load_spot_em_akshare() -> pd.DataFrame:
    """AKShare 东财全 A 实时快照兜底，失败再降级腾讯批量行情。"""
    import akshare as ak
    gw = SourceGateway.get()

    def _fetch_ak():
        with gw.guard("akshare", timeout=60.0):
            df = ak.stock_zh_a_spot_em()
        return df

    try:
        df = retry_with_backoff(
            _fetch_ak,
            max_attempts=2,
            base_delay=3.0,
            max_delay=15.0,
            retry_on=(Exception,),
            no_retry_on=(RateLimitError,),
            name="ak_spot_em",
        )
        if df is not None and not df.empty:
            # 标准化列名
            rename = {
                "代码": "symbol",
                "名称": "name",
                "最新价": "close",
                "涨跌幅": "pct_change",
                "成交量": "volume",
                "成交额": "amount",
                "换手率": "turnover",
                "市盈率-动态": "pe_ttm",
                "市净率": "pb",
                "总市值": "total_mv",
                "流通市值": "float_mv",
                "60日涨跌幅": "pct_60d",
                "年初至今涨跌幅": "pct_ytd",
            }
            df = df.rename(columns=rename)
            keep = [c for c in rename.values() if c in df.columns]
            df = df[keep].copy()
            df["symbol"] = df["symbol"].astype(str).str.zfill(6)
            logger.success(f"[Universe] 东财快照加载成功: {len(df)} 只")
            return df
    except Exception as e:
        logger.warning(f"[Universe] 东财快照失败: {e}, 降级到腾讯批量行情")

    # ── 降级: 用 AKShare 拿代码列表 + 腾讯批量行情补数据 ──
    return _load_via_tencent()


def _load_via_tencent() -> pd.DataFrame:
    """降级方案: AKShare 拿全 A 代码 + 腾讯批量行情 (60 只/请求)
    
    腾讯接口没被服务器 IP 拉黑, 东财有时会拒拒绚机房 IP
    仅拿: symbol, name, close, pct_change, turnover, pe_ttm, pb, total_mv, float_mv
    """
    import akshare as ak
    gw = SourceGateway.get()

    # 1. 拿全 A 代码列表
    try:
        with gw.guard("akshare", timeout=60.0):
            code_df = ak.stock_info_a_code_name()
        symbols = code_df.rename(columns={"code": "symbol"})
        symbols["symbol"] = symbols["symbol"].astype(str).str.zfill(6)
        logger.info(f"[Universe] 降级路径: AKShare 拿到 {len(symbols)} 只代码")
    except Exception as e:
        logger.error(f"[Universe] AKShare 代码列表也失败: {e}")
        return pd.DataFrame()

    # 2. 腾讯批量行情补数据 (60 只/请求)
    from data.fetchers.tencent_fetcher import TencentFetcher
    tf = TencentFetcher()
    sym_list = symbols["symbol"].tolist()
    batch_size = 60
    all_rows = []
    for i in range(0, len(sym_list), batch_size):
        batch = sym_list[i:i + batch_size]
        try:
            rt = tf.get_realtime_quotes(batch)
            if not rt.empty:
                all_rows.append(rt)
        except Exception as e:
            logger.debug(f"[Universe] 腾讯批量行情失败 batch {i}: {e}")
        # 限流已由 gateway 控制, 但进度提示
        if i // batch_size % 10 == 0 and i > 0:
            logger.info(f"[Universe] 腾讯批量进度 {i}/{len(sym_list)}")

    if not all_rows:
        # 实时行情拿不到, 只返回代码列表 (缺价格/市值字段)
        logger.warning("[Universe] 腾讯行情全部失败, 仅返回代码列表")
        symbols["close"] = 0
        symbols["pct_change"] = 0
        symbols["turnover"] = 0
        symbols["pe_ttm"] = None
        symbols["pb"] = None
        symbols["total_mv"] = None
        symbols["float_mv"] = None
        symbols["amount"] = 0
        return symbols

    quotes = pd.concat(all_rows, ignore_index=True).drop_duplicates(subset=["symbol"])
    quotes["symbol"] = quotes["symbol"].astype(str).str.zfill(6)

    # 3. 合并: 代码列表 + 行情
    merged = symbols.merge(
        quotes[["symbol", "close", "pct_change", "turnover",
                "pe_ttm", "pb", "total_mv", "float_mv", "amount", "volume"]],
        on="symbol", how="left", suffixes=("", "_q"),
    )
    logger.success(f"[Universe] 腾讯降级路径完成: {len(merged)} 只 (其中 {quotes.shape[0]} 只有行情)")
    return merged


def _load_listing_dates() -> pd.DataFrame:
    """加载沪深两市上市日期 (symbol -> list_date)"""
    import akshare as ak
    gw = SourceGateway.get()
    frames = []

    def _fetch_sh():
        with gw.guard("akshare", timeout=60.0):
            return ak.stock_info_sh_name_code(symbol="主板A股")

    def _fetch_sh_kc():
        with gw.guard("akshare", timeout=60.0):
            return ak.stock_info_sh_name_code(symbol="科创板")

    def _fetch_sz():
        with gw.guard("akshare", timeout=60.0):
            return ak.stock_info_sz_name_code(symbol="A股列表")

    for fetcher_fn, name, code_col, date_col in [
        (_fetch_sh, "sh_main", "证券代码", "上市日期"),
        (_fetch_sh_kc, "sh_kc", "证券代码", "上市日期"),
        (_fetch_sz, "sz", "A股代码", "A股上市日期"),
    ]:
        try:
            df = retry_with_backoff(
                fetcher_fn,
                max_attempts=3, base_delay=1.0, max_delay=10.0,
                retry_on=(Exception,), no_retry_on=(RateLimitError,),
                name=f"ak_listing_{name}",
            )
            if df is None or df.empty:
                continue
            if code_col not in df.columns or date_col not in df.columns:
                logger.warning(f"[Universe] {name} 列缺失")
                continue
            sub = df[[code_col, date_col]].rename(
                columns={code_col: "symbol", date_col: "list_date"}
            )
            sub["symbol"] = sub["symbol"].astype(str).str.zfill(6)
            sub["list_date"] = pd.to_datetime(sub["list_date"], errors="coerce")
            frames.append(sub)
        except Exception as e:
            logger.warning(f"[Universe] 加载 {name} 上市日期失败: {e}")

    if not frames:
        return pd.DataFrame(columns=["symbol", "list_date"])
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["symbol"])
    return out


# ════════════════════════════════════════════════════════════
# 过滤器
# ════════════════════════════════════════════════════════════
class UniverseFilter:
    """对股票池 DataFrame 做链式过滤, 每一步都打日志记录淘汰数"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_count = len(df)
        self.steps: list[dict] = []

    def _step(self, name: str, mask: pd.Series, reason: str):
        before = len(self.df)
        self.df = self.df[mask].copy()
        after = len(self.df)
        removed = before - after
        if removed > 0:
            logger.info(f"[Universe] 过滤 {name}: -{removed} ({reason}), 剩 {after}")
        self.steps.append({
            "name": name, "before": before, "after": after, "removed": removed,
            "reason": reason,
        })
        return self

    # ── 各类过滤 ──
    def exclude_bj(self, enabled: bool):
        """北交所: 代码 43/83/87/88/92 开头"""
        if not enabled or self.df.empty:
            return self
        mask = ~self.df["symbol"].str.startswith(("43", "83", "87", "88", "92"))
        return self._step("exclude_bj", mask, "北交所")

    def exclude_st(self, enabled: bool):
        """ST / *ST / SST / S / PT"""
        if not enabled or "name" not in self.df.columns or self.df.empty:
            return self
        n = self.df["name"].fillna("").str.upper()
        mask = ~(n.str.contains("ST", na=False) | n.str.startswith("S "))
        return self._step("exclude_st", mask, "ST/*ST/SST")

    def exclude_delisting(self, enabled: bool):
        """退市整理: 名字含 '退'"""
        if not enabled or "name" not in self.df.columns or self.df.empty:
            return self
        mask = ~self.df["name"].fillna("").str.contains("退", na=False)
        return self._step("exclude_delisting", mask, "退市")

    def exclude_suspended(self, enabled: bool):
        """停牌: 当日 close=0 或 amount=0"""
        if not enabled or self.df.empty:
            return self
        mask = pd.Series(True, index=self.df.index)
        if "close" in self.df.columns:
            mask &= self.df["close"].fillna(0) > 0
        if "amount" in self.df.columns:
            mask &= self.df["amount"].fillna(0) > 0
        return self._step("exclude_suspended", mask, "停牌(无成交)")

    def exclude_new_stock(self, days: int):
        """次新股: 上市不足 N 天"""
        if days <= 0 or "list_date" not in self.df.columns or self.df.empty:
            return self
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
        # list_date 为 NaT 的 (查不到上市日期) 也保留
        mask = self.df["list_date"].isna() | (self.df["list_date"] <= cutoff)
        return self._step(
            "exclude_new_stock", mask, f"上市不足 {days} 天"
        )

    def float_mv_range(self, min_yi: float, max_yi: Optional[float]):
        """流通市值过滤 (亿元)"""
        if "float_mv" not in self.df.columns or self.df.empty:
            return self
        # 单位: 元 -> 亿元
        mv_yi = self.df["float_mv"].fillna(0) / 1e8
        mask = pd.Series(True, index=self.df.index)
        if min_yi and min_yi > 0:
            mask &= mv_yi >= min_yi
        if max_yi is not None and max_yi > 0:
            mask &= mv_yi <= max_yi
        if min_yi or max_yi:
            return self._step(
                "float_mv_range", mask,
                f"流通市值 [{min_yi}, {max_yi}] 亿"
            )
        return self

    def min_turnover(self, threshold: float):
        """最小换手率%"""
        if threshold <= 0 or "turnover" not in self.df.columns or self.df.empty:
            return self
        mask = self.df["turnover"].fillna(0) >= threshold
        return self._step(
            "min_turnover", mask, f"换手率 >= {threshold}%"
        )

    def limit(self, max_count: int):
        """截取前 N 只 (按当前顺序)"""
        if max_count <= 0 or len(self.df) <= max_count:
            return self
        before = len(self.df)
        self.df = self.df.head(max_count).copy()
        logger.info(f"[Universe] 截取前 {max_count}: {before} -> {len(self.df)}")
        self.steps.append({
            "name": "limit", "before": before, "after": len(self.df),
            "removed": before - max_count, "reason": f"最多 {max_count} 只",
        })
        return self

    def report(self) -> dict:
        return {
            "original": self.original_count,
            "final": len(self.df),
            "removed_total": self.original_count - len(self.df),
            "steps": self.steps,
        }


# ════════════════════════════════════════════════════════════
# 高层 API
# ════════════════════════════════════════════════════════════
class Universe:
    """股票池构建器 - 带缓存的全 A 加载 + 配置化过滤"""

    CACHE_KEY = "universe_all_a_filtered"
    CACHE_TTL_SECONDS = 3600 * 4  # 4 小时

    @classmethod
    def load(cls, use_cache: bool = True, **filter_overrides) -> pd.DataFrame:
        """加载并过滤股票池
        
        Args:
            use_cache: 是否使用 L1/L2 缓存
            filter_overrides: 覆盖 settings 中的过滤参数
                exclude_st, exclude_bj, exclude_delisting, exclude_suspended,
                exclude_new_stock_days, min_float_mv_yi, max_float_mv_yi,
                min_avg_turnover, max_stocks, universe_source
        """
        # 缓存 key 把过滤参数也考虑进去
        cfg_hash = cls._config_signature(filter_overrides)
        cache_key = f"{cls.CACHE_KEY}_{cfg_hash}"

        cache = CacheManager.get()

        def _build():
            return cls._build(**filter_overrides)

        if not use_cache:
            return _build()

        return cache.get_or_fetch_snapshot(
            key=cache_key,
            fetch_fn=_build,
            l2_ttl_seconds=cls.CACHE_TTL_SECONDS,
            l1_ttl_seconds=900,  # 内存 15 分钟
        )

    @classmethod
    def _build(cls, **overrides) -> pd.DataFrame:
        """实际构建逻辑"""
        cfg = cls._merged_config(overrides)
        logger.info(f"[Universe] 开始构建股票池, 配置: {cfg}")

        # 1. 主表
        df = _load_spot_em()
        if df.empty:
            logger.error("[Universe] 主表为空, 无法构建股票池")
            return df

        # 2. merge 上市日期。Tushare 主路径已带 list_date，只有缺失时才用 AKShare 兜底。
        if "list_date" not in df.columns or df["list_date"].isna().all():
            listing = _load_listing_dates()
        else:
            listing = pd.DataFrame()
        if not listing.empty:
            if "list_date" in df.columns:
                df = df.drop(columns=["list_date"])
            df = df.merge(listing, on="symbol", how="left")
        elif "list_date" not in df.columns:
            df["list_date"] = pd.NaT

        # 3. 链式过滤
        f = UniverseFilter(df)
        (f
         .exclude_bj(cfg["exclude_bj"])
         .exclude_st(cfg["exclude_st"])
         .exclude_delisting(cfg["exclude_delisting"])
         .exclude_suspended(cfg["exclude_suspended"])
         .exclude_new_stock(cfg["exclude_new_stock_days"])
         .float_mv_range(cfg["min_float_mv_yi"], cfg["max_float_mv_yi"])
         .min_turnover(cfg["min_avg_turnover"])
         .limit(cfg["max_stocks"])
        )

        report = f.report()
        logger.success(
            f"[Universe] 构建完成: {report['original']} -> {report['final']} "
            f"(过滤 {report['removed_total']})"
        )
        result = f.df.reset_index(drop=True)
        result.attrs["filter_report"] = report
        return result

    @classmethod
    def _merged_config(cls, overrides: dict) -> dict:
        return {
            "exclude_bj": overrides.get("exclude_bj", settings.exclude_bj),
            "exclude_st": overrides.get("exclude_st", settings.exclude_st),
            "exclude_delisting": overrides.get("exclude_delisting", settings.exclude_delisting),
            "exclude_suspended": overrides.get("exclude_suspended", settings.exclude_suspended),
            "exclude_new_stock_days": overrides.get(
                "exclude_new_stock_days", settings.exclude_new_stock_days
            ),
            "min_float_mv_yi": overrides.get("min_float_mv_yi", settings.min_float_mv_yi),
            "max_float_mv_yi": overrides.get("max_float_mv_yi", settings.max_float_mv_yi),
            "min_avg_turnover": overrides.get("min_avg_turnover", settings.min_avg_turnover),
            "max_stocks": overrides.get("max_stocks", settings.max_stocks),
        }

    @classmethod
    def _config_signature(cls, overrides: dict) -> str:
        """生成配置签名作为缓存 key 后缀"""
        cfg = cls._merged_config(overrides)
        import hashlib
        s = "|".join(f"{k}={v}" for k, v in sorted(cfg.items()))
        return hashlib.md5(s.encode()).hexdigest()[:10]

    @classmethod
    def list_symbols(cls, **filter_overrides) -> List[str]:
        """便捷方法: 直接返回符合条件的 symbol 列表"""
        df = cls.load(**filter_overrides)
        return df["symbol"].tolist() if not df.empty else []


# 兼容旧代码: 提供模块级便捷函数
def get_universe(use_cache: bool = True, **overrides) -> pd.DataFrame:
    return Universe.load(use_cache=use_cache, **overrides)


def get_universe_symbols(**overrides) -> List[str]:
    return Universe.list_symbols(**overrides)
