"""Baostock 数据获取器 - 免费 A 股日线兜底源

定位：
- Baostock 免费、无需 token，适合作为 Tushare/AKShare/腾讯之外的日 K 兜底。
- Baostock 不提供高质量实时全市场快照；股票列表仅返回代码/名称/上市日期等基础字段。
- 交易撮合仍应使用不复权价格；本 Fetcher 支持 adjust="" 取不复权，默认 qfq 与现有看板因子趋势计算保持兼容。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from .base import BaseFetcher
from config.settings import settings
from data.cache_manager import CacheManager
from data.bars_normalizer import normalize_daily_bars

try:
    import baostock as bs
except ImportError:  # pragma: no cover - 依赖可选
    bs = None


class BaostockFetcher(BaseFetcher):
    """Baostock 数据源适配器。"""

    def __init__(self):
        super().__init__()
        self._logged_in = False
        self.cache = CacheManager.get()
        if bs is None:
            logger.warning("[Baostock] baostock 未安装，运行: pip install baostock")

    @staticmethod
    def _health_path() -> Path:
        d = settings.cache_dir / "source_health"
        d.mkdir(parents=True, exist_ok=True)
        return d / "baostock.json"

    @classmethod
    def _write_health(cls, operation: str, ok: bool, rows: int = 0, **extra) -> None:
        payload = {
            "source": "baostock",
            "installed": bs is not None,
            "version": getattr(bs, "__version__", "unknown") if bs is not None else "missing",
            "last_checked_at": datetime.now().isoformat(timespec="seconds"),
            "last_operation": operation,
            "last_ok": bool(ok),
            "last_rows": int(rows or 0),
            **extra,
        }
        try:
            cls._health_path().write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug(f"[Baostock] 写入健康状态失败: {e}")

    @classmethod
    def status(cls) -> dict:
        """读取 Baostock 本地可用性状态，不触发外部网络请求。"""
        health = {
            "source": "baostock",
            "installed": bs is not None,
            "version": getattr(bs, "__version__", "unknown") if bs is not None else "missing",
            "last_ok": False,
            "last_rows": 0,
        }
        p = cls._health_path()
        if p.exists():
            try:
                health.update(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass
        try:
            snap_dir = settings.parquet_dir / "snapshots"
            health["cached_daily_files"] = sum(1 for _ in snap_dir.glob("baostock_daily_*.parquet")) if snap_dir.exists() else 0
        except Exception:
            health["cached_daily_files"] = 0
        return health

    def _ensure_login(self) -> bool:
        if bs is None:
            return False
        if self._logged_in:
            return True
        try:
            rs = bs.login()
            if getattr(rs, "error_code", "1") != "0":
                logger.warning(f"[Baostock] 登录失败: {getattr(rs, 'error_msg', '')}")
                return False
            self._logged_in = True
            return True
        except Exception as e:
            logger.warning(f"[Baostock] 登录异常: {e}")
            return False

    @staticmethod
    def _to_bs_code(symbol: str) -> str:
        """纯数字 A 股代码转 baostock 格式。"""
        s = str(symbol).strip().lower()
        if s.startswith(("sh.", "sz.")):
            return s
        if s.endswith((".sh", ".sz")):
            code, ex = s.split(".", 1)
            return f"{ex}.{code}"
        if s.startswith(("6", "9")):
            return f"sh.{s}"
        return f"sz.{s}"

    @staticmethod
    def _from_bs_code(code: str) -> str:
        return str(code).split(".")[-1].zfill(6)

    @staticmethod
    def _fmt_date(value: Optional[str], default: str) -> str:
        raw = value or default
        if "-" in raw:
            return raw[:10]
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"

    def _query_to_df(self, rs) -> pd.DataFrame:
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if rs.error_code != "0":
            logger.warning(f"[Baostock] 查询失败: {rs.error_msg}")
            return pd.DataFrame()
        return pd.DataFrame(rows, columns=rs.fields)

    def get_stock_list(self) -> pd.DataFrame:
        """获取当前日期 A 股基础股票列表。"""
        if not self._ensure_login():
            return pd.DataFrame()

        def _fetch():
            # query_all_stock(day=当天) 在非交易日/未来日期可能为空；query_stock_basic 更适合作为股票池基础表。
            rs = bs.query_stock_basic()
            df = self._query_to_df(rs)
            if df.empty:
                return df
            df = df.rename(columns={"code": "bs_code", "code_name": "name"})
            # type=1 股票，status=1 上市；过滤指数/退市，再保留沪深 A 股常见代码段。
            if "type" in df.columns:
                df = df[df["type"].astype(str) == "1"]
            if "status" in df.columns:
                df = df[df["status"].astype(str) == "1"]
            df = df[df["bs_code"].str.match(r"^(sh\.(60|68)|sz\.(00|30))\d{4}$", na=False)].copy()
            df["symbol"] = df["bs_code"].map(self._from_bs_code)
            df["close"] = 0.0
            return df[["symbol", "name", "close"]].reset_index(drop=True)

        result = self._safe_fetch(_fetch)
        if result is not None and not result.empty:
            self._write_health("get_stock_list", True, len(result))
            return result
        self._write_health("get_stock_list", False, 0)
        return pd.DataFrame()

    def get_daily_bars(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = "qfq",
        **kwargs,
    ) -> pd.DataFrame:
        """获取日 K。

        Args:
            adjust: qfq=前复权, hfq=后复权, ""=不复权。
        """
        if not self._ensure_login():
            return pd.DataFrame()

        bs_code = self._to_bs_code(symbol)
        sd = self._fmt_date(start_date, "19900101")
        ed = self._fmt_date(end_date, datetime.now().strftime("%Y%m%d"))
        adjustflag = {"": "3", "none": "3", "qfq": "2", "hfq": "1"}.get(str(adjust).lower(), "2")

        def _fetch():
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
                start_date=sd,
                end_date=ed,
                frequency="d",
                adjustflag=adjustflag,
            )
            df = self._query_to_df(rs)
            if df.empty:
                return df
            df = df.rename(columns={
                "date": "trade_date",
                "turn": "turnover",
                "pctChg": "pct_change",
                "preclose": "pre_close",
            })
            df["symbol"] = self._from_bs_code(bs_code)
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            for col in ["open", "high", "low", "close", "pre_close", "volume", "amount", "turnover", "pct_change"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df["change"] = df["close"] - df["pre_close"]
            cols = ["symbol", "trade_date", "open", "high", "low", "close",
                    "volume", "amount", "pct_change", "change", "turnover"]
            df = df[[c for c in cols if c in df.columns]].dropna(subset=["close"]).sort_values("trade_date").reset_index(drop=True)
            df = normalize_daily_bars(df, source="baostock", symbol=self._from_bs_code(bs_code), adjust=adjust)
            return df

        result = self._safe_fetch(_fetch)
        if result is not None and not result.empty:
            self._write_health(
                "get_daily_bars", True, len(result),
                symbol=self._from_bs_code(bs_code), start_date=sd, end_date=ed, adjust=adjust or "raw",
            )
            return result
        self._write_health(
            "get_daily_bars", False, 0,
            symbol=self._from_bs_code(bs_code), start_date=sd, end_date=ed, adjust=adjust or "raw",
        )
        return pd.DataFrame()

    def get_daily_bars_cached(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """带 L2 快照缓存的 Baostock 日 K。

        注意：不复权/前复权/后复权价格不能混入通用 bars/{symbol}.parquet，
        因此这里使用带 source+日期+复权参数的 snapshot key，避免污染撮合用不复权 K 线。
        """
        key_adjust = adjust or "raw"
        key = f"baostock_daily_{symbol}_{start_date}_{end_date}_{key_adjust}"
        cached = self.cache.l2.get_snapshot(key, ttl_seconds=86400 * 365)
        if cached is not None and not cached.empty:
            cached = cached.copy()
            if "trade_date" in cached.columns:
                cached["trade_date"] = pd.to_datetime(cached["trade_date"])
            cached.attrs["source"] = "baostock_cache"
            return cached

        df = self.get_daily_bars(symbol, start_date=start_date, end_date=end_date, adjust=adjust)
        if df is not None and not df.empty:
            df = df.copy()
            df.attrs["source"] = "baostock_api"
            self.cache.l2.set_snapshot(key, df)
        return df

    def get_sector_list(self) -> pd.DataFrame:
        """Baostock 无行业板块列表能力，返回空 DataFrame 交给降级链下一源。"""
        return pd.DataFrame()
