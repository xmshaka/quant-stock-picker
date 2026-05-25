"""数据校验器 - 写入缓存/DB 前的最后防线

校验项:
1. 必填字段非空
2. 价格逻辑一致性 (low <= close <= high 等)
3. 单日涨跌幅合理性
4. 成交量/额非负
5. 日期格式与连续性

使用:
    v = Validator()
    ok, msg = v.validate_bar({"close": 10, "high": 9, ...})
    if not ok:
        logger.warning(msg)
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
from loguru import logger

from config.settings import settings


class Validator:
    """行情数据校验器"""

    BAR_REQUIRED = {"symbol", "trade_date", "open", "high", "low", "close"}
    QUOTE_REQUIRED = {"symbol", "name", "close"}

    def __init__(
        self,
        price_jump_ratio: float | None = None,
        strict: bool | None = None,
    ):
        self.price_jump_ratio = price_jump_ratio or settings.validate_price_jump_ratio
        self.strict = strict if strict is not None else settings.validate_strict
        self._issues: list[dict] = []

    @property
    def issues(self) -> list[dict]:
        return self._issues

    def _record(self, symbol: str, level: str, msg: str):
        entry = {"symbol": symbol, "level": level, "msg": msg}
        self._issues.append(entry)
        if level == "error":
            logger.warning(f"[Validate] ❌ {symbol}: {msg}")
        else:
            logger.debug(f"[Validate] ⚠️ {symbol}: {msg}")

    def validate_bar(
        self, row: dict, prev_close: Optional[float] = None
    ) -> Tuple[bool, str]:
        """校验单条日 K 数据

        Returns: (ok, message)
        """
        symbol = row.get("symbol", "UNKNOWN")

        # 1. 必填字段
        for field in self.BAR_REQUIRED:
            val = row.get(field)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                msg = f"必填字段缺失: {field}"
                self._record(symbol, "error", msg)
                if self.strict:
                    return False, msg

        close = row.get("close", 0)
        high = row.get("high", 0)
        low = row.get("low", 0)
        open_ = row.get("open", 0)

        # 2. 价格逻辑
        if low > high:
            msg = f"最低价({low}) > 最高价({high})"
            self._record(symbol, "error", msg)
            if self.strict:
                return False, msg

        if close > 0:
            if close > high * 1.001:
                msg = f"收盘价({close}) > 最高价({high})"
                self._record(symbol, "error", msg)
                if self.strict:
                    return False, msg
            if close < low * 0.999:
                msg = f"收盘价({close}) < 最低价({low})"
                self._record(symbol, "error", msg)
                if self.strict:
                    return False, msg

        if open_ > 0:
            if open_ > high * 1.001 or open_ < low * 0.999:
                msg = f"开盘价({open_}) 超出高低价范围 [{low}, {high}]"
                self._record(symbol, "warn", msg)

        # 3. 涨跌幅
        if prev_close and prev_close > 0 and close > 0:
            pct = abs(close / prev_close - 1)
            if pct > self.price_jump_ratio:
                msg = f"涨跌幅超限: {pct*100:.1f}% (close={close}, prev={prev_close})"
                self._record(symbol, "warn", msg)
                # 不算 error, 因为 ST/涨跌停确实可能超预期

        # 4. 成交量
        vol = row.get("volume", 0)
        if vol is not None and vol < 0:
            msg = f"成交量为负: {vol}"
            self._record(symbol, "error", msg)
            if self.strict:
                return False, msg

        # 5. 日期格式
        td = row.get("trade_date")
        if td is not None:
            try:
                if isinstance(td, str):
                    datetime.strptime(td[:10], "%Y-%m-%d")
            except ValueError:
                msg = f"日期格式异常: {td}"
                self._record(symbol, "error", msg)
                if self.strict:
                    return False, msg

        return True, ""

    def validate_quote(self, row: dict) -> Tuple[bool, str]:
        """校验实时行情快照"""
        symbol = row.get("symbol", "UNKNOWN")

        for field in self.QUOTE_REQUIRED:
            val = row.get(field)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                msg = f"必填字段缺失: {field}"
                self._record(symbol, "error", msg)
                if self.strict:
                    return False, msg

        close = row.get("close", 0)
        if close is not None and close < 0:
            msg = f"股价为负: {close}"
            self._record(symbol, "error", msg)
            if self.strict:
                return False, msg

        return True, ""

    def validate_stock_list(self, df: pd.DataFrame) -> pd.DataFrame:
        """校验股票列表, 过滤异常行"""
        if df.empty:
            return df
        before = len(df)

        # 去除 symbol 为空或 NaN
        df = df[df["symbol"].notna() & (df["symbol"] != "")]

        # 去除 close 全为 0 且 name 为空 (接口返回的占位行)
        if "close" in df.columns and "name" in df.columns:
            df = df[~((df["close"] == 0) & (df["name"].isna() | (df["name"] == "")))]

        after = len(df)
        if before != after:
            logger.info(f"[Validate] 股票列表过滤: {before} -> {after} (移除 {before-after} 异常行)")
        return df

    def validate_bars_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """批量校验日 K DataFrame, 标记异常行但默认不过滤"""
        if df.empty:
            return df

        issues = 0
        result = df.copy()

        # 价格逻辑
        mask_bad_price = result["low"] > result["high"]
        issues += mask_bad_price.sum()

        # 收盘价超最高/低于最低
        mask_bad_close = (result["close"] > result["high"] * 1.001) | \
                         (result["close"] < result["low"] * 0.999)
        issues += mask_bad_close.sum()

        # 成交量负
        if "volume" in result.columns:
            mask_neg_vol = result["volume"] < 0
            issues += mask_neg_vol.sum()

        if issues:
            logger.info(f"[Validate] {df['symbol'].iloc[0] if 'symbol' in df.columns else '?'}: "
                        f"{issues} 条异常 (strict={self.strict})")
            if self.strict:
                bad = mask_bad_price | mask_bad_close
                if "volume" in result.columns:
                    bad = bad | mask_neg_vol
                return result[~bad].reset_index(drop=True)

        return result
