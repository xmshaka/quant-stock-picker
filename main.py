"""最小闭环验证脚本

用 AKShare 真实数据跑通：拉数据 → 算因子 → 打分 → 输出 Top 股票
"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import akshare as ak

from factors.base import FactorRegistry
from factors.valuation import *  # noqa: F401,F403
from factors.momentum import *   # noqa: F401,F403
from factors.quality import *    # noqa: F401,F403
from factors.technical import *  # noqa: F401,F403
from models.stock_picker import MultiFactorScorer


def main():
    print("=" * 60)
    print("量化选股系统 - 最小闭环验证")
    print("=" * 60)
    
    # 拉取A股列表
    print("\n[1/4] 拉取A股列表...")
    stock_list = ak.stock_zh_a_spot_em()
    stock_list = stock_list[stock_list["代码"].str.match(r"^\d{6}$")]
    print(f"  A股列表: {len(stock_list)} 只股票")
    
    # 缩小范围到沪深300成分股，加快速度
    print("\n[2/4] 拉取实时行情...")
    try:
        hs300 = ak.index_stock_cons_weight_csindex(symbol="000300")
        hs300_symbols = set(hs300["成分券代码"].astype(str).str.zfill(6).tolist())
        symbols = list(hs300_symbols)[:100]  # 取前100只
        print(f"  沪深300成分股: {len(symbols)} 只")
    except Exception as e:
        print(f"  获取沪深300失败，用A股前50只: {e}")
        symbols = stock_list["代码"].tolist()[:50]
    
    # 用spot_em的实时行情（速度快，覆盖全）
    df_spot = ak.stock_zh_a_spot_em()
    df_spot = df_spot[df_spot["代码"].isin(symbols)].copy()
    df_spot["symbol"] = df_spot["代码"]
    df_spot["trade_date"] = datetime.now().date()
    
    # 列名映射到统一格式
    rename_map = {
        "今开": "open", "最高": "high", "最低": "low", 
        "最新价": "close", "成交量": "volume", "成交额": "amount",
        "涨跌幅": "pct_change", "换手率": "turnover",
        "市盈率-动态": "pe_ttm", "市净率": "pb",
        "总市值": "total_mv", "流通市值": "float_mv"
    }
    df_spot = df_spot.rename(columns=rename_map)
    
    cols = ["symbol", "trade_date", "open", "high", "low", "close", 
            "volume", "amount", "pct_change", "turnover",
            "pe_ttm", "pb", "total_mv", "float_mv"]
    df = df_spot[[c for c in cols if c in df_spot.columns]].copy()
    
    print(f"  实时行情: {len(df)} 只股票")
    if df.empty:
        print("  无行情数据，退出")
        return
    
    # 转换数值类型
    for col in ["open", "high", "low", "close", "volume", "amount", 
                "pct_change", "turnover", "pe_ttm", "pb", "total_mv", "float_mv"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    print("\n[3/4] 计算因子...")
    factors = FactorRegistry.build_all()
    print(f"  已注册因子: {len(factors)} 个")
    
    available_cols = set(df.columns)
    results = []
    for f in factors:
        try:
            if f.group == "valuation":
                needed = {"pe_ttm", "pb"}
            elif f.group == "quality":
                needed = {"roe", "roa", "gross_margin", "net_margin", 
                         "revenue_growth", "profit_growth"}
            elif f.group in ("momentum", "technical"):
                needed = {"close"}  # 截面数据无法计算，需要历史K线
            else:
                needed = set()
            
            if needed.issubset(available_cols):
                result = f.calculate(df)
                if result is not None and not result.values.empty:
                    results.append(result)
                    print(f"    ✅ {f.name} ({f.group}): {len(result.values)} 只")
            else:
                print(f"    ⏭️ {f.name} ({f.group}): 缺 {needed - available_cols}")
        except Exception as e:
            print(f"    ❌ {f.name}: {e}")
    
    if not results:
        print("  没有因子可计算，退出")
        return
    
    # 多因子打分
    print("\n[4/4] 多因子打分...")
    scorer = MultiFactorScorer(factor_weights={})
    
    factor_matrix = pd.DataFrame({r.name: r.ranked for r in results})
    factor_matrix = factor_matrix.fillna(0.5)
    
    scores = scorer.score(factor_matrix)
    top20 = scores.head(20)
    
    print(f"\n{'='*60}")
    print("🏆 Top 20 选股结果")
    print(f"{'='*60}")
    
    name_map = stock_list.set_index("代码")["名称"].to_dict()
    for i, symbol in enumerate(top20.index, 1):
        name = name_map.get(symbol, "未知")
        score = top20.loc[symbol, "total_score"]
        pe = df[df["symbol"]==symbol]["pe_ttm"].values[0] if "pe_ttm" in df.columns else "N/A"
        pb = df[df["symbol"]==symbol]["pb"].values[0] if "pb" in df.columns else "N/A"
        mv = df[df["symbol"]==symbol]["total_mv"].values[0] if "total_mv" in df.columns else 0
        mv_str = f"{mv/1e8:.1f}亿" if isinstance(mv, (int, float)) and mv > 0 else "N/A"
        print(f"  {i:2d}. {symbol} {name:8s} | 得分:{score:.4f} | PE:{pe} PB:{pb} | 市值:{mv_str}")
    
    print(f"\n{'='*60}")
    print("✅ 最小闭环验证完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
