"""测试热点聚合器"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

from loguru import logger
logger.remove()
logger.add(sys.stdout, level="INFO")

from hotspot.aggregator import HotspotAggregator

def main():
    print("=" * 50)
    print("测试热点聚合器")
    print("=" * 50)

    agg = HotspotAggregator(max_news=50)
    result = agg.run()

    print("\n--- 情感统计 ---")
    print(result["sentiment_summary"])

    print("\n--- 热门新闻 Top 10 ---")
    scored = result["scored"]
    if not scored.empty:
        for _, row in scored.head(10).iterrows():
            emoji = "🟢" if row["sentiment_label"] == "positive" else "🔴" if row["sentiment_label"] == "negative" else "⚪"
            stocks = f" [{row['stocks']}]" if row["stocks"] else ""
            inds = f" ({row['industries']})" if row["industries"] else ""
            print(f"  {emoji} {row['title'][:50]}{stocks}{inds} | 热度:{row['heat_score']}")
    else:
        print("  无数据")

    print("\n--- 热门行业 Top 10 ---")
    sectors = result["hot_sectors"]
    if not sectors.empty:
        for _, row in sectors.head(10).iterrows():
            print(f"  {row['industry']}: 热度{row['heat_score']:.1f} 新闻{row['news_count']}条 情感{row['sentiment_score']:.2f}")
    else:
        print("  无数据")

    print("\n--- 热门股票 Top 10 ---")
    stocks = result["hot_stocks"]
    if not stocks.empty:
        for _, row in stocks.head(10).iterrows():
            emoji = "🟢" if row["avg_sentiment"] > 0.2 else "🔴" if row["avg_sentiment"] < -0.2 else "⚪"
            print(f"  {emoji} {row['stock']}: 热度{row['heat_score']:.1f} 提及{row['mention_count']}次 情感{row['avg_sentiment']:.2f}")
    else:
        print("  无数据")

    print("\n✅ 测试完成")

if __name__ == "__main__":
    main()
