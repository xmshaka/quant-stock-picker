"""NLP模块 - 中文情感分析 + 实体提取 + 热点评分

支持动态加载全量A股股票关键词库
"""
import re
import json
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
from functools import lru_cache

import jieba
import pandas as pd
import numpy as np
from loguru import logger


# ── 内置情感词典 (简化版，可扩展) ──
POSITIVE_WORDS = {
    "上涨", "涨停", "大涨", "飙升", "突破", "创新高", "利好", "强劲",
    "反弹", "回升", "回暖", "复苏", "增长", "盈利", "超预期", "看好",
    "买入", "增持", "推荐", "优", "强", "牛", "火爆", "火热", "飙升",
    "暴涨", "拉升", "走高", "上扬", "翻红", "领涨", "龙头", "黑马",
    "并购", "收购", "合作", "签约", "订单", "中标", "获批", "通过",
    "放量", "企稳", "反攻", "冲高", "井喷", "爆发", "放量上涨",
}

NEGATIVE_WORDS = {
    "下跌", "跌停", "大跌", "暴跌", "跳水", "创新低", "利空", "疲软",
    "回落", "回调", "降温", "衰退", "亏损", "不及预期", "看空",
    "卖出", "减持", "回避", "差", "弱", "熊", "惨淡", "低迷", "崩盘",
    "腰斩", "砸盘", "走低", "下探", "翻绿", "领跌", "暴雷", "退市",
    "调查", "处罚", "违规", "裁员", "债务", "违约", "爆亏", "巨亏",
    "缩量", "破位", "阴跌", "恐慌", "踩踏", "抛售", "放量下跌",
}

INTENSIFIERS = {
    "大幅", "剧烈", "明显", "显著", "严重", "急剧", "疯狂", "彻底",
    "全面", "深度", "超级", "极度", "绝对", "完全", "非常", "格外",
}

# ── 行业/概念关键词 ──
INDUSTRY_KEYWORDS = {
    "银行": "bank", "保险": "insurance", "券商": "broker",
    "白酒": "liquor", "医药": "pharma", "新能源": "new_energy",
    "光伏": "solar", "半导体": "semiconductor", "芯片": "chip",
    "AI": "ai", "人工智能": "ai", "算力": "computing",
    "房地产": "realestate", "汽车": "auto", "军工": "defense",
    "煤炭": "coal", "钢铁": "steel", "有色": "metal",
    "石油": "oil", "化工": "chemical", "电力": "power",
    "消费电子": "consumer_electronics", "通信": "telecom",
    "游戏": "gaming", "传媒": "media", "电商": "ecommerce",
    "稀土": "rare_earth", "锂": "lithium", "储能": "energy_storage",
    "机器人": "robotics", "无人驾驶": "autonomous_driving",
    "低空经济": "low_altitude_economy", "人形机器人": "humanoid_robot",
    "固态电池": "solid_state_battery", "6G": "6g",
}


# ── 动态加载股票关键词 ──
@lru_cache(maxsize=1)
def load_stock_keywords() -> Dict[str, str]:
    """动态加载A股股票名称映射
    
    优先从腾讯API获取，失败则用内置fallback
    """
    # 内置fallback（核心权重股）
    fallback = {
        "平安银行": "000001", "浦发银行": "600000", "招商银行": "600036",
        "工商银行": "601398", "建设银行": "601939", "农业银行": "601288",
        "贵州茅台": "600519", "五粮液": "000858", "泸州老窖": "000568",
        "洋河股份": "002304", "山西汾酒": "600809",
        "宁德时代": "300750", "比亚迪": "002594", "隆基绿能": "601012",
        "通威股份": "600438", "阳光电源": "300274",
        "中芯国际": "688981", "海康威视": "002415", "立讯精密": "002475",
        "韦尔股份": "603501", "兆易创新": "603986",
        "恒瑞医药": "600276", "药明康德": "603259", "迈瑞医疗": "300760",
        "万科": "000002", "中国平安": "601318", "中信证券": "600030",
        "东方财富": "300059", "中国中免": "601888", "爱美客": "300896",
        "迈瑞医疗": "300760", "爱尔眼科": "300015", "智飞生物": "300122",
        "万华化学": "600309", "海尔智家": "600690", "美的集团": "000333",
        "格力电器": "000651", "伊利股份": "600887", "海天味业": "603288",
        "三一重工": "600031", "中国中车": "601766", "中国建筑": "601668",
        "中国石化": "600028", "中国石油": "601857", "中国神华": "601088",
        "长江电力": "600900", "中国核电": "601985", "特变电工": "600089",
        "紫金矿业": "601899", "洛阳钼业": "603993", "北方稀土": "600111",
        "天齐锂业": "002466", "赣锋锂业": "002460", "华友钴业": "603799",
        "牧原股份": "002714", "温氏股份": "300498", "金龙鱼": "300999",
        "顺丰控股": "002352", "京沪高铁": "601816", "中远海控": "601919",
        "京东方A": "000725", "TCL科技": "000100", "科大讯飞": "002230",
        "中兴通讯": "000063", "中国移动": "600941", "中国电信": "601728",
        "中国联通": "600050", "分众传媒": "002027", "芒果超媒": "300413",
        "三七互娱": "002555", "世纪华通": "002602", "金山办公": "688111",
        "用友网络": "600588", "恒生电子": "600570", "广联达": "002410",
        "宝信软件": "600845", "深信服": "300454", "奇安信": "688561",
        "汇川技术": "300124", "汇顶科技": "603160", "卓胜微": "300782",
        "中微公司": "688012", "北方华创": "002371", "长电科技": "600584",
        "闻泰科技": "600745", "歌尔股份": "002241", "蓝思科技": "300433",
        "立讯精密": "002475", "鹏鼎控股": "002938", "工业富联": "601138",
        "药明康德": "603259", "泰格医药": "300347", "康龙化成": "300759",
        "凯莱英": "002821", "昭衍新药": "603127", "药石科技": "300725",
        "百济神州": "688235", "荣昌生物": "688331", "百克生物": "688276",
        "欧普康视": "300595", "我武生物": "300357", "长春高新": "000661",
        "通策医疗": "600763", "国际医学": "000516", "金域医学": "603882",
    }

    # 尝试从腾讯API加载
    try:
        import sys
        sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")
        from data.fetchers import TencentFetcher

        tf = TencentFetcher()
        stocks = tf.get_stock_list()
        if not stocks.empty and len(stocks) > 50:
            loaded = {}
            for _, row in stocks.iterrows():
                name = str(row.get("name", "")).strip()
                code = str(row.get("symbol", "")).strip()
                if name and code and len(code) == 6:
                    loaded[name] = code
            if len(loaded) > 100:
                logger.info(f"[NLP] 从腾讯API加载 {len(loaded)} 只股票关键词")
                # 合并fallback（确保核心股一定有）
                loaded.update(fallback)
                return loaded
    except Exception as e:
        logger.warning(f"[NLP] 动态加载股票关键词失败: {e}，使用fallback")

    return fallback


class SentimentAnalyzer:
    """情感分析器 - 基于词典规则"""

    def __init__(self):
        self.positive = POSITIVE_WORDS
        self.negative = NEGATIVE_WORDS
        self.intensifiers = INTENSIFIERS

    def analyze(self, text: str) -> Dict[str, float]:
        """
        分析文本情感

        Returns:
            {
                'score': 情感得分 (-1 ~ +1),
                'positive_count': 正向词数,
                'negative_count': 负向词数,
                'intensity': 情感强度 (0~1),
                'label': 'positive'|'negative'|'neutral'
            }
        """
        text = str(text)

        # 分词
        words = set(jieba.lcut(text))

        pos_count = len(words & self.positive)
        neg_count = len(words & self.negative)
        intens_count = len(words & self.intensifiers)

        # 计算得分
        total = pos_count + neg_count
        score = (pos_count - neg_count) / max(total, 1)

        # 强度调整
        intensity = min(1.0, (total + intens_count * 0.5) / 5)

        # 标签
        if score > 0.2:
            label = "positive"
        elif score < -0.2:
            label = "negative"
        else:
            label = "neutral"

        return {
            "score": round(score, 4),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "intensity": round(intensity, 4),
            "label": label
        }

    def analyze_batch(self, texts: List[str]) -> pd.DataFrame:
        """批量分析"""
        results = []
        for text in texts:
            r = self.analyze(text)
            r["text"] = text[:100]
            results.append(r)
        return pd.DataFrame(results)

    def analyze_news(self, news_items: List[Dict]) -> pd.DataFrame:
        """分析新闻列表"""
        results = []
        for item in news_items:
            text = item.get("title", "")
            r = self.analyze(text)
            r["title"] = text
            r["source"] = item.get("source", "")
            r["time_str"] = item.get("time_str", "")
            results.append(r)
        return pd.DataFrame(results)


class EntityExtractor:
    """实体提取器 - 提取股票、行业、概念"""

    def __init__(self, stock_keywords: Optional[Dict[str, str]] = None):
        self.stock_map = stock_keywords or load_stock_keywords()
        self.industry_map = INDUSTRY_KEYWORDS

        # 为jieba添加自定义词典
        for name in self.stock_map.keys():
            jieba.add_word(name, freq=1000)
        for ind in self.industry_map.keys():
            jieba.add_word(ind, freq=800)

    def extract_stocks(self, text: str) -> List[Dict]:
        """提取提到的股票"""
        text = str(text)
        found = []

        for name, code in self.stock_map.items():
            if name in text:
                found.append({
                    "name": name,
                    "symbol": code,
                    "type": "stock"
                })

        # 同时匹配6位数字代码
        code_pattern = r'\b(\d{6})\b'
        for match in re.finditer(code_pattern, text):
            code = match.group(1)
            if not any(s["symbol"] == code for s in found):
                found.append({
                    "name": code,
                    "symbol": code,
                    "type": "stock_code"
                })

        return found

    def extract_industries(self, text: str) -> List[Dict]:
        """提取提到的行业/概念"""
        text = str(text)
        found = []

        for name, code in self.industry_map.items():
            if name in text:
                found.append({
                    "name": name,
                    "category": code,
                    "type": "industry"
                })

        return found

    def extract_all(self, text: str) -> Dict:
        """提取所有实体"""
        return {
            "stocks": self.extract_stocks(text),
            "industries": self.extract_industries(text),
            "keywords": list(set(jieba.lcut(text)) - {" ", "\n", "\t"})
        }

    def extract_from_news(self, news_items: List[Dict]) -> pd.DataFrame:
        """从新闻列表提取实体"""
        results = []
        for item in news_items:
            text = item.get("title", "")
            entities = self.extract_all(text)

            stock_names = [s["name"] for s in entities["stocks"]]
            industry_names = [i["name"] for i in entities["industries"]]

            results.append({
                "title": text,
                "source": item.get("source", ""),
                "stocks": ",".join(stock_names) if stock_names else "",
                "industries": ",".join(industry_names) if industry_names else "",
                "stock_count": len(stock_names),
                "industry_count": len(industry_names),
            })

        return pd.DataFrame(results)

    def get_hot_sectors(self, news_items: List[Dict], top_n: int = 10) -> pd.DataFrame:
        """统计热门行业"""
        industry_counts = {}

        for item in news_items:
            text = item.get("title", "")
            industries = self.extract_industries(text)
            for ind in industries:
                name = ind["name"]
                industry_counts[name] = industry_counts.get(name, 0) + 1

        df = pd.DataFrame([
            {"industry": k, "mention_count": v}
            for k, v in industry_counts.items()
        ])

        if df.empty:
            return df

        return df.sort_values("mention_count", ascending=False).head(top_n).reset_index(drop=True)


class HotspotScorer:
    """热点得分计算"""

    def __init__(self):
        self.sentiment = SentimentAnalyzer()
        self.entity = EntityExtractor()

    def score_news(self, news_items: List[Dict]) -> pd.DataFrame:
        """
        计算每条新闻的热点得分

        得分 = 情感强度 * (股票提及数 + 行业提及数) * 时间衰减
        """
        results = []

        for item in news_items:
            text = item.get("title", "")

            # 情感
            sent = self.sentiment.analyze(text)

            # 实体
            stocks = self.entity.extract_stocks(text)
            industries = self.entity.extract_industries(text)

            # 热度得分
            entity_score = len(stocks) * 2 + len(industries)
            sentiment_weight = 1 + abs(sent["score"]) * 0.5

            heat_score = entity_score * sentiment_weight * sent["intensity"]

            results.append({
                "title": text,
                "source": item.get("source", ""),
                "sentiment_score": sent["score"],
                "sentiment_label": sent["label"],
                "intensity": sent["intensity"],
                "stocks": ",".join([s["name"] for s in stocks]),
                "industries": ",".join([i["name"] for i in industries]),
                "heat_score": round(heat_score, 2),
            })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values("heat_score", ascending=False).reset_index(drop=True)

        return df

    def get_daily_hotspots(self, news_items: List[Dict]) -> pd.DataFrame:
        """按行业聚合每日热点"""
        scored = self.score_news(news_items)
        if scored.empty:
            return scored

        # 展开行业
        hotspot_rows = []
        for _, row in scored.iterrows():
            industries = row["industries"].split(",") if row["industries"] else ["其他"]
            for ind in industries:
                if ind.strip():
                    hotspot_rows.append({
                        "industry": ind.strip(),
                        "heat_score": row["heat_score"],
                        "sentiment_score": row["sentiment_score"],
                        "title": row["title"],
                    })

        if not hotspot_rows:
            return pd.DataFrame()

        df = pd.DataFrame(hotspot_rows)

        # 聚合
        agg = df.groupby("industry").agg({
            "heat_score": "sum",
            "sentiment_score": "mean",
            "title": "count"
        }).rename(columns={"title": "news_count"}).reset_index()

        agg = agg.sort_values("heat_score", ascending=False).reset_index(drop=True)
        return agg
