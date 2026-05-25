"""财经热点爬虫 - 多源抓取 + 自动回退"""
import re
import json
import time
from typing import List, Dict, Optional
from datetime import datetime
from urllib.parse import urlencode

import requests
import pandas as pd
from loguru import logger


class NewsItem:
    """单条新闻"""
    def __init__(self, title: str, source: str, time_str: str, url: str = "", content: str = ""):
        self.title = title.strip()
        self.source = source
        self.time_str = time_str
        self.url = url
        self.content = content or title
        self.crawl_time = datetime.now()

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "source": self.source,
            "time_str": self.time_str,
            "url": self.url,
            "content": self.content,
            "crawl_time": self.crawl_time.isoformat()
        }


class HotspotCrawler:
    """热点爬虫 - 多源抓取，自动回退"""

    def __init__(self, timeout: int = 15, delay: float = 0.5):
        self.timeout = timeout
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "zh-CN,zh;q=0.9",
        })

    def _get(self, url: str, params: Dict = None, headers: Dict = None) -> Optional[requests.Response]:
        """带重试的GET请求"""
        h = {**self.session.headers}
        if headers:
            h.update(headers)
        for attempt in range(3):
            try:
                time.sleep(self.delay * attempt)
                resp = self.session.get(url, params=params, headers=h, timeout=self.timeout)
                resp.raise_for_status()
                return resp
            except Exception as e:
                logger.warning(f"请求失败 {url} (attempt {attempt+1}): {e}")
        return None

    # ── 主源: 新浪RSS财经要闻 ──
    def fetch_sina_rss(self, pages: int = 3) -> List[NewsItem]:
        """新浪RSS财经要闻 - 主推荐源"""
        items = []
        for page in range(1, pages + 1):
            try:
                url = f"https://feed.sina.com.cn/api/roll/get?pageid=153&lid=2516&num=30&page={page}"
                resp = self._get(url, headers={"Referer": "https://finance.sina.com.cn/"})
                if not resp:
                    continue

                data = resp.json()
                result = data.get("result", {})
                if result.get("status", {}).get("code") != 0:
                    continue

                for item in result.get("data", []):
                    ts = item.get("ctime", "")
                    # ctime是unix时间戳
                    try:
                        dt = datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")
                    except (ValueError, TypeError):
                        dt = ""

                    items.append(NewsItem(
                        title=item.get("title", ""),
                        source="新浪财经",
                        time_str=dt,
                        url=item.get("url", ""),
                    ))

            except Exception as e:
                logger.warning(f"新浪RSS page {page} 失败: {e}")
                continue

        logger.info(f"[新浪RSS] 抓取 {len(items)} 条")
        return items

    # ── 备用源1: 东方财富公告 ──
    def fetch_eastmoney_announcement(self, pages: int = 2) -> List[NewsItem]:
        """东方财富公告 - 备用源"""
        items = []
        for page in range(1, pages + 1):
            try:
                url = f"https://np-anotice-stock.eastmoney.com/api/security/ann?page_size=30&page_index={page}&client_source=web&f_key=security"
                resp = self._get(url)
                if not resp:
                    continue

                data = resp.json()
                if data.get("success") != 1:
                    continue

                for item in data.get("data", {}).get("list", []):
                    codes = item.get("codes", [])
                    stock_name = codes[0].get("short_name", "") if codes else ""
                    title = item.get("title", "")
                    if stock_name:
                        title = f"【{stock_name}】{title}"

                    items.append(NewsItem(
                        title=title,
                        source="东方财富公告",
                        time_str=item.get("display_time", "").split(":")[0] + ":" + item.get("display_time", "").split(":")[1] if ":" in item.get("display_time", "") else item.get("display_time", ""),
                        url=f"https://data.eastmoney.com/notices/detail/{item.get('art_code', '')}.html",
                    ))

            except Exception as e:
                logger.warning(f"东方财富公告 page {page} 失败: {e}")
                continue

        logger.info(f"[东财公告] 抓取 {len(items)} 条")
        return items

    # ── 备用源2: AKShare新闻 ──
    def fetch_akshare_news(self) -> List[NewsItem]:
        """AKShare财经新闻"""
        items = []
        try:
            import akshare as ak
            df = ak.stock_news_em()
            for _, row in df.head(50).iterrows():
                items.append(NewsItem(
                    title=row.get("title", ""),
                    source="东方财富",
                    time_str=str(row.get("datetime", "")),
                    url=row.get("url", ""),
                ))
            logger.info(f"[AKShare] 抓取 {len(items)} 条")
        except Exception as e:
            logger.warning(f"AKShare新闻获取失败: {e}")
        return items

    # ── 备用源3: 新浪API v2 ──
    def fetch_sina_api_v2(self, pages: int = 2) -> List[NewsItem]:
        """新浪API v2 - 另一种接口"""
        items = []
        for page in range(1, pages + 1):
            try:
                url = f"https://interface.sina.cn/wap_api/layout_col.d.json?showcid=56592&col=56592&show_num=30&page={page}"
                resp = self._get(url)
                if not resp:
                    continue

                data = resp.json()
                result = data.get("result", {})
                if result.get("status", {}).get("code") != 0:
                    continue

                for item in result.get("data", {}).get("list", []):
                    items.append(NewsItem(
                        title=item.get("title", ""),
                        source="新浪财经v2",
                        time_str="",
                        url=item.get("URL", ""),
                    ))

            except Exception as e:
                logger.warning(f"新浪APIv2 page {page} 失败: {e}")
                continue

        logger.info(f"[新浪APIv2] 抓取 {len(items)} 条")
        return items

    # ── 综合抓取 ──
    def fetch_all(self, max_items: int = 100) -> List[NewsItem]:
        """抓取所有来源，自动回退"""
        all_items = []

        # 主源
        all_items.extend(self.fetch_sina_rss(pages=3))

        # 如果主源不足，启用备用
        if len(all_items) < max_items // 2:
            logger.info("主源不足，启用备用源...")
            all_items.extend(self.fetch_eastmoney_announcement(pages=2))
            all_items.extend(self.fetch_sina_api_v2(pages=2))
            all_items.extend(self.fetch_akshare_news())

        # 去重 (标题前30字)
        seen = set()
        unique = []
        for item in all_items:
            key = item.title[:30]
            if key not in seen:
                seen.add(key)
                unique.append(item)

        # 按时间排序（如果有的话）
        unique.sort(key=lambda x: x.time_str, reverse=True)

        logger.info(f"[爬虫] 共抓取 {len(unique)} 条去重新闻")
        return unique[:max_items]

    def to_dataframe(self, items: List[NewsItem]) -> pd.DataFrame:
        """转为DataFrame"""
        return pd.DataFrame([item.to_dict() for item in items])
