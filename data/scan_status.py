"""扫描状态读取工具

读取 ``logs/daily_scan_alerts.log`` 这类 JSONL 扫描报告，供 Streamlit 看板和运维脚本展示。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from config.settings import settings


def resolve_alert_path(path: Optional[str | Path] = None) -> Path:
    """解析扫描告警文件路径。"""
    p = Path(path or settings.daily_scan_alert_file)
    if not p.is_absolute():
        p = settings.project_root / p
    return p


def read_scan_reports(path: Optional[str | Path] = None, limit: int = 50) -> list[dict[str, Any]]:
    """读取最近 N 条扫描报告。坏行会被跳过。"""
    alert_path = resolve_alert_path(path)
    if not alert_path.exists():
        return []

    lines = alert_path.read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    for line in lines[-max(limit * 3, limit):]:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                out.append(item)
        except json.JSONDecodeError:
            continue
    return out[-limit:]


def load_scan_reports(path: Optional[str | Path] = None, limit: int = 50) -> pd.DataFrame:
    """兼容旧看板接口：返回最近扫描报告 DataFrame，最新记录在前。

    Streamlit 首页和「数据状态」页历史代码使用 ``load_scan_reports()``，
    并读取 ``ts/total_symbols/updated_count/failed_count/elapsed_seconds`` 等原始字段。
    因此这里保留原始 JSON 字段，不调用 ``reports_to_frame()`` 的中文列转换。
    """
    reports = read_scan_reports(path=path, limit=limit)
    if not reports:
        return pd.DataFrame()
    return pd.DataFrame(reports).iloc[::-1].reset_index(drop=True)


def latest_scan_report(path: Optional[str | Path] = None) -> dict[str, Any]:
    """返回最近一次扫描报告；不存在则返回空 dict。"""
    reports = read_scan_reports(path=path, limit=1)
    return reports[-1] if reports else {}


def reports_to_frame(reports: list[dict[str, Any]]) -> pd.DataFrame:
    """将报告列表转为适合展示的 DataFrame。"""
    if not reports:
        return pd.DataFrame()

    rows = []
    for r in reports:
        ts = r.get("ts", "")
        rows.append({
            "时间": ts,
            "级别": r.get("level", ""),
            "股票数": r.get("total_symbols", 0),
            "已最新": r.get("skipped_up_to_date", 0),
            "成功更新": r.get("updated_count", 0),
            "失败": r.get("failed_count", 0),
            "新增行数": r.get("new_rows", 0),
            "耗时(s)": round(float(r.get("elapsed_seconds", 0) or 0), 1),
        })
    return pd.DataFrame(rows)


def scan_health(report: dict[str, Any]) -> tuple[str, str]:
    """根据最近报告给出健康状态。返回 (status, message)。"""
    if not report:
        return "UNKNOWN", "还没有扫描报告"

    level = str(report.get("level", "INFO")).upper()
    failed = int(report.get("failed_count", 0) or 0)
    total = int(report.get("total_symbols", 0) or 0)
    updated = int(report.get("updated_count", 0) or 0)
    skipped = int(report.get("skipped_up_to_date", 0) or 0)

    if level == "ERROR" or failed >= settings.daily_scan_failure_threshold:
        return "ERROR", f"失败 {failed}/{total}，超过阈值"
    if failed > 0:
        return "WARN", f"有 {failed} 只失败，需观察"
    if updated == 0 and skipped >= total and total > 0:
        return "OK", "全部已是最新"
    return "OK", f"成功更新 {updated} 只，失败 {failed} 只"


def parse_report_time(report: dict[str, Any]) -> Optional[datetime]:
    """解析报告时间。"""
    ts = report.get("ts")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts))
    except ValueError:
        return None
