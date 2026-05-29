"""每日扫描调度器

P5: 使用 APScheduler 在每日收盘后执行全量/增量扫描，并把结果写入日志/告警文件。
P6 (2026-05-26): 增加全A股池因子预计算 job，看板可直接读快照秒开。

典型用法：
    python -m data.scheduler --once          # 立即跑一次增量扫描
    python -m data.scheduler --factor-once   # 立即跑一次全池因子预计算
    python -m data.scheduler                 # 启动常驻调度器
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import settings
from data.incremental import IncrementalUpdater, ScanReport


def _alert_path() -> Path:
    path = Path(settings.daily_scan_alert_file)
    if not path.is_absolute():
        path = settings.project_root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_scan_alert(report: ScanReport, level: str = "INFO") -> Path:
    """把扫描结果以 JSONL 形式写入告警文件。"""
    payload = asdict(report)
    payload.update({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "level": level,
        "summary": report.summary(),
    })
    path = _alert_path()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    return path


def run_daily_job(
    lookback_days: Optional[int] = None,
    max_workers: Optional[int] = None,
    symbols: Optional[list[str]] = None,
) -> ScanReport:
    """执行一次每日增量扫描，并落告警。"""
    logger.info("[Scheduler] 开始每日增量扫描")
    updater = IncrementalUpdater(max_workers=max_workers)
    report = updater.run_daily_scan(
        lookback_days=lookback_days or settings.daily_scan_lookback_days,
        symbols=symbols,
        max_workers=max_workers,
    )

    level = "ERROR" if report.failed_count >= settings.daily_scan_failure_threshold else "INFO"
    path = write_scan_alert(report, level=level)
    if level == "ERROR":
        logger.error(f"[Scheduler] 扫描失败数过高，已写告警: {path}")
    else:
        logger.success(f"[Scheduler] 扫描完成，报告已写入: {path}")
    return report


def run_daily_factor_job(max_workers: Optional[int] = None) -> dict:
    """执行一次每日全A股池因子预计算，并返回 meta。"""
    from data.daily_factors import compute_daily_factors, clean_old_factors, load_snapshot_meta

    workers = max_workers or settings.daily_factor_max_workers
    logger.info(f"[Scheduler] 开始全池因子预计算 (workers={workers})")
    try:
        compute_daily_factors(max_workers=workers)
        clean_old_factors(keep_days=settings.daily_factor_keep_days)
        meta = load_snapshot_meta() or {}
        if meta:
            logger.success(
                f"[Scheduler] 快照完成: date={meta.get('date')} "
                f"size={meta.get('universe_size')} "
                f"elapsed={meta.get('elapsed_seconds', 0):.1f}s"
            )
        else:
            logger.success("[Scheduler] 快照完成 (meta 未读取到)")
        return meta
    except Exception as e:  # pragma: no cover - 容错路径
        logger.exception(f"[Scheduler] 全池因子预计算失败: {e}")
        return {"error": str(e)}


def build_scheduler():
    """构建 BlockingScheduler。延迟导入，避免测试环境强依赖启动。"""
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler(timezone="Asia/Shanghai")
    scheduler.add_job(
        run_daily_job,
        trigger=CronTrigger(
            hour=settings.daily_scan_hour,
            minute=settings.daily_scan_minute,
            timezone="Asia/Shanghai",
        ),
        id="daily_incremental_scan",
        name="每日A股增量扫描",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )

    if settings.daily_factor_enabled:
        scheduler.add_job(
            run_daily_factor_job,
            trigger=CronTrigger(
                hour=settings.daily_factor_hour,
                minute=settings.daily_factor_minute,
                timezone="Asia/Shanghai",
            ),
            id="daily_factor_snapshot",
            name="每日全池因子预计算",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=3600,
        )
    return scheduler


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="量化选股每日扫描调度器")
    parser.add_argument("--once", action="store_true", help="立即执行一次增量扫描后退出")
    parser.add_argument("--factor-once", action="store_true", help="立即执行一次全池因子预计算后退出")
    parser.add_argument("--symbols", default="", help="逗号分隔的股票代码，仅用于测试/小范围扫描")
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] or None

    if args.factor_once:
        meta = run_daily_factor_job(max_workers=args.max_workers)
        print(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
        return 0 if not meta.get("error") else 2

    if args.once:
        report = run_daily_job(
            lookback_days=args.lookback_days,
            max_workers=args.max_workers,
            symbols=symbols,
        )
        print(report.summary())
        return 0 if report.failed_count < settings.daily_scan_failure_threshold else 2

    if not settings.daily_scan_enabled:
        logger.warning("[Scheduler] daily_scan_enabled=False，调度器未启动")
        return 0

    scheduler = build_scheduler()

    def _shutdown(signum, frame):  # pragma: no cover - 信号路径
        logger.info(f"[Scheduler] 收到信号 {signum}，准备退出")
        scheduler.shutdown(wait=False)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    logger.info(
        f"[Scheduler] 启动，每日 {settings.daily_scan_hour:02d}:"
        f"{settings.daily_scan_minute:02d} 执行增量扫描"
    )
    if settings.daily_factor_enabled:
        logger.info(
            f"[Scheduler] 每日 {settings.daily_factor_hour:02d}:"
            f"{settings.daily_factor_minute:02d} 执行全池因子预计算"
        )
    scheduler.start()
    while scheduler.running:  # pragma: no cover
        time.sleep(1)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
