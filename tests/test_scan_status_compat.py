"""扫描状态兼容接口测试。"""

import json

from data.scan_status import load_scan_reports


def test_load_scan_reports_returns_raw_fields_latest_first(tmp_path):
    log_path = tmp_path / "daily_scan_alerts.log"
    rows = [
        {
            "ts": "2026-06-13T09:00:00+08:00",
            "total_symbols": 10,
            "updated_count": 3,
            "failed_count": 1,
            "elapsed_seconds": 12.5,
        },
        {
            "ts": "2026-06-13T10:00:00+08:00",
            "total_symbols": 11,
            "updated_count": 4,
            "failed_count": 0,
            "elapsed_seconds": 8.0,
        },
    ]
    log_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    df = load_scan_reports(path=log_path)

    assert list(df["ts"]) == ["2026-06-13T10:00:00+08:00", "2026-06-13T09:00:00+08:00"]
    assert df.loc[0, "total_symbols"] == 11
    assert df.loc[0, "updated_count"] == 4
    assert df.loc[0, "failed_count"] == 0
    assert df.loc[0, "elapsed_seconds"] == 8.0


def test_load_scan_reports_missing_file_returns_empty_frame(tmp_path):
    df = load_scan_reports(path=tmp_path / "missing.log")
    assert df.empty
