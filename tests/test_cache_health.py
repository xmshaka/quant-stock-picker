"""缓存健康统计测试。"""
from __future__ import annotations


def test_l2_kline_cache_health_counts_new_and_old_layout(tmp_path):
    from data.cache_health import l2_kline_cache_health

    # 新结构 bars/source/adjust/prefix/symbol.parquet
    (tmp_path / "tencent" / "raw" / "60").mkdir(parents=True)
    (tmp_path / "tencent" / "raw" / "60" / "600519.parquet").write_text("x")
    (tmp_path / "tencent" / "qfq" / "60").mkdir(parents=True)
    (tmp_path / "tencent" / "qfq" / "60" / "600519.parquet").write_text("x")
    (tmp_path / "baostock" / "raw" / "60").mkdir(parents=True)
    (tmp_path / "baostock" / "raw" / "60" / "600519.parquet").write_text("x")

    # 旧结构 bars/prefix/symbol.parquet
    (tmp_path / "00").mkdir()
    (tmp_path / "00" / "000001.parquet").write_text("x")

    health = l2_kline_cache_health(tmp_path)

    assert health["new_files"] == 3
    assert health["old_files"] == 1
    assert {f"{d['source']}/{d['adjust']}": d["files"] for d in health["distribution"]} == {
        "baostock/raw": 1,
        "tencent/qfq": 1,
        "tencent/raw": 1,
    }
    assert health["old_examples"] == ["00/000001.parquet"]
