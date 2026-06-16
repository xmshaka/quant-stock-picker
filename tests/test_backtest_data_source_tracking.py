"""回测数据源追踪测试。"""
from __future__ import annotations

import pandas as pd
from datetime import date, datetime
from dataclasses import dataclass, field
from typing import Dict, List
import json
import tempfile
from pathlib import Path


def test_scheme_backtest_result_data_source_fields():
    """测试 SchemeBacktestResult 类的数据源字段。"""
    from backtest.scheme_backtest import SchemeBacktestResult
    
    # 创建带数据源的结果
    result = SchemeBacktestResult(
        scheme_id="test_scheme",
        scheme_name="测试方案",
        start_date="2026-01-01",
        end_date="2026-06-15",
        data_source="tencent",
        data_adjust="raw",
        data_version="source=tencent, adjust=raw",
        total_return=0.1,
        annual_return=0.2,
        sharpe_ratio=1.5,
        max_drawdown=0.05,
        win_rate=0.6,
        trade_count=10,
        buy_count=5,
        sell_count=5,
        final_value=1100000.0,
        run_id="20260615_120000_test_scheme",
    )
    
    # 验证字段存在
    assert hasattr(result, "data_source")
    assert hasattr(result, "data_adjust")
    assert hasattr(result, "data_version")
    
    # 验证字段值
    assert result.data_source == "tencent"
    assert result.data_adjust == "raw"
    assert result.data_version == "source=tencent, adjust=raw"
    
    # 验证summary_text包含数据源信息
    summary = result.summary_text()
    assert "数据源: tencent/raw" in summary
    assert "数据版本: source=tencent, adjust=raw" in summary


def test_persist_data_source_in_metrics():
    """测试数据源信息是否被正确保存到metrics.json。"""
    from backtest.scheme_backtest import SchemeBacktestResult
    from backtest.records import persist_backtest_run, BacktestRunConfig, list_backtest_runs
    
    # 创建测试数据
    result = SchemeBacktestResult(
        scheme_id="test_persist",
        scheme_name="测试持久化",
        start_date="2026-01-01",
        end_date="2026-06-15",
        data_source="tushare",
        data_adjust="qfq",
        data_version="source=tushare, adjust=qfq, timestamp=20260615",
        total_return=0.15,
        annual_return=0.25,
        sharpe_ratio=1.8,
        max_drawdown=0.04,
        win_rate=0.65,
        trade_count=12,
        buy_count=6,
        sell_count=6,
        final_value=1150000.0,
        run_id="test_persist_001",
    )
    
    config = BacktestRunConfig(
        run_id="test_persist_001",
        scheme_id="test_persist",
        scheme_name="测试持久化",
        start_date="2026-01-01",
        end_date="2026-06-15",
        lookback_days=60,
        top_n=20,
        initial_capital=1000000.0,
    )
    
    # 使用临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "backtest_runs"
        tmp_path.mkdir(parents=True, exist_ok=True)
        
        # 持久化
        run_dir = persist_backtest_run(
            result=result,
            config=config,
            trades=pd.DataFrame(),
            signals_raw=pd.DataFrame(),
            signals_executed=pd.DataFrame(),
            equity=pd.DataFrame(),
            positions=pd.DataFrame(),
            factor_snapshot=pd.DataFrame(),
            root=tmp_path,
        )
        
        # 读取metrics.json
        metrics_file = run_dir / "metrics.json"
        assert metrics_file.exists()
        
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        # 验证数据源字段
        assert "data_source" in metrics
        assert "data_adjust" in metrics
        assert "data_version" in metrics
        assert metrics["data_source"] == "tushare"
        assert metrics["data_adjust"] == "qfq"
        assert metrics["data_version"] == "source=tushare, adjust=qfq, timestamp=20260615"
        
        # 验证list_backtest_runs包含数据源
        runs_df = list_backtest_runs(root=tmp_path)
        assert not runs_df.empty
        assert "data_source" in runs_df.columns
        assert "data_adjust" in runs_df.columns
        assert "data_version" in runs_df.columns
        assert runs_df.iloc[0]["data_source"] == "tushare"
        assert runs_df.iloc[0]["data_adjust"] == "qfq"


def test_data_source_formatting():
    """测试数据源格式化逻辑。"""
    import pandas as pd
    
    # 测试格式化函数
    def format_data_source(row):
        source = row["data_source"] if pd.notna(row["data_source"]) else ""
        adjust = row["data_adjust"] if pd.notna(row["data_adjust"]) else "raw"
        version = row["data_version"] if pd.notna(row["data_version"]) else ""
        if not source:
            return "未知"
        base = f"{source}/{adjust}"
        # 如果版本信息是默认格式，不显示
        default_version = f"source={source}, adjust={adjust}"
        if version and version != default_version and not version.startswith(default_version + ","):
            # 简化显示
            if len(version) > 20:
                version = version[:17] + "..."
            return f"{base} ({version})"
        return base
    
    # 测试用例
    test_cases = [
        {"data_source": "tencent", "data_adjust": "raw", "data_version": "", "expected": "tencent/raw"},
        {"data_source": "tushare", "data_adjust": "qfq", "data_version": "source=tushare, adjust=qfq", "expected": "tushare/qfq"},
        {"data_source": "akshare", "data_adjust": "hfq", "data_version": "custom_version", "expected": "akshare/hfq (custom_version)"},
        {"data_source": "", "data_adjust": "raw", "data_version": "", "expected": "未知"},
        {"data_source": "baostock", "data_adjust": "raw", "data_version": "source=baostock, adjust=raw, timestamp=20260615", "expected": "baostock/raw"},
    ]
    
    for tc in test_cases:
        row = pd.Series(tc)
        result = format_data_source(row)
        assert result == tc["expected"], f"Failed for {tc}: got {result}"
    
    print("✅ 数据源格式化测试通过")


if __name__ == "__main__":
    test_scheme_backtest_result_data_source_fields()
    print("✅ test_scheme_backtest_result_data_source_fields 通过")
    
    test_persist_data_source_in_metrics()
    print("✅ test_persist_data_source_in_metrics 通过")
    
    test_data_source_formatting()
    print("✅ test_data_source_formatting 通过")
    
    print("🎉 所有数据源追踪测试通过")
