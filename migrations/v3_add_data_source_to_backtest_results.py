"""迁移脚本 v3：向 backtest_results 表添加数据源追踪字段。

新增字段：
- data_source (String(20), default="")
- data_adjust (String(10), default="raw")
- data_version (String(40), default="")

新增索引：
- idx_bt_data_source (data_source, data_adjust)

执行方式（在项目根目录）：
1. 备份数据（可选）：pg_dump -h localhost -U postgres -d quant_picker -t backtest_results -F c -f backtest_results_backup.dump
2. 运行迁移：venv/bin/python migrations/v3_add_data_source_to_backtest_results.py
"""
import sys
sys.path.insert(0, ".")

from data.storage.models import get_engine
from sqlalchemy import text, inspect
from loguru import logger


def migration_needed(engine) -> bool:
    """检查是否需要迁移"""
    inspector = inspect(engine)
    columns = [c["name"] for c in inspector.get_columns("backtest_results")]
    return not all(col in columns for col in ["data_source", "data_adjust", "data_version"])


def main():
    engine = get_engine()
    
    if not migration_needed(engine):
        logger.info("✅ backtest_results 表已包含 data_source/data_adjust/data_version 字段，无需迁移")
        return
    
    logger.info("检测到 backtest_results 表需要添加数据源字段，开始迁移...")
    
    with engine.begin() as conn:
        # 添加字段
        conn.execute(text("""
            ALTER TABLE backtest_results 
            ADD COLUMN data_source VARCHAR(20) DEFAULT '' NOT NULL
        """))
        conn.execute(text("""
            ALTER TABLE backtest_results 
            ADD COLUMN data_adjust VARCHAR(10) DEFAULT 'raw' NOT NULL
        """))
        conn.execute(text("""
            ALTER TABLE backtest_results 
            ADD COLUMN data_version VARCHAR(40) DEFAULT '' NOT NULL
        """))
        
        # 创建索引
        conn.execute(text("""
            CREATE INDEX idx_bt_data_source ON backtest_results (data_source, data_adjust)
        """))
        
        # 更新现有记录的默认值
        conn.execute(text("""
            UPDATE backtest_results 
            SET data_source = '', data_adjust = 'raw', data_version = ''
            WHERE data_source IS NULL OR data_adjust IS NULL OR data_version IS NULL
        """))
    
    logger.info("✅ 迁移完成：已添加 data_source/data_adjust/data_version 字段到 backtest_results 表")


if __name__ == "__main__":
    main()
