#!/usr/bin/env python3
"""迁移脚本：在 stock_bars 表添加 source/adjust 字段，并调整唯一索引。

注意：这是一个破坏性变更，因为旧表有唯一约束 (symbol, trade_date)。
如果已存在冲突数据（例如同一 symbol+date 有 raw 和 qfq），可能需要先清理。

步骤：
1. 备份当前表（可选）
2. 添加字段
3. 调整唯一索引
4. 为现有数据填充默认值

**使用前注意事项**：
- 确保 PG 连接正常
- 确保当前应用已停止
- 可选：先运行 `python -m data.storage.models` 检查表是否存在
- 可选：创建备份表 `pg_dump -t stock_bars -U quant_user quant_db > stock_bars_backup.sql`

**执行**：
PG 连接信息来自 settings 的 database_url。

调用方式（两种）：
1. 直接运行： python migrations/v2_add_source_adjust_to_stock_bars.py
2. 通过 Alembic（推荐但当前未集成）：先配置 Alembic 再生成迁移
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.storage.models import Base, StockBar
from config.settings import settings
from sqlalchemy import create_engine, text
from sqlalchemy.schema import AddConstraint, DropConstraint, CreateIndex, DropIndex
from sqlalchemy.schema import MetaData

engine = create_engine(settings.database_url, echo=True)


def check_table_exists(engine, table_name):
    """检查表是否存在"""
    with engine.connect() as conn:
        res = conn.execute(
            text("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = :name)"),
            {"name": table_name}
        )
        return bool(res.scalar())


def get_current_constraints(engine, table_name):
    """获取当前表的唯一约束"""
    with engine.connect() as conn:
        # PostgreSQL 查询唯一约束
        sql = """
        SELECT conname, conkey, contype, pg_get_constraintdef(c.oid) as def
        FROM pg_constraint c
        JOIN pg_class t ON c.conrelid = t.oid
        WHERE t.relname = :table_name AND c.contype IN ('u','p')
        ORDER BY conname;
        """
        res = conn.execute(text(sql), {"table_name": table_name})
        return [dict(row) for row in res]


def get_current_indices(engine, table_name):
    """获取当前表的索引"""
    with engine.connect() as conn:
        sql = """
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE tablename = :table_name
        ORDER BY indexname;
        """
        res = conn.execute(text(sql), {"table_name": table_name})
        return [dict(row) for row in res]


def run_migration():
    """执行迁移"""
    if not check_table_exists(engine, "stock_bars"):
        print("表 stock_bars 不存在，跳过迁移。")
        return

    print("当前唯一约束:")
    for c in get_current_constraints(engine, "stock_bars"):
        print(f"  {c['conname']}: {c['def']}")
    print("当前索引:")
    for idx in get_current_indices(engine, "stock_bars"):
        print(f"  {idx['indexname']}")

    # 1. 删除旧唯一约束
    with engine.begin() as conn:
        conn.execute(text("""
        ALTER TABLE stock_bars DROP CONSTRAINT IF EXISTS uix_symbol_date;
        """))
        print("已删除旧约束 uix_symbol_date")

    # 2. 删除旧索引
    with engine.begin() as conn:
        conn.execute(text("""
        DROP INDEX IF EXISTS idx_symbol_date;
        """))
        print("已删除旧索引 idx_symbol_date")

    # 3. 添加 source 和 adjust 字段
    with engine.begin() as conn:
        # source: 默认 ''，原表已有数据 source 会留空，兼容旧链路
        conn.execute(text("""
        ALTER TABLE stock_bars ADD COLUMN IF NOT EXISTS source VARCHAR(20) DEFAULT '';
        """))
        # adjust: 默认 'raw'，因为旧表数据通常为原始价格
        conn.execute(text("""
        ALTER TABLE stock_bars ADD COLUMN IF NOT EXISTS adjust VARCHAR(10) DEFAULT 'raw';
        """))
        print("已添加 source 和 adjust 字段")

    # 4. 创建新唯一约束
    with engine.begin() as conn:
        conn.execute(text("""
        ALTER TABLE stock_bars ADD CONSTRAINT uix_symbol_date_source_adjust
        UNIQUE (symbol, trade_date, source, adjust);
        """))
        print("已添加新唯一约束 uix_symbol_date_source_adjust")

    # 5. 创建新索引
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_symbol_date_src_adj 
        ON stock_bars (symbol, trade_date, source, adjust);
        """))
        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_source_adjust 
        ON stock_bars (source, adjust);
        """))
        print("已添加新索引 idx_symbol_date_src_adj 和 idx_source_adjust")

    print("迁移完成。")
    print("注意：现有数据的 source 字段为空字符串，adjust 为 'raw'。")


if __name__ == "__main__":
    run_migration()
