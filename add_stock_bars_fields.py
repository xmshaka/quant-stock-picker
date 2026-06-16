#!/usr/bin/env python3
"""直接添加 stock_bars 表的 source 和 adjust 字段"""
import sys
sys.path.insert(0, '.')

from sqlalchemy import text
from data.storage.models import get_engine

def main():
    engine = get_engine()
    
    with engine.begin() as conn:
        # 检查字段是否已存在
        result = conn.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'stock_bars' AND column_name = 'source'
        """))
        if result.rowcount > 0:
            print("✅ source 字段已存在")
        else:
            conn.execute(text("""
                ALTER TABLE stock_bars ADD COLUMN source VARCHAR(20) DEFAULT ''
            """))
            print("✅ 已添加 source 字段")
        
        result = conn.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'stock_bars' AND column_name = 'adjust'
        """))
        if result.rowcount > 0:
            print("✅ adjust 字段已存在")
        else:
            conn.execute(text("""
                ALTER TABLE stock_bars ADD COLUMN adjust VARCHAR(10) DEFAULT 'raw'
            """))
            print("✅ 已添加 adjust 字段")
        
        # 检查唯一约束
        result = conn.execute(text("""
            SELECT constraint_name 
            FROM information_schema.table_constraints 
            WHERE table_name = 'stock_bars' AND constraint_name = 'uix_symbol_date_source_adjust'
        """))
        if result.rowcount > 0:
            print("✅ 新唯一约束已存在")
        else:
            # 删除旧约束（如果存在）
            conn.execute(text("""
                ALTER TABLE stock_bars DROP CONSTRAINT IF EXISTS uix_symbol_date
            """))
            print("✅ 已删除旧约束 uix_symbol_date")
            
            # 添加新约束
            conn.execute(text("""
                ALTER TABLE stock_bars ADD CONSTRAINT uix_symbol_date_source_adjust
                UNIQUE (symbol, trade_date, source, adjust)
            """))
            print("✅ 已添加新唯一约束 uix_symbol_date_source_adjust")
        
        # 添加索引
        result = conn.execute(text("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE tablename = 'stock_bars' AND indexname = 'idx_source_adjust'
        """))
        if result.rowcount > 0:
            print("✅ idx_source_adjust 索引已存在")
        else:
            conn.execute(text("""
                CREATE INDEX idx_source_adjust ON stock_bars (source, adjust)
            """))
            print("✅ 已添加 idx_source_adjust 索引")
    
    print("\n🎉 数据库字段添加完成")

if __name__ == "__main__":
    main()