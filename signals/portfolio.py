"""投资组合管理 - 持仓池与观察池"""
import json
import os
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from datetime import date, datetime

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "portfolio")
os.makedirs(DATA_DIR, exist_ok=True)


@dataclass
class PoolItem:
    """池子中的股票"""
    symbol: str
    add_date: str           # 加入日期 YYYY-MM-DD
    add_reason: str         # 加入原因（策略名）
    signal_strength: float  # 信号强度
    signal_score: float     # 综合得分
    note: str = ""          # 备注
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "PoolItem":
        return cls(**d)


class PortfolioManager:
    """组合管理器"""
    
    def __init__(self, portfolio_name: str = "default"):
        self.portfolio_name = portfolio_name
        self.watch_file = os.path.join(DATA_DIR, f"{portfolio_name}_watch.json")
        self.hold_file = os.path.join(DATA_DIR, f"{portfolio_name}_hold.json")
        self._watch_pool: Dict[str, PoolItem] = {}
        self._hold_pool: Dict[str, PoolItem] = {}
        self._load()
    
    def _load(self):
        """从文件加载"""
        if os.path.exists(self.watch_file):
            try:
                with open(self.watch_file, 'r') as f:
                    data = json.load(f)
                    self._watch_pool = {k: PoolItem.from_dict(v) for k, v in data.items()}
            except Exception:
                self._watch_pool = {}
        
        if os.path.exists(self.hold_file):
            try:
                with open(self.hold_file, 'r') as f:
                    data = json.load(f)
                    self._hold_pool = {k: PoolItem.from_dict(v) for k, v in data.items()}
            except Exception:
                self._hold_pool = {}
    
    def _save(self):
        """保存到文件"""
        with open(self.watch_file, 'w') as f:
            json.dump({k: v.to_dict() for k, v in self._watch_pool.items()}, f, indent=2)
        with open(self.hold_file, 'w') as f:
            json.dump({k: v.to_dict() for k, v in self._hold_pool.items()}, f, indent=2)
    
    # ========== 观察池操作 ==========
    def add_to_watch(self, item: PoolItem):
        """加入观察池"""
        self._watch_pool[item.symbol] = item
        self._save()
    
    def remove_from_watch(self, symbol: str):
        """从观察池移除"""
        if symbol in self._watch_pool:
            del self._watch_pool[symbol]
            self._save()
    
    def move_to_hold(self, symbol: str, note: str = ""):
        """从观察池移到持仓池"""
        if symbol in self._watch_pool:
            item = self._watch_pool[symbol]
            item.note = note
            self._hold_pool[symbol] = item
            del self._watch_pool[symbol]
            self._save()
    
    # ========== 持仓池操作 ==========
    def add_to_hold(self, item: PoolItem):
        """直接加入持仓池"""
        self._hold_pool[item.symbol] = item
        self._save()
    
    def remove_from_hold(self, symbol: str):
        """从持仓池移除（卖出）"""
        if symbol in self._hold_pool:
            del self._hold_pool[symbol]
            self._save()
    
    # ========== 查询 ==========
    @property
    def watch_list(self) -> List[PoolItem]:
        return list(self._watch_pool.values())
    
    @property
    def hold_list(self) -> List[PoolItem]:
        return list(self._hold_pool.values())
    
    def is_in_watch(self, symbol: str) -> bool:
        return symbol in self._watch_pool
    
    def is_in_hold(self, symbol: str) -> bool:
        return symbol in self._hold_pool
    
    def clear_all(self):
        """清空所有"""
        self._watch_pool.clear()
        self._hold_pool.clear()
        self._save()
