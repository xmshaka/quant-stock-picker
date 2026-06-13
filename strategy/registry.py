"""策略方案注册表 — 持久化自定义方案 + 内置方案管理"""
import sys
sys.path.insert(0, "/root/.openclaw/workspace/quant-stock-picker")

import json
from pathlib import Path
from typing import Dict, List, Optional

from strategy.schemes import StrategyScheme, BUILTIN_SCHEMES


REGISTRY_PATH = Path(__file__).parent.parent / "data" / "custom_schemes.json"


class SchemeRegistry:
    """策略方案注册表"""

    def __init__(self):
        self._custom: Dict[str, StrategyScheme] = {}
        self._load_custom()

    def _load_custom(self):
        if REGISTRY_PATH.exists():
            try:
                data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
                for d in data:
                    scheme = StrategyScheme.from_dict(d)
                    self._custom[scheme.scheme_id] = scheme
            except Exception as e:
                print(f"[SchemeRegistry] 加载自定义方案失败: {e}")

    def _save_custom(self):
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = [s.to_dict() for s in self._custom.values()]
        REGISTRY_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_all(self) -> List[StrategyScheme]:
        """返回所有方案（内置 + 自定义）"""
        result = list(BUILTIN_SCHEMES.values()) + list(self._custom.values())
        return sorted(result, key=lambda s: (not s.is_builtin, s.name))

    def get(self, scheme_id: str) -> Optional[StrategyScheme]:
        return BUILTIN_SCHEMES.get(scheme_id) or self._custom.get(scheme_id)

    def save(self, scheme: StrategyScheme):
        """保存自定义方案"""
        scheme.is_builtin = False
        self._custom[scheme.scheme_id] = scheme
        self._save_custom()

    def delete(self, scheme_id: str) -> bool:
        if scheme_id in self._custom:
            del self._custom[scheme_id]
            self._save_custom()
            return True
        return False

    def duplicate(self, scheme_id: str, new_id: str, new_name: str) -> Optional[StrategyScheme]:
        """复制一个方案为新的自定义方案"""
        source = self.get(scheme_id)
        if not source:
            return None
        import copy
        new_scheme = copy.deepcopy(source)
        new_scheme.scheme_id = new_id
        new_scheme.name = new_name
        new_scheme.is_builtin = False
        self.save(new_scheme)
        return new_scheme

    def suggest_regime(self, regime: str) -> List[StrategyScheme]:
        """根据行情分类推荐适配方案"""
        result = []
        for s in self.list_all():
            if "*" in s.regime_fit or regime in s.regime_fit:
                result.append(s)
        return result
