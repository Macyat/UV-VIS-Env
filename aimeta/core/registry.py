"""通用注册表：驱动全库「新增能力 = 写一个类 + 一个装饰器」。

用法::

    @MODELS.register("pls", family="latent")
    class PLS: ...

    est = MODELS.build("pls", n_components=10)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple


class Registry:
    """键值注册表，附带元数据（family / tags / 是否需要 y 等）。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: Dict[str, Tuple[Any, Dict[str, Any]]] = {}

    def register(self, key: str, **meta: Any) -> Callable[[Any], Any]:
        if key in self._items:
            raise KeyError(f"[{self.name}] duplicated key: {key!r}")

        def deco(obj: Any) -> Any:
            self._items[key] = (obj, dict(meta))
            try:                       # sklearn 等第三方类也能挂标记
                obj._aimeta_key = key
            except Exception:          # pragma: no cover - builtin/immutable
                pass
            return obj

        return deco

    def get(self, key: str) -> Any:
        return self._items[key][0]

    def build(self, key: str, **kwargs: Any) -> Any:
        return self._items[key][0](**kwargs)

    def meta(self, key: str) -> Dict[str, Any]:
        return self._items[key][1]

    def keys(self, family: Optional[str] = None) -> List[str]:
        if family is None:
            return sorted(self._items)
        return sorted(k for k, (_, m) in self._items.items() if m.get("family") == family)

    def families(self) -> List[str]:
        return sorted({m.get("family", "misc") for _, m in self._items.values()})

    def items(self) -> Iterator[Tuple[str, Any, Dict[str, Any]]]:
        for k in sorted(self._items):
            obj, meta = self._items[k]
            yield k, obj, meta

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<Registry {self.name}: {sorted(self._items)}>"


# 全局注册表。各子包负责往里注册具体实现。
MODELS = Registry("models")        # + 2.x
PREPROC = Registry("preproc")      #
SELECTORS = Registry("selectors")  #
TRANSFER = Registry("transfer")    #
METRICS = Registry("metrics")      #
