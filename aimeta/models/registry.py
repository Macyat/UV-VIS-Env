"""模型注册表。

新增模型 = 一个装饰器，无需改动其他文件。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.registry import MODELS

__all__ = ["MODELS", "build_model", "list_models", "model_meta"]


def build_model(key: str, **kwargs: Any) -> Any:
    """按注册名构造模型。"""
    return MODELS.build(key, **kwargs)


def model_meta(key: str) -> Dict[str, Any]:
    return MODELS.meta(key)


def list_models(family: Optional[str] = None) -> List[str]:
    """列出可用模型名（可按 family 过滤）。"""
    return MODELS.keys(family)


def families() -> List[str]:
    return MODELS.families()
