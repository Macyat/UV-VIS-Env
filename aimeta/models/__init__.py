"""models：模型库+ MCR 专区。

- registry   注册表与便捷构造
- zoo        已注册的候选模型（线性 / GLM / 潜变量 / 核 / 树 / 神经网络）
- wrappers   WaterQualityModel：x/y 缩放 + 上下限截断
- curve_resolution.MCRALS  多变量曲线分辨（ALS + 约束 + 初值估计）
"""
from .registry import MODELS, build_model, list_models
from .wrappers import WaterQualityModel
from . import zoo  # noqa: F401

__all__ = ["MODELS", "build_model", "list_models", "WaterQualityModel"]
