"""core：地基层。

- registry.Registry   通用注册表，驱动全部可插拔能力
- spectra.SpectrumSet 波长轴 + 吸光度 + 元数据的显式数据契约
- artifact.ModelCard  模型产物：预处理链指纹 + 波长轴 + FOM + 仪器 ID
"""
from .registry import Registry, MODELS, PREPROC, SELECTORS, TRANSFER, METRICS
from .spectra import SpectrumSet
from .artifact import ModelCard

__all__ = [
    "Registry", "MODELS", "PREPROC", "SELECTORS", "TRANSFER", "METRICS",
    "SpectrumSet", "ModelCard",
]
