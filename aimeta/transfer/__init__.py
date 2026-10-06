"""transfer：仪器间模型迁移。

把一台仪器上训练的模型用到另一台仪器，避免每台设备各训一套。

- standardization  SBC / DS / PDS / GLSW / 均值方差对齐 + 标准化样本选择
- diagnose         仪器差异诊断：波长漂移、增益差、残差谱 → 推荐用哪种方法
"""
from .standardization import (
    SlopeBiasCorrection, DirectStandardization,
    PiecewiseDirectStandardization, GLSW, MeanVarianceAlign,
    kennard_stone,
)
from .diagnose import (
    estimate_wavelength_shift, estimate_gain_offset,
    residual_spectrum, recommend_method,
)

__all__ = [
    "SlopeBiasCorrection", "DirectStandardization",
    "PiecewiseDirectStandardization", "GLSW", "MeanVarianceAlign",
    "kennard_stone",
    "estimate_wavelength_shift", "estimate_gain_offset",
    "residual_spectrum", "recommend_method",
]
