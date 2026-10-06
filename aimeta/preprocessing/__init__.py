"""preprocessing：预处理工具箱。

- base.Transformer / Pipeline   可 YAML 序列化、可版本化的算子与链
- operators.*                   SG / DERIV(Gram) / 小波 / SNV / MSC / 缩放

核心约束：**训练端与部署端加载同一条链**，杜绝「同一条链复制五份各改一点」。
"""
from .base import Transformer, Pipeline, build_chain
from . import operators  # noqa: F401  触发算子注册
from .operators import DERIV, savgol, wavelet_denoise, snv_transform, msc_transform

__all__ = [
    "Transformer", "Pipeline", "build_chain",
    "DERIV", "savgol", "wavelet_denoise", "snv_transform", "msc_transform",
]
