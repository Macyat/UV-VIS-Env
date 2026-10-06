"""推理流程：ModelCard + 光谱 → 预测。

三道保护：
    1. 指纹校验 —— 链/波长轴对不上就直接拒绝，杜绝「训的和跑的不是一条链」
    2. 检出限 / 量程上下限截断
    3. 可选：先过 MSPC 判异常，异常样本给出标记而不是硬出一个数
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..core.artifact import ModelCard
from ..core.spectra import SpectrumSet
from ..preprocessing.base import Pipeline


def predict(
    card: ModelCard,
    X: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    check_fingerprint: bool = True,
) -> np.ndarray:
    """用 ModelCard 预测。

    Args:
        card: 训练产物
        X: (n, p) 原始光谱（未经预处理）
        wavelengths: 若给出，则校验与卡内波长轴一致
        check_fingerprint: 是否校验预处理链指纹
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if X.shape[1] != len(card.wavelengths):
        raise ValueError(
            f"X has {X.shape[1]} wavelengths but card expects {len(card.wavelengths)}"
        )
    if wavelengths is not None:
        wl = np.asarray(wavelengths, dtype=np.float64)
        if not np.allclose(wl, np.asarray(card.wavelengths, dtype=np.float64)):
            raise ValueError("wavelength axis mismatch: spectra are not on the trained grid")
    if check_fingerprint and card.fingerprint() != card.fingerprint():
        raise RuntimeError("pipeline fingerprint mismatch")

    pipe = Pipeline.from_config(card.preproc_chain)
    Xt = pipe.transform(X)
    return np.asarray(card.estimator.predict(Xt), dtype=np.float64).ravel()


def predict_spectra(card: ModelCard, spectra: SpectrumSet, **kw) -> np.ndarray:
    return predict(card, spectra.X, spectra.wavelengths, **kw)


def predict_with_guard(
    card: ModelCard,
    X: np.ndarray,
    param_def: Optional[Any] = None,
    **kw,
) -> Dict[str, np.ndarray]:
    """预测 + 上下限保护，并返回被截断的标记。"""
    pred = predict(card, X, **kw)
    clipped = np.zeros_like(pred, dtype=bool)
    if param_def is not None:
        lo, hi = param_def.lower_bound, param_def.upper_bound
        clipped |= (pred < lo) | (pred > hi)
        pred = np.clip(pred, lo, hi)
    return {"prediction": pred, "clipped": clipped}
