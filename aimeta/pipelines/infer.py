"""推理流程：ModelCard + 光谱 → 预测。

三道保护：
    1. 指纹校验 —— 链/波长轴对不上就直接拒绝，杜绝「训的和跑的不是一条链」
    2. 检出限 / 复核限 / 死限处理（低于检出限替为 LOD/2；超出复核限仍显示并
       打"需复核"标；超出死限置 NaN 不显示）
    3. 可选：先过 MSPC 判异常，异常样本给出标记而不是硬出一个数
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..core.artifact import ModelCard
from ..core.spectra import SpectrumSet
from ..metrics.water_standards import apply_bounds
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
    """预测 + 检出限 / 复核限 / 死限处理，并返回标记。

    返回：
        prediction:  处理后的预测值
                     （低于检出限替为 LOD/2；高于死限置 NaN 不显示）
        below_lod:   布尔数组，True 表示该样本低于检出限
        above_upper: 布尔数组，True 表示该样本超出复核限（需人工 / 金标准复核），
                     但仍在死限内，仍显示
        not_display: 布尔数组，True 表示该样本超过死限（物理不可能值），不显示
    """
    # 用 raw 预测算标记，避免被模型内部的边界处理掩盖
    raw = np.asarray(card.estimator.predict_raw(
        Pipeline.from_config(card.preproc_chain).transform(
            np.asarray(X, dtype=np.float64))), dtype=np.float64).ravel()
    # 死限优先取卡内已算好的有效死限；卡内缺失时退化为仅河流维（2 × V类）
    dead = getattr(card, "dead_bound", None)
    if dead is None and param_def is not None:
        dead = param_def.dead_bound(None)
    if param_def is not None:
        pred, below, above, suppress = apply_bounds(
            raw, param_def.lower_bound, param_def.upper_bound, dead)
    else:
        pred = raw
        below = np.zeros(raw.shape, dtype=bool)
        above = np.zeros(raw.shape, dtype=bool)
        suppress = np.zeros(raw.shape, dtype=bool)
    return {"prediction": pred, "below_lod": below,
            "above_upper": above, "not_display": suppress}
