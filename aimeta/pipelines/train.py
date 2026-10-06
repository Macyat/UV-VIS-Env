"""训练流程：SpectrumSet → ModelCard。

与老 ``Train.py`` 的差别：
    - 预处理链来自 YAML，与部署端同源
    - 模型从注册表取，不再 match-case
    - 产出是 ModelCard（含指纹与 FOM），不是裸 .pkl
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from sklearn.model_selection import KFold

from ..core.artifact import ModelCard, data_fingerprint
from ..core.spectra import SpectrumSet
from ..preprocessing.base import Pipeline
from ..models.registry import build_model, list_models
from ..models.wrappers import WaterQualityModel
from ..metrics.water_standards import WaterParam


def _cv_rmse(X: np.ndarray, y: np.ndarray, model_key: str, params: Dict[str, Any],
             folds: int = 5, seed: int = 0) -> float:
    kf = KFold(n_splits=min(folds, len(y)), shuffle=True, random_state=seed)
    errs = []
    for tr, te in kf.split(X):
        m = WaterQualityModel(build_model(model_key, **params))
        m.fit(X[tr], y[tr])
        errs.append(float(np.sqrt(np.mean((y[te] - m.predict(X[te])) ** 2))))
    return float(np.mean(errs)) if errs else float("nan")


def train_model(
    spectra: SpectrumSet,
    label: str,
    model_key: str = "pls",
    chain: Optional[Iterable[Dict[str, Any]]] = None,
    model_params: Optional[Dict[str, Any]] = None,
    param_def: Optional[WaterParam] = None,
    folds: int = 5,
    instrument_id: str = "unknown",
) -> ModelCard:
    """训练单个「参数 × 模型」，返回 ModelCard。

    Args:
        spectra: 含 y 的光谱集（y 为单目标）
        label:   水质参数名
        model_key: 注册表中的模型名
        chain:   预处理链（YAML 字典列表）
        model_params: 模型超参
        param_def: 该参数的计量定义（决定预测上下限）
        folds:   CV 折数
    """
    if spectra.y is None:
        raise ValueError("spectra.y is required for training")
    y = np.asarray(spectra.y, dtype=np.float64).ravel()

    chain_cfg = list(chain) if chain is not None else []
    pipe = Pipeline.from_config(chain_cfg)
    X = pipe.fit_transform(spectra.X, y)

    params = model_params or {}
    wrapped = WaterQualityModel(
        build_model(model_key, **params),
        lower_bound=param_def.lower_bound if param_def else None,
        upper_bound=param_def.upper_bound if param_def else None,
    )
    wrapped.fit(X, y)

    rmse_cv = _cv_rmse(X, y, model_key, params, folds=folds)
    return ModelCard(
        model_key=model_key,
        label=label,
        estimator=wrapped,
        preproc_chain=pipe.to_config(),
        wavelengths=[float(w) for w in spectra.wavelengths],
        instrument_id=instrument_id,
        data_version=data_fingerprint(spectra.X, y),
        fom={"rmse_cv": rmse_cv,
             "rmse_fit": float(np.sqrt(np.mean((y - wrapped.predict(X)) ** 2)))},
        params=params,
    )


def sweep(
    spectra: SpectrumSet,
    label: str,
    model_keys: Sequence[str] = ("pls", "ridge", "lasso", "ols"),
    chain: Optional[Iterable[Dict[str, Any]]] = None,
    model_params: Optional[Dict[str, Dict[str, Any]]] = None,
    param_def: Optional[WaterParam] = None,
    folds: int = 5,
    instrument_id: str = "unknown",
) -> List[ModelCard]:
    """模型 sweep：取代老 run.py 的 subprocess fork。

    返回按 rmse_cv 升序排列的 ModelCard 列表。
    """
    model_params = model_params or {}
    cards = [
        train_model(spectra, label, k, chain, model_params.get(k), param_def,
                    folds, instrument_id)
        for k in model_keys
    ]
    return sorted(cards, key=lambda c: c.fom.get("rmse_cv", np.inf))
