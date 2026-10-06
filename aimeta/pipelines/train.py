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
from ..models.registry import build_model, list_models, model_meta
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


def select_n_components(
    X: np.ndarray, y: np.ndarray, *, model_key: str = "pls",
    max_components: int = 20, folds: int = 5,
    model_params: Optional[Dict[str, Any]] = None,
) -> int:
    """按 PLS_toolbox 惯例为潜变量模型选择维数 n_components。

    对 n_components = 1..k 逐一做 K 折交叉验证（取 RMSE），返回误差最小者。
    k = min(max_components, n_samples // folds, n_features)，保证每个 CV 折的
    训练集都放得下该维数。仅对 latent 族（pls / pcr）有意义。
    """
    if model_meta(model_key).get("family") != "latent":
        raise ValueError(f"select_n_components 仅适用于 latent 族模型，收到 {model_key!r}")
    n, p = X.shape
    k = min(max_components, max(1, n // folds), p)
    base = dict(model_params or {})
    errs: List[float] = []
    for nc in range(1, k + 1):
        errs.append(_cv_rmse(X, y, model_key, {**base, "n_components": nc}, folds=folds))
    return int(np.nanargmin(errs)) + 1


def train_model(
    spectra: SpectrumSet,
    label: str,
    model_key: str = "pls",
    chain: Optional[Iterable[Dict[str, Any]]] = None,
    model_params: Optional[Dict[str, Any]] = None,
    param_def: Optional[WaterParam] = None,
    folds: int = 5,
    instrument_id: str = "unknown",
    calibration_upper: Optional[float] = None,
) -> ModelCard:
    """训练单个「参数 × 模型」，返回 ModelCard。

    Args:
        spectra: 含 y 的光谱集（y 为单目标，是现场水样的化学法参考值）
        label:   水质参数名
        model_key: 注册表中的模型名
        chain:   预处理链（YAML 字典列表）
        model_params: 模型超参
        param_def: 该参数的计量定义（决定预测上下限）
        folds:   CV 折数
        calibration_upper: 该校准标样的最高浓度（由专门的标样初始化流程给出）。
            给出时设备死限 = 2 × calibration_upper；不给出则设备死限不启用，
            仅用河流死限（param_def 的河流死限或默认 2 × V类）。注意：现场
            参考值 y 不是校准标样，绝不能拿它的 max 当设备死限。
    """
    if spectra.y is None:
        raise ValueError("spectra.y is required for training")
    y = np.asarray(spectra.y, dtype=np.float64).ravel()

    chain_cfg = list(chain) if chain is not None else []
    pipe = Pipeline.from_config(chain_cfg)
    X = pipe.fit_transform(spectra.X, y)

    # 设备死限来自校准标样的最高浓度，由专门的标样初始化流程给出，
    # 不是训练用的现场参考值 y（y 是河水样的化学法结果，不是标样）。
    # 未跑标样初始化（calibration_upper 为 None）时设备死限不启用，
    # 仅用河流死限（见 WaterParam.dead_bound）。也可在 params.yaml 用
    # device_dead_bound 手动覆盖。
    auto_device_dead = 2.0 * float(calibration_upper) if calibration_upper is not None else None
    if param_def is not None:
        dead = param_def.dead_bound(auto_device_dead)
    else:
        dead = auto_device_dead

    params = dict(model_params or {})
    # 潜变量模型（PLS/PCR）：未显式给 n_components 时，按 CV 遍历 1..k 取交叉验证
    # 误差最小的维数（PLS_toolbox 惯例）。显式给了就以手填为准。
    if model_meta(model_key).get("family") == "latent" and "n_components" not in params:
        params["n_components"] = select_n_components(
            X, y, model_key=model_key, folds=folds)

    wrapped = WaterQualityModel(
        build_model(model_key, **params),
        lower_bound=param_def.lower_bound if param_def else None,
        upper_bound=param_def.upper_bound if param_def else None,
        dead_bound=dead,
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
             "rmse_fit": float(np.sqrt(np.mean(
                 (y - wrapped.predict_raw(X)) ** 2)))},
        params=params,
        dead_bound=dead,
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
