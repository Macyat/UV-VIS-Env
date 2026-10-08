"""训练流程：SpectrumSet → ModelCard。

与老 ``Train.py`` 的差别：
    - 预处理链来自 YAML，与部署端同源
    - 模型从注册表取，不再 match-case
    - 产出是 ModelCard（含指纹与指标），不是裸 .pkl
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import KFold

from ..core.artifact import ModelCard, data_fingerprint
from ..core.spectra import SpectrumSet
from ..preprocessing.base import Pipeline
from ..models.registry import build_model, list_models, model_meta
from ..models.wrappers import WaterQualityModel
from ..metrics.water_standards import WaterParam


def _cv_splits(n: int, day_idx: Optional[np.ndarray], cv: str, folds: int,
               seed: int = 0, window: Optional[int] = None
               ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """按 ``cv`` 策略生成 (train, test) 索引对。

    - ``kfold``：随机打乱 K 折（无需 day_idx）。
    - ``logo``：按日留一（训练集 = 除该天外的所有天；含未来天，存在时序泄漏，仅作对照）。
    - ``expanding``：按日扩窗（训练集 = 该天之前的所有天，不用未来预测过去）。
    - ``rolling``：按日滚动窗（训练集 = 该天之前最近的 ``window`` 天，不用未来预测过去）。
      前 ``window`` 天作为 warm-up 期跳过（过去天数不足 ``window`` 不产生折）。

    ``logo`` / ``expanding`` / ``rolling`` 需要 ``day_idx``（每样本所属的「第几天」标签）。
    """
    if n < 2:
        return []
    idx = np.arange(n)
    if cv == "kfold":
        kf = KFold(n_splits=min(folds, n), shuffle=True, random_state=seed)
        return [(idx[tr], idx[te]) for tr, te in kf.split(idx)]
    if day_idx is None:
        raise ValueError(f"cv={cv!r} 是按日交叉验证，需要提供 day_idx")
    days = np.asarray(day_idx).ravel()
    if len(days) != n:
        raise ValueError("day_idx 长度必须等于样本数")
    unique = np.unique(days)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for d in unique:
        test = np.where(days == d)[0]
        if cv == "logo":
            train = np.where(days != d)[0]
        elif cv == "expanding":
            train = np.where(days < d)[0]
        elif cv == "rolling":
            if window is None:
                raise ValueError("cv='rolling' 需要给定 cv_window（滚动窗口的天数）")
            if window <= 0:
                raise ValueError(f"cv='rolling' 的 cv_window 必须为正整数，收到 {window}。")
            if window >= len(unique):
                raise ValueError(
                    f"cv='rolling' 的 cv_window={window} 天 ≥ 总天数 {len(unique)}，"
                    "warm-up 会跳过所有天、无法产生任何训练折；请减小 cv_window。")
            past = unique[unique < d]
            if len(past) < window:
                continue  # warm-up 期：过去天数不足 window，跳过（不做不满窗的训练）
            win = past[-window:]
            train = np.where(np.isin(days, win))[0]
        else:
            raise ValueError(f"unknown cv: {cv!r}")
        if len(train) and len(test):
            splits.append((train, test))
    return splits


def _cv_rmse(X: np.ndarray, y: np.ndarray, model_key: str, params: Dict[str, Any],
             day_idx: Optional[np.ndarray] = None, cv: str = "kfold",
             folds: int = 5, seed: int = 0,
             cv_window: Optional[int] = None) -> float:
    errs = []
    for tr, te in _cv_splits(len(y), day_idx, cv, folds, seed, cv_window):
        m = WaterQualityModel(build_model(model_key, **params))
        m.fit(X[tr], y[tr])
        errs.append(float(np.sqrt(np.mean((y[te] - m.predict(X[te])) ** 2))))
    return float(np.mean(errs)) if errs else float("nan")


def select_n_components(
    X: np.ndarray, y: np.ndarray, *, model_key: str = "pls",
    max_components: int = 20, folds: int = 5,
    model_params: Optional[Dict[str, Any]] = None,
    day_idx: Optional[np.ndarray] = None,
    cv: str = "kfold", cv_window: Optional[int] = None,
) -> int:
    """按 PLS_toolbox routine 为潜变量模型选择维数 n_components。

    对 n_components = 1..k 逐一做交叉验证（取 RMSE），返回误差最小者。
    k = min(max_components, 各折训练集最小样本数, n_features)，保证每个 CV 折的
    训练集都放得下该维数（潜变量数不能超过该折训练样本数）。仅对 latent 族
    （pls / pcr）有意义。
    """
    if model_meta(model_key).get("family") != "latent":
        raise ValueError(f"select_n_components 仅适用于 latent 族模型，收到 {model_key!r}")
    n, p = X.shape
    splits = _cv_splits(n, day_idx, cv, folds, seed=0, window=cv_window)
    min_train = min(len(tr) for tr, _ in splits) if splits else n
    k = min(max_components, max(1, min_train), p)
    base = dict(model_params or {})
    errs: List[float] = []
    for nc in range(1, k + 1):
        errs.append(_cv_rmse(X, y, model_key, {**base, "n_components": nc},
                             day_idx=day_idx, cv=cv, folds=folds, cv_window=cv_window))
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
    day_idx: Optional[np.ndarray] = None,
    cv: str = "kfold",
    cv_window: Optional[int] = None,
) -> ModelCard:
    """训练单个「参数 × 模型」，返回 ModelCard。

    标签（``spectra.y``）中为 NaN 的样本会自动剔除（对应的光谱行与 ``day_idx`` 同步剔除），
    外层无需再手动做掩码。

    Args:
        spectra: 含 y 的光谱集（y 为单目标，是现场水样的化学法参考值）
        label:   水质参数名
        model_key: 注册表中的模型名
        chain:   预处理链（YAML 字典列表）
        model_params: 模型超参
        param_def: 该参数的计量定义（决定预测上下限）
        folds:   CV 折数（kfold 用）
        day_idx: 每样本所属的「第几天」标签；按日 CV（logo/expanding/rolling）时必填
        cv:      CV 策略：'kfold'（随机打乱）/ 'logo'（按日留一，含未来、仅对照）/
                 'expanding'（按日扩窗）/ 'rolling'（按日滚动窗）。
                 expanding / rolling 只用过去的天，不用未来预测过去
        cv_window: 滚动窗口天数（仅 cv='rolling' 时需要）
        calibration_upper: 该校准标样的最高浓度（由专门的标样初始化流程给出）。
            给出时设备理论上限 = 2 × calibration_upper；不给出则设备理论上限不启用，
            仅用河流理论上限（param_def 的河流理论上限或默认 2 × V类）。注意：现场
            参考值 y 不是校准标样，绝不能拿它的 max 当设备理论上限。
    """
    if spectra.y is None:
        raise ValueError("spectra.y is required for training")
    y = np.asarray(spectra.y, dtype=np.float64).ravel()
    X_raw = np.asarray(spectra.X, dtype=np.float64)
    day = np.asarray(day_idx).ravel() if day_idx is not None else None

    # 自动剔除标签缺失（NaN）的样本，外层无需再手动做掩码；
    # 对应的光谱行与采样日一并剔除，保持对齐。
    mask = ~np.isnan(y)
    if not mask.all():
        y = y[mask]
        X_raw = X_raw[mask]
        if day is not None:
            day = day[mask]

    chain_cfg = list(chain) if chain is not None else []
    pipe = Pipeline.from_config(chain_cfg)
    X = pipe.fit_transform(X_raw, y)

    # 设备理论上限来自校准标样的最高浓度，由专门的标样初始化流程给出，
    # 不是训练用的现场参考值 y（y 是河水样的化学法结果，不是标样）。
    # 未跑标样初始化（calibration_upper 为 None）时设备理论上限不启用，
    # 仅用河流理论上限（见 WaterParam.theoretical_upper）。也可在 params.yaml 用
    # device_theoretical_upper 手动覆盖。
    auto_device_theoretical_upper = 2.0 * float(calibration_upper) if calibration_upper is not None else None
    if param_def is not None:
        theo_upper = param_def.theoretical_upper(auto_device_theoretical_upper)
    else:
        theo_upper = auto_device_theoretical_upper

    params = dict(model_params or {})
    # 潜变量模型（PLS/PCR）：未显式给 n_components 时，按 CV 遍历 1..k 取交叉验证
    # 误差最小的维数（PLS_toolbox routine）。显式给了就以手填为准。
    if model_meta(model_key).get("family") == "latent" and "n_components" not in params:
        params["n_components"] = select_n_components(
            X, y, model_key=model_key, folds=folds, day_idx=day,
            cv=cv, cv_window=cv_window)

    wrapped = WaterQualityModel(
        build_model(model_key, **params),
        lower_bound=param_def.lower_bound if param_def else None,
        review_upper=param_def.review_upper if param_def else None,
        theoretical_upper=theo_upper,
    )
    wrapped.fit(X, y)

    rmse_cv = _cv_rmse(X, y, model_key, params, day_idx=day, cv=cv,
                       folds=folds, cv_window=cv_window)
    return ModelCard(
        model_key=model_key,
        label=label,
        estimator=wrapped,
        preproc_chain=pipe.to_config(),
        wavelengths=[float(w) for w in spectra.wavelengths],
        instrument_id=instrument_id,
        data_version=data_fingerprint(X_raw, y),
        fom={"rmse_cv": rmse_cv,
             "rmse_fit": float(np.sqrt(np.mean(
                 (y - wrapped.predict_raw(X)) ** 2)))},
        params=params,
        theoretical_upper=theo_upper,
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
    day_idx: Optional[np.ndarray] = None,
    cv: str = "kfold",
    cv_window: Optional[int] = None,
) -> List[ModelCard]:
    """模型 sweep：取代老 run.py 的 subprocess fork。

    返回按 rmse_cv 升序排列的 ModelCard 列表。
    """
    model_params = model_params or {}
    cards = [
        train_model(spectra, label, k, chain, model_params.get(k), param_def,
                    folds=folds, instrument_id=instrument_id, day_idx=day_idx,
                    cv=cv, cv_window=cv_window)
        for k in model_keys
    ]
    return sorted(cards, key=lambda c: c.fom.get("rmse_cv", np.inf))


def sweep_grid(
    spectra: SpectrumSet,
    label: str,
    model_keys: Sequence[str] = ("pls", "ridge", "lasso", "ols"),
    chains: Optional[Sequence[Iterable[Dict[str, Any]]]] = None,
    model_params: Optional[Dict[str, Dict[str, Any]]] = None,
    param_def: Optional[WaterParam] = None,
    folds: int = 5,
    instrument_id: str = "unknown",
    day_idx: Optional[np.ndarray] = None,
    cv: str = "kfold",
    cv_window: Optional[int] = None,
) -> List[ModelCard]:
    """候选池：跨「预处理链 × 模型」全量枚举（取代手填单链 sweep）。

    每个 (chain, model) 组合训一个 ModelCard，返回全部候选（未排序）。
    排序/选 TOP K 交给 ``scoring.score_cards`` / ``select_top_k``。

    Args:
        chains: 预处理链列表；为 None 时退化为单条空链（等价于 ``sweep``）。
    """
    model_params = model_params or {}
    chain_list = list(chains) if chains is not None else [[]]
    cards: List[ModelCard] = []
    for chain in chain_list:
        for k in model_keys:
            cards.append(train_model(
                spectra, label, k, chain, model_params.get(k),
                param_def, folds=folds, instrument_id=instrument_id,
                day_idx=day_idx, cv=cv, cv_window=cv_window))
    return cards
