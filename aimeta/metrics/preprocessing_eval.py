"""预处理评价（§3.02）：用客观判据评价预处理链，而非只凭 RMSECV。

判据（见 ``docs/术语表.md`` 与 Comprehensive Chemometrics §3.02）：
    - ``s_x``      空白光谱噪声：预处理前后对比，好的预处理应压低噪声
    - ``gamma``    分析灵敏度 γ = SEN / s_x：信号保真 + 噪声压低的综合判据
    - ``overfit``  训练 vs 交叉验证误差的 gap：gap 大 = 预处理/模型在拟合噪声
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.model_selection import KFold

from ..models.registry import build_model
from ..preprocessing.base import Pipeline
from .figures_of_merit import figures_of_merit, spectral_noise


def evaluate_preprocessing(
    X_train: np.ndarray,
    y: np.ndarray,
    X_blank: np.ndarray,
    chain: Sequence[Dict],
    model_key: str = "pls",
    model_params: Optional[Dict] = None,
    folds: int = 5,
    random_state: int = 0,
) -> Dict[str, float]:
    """评价一条预处理链的客观判据。

    Args:
        X_train: (n, p) 训练光谱
        y: (n,) 参考浓度
        X_blank: (n_blank, p) 空白光谱（与样品基质一致、不含待测物；估计 s_x 建议
                 ≥10 条重复，最少 1 条）
        chain: 预处理链（YAML 字典列表）
        model_key: 用于算 γ 的模型（默认 pls）
        model_params: 模型超参
        folds: 过拟合 gap 用的 CV 折数

    Returns:
        dict:
            ``s_x_before`` / ``s_x_after``: 预处理前后空白光谱噪声（好的预处理压低它）
            ``sen`` / ``gamma``: 灵敏度 / 分析灵敏度（越大越好）
            ``rmse_fit`` / ``rmse_cv``: 训练拟合 / 交叉验证 RMSE
            ``overfit_gap``: ``rmse_cv / rmse_fit``，明显 > 1 表示在拟合噪声
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    X_blank = np.asarray(X_blank, dtype=np.float64)

    s_x_before = spectral_noise(X_blank)

    pipe = Pipeline.from_config(chain)
    pipe.fit(X_train, y)
    Xt = pipe.transform(X_train)
    Xb = pipe.transform(X_blank)

    s_x_after = spectral_noise(Xb)

    model = build_model(model_key, **(model_params or {}))
    model.fit(Xt, y)
    fom = figures_of_merit(model, Xb, X_cal=Xt)

    rmse_fit = float(np.sqrt(np.mean((y - np.asarray(model.predict(Xt)).ravel()) ** 2)))

    errs = []
    kf = KFold(n_splits=min(folds, len(y)), shuffle=True, random_state=random_state)
    for tr, te in kf.split(Xt):
        m = build_model(model_key, **(model_params or {}))
        m.fit(Xt[tr], y[tr])
        errs.append(float(np.sqrt(np.mean(
            (y[te] - np.asarray(m.predict(Xt[te])).ravel()) ** 2))))
    rmse_cv = float(np.mean(errs)) if errs else float("nan")

    return {
        "s_x_before": s_x_before,
        "s_x_after": s_x_after,
        "sen": fom["SEN"],
        "gamma": fom["gamma"],
        "rmse_fit": rmse_fit,
        "rmse_cv": rmse_cv,
        "overfit_gap": rmse_cv / rmse_fit if rmse_fit > 0 else float("inf"),
    }
