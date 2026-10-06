"""多模型融合：精度加权（1/RMSE²）与等权平均。

候选池（``sweep_grid`` 产出）里挑出的 TOP K 卡，各跑各的预处理链得到 raw
预测，再按权重融合，得到比任一单模型更稳的最终读数。

权重来源（你定的口径）：
    - ``inv_rmse2``：w_i = 1 / rmse_cv_i²（误差方差反比加权）。
      任一张卡 rmse_cv 缺失 / 非正 / 非有限 → 自动退化成等权，避免除零。
    - ``equal``：等权平均。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.artifact import ModelCard
from ..preprocessing.base import Pipeline


def _raw_predict(card: ModelCard, X: np.ndarray) -> np.ndarray:
    """用卡片自己的预处理链做 raw 预测（不去检出限/量程裁剪）。"""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    pipe = Pipeline.from_config(card.preproc_chain)
    Xt = pipe.transform(X)
    return np.asarray(card.estimator.predict_raw(Xt), dtype=np.float64).ravel()


def fuse_predict(
    cards: Sequence[ModelCard],
    X: np.ndarray,
    scheme: str = "inv_rmse2",
    param: Optional[object] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """精度加权融合 TOP K 候选卡的预测。

    Args:
        cards: 参与融合的 ModelCard 列表（通常来自 ``select_top_k``）。
        X: (n, p) 原始光谱（未经预处理）。
        scheme: ``"inv_rmse2"``（默认，1/RMSE² 加权）或 ``"equal"``。
        param: 可选 ``WaterParam``；给出时融合结果再统一过检出限/复核限。

    Returns:
        (fused, info)：融合预测值，以及 ``{"scheme", "weights", "rmse_cv"}``。
        融合前各卡 raw 预测形如 (n, m)，权重 w 满足 Σw=1。
    """
    if not cards:
        raise ValueError("fuse_predict 需要至少一张候选卡")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    preds = np.stack([_raw_predict(c, X) for c in cards], axis=1)  # (n, m)
    rmse = np.array([float(c.fom.get("rmse_cv", np.nan)) for c in cards],
                    dtype=np.float64)

    if scheme == "equal" or np.any(~np.isfinite(rmse)) or np.any(rmse <= 0):
        w = np.ones(preds.shape[1], dtype=np.float64) / preds.shape[1]
        used = "equal"
    else:
        w = 1.0 / (rmse ** 2)
        w = w / w.sum()
        used = "inv_rmse2"

    fused = preds @ w

    if param is not None:
        from ..metrics.water_standards import apply_bounds
        fused, *_ = apply_bounds(
            fused, getattr(param, "lower_bound", None),
            getattr(param, "upper_bound", None), None)

    info: Dict[str, object] = {
        "scheme": used,
        "weights": [float(x) for x in w],
        "rmse_cv": [float(x) for x in rmse],
    }
    return fused, info
