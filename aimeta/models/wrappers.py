"""WaterQualityModel：x/y 缩放 + 上下限截断的包装器。

统一训练与推理时的缩放方式，并支持导出为 numpy-only 的部署产物
（见 ``edge/runtime.py``）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class WaterQualityModel:
    """包装任意估计器：X 中心化 → 估计器 → y 反标准化 → 上下限截断。

    Args:
        estimator: 任何实现了 fit/predict 的对象
        lower_bound: 预测下限（通常是检出限）
        upper_bound: 预测上限（量程上限）
    """

    def __init__(self, estimator, lower_bound: Optional[float] = None,
                 upper_bound: Optional[float] = None):
        self.estimator = estimator
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.x_mean_: Optional[np.ndarray] = None
        self.y_mean_: float = 0.0
        self.y_scale_: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WaterQualityModel":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.x_mean_ = X.mean(axis=0)
        self.y_mean_ = float(np.mean(y))
        sd = float(np.std(y))
        self.y_scale_ = sd if sd > 0 else 1.0
        self.estimator.fit(X - self.x_mean_, (y - self.y_mean_) / self.y_scale_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        pred = self.estimator.predict(X - self.x_mean_) * self.y_scale_ + self.y_mean_
        pred = np.asarray(pred, dtype=np.float64).ravel()
        if self.lower_bound is not None:
            pred = np.maximum(pred, self.lower_bound)
        if self.upper_bound is not None:
            pred = np.minimum(pred, self.upper_bound)
        return pred

    # ---- 供 edge 导出 ----
    def linear_coef(self) -> Optional[np.ndarray]:
        """若底层是线性模型，返回等效回归系数（已含 x/y 缩放）。"""
        est = self.estimator
        coef = None
        if hasattr(est, "coef_"):
            coef = np.asarray(est.coef_, dtype=np.float64).reshape(-1)
        elif hasattr(est, "pipe") and hasattr(est, "coef_"):   # PCR
            coef = np.asarray(est.coef_, dtype=np.float64).reshape(-1)
        if coef is None:
            return None
        return coef * self.y_scale_

    def linear_intercept(self) -> Optional[float]:
        """等效截距：y = X @ coef + intercept（已含 X 中心化与 y 反标准化）。

        注意必须把 X 中心化的偏移补回来，否则部署端会整体偏一个常数。
        """
        est = self.estimator
        if not hasattr(est, "intercept_") or self.x_mean_ is None:
            return None
        coef = self.linear_coef()
        if coef is None:
            return None
        b = float(np.asarray(est.intercept_).reshape(-1)[0])
        return float(self.y_mean_ - float(self.x_mean_ @ coef) + b * self.y_scale_)
