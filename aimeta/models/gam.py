"""广义加性模型 GAM（§4.04）：样条基 + 光滑惩罚的非线性回归。

用于「水质指标 vs 多环境因子」这类非线性、非参数建模（如预警驱动因子分析）。
每个特征展开成 B-spline 基，用 ridge 惩罚（``smooth``）控制光滑度。
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer

from ..core.registry import MODELS


class GAM:
    """P-splines 风格 GAM：B-spline 基 + ridge 惩罚。

    ``n_splines`` 决定基的精细度，``smooth`` 越大越光滑（惩罚越强）。
    与 PLS 不同，它不假设线性关系，可拟合单波长/环境因子的非线性趋势。
    """

    def __init__(self, n_splines: int = 8, degree: int = 3, smooth: float = 1.0):
        self.n_splines = n_splines
        self.degree = degree
        self.smooth = smooth

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GAM":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.spline_ = SplineTransformer(
            n_knots=self.n_splines, degree=self.degree, include_bias=True)
        self.ridge_ = Ridge(alpha=self.smooth)
        B = self.spline_.fit_transform(X)
        self.ridge_.fit(B, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        B = self.spline_.transform(np.asarray(X, dtype=np.float64))
        return np.asarray(self.ridge_.predict(B), dtype=np.float64).ravel()


MODELS.register("gam", family="gam")(GAM)
