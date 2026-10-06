"""CARS：竞争性自适应重加权采样（Competitive Adaptive Reweighted Sampling）。

思路（Li et al.）：
    1. 每次 Monte-Carlo 抽 80% 样本建 PLS
    2. 用 |回归系数| 作为波长权重（EDF / 或直接用权重）
    3. 按指数衰减保留波长数（从 p 个衰减到 2 个）
    4. 取 RMSECV 最小的那一轮作为最终子集

提供 fit/transform 接口，可放进流程中，也能用 ``permutation_test`` 做显著性检验。
"""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from ..core.registry import SELECTORS
from .base import SelectorBase


@SELECTORS.register("cars", family="wrapper")
class CARS(SelectorBase):
    """竞争性自适应重加权采样。

    Args:
        n_iter: 衰减迭代次数
        n_components: PLS 主成分数
        mc_fraction: 每轮 Monte-Carlo 抽样比例
        cv_folds: RMSECV 折数
        keep_min: 最少保留波长数
        random_state: 随机种子
    """

    def __init__(
        self,
        n_iter: int = 30,
        n_components: int = 10,
        mc_fraction: float = 0.8,
        cv_folds: int = 5,
        keep_min: int = 2,
        random_state: int = 0,
    ) -> None:
        self.n_iter = n_iter
        self.n_components = n_components
        self.mc_fraction = mc_fraction
        self.cv_folds = cv_folds
        self.keep_min = keep_min
        self.random_state = random_state
        self.rmsecv_curve_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CARS":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, p = X.shape
        rng = np.random.default_rng(self.random_state)

        k_eff = max(1, min(self.n_components, min(n, p) - 1))
        decay = np.log(max(p, 2) / max(self.keep_min, 1)) / max(self.n_iter - 1, 1)

        candidates = list(range(p))
        best_rmse, best_subset = np.inf, list(range(p))

        for it in range(self.n_iter):
            keep_n = max(self.keep_min, int(np.ceil(p * np.exp(-decay * it))))
            sub_idx = rng.choice(n, size=max(2, int(self.mc_fraction * n)), replace=False)
            Xs, ys = X[np.ix_(sub_idx, candidates)], y[sub_idx]

            pls = PLSRegression(n_components=min(k_eff, Xs.shape[1]))
            pls.fit(Xs, ys)
            b = np.abs(np.asarray(pls.coef_).reshape(-1))
            w = b / (b.sum() + 1e-12)

            order = np.argsort(-w)
            keep = [candidates[i] for i in order[:keep_n]]
            rmse = self._rmsecv(X[:, keep], y, k_eff)
            self.rmsecv_curve_.append(rmse)
            if rmse < best_rmse:
                best_rmse, best_subset = rmse, keep
            candidates = keep

        self.selected_ = np.sort(np.asarray(best_subset, dtype=int))
        self.best_rmsecv_ = float(best_rmse)
        return self

    def _rmsecv(self, X: np.ndarray, y: np.ndarray, k: int) -> float:
        n = len(y)
        folds = max(2, min(self.cv_folds, n))
        kf = KFold(n_splits=folds, shuffle=True, random_state=self.random_state)
        errs = []
        for tr, te in kf.split(X):
            kk = max(1, min(k, X[tr].shape[1], len(tr) - 1))
            pls = PLSRegression(n_components=kk)
            pls.fit(X[tr], y[tr])
            pred = np.asarray(pls.predict(X[te])).ravel()
            errs.append(np.sqrt(np.mean((y[te] - pred) ** 2)))
        return float(np.mean(errs))
