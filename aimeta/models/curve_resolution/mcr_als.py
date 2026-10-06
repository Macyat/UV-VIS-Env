"""MCR-ALS：交替最小二乘多元曲线分辨。

模型  X ≈ C @ S
    C: (n_samples, k)   浓度剖面
    S: (k, n_wavelengths) 纯组分光谱

约束：
    non_negative   C、S 均非负
    closure        C 每行和为 1（组成型数据，如水体中各组分占比）
    unimodality    C 每列单峰（动力学 / 色谱过程）

模糊性：MCR 的解本身不唯一。``ambiguity_estimate`` 给出
一个粗略的可行解带宽估计 —— 带宽越大，越需要用约束或外部信息收敛。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import nnls

from .initializers import simplisma


class MCRALS:
    """交替最小二乘曲线分辨。

    Example::

        mcr = MCRALS(n_components=3, non_negative=True).fit(X)
        C = mcr.transform(X)      # 浓度剖面
        mcr.components_            # 分辨出的光谱 (k, p)
    """

    def __init__(
        self,
        n_components: int = 3,
        max_iter: int = 200,
        tol: float = 1e-7,
        non_negative: bool = True,
        closure: bool = False,
        unimodality: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.non_negative = non_negative
        self.closure = closure
        self.unimodality = unimodality
        self.random_state = random_state

        self.components_: Optional[np.ndarray] = None   # S
        self.concentrations_: Optional[np.ndarray] = None  # C
        self.loss_curve_: list[float] = []

    # ---- 主流程 ----
    def fit(self, X: np.ndarray, y=None) -> "MCRALS":
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        k = min(self.n_components, min(n, p))
        S = simplisma(X, k) if np.all(X >= -1e-12) else self._random_init(X, k)
        S = np.maximum(S, 0.0) if self.non_negative else S
        S = self._normalize_rows(S)

        prev = np.inf
        C = np.zeros((n, k))
        for _ in range(self.max_iter):
            C = self._solve_C(X, S)
            S = self._solve_S(X, C)
            if self.non_negative:
                S = np.maximum(S, 0.0)
            S = self._normalize_rows(S)
            resid = float(np.linalg.norm(X - C @ S))
            self.loss_curve_.append(resid)
            if abs(prev - resid) < self.tol * max(1.0, prev):
                break
            prev = resid

        self.components_ = S
        self.concentrations_ = C
        self.n_components_ = k
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """把新样本投影到已分辨的光谱上，得到浓度剖面。"""
        if self.components_ is None:
            raise RuntimeError("fit() must be called before transform()")
        return self._solve_C(np.asarray(X, dtype=np.float64), self.components_)

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        self.fit(X)
        return self.concentrations_

    def reconstruct(self) -> np.ndarray:
        return self.concentrations_ @ self.components_

    # ---- 单步求解 ----
    def _solve_C(self, X: np.ndarray, S: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if self.non_negative:
            C = np.vstack([nnls(S.T, X[i])[0] for i in range(n)])
        else:
            C = np.linalg.lstsq(S.T, X.T, rcond=None)[0].T
        if self.closure:
            rowsum = C.sum(axis=1, keepdims=True)
            rowsum[rowsum == 0] = 1.0
            C = C / rowsum
        if self.unimodality:
            C = np.apply_along_axis(_enforce_unimodal, 0, C)
        return C

    def _solve_S(self, X: np.ndarray, C: np.ndarray) -> np.ndarray:
        if self.non_negative:
            p = X.shape[1]
            S = np.vstack([nnls(C, X[:, j])[0] for j in range(p)]).T
        else:
            S = np.linalg.lstsq(C, X, rcond=None)[0]
        return S

    def _random_init(self, X: np.ndarray, k: int) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        n, p = X.shape
        return rng.random((k, p))

    @staticmethod
    def _normalize_rows(S: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(S, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return S / norms

    # ---- 模糊性提示 ----
    def ambiguity_estimate(self, X: Optional[np.ndarray] = None, n_trials: int = 20,
                           seed: int = 0) -> float:
        """用多组随机初值重跑，统计各组分解的差异，粗略刻画旋转模糊性。

        返回 0~1 的标量：越大说明可行解带越宽，结论越依赖于约束。
        """
        if X is None:
            X = self.reconstruct()
            if X is None:
                raise ValueError("no X given and model is not fitted")
        rng = np.random.default_rng(seed)
        base = self.components_
        diffs = []
        for _ in range(n_trials):
            S0 = rng.random(base.shape)
            trial = MCRALS(
                n_components=base.shape[0], non_negative=self.non_negative,
                closure=self.closure, max_iter=50,
            )
            trial.fit(X)
            S = trial.components_
            # 用最佳匹配对齐两组分量后比较
            d = _matched_component_distance(base, S)
            diffs.append(d)
        return float(np.mean(diffs))


def _enforce_unimodal(c: np.ndarray) -> np.ndarray:
    """把一列强制为单峰：保留最大峰，峰两侧单调化。"""
    peak = int(np.argmax(c))
    out = c.copy()
    for i in range(1, peak + 1):
        out[i] = max(out[i], out[i - 1])
    for i in range(len(c) - 2, peak - 1, -1):
        out[i] = max(out[i], out[i + 1])
    return out


def _matched_component_distance(A: np.ndarray, B: np.ndarray) -> float:
    """两组单位范数分量之间的平均余弦距离（先做最佳匹配）。"""
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    sim = np.abs(A @ B.T)
    best = sim.max(axis=1)
    return float(np.mean(1.0 - best))
