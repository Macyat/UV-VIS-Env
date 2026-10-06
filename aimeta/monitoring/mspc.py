"""多变量统计过程控制。

用潜变量把整条光谱压成两个统计量：

    T²   Hotelling T²：模型内部的变异（正常波动方向上的偏离）
    SPE  Q 残差：模型**解释不了**的新变异 ← 探头污染、气泡、异物最常体现在这

贡献图把 SPE 拆回波长，直接指出「是哪个波长段出问题」——
这才是现场真正能动手的信息。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import stats


class MSPC:
    """PCA 版 MSPC。

    Example::

        m = MSPC(n_components=5).fit(X_normal)
        r = m.monitor(X_new)
        r["spe_alarm"]      # 是否异常
        r["contribution"]   # 各波长对 SPE 的贡献
    """

    def __init__(self, n_components: int = 5, alpha: float = 0.05) -> None:
        self.n_components = n_components
        self.alpha = alpha
        self.mean_: Optional[np.ndarray] = None
        self.P_: Optional[np.ndarray] = None      # (p, k) 载荷
        self.t2_limit_: float = 0.0
        self.spe_limit_: float = 0.0

    def fit(self, X: np.ndarray, y=None) -> "MSPC":
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        k = int(min(self.n_components, min(n, p)))
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # 用 SVD 直接取载荷，避免依赖 sklearn
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.P_ = Vt[:k].T
        self.s_ = s[:k]

        T = Xc @ self.P_
        self.t2_limit_ = self._t2_limit(T)
        resid = Xc - T @ self.P_.T
        spe = np.sum(resid ** 2, axis=1)
        self.spe_limit_ = self._spe_limit(spe)
        return self

    def _t2_limit(self, T: np.ndarray) -> float:
        n, k = T.shape
        f = stats.f.ppf(1 - self.alpha, k, n - k) if n > k else np.inf
        return float(k * (n - 1) / (n - k) * f) if np.isfinite(f) else float(np.inf)

    def _spe_limit(self, spe: np.ndarray) -> float:
        """Jackson-Mudholkar 近似：用 SPE 的均值与方差配一个加权卡方。"""
        mu = float(np.mean(spe))
        var = float(np.var(spe, ddof=1)) if len(spe) > 1 else 0.0
        if var <= 0:
            return mu
        g = var / (2 * mu)
        h = 2 * mu ** 2 / var
        return float(g * stats.chi2.ppf(1 - self.alpha, h))

    def transform(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        Xc = np.asarray(X, dtype=np.float64) - self.mean_
        T = Xc @ self.P_
        resid = Xc - T @ self.P_.T
        cov = np.diag(self.s_[: T.shape[1]] ** 2 / max(len(T) - 1, 1))
        cov_inv = np.linalg.pinv(cov)
        t2 = np.array([float(t @ cov_inv @ t) for t in T])
        return {"scores": T, "T2": t2, "SPE": np.sum(resid ** 2, axis=1),
                "residual": resid}

    def monitor(self, X: np.ndarray) -> Dict[str, object]:
        r = self.transform(X)
        r["t2_alarm"] = r["T2"] > self.t2_limit_
        r["spe_alarm"] = r["SPE"] > self.spe_limit_
        r["t2_limit"] = self.t2_limit_
        r["spe_limit"] = self.spe_limit_
        r["contribution"] = r["residual"] ** 2
        return r
