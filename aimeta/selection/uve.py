"""UVE-PLS（无信息变量消除，§3.15）。

与 CARS 互补：CARS 是「竞争淘汰」，UVE 是「与噪声比较」——往 X 里塞人工噪声变量，
用噪声变量系数的稳定性分布作为阈值，剔除稳定性低于噪声的波长。
"""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from .base import SelectorBase


class UVE(SelectorBase):
    """无信息变量消除（Uninformative Variable Elimination）。

    步骤（Centner et al. 1996 的简化版）：
        1. 中心化 X、y；
        2. 拼接人工随机噪声矩阵 R；
        3. 对 [X, R] 用 PLS 回归，bootstrap 多轮得到各变量系数的分布；
        4. 每个变量的稳定性 c = mean(b) / std(b)，用噪声变量的 |c| 最大值作阈值；
        5. 选出 |c| > 阈值的波长。
    """

    def __init__(self, n_components: int = 10, n_rounds: int = 30,
                 random_state: int = 0):
        self.n_components = n_components
        self.n_rounds = n_rounds
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "UVE":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, p = X.shape
        Xc = X - X.mean(axis=0)
        yc = y - y.mean()
        rng = np.random.default_rng(self.random_state)
        nc = max(1, min(self.n_components, p, n - 1))

        coefs = np.zeros((self.n_rounds, p))
        noise = np.zeros((self.n_rounds, p))
        for r in range(self.n_rounds):
            idx = rng.choice(n, size=n, replace=True)      # bootstrap
            R = rng.normal(0, 1, (n, p))                   # 人工噪声变量
            Xa = np.hstack([Xc[idx], R[idx]])
            pls = PLSRegression(n_components=nc)
            pls.fit(Xa, yc[idx])
            b = pls.coef_.ravel()                          # 长度 2p
            coefs[r] = b[:p]
            noise[r] = b[p:]

        c = coefs.mean(axis=0) / (coefs.std(axis=0) + 1e-12)
        c_noise = noise.mean(axis=0) / (noise.std(axis=0) + 1e-12)
        threshold = float(np.max(np.abs(c_noise)))

        self.stability_ = c
        self.threshold_ = threshold
        self.selected_ = np.where(np.abs(c) > threshold)[0]
        return self
