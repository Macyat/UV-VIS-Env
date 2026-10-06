"""标准化 / 迁移方法。

四类手段，复杂度与所需样本数递增：

    SBC      y' = a·y + b                     只需两台机的预测值对，最轻
    DS       X_master ≈ X_slave @ F            全波长的全局线性映射
    PDS      逐波长窗口回归 → 带状 F           最常用，能处理波长相关差异
    GLSW     用主从差的子空间构造滤波器         属于「不做标准化的传递」
    MeanVar  均值/方差对齐（CORAL 思路）        极简兜底

前提：需要一批**在两台仪器上都测过的标准化样本**（对本项目＝同一套标液）。
样本不够时 ``recommend_method`` 会建议降级到 SBC。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.registry import TRANSFER


class _Base:
    def fit(self, X_slave: np.ndarray, X_master: np.ndarray) -> "_Base":
        raise NotImplementedError

    def transform(self, X_slave: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, X_slave: np.ndarray, X_master: np.ndarray) -> np.ndarray:
        return self.fit(X_slave, X_master).transform(X_slave)


@TRANSFER.register("sbc", family="response", min_samples=2)
class SlopeBiasCorrection(_Base):
    """响应校正：y' = a·y + b。

    只需要两台机对同批样本的**预测值**，不需要光谱 —— 现场最容易落地。
    """

    def __init__(self) -> None:
        self.slope_: float = 1.0
        self.intercept_: float = 0.0

    def fit(self, y_slave: np.ndarray, y_master: np.ndarray) -> "SlopeBiasCorrection":
        y_slave = np.asarray(y_slave, dtype=np.float64).ravel()
        y_master = np.asarray(y_master, dtype=np.float64).ravel()
        A = np.vstack([y_slave, np.ones_like(y_slave)]).T
        sol, *_ = np.linalg.lstsq(A, y_master, rcond=None)
        self.slope_, self.intercept_ = float(sol[0]), float(sol[1])
        return self

    def transform(self, y_slave: np.ndarray) -> np.ndarray:
        y = np.asarray(y_slave, dtype=np.float64)
        return self.slope_ * y + self.intercept_


@TRANSFER.register("ds", family="spectral", min_samples=20)
class DirectStandardization(_Base):
    """直接标准化：X_master ≈ X_slave @ F，F 为全局 (p, p) 矩阵。

    需要样本数 >= 波长数，否则退化为最小范数解（此时应改用 PDS）。
    """

    def __init__(self, ridge: float = 1e-8) -> None:
        self.ridge = ridge
        self.F_: Optional[np.ndarray] = None

    def fit(self, X_slave: np.ndarray, X_master: np.ndarray) -> "DirectStandardization":
        S = np.asarray(X_slave, dtype=np.float64)
        M = np.asarray(X_master, dtype=np.float64)
        p = S.shape[1]
        G = S.T @ S + self.ridge * np.eye(p)
        self.F_ = np.linalg.solve(G, S.T @ M)
        return self

    def transform(self, X_slave: np.ndarray) -> np.ndarray:
        return np.asarray(X_slave, dtype=np.float64) @ self.F_


@TRANSFER.register("pds", family="spectral", min_samples=5)
class PiecewiseDirectStandardization(_Base):
    """分段直接标准化（最常用）。

    对每个波长 j，用 slave 端 [j-k, j+k] 窗口（加截距）回归 master 端第 j 列::

        x_master[:, j] = [1, X_slave[:, j-k:j+k+1]] @ b_j

    所有 b_j 组装成带状转移矩阵 F，因此 PDS 能处理**随波长变化**的仪器差异
    （增益随波长漂移、单色器误差等），而 DS 只能处理全局线性差异。
    """

    def __init__(self, window: int = 5, ridge: float = 1e-8) -> None:
        self.window = window
        self.ridge = ridge
        self.F_: Optional[np.ndarray] = None

    def fit(self, X_slave: np.ndarray, X_master: np.ndarray) -> "PiecewiseDirectStandardization":
        S = np.asarray(X_slave, dtype=np.float64)
        M = np.asarray(X_master, dtype=np.float64)
        if S.shape[1] != M.shape[1]:
            raise ValueError("slave and master must share the same wavelength axis")
        n, p = S.shape
        k = int(self.window)
        # F 的第一行是截距，其余 p 行对应 slave 的波长；
        # 边界窗口更短，因此系数要写入对应的行区间，不能整列覆盖
        F = np.zeros((p + 1, p))
        for j in range(p):
            lo, hi = max(0, j - k), min(p, j + k + 1)
            B = np.hstack([np.ones((n, 1)), S[:, lo:hi]])
            G = B.T @ B + self.ridge * np.eye(B.shape[1])
            coef = np.linalg.solve(G, B.T @ M[:, j])
            F[0, j] = coef[0]
            F[1 + lo:1 + hi, j] = coef[1:]
        self.F_ = F
        return self

    def transform(self, X_slave: np.ndarray) -> np.ndarray:
        S = np.asarray(X_slave, dtype=np.float64)
        return np.hstack([np.ones((len(S), 1)), S]) @ self.F_


@TRANSFER.register("glsw", family="filter", min_samples=5)
class GLSW(_Base):
    """广义最小二乘加权滤波。

    用主从差矩阵构造协方差 C，再取 G = (C/trace·p + λI)^(-1/2) 作为滤波器，
    把 slave 光谱投影到「仪器差异被压低」的子空间，之后主模型可直接用。
    """

    def __init__(self, lam: float = 1e-3) -> None:
        self.lam = lam
        self.W_: Optional[np.ndarray] = None

    def fit(self, X_slave: np.ndarray, X_master: np.ndarray) -> "GLSW":
        D = np.asarray(X_master, dtype=np.float64) - np.asarray(X_slave, dtype=np.float64)
        p = D.shape[1]
        C = D.T @ D / max(len(D), 1)
        tr = np.trace(C)
        if tr <= 0:
            self.W_ = np.eye(p)
            return self
        Cn = C / tr * p
        G = np.linalg.inv(Cn + self.lam * np.eye(p))
        vals, vecs = np.linalg.eigh(G)
        self.W_ = vecs @ np.diag(np.sqrt(np.maximum(vals, 0.0))) @ vecs.T
        return self

    def transform(self, X_slave: np.ndarray) -> np.ndarray:
        return np.asarray(X_slave, dtype=np.float64) @ self.W_


@TRANSFER.register("meanvar", family="spectral", min_samples=2)
class MeanVarianceAlign(_Base):
    """逐波长均值/方差对齐（CORAL 的一维版本），极简兜底。"""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.shift_ = None
        self.scale_ = None

    def fit(self, X_slave: np.ndarray, X_master: np.ndarray) -> "MeanVarianceAlign":
        S = np.asarray(X_slave, dtype=np.float64)
        M = np.asarray(X_master, dtype=np.float64)
        ss, sm = S.std(axis=0), M.std(axis=0)
        ss[ss < self.eps] = 1.0
        sm[sm < self.eps] = 1.0
        self.scale_ = sm / ss
        self.shift_ = M.mean(axis=0) - S.mean(axis=0) * self.scale_
        return self

    def transform(self, X_slave: np.ndarray) -> np.ndarray:
        return np.asarray(X_slave, dtype=np.float64) * self.scale_ + self.shift_


# ------------------------------------------------------------ 样本选择
def kennard_stone(X: np.ndarray, n_select: int) -> np.ndarray:
    """Kennard-Stone：选在空间上最均匀铺开的一批样本作为标准化样本。

    样本贵（要跑标液），所以要选得值。
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    n_select = int(min(n_select, n))
    if n_select <= 0:
        return np.array([], dtype=int)
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    selected = [int(np.argmax(D.sum(axis=1)))]
    while len(selected) < n_select:
        rest = [i for i in range(n) if i not in selected]
        d = D[np.ix_(rest, selected)].min(axis=1)
        selected.append(rest[int(np.argmax(d))])
    return np.asarray(sorted(selected), dtype=int)
