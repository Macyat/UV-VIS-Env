"""纯变量 / 最纯变量初值估计。

MCR-ALS 的解依赖初值。随机初值容易落到旋转模糊区里的任意一点；
最纯变量法给出「物理上可解释」的起点。
"""
from __future__ import annotations

import numpy as np


def simplisma(X: np.ndarray, n_components: int, offset: float = 0.05) -> np.ndarray:
    """SIMPLISMA：选最纯波长 → 得到浓度初值 → 最小二乘求初始光谱。

    纯度定义（在**波长方向**上）：
        purity_j = std_j / (|mean_j| + offset·max|mean|)
    选出一个波长后，用行列式权重压低与之相关的波长，再选下一个。

    注意一个容易搞反的点：矩阵 X = C @ S 中，
    纯度是在**列（波长）**上算的，选出的列给出的是**浓度剖面** C 的估计，
    而不是光谱。因此最后要用最小二乘 S0 = lstsq(C0, X) 还原光谱。

    Args:
        X: (n_samples, n_wavelengths)
        n_components: 组分数（可用 EFA 或奇异值确定）
        offset: 防止均值接近 0 时纯度爆炸

    Returns:
        (n_components, n_wavelengths) 的初始光谱矩阵 S0（每行已归一化）
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    alpha = offset * float(np.max(np.abs(mean)))
    denom = np.abs(mean) + alpha
    denom[denom == 0] = 1e-12
    purity = std / denom

    idx: list[int] = []
    det_weight = np.ones(p)
    for _ in range(n_components):
        score = purity * det_weight
        for j in idx:
            score[j] = -np.inf
        j = int(np.argmax(score))
        idx.append(j)
        if len(idx) < n_components:
            det_weight = _update_determinant_weight(X, idx)

    # 选中的列 ≈ 浓度剖面；再用最小二乘还原光谱
    C0 = X[:, idx]                                     # (n, k)
    S0 = np.linalg.lstsq(C0, X, rcond=None)[0]         # (k, p)
    S0 = np.maximum(S0, 0.0) if np.all(X >= -1e-12) else S0
    norms = np.linalg.norm(S0, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return S0 / norms


def _update_determinant_weight(X: np.ndarray, idx: list[int]) -> np.ndarray:
    """SIMPLISMA 的行列式权重：已选变量张成的空间中，相关性越高权重越低。"""
    n, p = X.shape
    S = X[:, idx]                                          # (n, k)
    # 把每个「波长向量」(长度 n) 投影到已选波长张成的子空间，取残差占比
    proj = np.linalg.lstsq(S, X, rcond=None)[0]            # (k, p)
    resid = X - S @ proj                                   # (n, p)
    norm_resid = np.linalg.norm(resid, axis=0)             # (p,)
    norm_x = np.linalg.norm(X, axis=0)
    norm_x[norm_x == 0] = 1e-12
    return norm_resid / norm_x


def orthogonal_projection(X: np.ndarray, n_components: int) -> np.ndarray:
    """OPA：正交投影法选最纯变量。

    逐步选取与已选光谱张成空间「最正交」的那条光谱。
    """
    X = np.asarray(X, dtype=np.float64)
    X = np.maximum(X, 0.0)
    n, p = X.shape
    # 第一个：范数最大的光谱（整体信号最强，通常最接近纯组分）
    first = int(np.argmax(np.linalg.norm(X, axis=1)))
    idx = [first]
    S = X[[first]]
    for _ in range(1, n_components):
        proj = np.linalg.lstsq(S.T, X.T, rcond=None)[0]     # (k, n)
        resid = X.T - S.T @ proj                            # (p, n)
        dets = np.linalg.norm(resid, axis=0)
        for j in idx:
            dets[j] = -np.inf
        j = int(np.argmax(dets))
        idx.append(j)
        S = X[idx]
    return X[idx]
