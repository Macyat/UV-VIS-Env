"""MCR 可行解域 AFS（§2.12）：Lawton-Sylvestre 二组分旋转模糊性刻画。

回答「解析解是否唯一」——对二组分体系，扫描旋转角 θ，用「纯光谱非负 + 浓度非负」
约束界定纯组分光谱/浓度的可行角度范围。可行范围越窄，解越确定。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def afs_lawton_sylvestre(
    X: np.ndarray,
    n_points: int = 360,
    tol: Optional[float] = None,
) -> Dict[str, object]:
    """二组分 Lawton-Sylvestre 可行解域。

    Args:
        X: (n, p) 二组分混合光谱矩阵（每行一个混合样本）
        n_points: 旋转角扫描点数（默认 360）
        tol: 非负约束的容差；默认 None = 0.01 × max|X|（相对容差，容纳秩 2 截断
             与光谱重叠带来的轻微负值）

    Returns:
        dict:
            ``angles``: (n_points,) 扫描的旋转角（0..2π）
            ``feasible``: (n_points,) bool，每个角度是否满足「光谱非负 + 浓度非负」
            ``boundaries``: 可行区间的角度边界列表 [(θ_lo, θ_hi), ...]
            ``feasible_frac``: 可行角度占比（0~1，越小解越确定）
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("AFS 需要 (n, p) 且 p >= 2")
    if tol is None:
        tol = 0.01 * float(np.abs(X).max())
    # 不中心化：浓度与光谱物理上非负，直接用原始数据做 SVD 才满足非负约束
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    T = U[:, :2] * s[:2]          # 得分 (n, 2)
    V = Vt[:2, :].T               # 载荷 (p, 2)
    # 符号校正：SVD 符号约定不确定，让每列主符号为正（平均光谱/平均浓度应为非负）
    for j in range(2):
        i = int(np.argmax(np.abs(V[:, j])))
        if V[i, j] < 0:
            V[:, j] *= -1
            T[:, j] *= -1

    angles = np.linspace(0.0, 2.0 * np.pi, n_points)
    feasible = np.zeros(n_points, dtype=bool)
    for i, th in enumerate(angles):
        R = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th), np.cos(th)]])
        S = V @ R                  # 纯组分光谱 (p, 2)
        C = T @ R                  # 浓度剖面 (n, 2)
        feasible[i] = bool(S.min() >= -tol) and bool(C.min() >= -tol)

    boundaries: List[Tuple[float, float]] = []
    if feasible.any():
        idx = np.where(feasible)[0]
        start = idx[0]
        prev = idx[0]
        for cur in idx[1:]:
            if cur != prev + 1:
                boundaries.append((float(angles[start]), float(angles[prev])))
                start = cur
            prev = cur
        boundaries.append((float(angles[start]), float(angles[prev])))

    return {
        "angles": angles,
        "feasible": feasible,
        "boundaries": boundaries,
        "feasible_frac": float(np.mean(feasible)),
    }
