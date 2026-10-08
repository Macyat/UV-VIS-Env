"""多组分校正集实验设计（§1.15）：对角设计 Diagonal Design。

用于同时测多个组分时配制校正集——各组分浓度沿对角线铺开，使浓度两两相关性低，
避免多元校正中的共线性（谱面重叠时尤其重要）。
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def diagonal_design(
    n_levels: int = 5,
    ranges: Optional[Sequence[Tuple[float, float]]] = None,
    k: int = 2,
    include_mixture: bool = False,
) -> np.ndarray:
    """生成对角设计的浓度矩阵（各组分浓度相关性低）。

    二组分示例（n_levels=5）：组分 A 沿一条对角线从低到高（B 恒为最低），
    组分 B 沿另一条对角线从低到高（A 恒为最低），形成「十字/对角」铺开。

    Args:
        n_levels: 每个组分一侧的浓度水平数（默认 5）
        ranges: [(min, max), ...] 每个组分的浓度范围，长度 = 组分数
        k: 组分数（ranges 未给时使用）
        include_mixture: 是否附加一个各组分取中值的混合点（默认 False）

    Returns:
        C: (n, k) 浓度设计矩阵，行 = 校正样本，列 = 组分
    """
    if ranges is None:
        ranges = [(0.0, 1.0)] * k
    ranges = [tuple(r) for r in ranges]
    if len(ranges) != k:
        raise ValueError("ranges 长度必须等于组分数 k")
    if n_levels < 2:
        raise ValueError("n_levels 至少 2")

    blocks = []
    for j in range(k):
        row = np.full((n_levels, k), 0.0)
        for jj, (lo, _) in enumerate(ranges):
            row[:, jj] = lo                       # 其他组分恒为最低
        row[:, j] = np.linspace(ranges[j][0], ranges[j][1], n_levels)
        blocks.append(row)
    C = np.vstack(blocks)

    if include_mixture:
        mid = np.array([[(lo + hi) / 2.0 for lo, hi in ranges]])
        C = np.vstack([C, mid])
    return C
