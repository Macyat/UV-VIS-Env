"""假设检验与一致性分析。 把「数据不相容」的处理分成两条路：剔除数据 / 用稳健方法。
``consistency_report`` 把这套判断固化下来，避免每次靠人眼看图。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import stats


def _groups(groups: Sequence[np.ndarray]) -> List[np.ndarray]:
    return [np.asarray(g, dtype=np.float64).ravel() for g in groups]


def cochran_test(groups: Sequence[np.ndarray], alpha: float = 0.05) -> Dict[str, float]:
    """Cochran 检验：最大方差是否显著大于其余。"""
    gs = _groups(groups)
    k = len(gs)
    var = np.array([g.var(ddof=1) for g in gs])
    C = float(var.max() / var.sum()) if var.sum() > 0 else 0.0
    # 临界值近似：C ~= 1/(1 + (k-1)/F)，用 F 分布反解
    df = float(np.mean([len(g) - 1 for g in gs]))
    f_crit = stats.f.ppf(1 - alpha / k, df, (k - 1) * df) if k > 1 else np.inf
    c_crit = 1.0 / (1.0 + (k - 1) / f_crit) if np.isfinite(f_crit) else 1.0
    return {"statistic": C, "critical": float(c_crit), "reject": bool(C > c_crit)}


def bartlett_test(groups: Sequence[np.ndarray], alpha: float = 0.05) -> Dict[str, float]:
    """Bartlett 方差齐性检验（要求正态；非正态请用 levene）。"""
    gs = _groups(groups)
    stat, p = stats.bartlett(*gs)
    return {"statistic": float(stat), "p_value": float(p), "reject": bool(p < alpha)}


def levene_test(groups: Sequence[np.ndarray], alpha: float = 0.05) -> Dict[str, float]:
    """Levene 方差齐性检验（对非正态稳健，）。"""
    gs = _groups(groups)
    stat, p = stats.levene(*gs)
    return {"statistic": float(stat), "p_value": float(p), "reject": bool(p < alpha)}


def grubbs_test(x: np.ndarray, alpha: float = 0.05) -> Dict[str, object]:
    """Grubbs 检验：找单个离群值。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < 3:
        return {"statistic": 0.0, "critical": np.inf, "outlier_index": None}
    mu, sd = x.mean(), x.std(ddof=1)
    if sd == 0:
        return {"statistic": 0.0, "critical": np.inf, "outlier_index": None}
    dev = np.abs(x - mu)
    i = int(np.argmax(dev))
    G = float(dev[i] / sd)
    t = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    crit = float((n - 1) / np.sqrt(n) * np.sqrt(t ** 2 / (n - 2 + t ** 2)))
    return {"statistic": G, "critical": crit, "outlier_index": i, "reject": bool(G > crit)}


def normality_test(x: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    """D'Agostino 正态性检验。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    stat, p = stats.normaltest(x)
    return {"statistic": float(stat), "p_value": float(p), "reject": bool(p < alpha)}


def consistency_report(
    groups: Sequence[np.ndarray], alpha: float = 0.05
) -> Dict[str, object]:
    """对一批重复测量做一致性体检。

    Args:
        groups: 每个元素是一组重复测量（如同一天的多次测定）

    Returns:
        方差齐性、离群、正态性三项结论 + 一条行动建议
    """
    gs = _groups(groups)
    pooled = np.concatenate(gs) if gs else np.array([])
    report: Dict[str, object] = {
        "n_groups": len(gs),
        "bartlett": bartlett_test(gs, alpha),
        "levene": levene_test(gs, alpha),
        "cochran": cochran_test(gs, alpha),
        "normality": normality_test(pooled, alpha) if pooled.size else {},
    }
    var_unequal = bool(report["levene"]["reject"] or report["cochran"]["reject"])
    report["variance_homogeneous"] = not var_unequal
    report["advice"] = (
        "方差不齐：优先用稳健估计或分组建模，不要简单剔除"
        if var_unequal else "方差齐性可接受，可合并估计精密度"
    )
    return report
