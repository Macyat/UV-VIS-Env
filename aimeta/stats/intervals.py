"""区间估计。

验收报告里最有说服力的两个数：
    置信区间    「方法均值落在哪」——评价正确度
    容忍区间    「未来单个测定值落在哪」——评价能否用于判定达标 区分三种容忍区间：b-content（至少覆盖比例 β 的总体）、
b-expectation（平均覆盖比例 β）、无分布假设。水质验收通常用 b-content。
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import stats


def confidence_interval_mean(
    x: np.ndarray, conf: float = 0.95, sigma: float | None = None
) -> Tuple[float, float]:
    """均值的置信区间（sigma 已知用 z，未知用 t，）。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    mu = float(x.mean())
    if sigma is not None:
        z = stats.norm.ppf(0.5 + conf / 2)
        half = z * sigma / np.sqrt(n)
    else:
        sd = float(x.std(ddof=1)) if n > 1 else 0.0
        t = stats.t.ppf(0.5 + conf / 2, n - 1) if n > 1 else 0.0
        half = t * sd / np.sqrt(n)
    return mu - half, mu + half


def tolerance_interval(
    x: np.ndarray, content: float = 0.95, conf: float = 0.95,
    kind: str = "b-content",
) -> Tuple[float, float]:
    """容忍区间。

    Args:
        content: 要覆盖的总体比例 β
        conf:    置信水平
        kind:    "b-content" | "b-expectation" | "nonparametric"
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < 2:
        return float(x.min()), float(x.max())
    mu, sd = float(x.mean()), float(x.std(ddof=1))

    if kind == "b-expectation":
        z = stats.norm.ppf(0.5 + content / 2)
        return mu - z * sd, mu + z * sd

    if kind == "nonparametric":
        # 无分布假设：用次序统计量，n 需足够大才覆盖得住
        k = int(np.ceil(n * content))
        xs = np.sort(x)
        lo = xs[max(0, (n - k) // 2)]
        hi = xs[min(n - 1, n - 1 - (n - k) // 2)]
        return float(lo), float(hi)

    z = stats.norm.ppf(0.5 + content / 2)
    chi2 = stats.chi2.ppf(1 - conf, n - 1)
    k = z * np.sqrt(1.0 + 1.0 / n) * np.sqrt((n - 1) / chi2)
    return mu - k * sd, mu + k * sd


def prediction_interval(
    x: np.ndarray, conf: float = 0.95, n_future: int = 1
) -> Tuple[float, float]:
    """单个（或若干）未来观测的预测区间。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    mu, sd = float(x.mean()), float(x.std(ddof=1)) if n > 1 else 0.0
    t = stats.t.ppf(0.5 + conf / 2, n - 1) if n > 1 else 0.0
    half = t * sd * np.sqrt(1.0 / n_future + 1.0 / n)
    return mu - half, mu + half


def precision_summary(x: np.ndarray, conf: float = 0.95) -> Dict[str, float]:
    """重复性/精密度的标准汇报口径：均值、SD、RSD、置信与容忍区间。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    mu = float(x.mean())
    sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    ci = confidence_interval_mean(x, conf)
    ti = tolerance_interval(x, content=0.95, conf=conf)
    return {
        "mean": mu,
        "sd": sd,
        "rsd": sd / mu if mu != 0 else float("nan"),
        "ci_low": ci[0], "ci_high": ci[1],
        "ti_low": ti[0], "ti_high": ti[1],
    }
