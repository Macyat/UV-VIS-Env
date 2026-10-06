"""控制图。

三种图的适用场景：
    Shewhart  抓大偏移（>= 1.5σ），简单直观
    CUSUM     抓持续的小漂移（累积偏差），现场探头慢污染最好用
    EWMA      介于两者之间，对自相关数据更友好

关键细节：水质时序**强自相关**，直接用 i.i.d. 控制限会疯狂误报。
必须先算有效样本量 n_eff 把控制限放宽。
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import stats


def effective_sample_size(x: np.ndarray, max_lag: int = 50) -> float:
    """用一阶自相关估计有效样本量：n_eff = n·(1-ρ)/(1+ρ)。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < 3:
        return float(n)
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    rho = float(np.dot(xc[:-1], xc[1:]) / denom) if denom > 0 else 0.0
    rho = float(np.clip(rho, -0.99, 0.99))
    return max(1.0, n * (1 - rho) / (1 + rho))


def shewhart_limits(x: np.ndarray, k: float = 3.0,
                    autocorr: bool = True) -> Tuple[float, float, float]:
    """均值图的中心线 / 上 / 下限。

    Args:
        k: 控制限宽度（3 = 常规 3σ）
        autocorr: 是否按自相关修正（用 n_eff 代替 n 计算标准误）
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    mu = float(x.mean())
    sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    n = effective_sample_size(x) if autocorr else float(len(x))
    half = k * sd / np.sqrt(max(n, 1.0))
    return mu, mu - half, mu + half


def cusum(x: np.ndarray, target: float | None = None, k: float = 0.5,
          h: float = 5.0) -> Dict[str, np.ndarray]:
    """CUSUM：累积偏差。|C+| 或 |C-| 超过 h·σ 即报警。

    Args:
        k: 允许 slack（通常 0.5σ）
        h: 决策区间（通常 4~5σ）
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    t = float(x.mean()) if target is None else float(target)
    sd = float(x.std(ddof=1)) if len(x) > 1 else 1.0
    sd = sd if sd > 0 else 1.0
    z = (x - t) / sd
    cp = np.zeros_like(z)
    cm = np.zeros_like(z)
    for i in range(1, len(z)):
        cp[i] = max(0.0, cp[i - 1] + z[i] - k)
        cm[i] = min(0.0, cm[i - 1] + z[i] + k)
    return {
        "cusum_pos": cp,
        "cusum_neg": cm,
        "limit": h,
        "alarm_pos": cp > h,
        "alarm_neg": cm < -h,
    }


def ewma(x: np.ndarray, lam: float = 0.2, k: float = 3.0,
         target: float | None = None) -> Dict[str, np.ndarray]:
    """EWMA：指数加权移动平均，控制限随时间收敛到稳态。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    t = float(x.mean()) if target is None else float(target)
    sd = float(x.std(ddof=1)) if len(x) > 1 else 1.0
    sd = sd if sd > 0 else 1.0
    n = len(x)
    z = np.empty(n)
    z[0] = lam * x[0] + (1 - lam) * t
    for i in range(1, n):
        z[i] = lam * x[i] + (1 - lam) * z[i - 1]
    idx = np.arange(1, n + 1)
    half = k * sd * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * idx)))
    return {"ewma": z, "ucl": t + half, "lcl": t - half, "alarm": np.abs(z - t) > half}


def arl0(k: float = 3.0) -> float:
    """受控状态下的平均运行长度 ARL0 ≈ 1/α。

    ARL0 = 370 意味着「平均 370 个点误报一次」，是选控制限宽度的依据。
    """
    alpha = 2.0 * (1.0 - stats.norm.cdf(k))
    return 1.0 / alpha if alpha > 0 else float("inf")
