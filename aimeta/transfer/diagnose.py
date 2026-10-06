"""仪器差异诊断：先量清楚差在哪，再决定用哪种迁移方法。

推荐逻辑：

    波长漂移显著      → 先做波长轴对齐，否则 DS/PDS 都在学一个错位映射
    差异是逐波长的    → PDS
    样本数 >= 波长数  → DS（更稳，参数更少）
    样本数很少        → SBC（只用预测值对）
    只想压掉差异子空间 → GLSW
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def estimate_wavelength_shift(
    wavelengths: np.ndarray, X_slave: np.ndarray, X_master: np.ndarray,
    max_shift_nm: float = 5.0,
) -> float:
    """用平均光谱的互相关估计 slave 相对 master 的波长偏移（nm）。

    正值表示 slave 的峰整体向长波方向移动。
    """
    wl = np.asarray(wavelengths, dtype=np.float64)
    step = float(np.median(np.diff(wl)))
    if step <= 0:
        return 0.0
    a = np.asarray(X_slave, dtype=np.float64).mean(axis=0)
    b = np.asarray(X_master, dtype=np.float64).mean(axis=0)
    a = a - a.mean()
    b = b - b.mean()
    max_lag = int(np.floor(max_shift_nm / step))
    if max_lag < 1:
        return 0.0
    best_lag, best_val = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        aa = np.roll(a, lag)
        if lag > 0:
            aa[:lag] = 0.0
        elif lag < 0:
            aa[lag:] = 0.0
        val = float(np.dot(aa, b) / (np.linalg.norm(aa) * np.linalg.norm(b) + 1e-12))
        if val > best_val:
            best_lag, best_val = lag, val
    return float(best_lag * step)


def estimate_gain_offset(X_slave: np.ndarray, X_master: np.ndarray) -> Dict[str, np.ndarray]:
    """逐波长的增益与偏置：master ≈ gain_j * slave + offset_j。

    增益随波长变化明显 → 说明差异是「逐波长」的，PDS 比 DS 合适。
    """
    S = np.asarray(X_slave, dtype=np.float64)
    M = np.asarray(X_master, dtype=np.float64)
    sm, mm = S.mean(axis=0), M.mean(axis=0)
    Sc, Mc = S - sm, M - mm
    denom = (Sc ** 2).sum(axis=0)
    denom[denom < 1e-12] = 1e-12
    gain = (Sc * Mc).sum(axis=0) / denom
    offset = mm - gain * sm
    return {"gain": gain, "offset": offset}


def residual_spectrum(X_slave: np.ndarray, X_master: np.ndarray) -> np.ndarray:
    """主从平均光谱之差 —— 看一眼就知道差异是全局倾斜还是局部结构。"""
    return np.asarray(X_master, dtype=np.float64).mean(axis=0) - \
        np.asarray(X_slave, dtype=np.float64).mean(axis=0)


def recommend_method(
    wavelengths: np.ndarray,
    X_slave: np.ndarray,
    X_master: np.ndarray,
    n_std_samples: Optional[int] = None,
) -> Dict[str, object]:
    """根据诊断结果推荐迁移方法与参数。

    Returns:
        dict(method, window, need_alignment, shift_nm, reason, n_samples)
    """
    shift = estimate_wavelength_shift(wavelengths, X_slave, X_master)
    n = len(X_slave) if n_std_samples is None else int(n_std_samples)
    p = X_slave.shape[1]

    # 样本数是最硬的约束：样本太少时连波长漂移都无从估计，只能做响应校正
    if n < 5:
        method, reason = "sbc", f"样本仅 {n} 个，不足以学光谱映射，退化为响应校正"
    elif abs(shift) > 0.5:
        method, reason = "align_then_pds", f"波长漂移 {shift:.2f} nm，必须先对齐波长轴"
    elif n >= p:
        method, reason = "ds", f"标准化样本 {n} >= 波长数 {p}，DS 参数更省更稳"
    else:
        method, reason = "pds", f"样本 {n} < 波长数 {p}，用窗口回归的 PDS"

    return {
        "method": method,
        "window": 5 if "pds" in method else None,
        "need_alignment": abs(shift) > 0.5,
        "shift_nm": shift,
        "n_samples": n,
        "reason": reason,
    }
