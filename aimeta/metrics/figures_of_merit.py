"""指标（Figures of Merit，FOM）。

多元校正下的定义（Olivieri 体系）：

    回归向量   b  —— 线性模型直接取 coef_；非线性模型用数值微分近似
    灵敏度     SEN = 1 / ||b||
    分析灵敏度 γ   = SEN / s_x     （s_x：空白光谱噪声）
    检出限     LOD = 3.3 · s_0 / SEN   （s_0：空白样本预测值的标准差）
    定量限     LOQ = 10 · s_0 / SEN
    选择性     SEL = 目标组分的净信号占比（0~1，越接近 1 越不受干扰）

注意：**不能**套用单变量的 3σ/slope。多元模型的噪声会被潜变量压缩，
用单变量公式会系统性低估 LOD。
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np


def regression_vector(model, X0: np.ndarray) -> np.ndarray:
    """求模型在某工作点 X0 处的回归向量 b（长度 = 波长数）。

    线性模型直接返回 coef_；非线性模型用中心差分数值求导。
    """
    X0 = np.asarray(X0, dtype=np.float64).ravel()
    if hasattr(model, "coef_"):
        b = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
        if b.shape[0] == X0.shape[0]:
            return b
    # 数值近似：b_j = dy/dx_j
    eps = 1e-4
    base = float(np.asarray(model.predict(X0[None, :])).ravel()[0])
    b = np.empty_like(X0)
    for j in range(X0.shape[0]):
        xp = X0.copy()
        xp[j] += eps
        b[j] = (float(np.asarray(model.predict(xp[None, :])).ravel()[0]) - base) / eps
    return b


def sensitivity(b: np.ndarray) -> float:
    """SEN = 1 / ||b||。"""
    n = float(np.linalg.norm(b))
    return 1.0 / n if n > 0 else float("inf")


def analytical_sensitivity(b: np.ndarray, s_x: float) -> float:
    """γ = SEN / s_x，单位浓度对应的信噪比。"""
    return sensitivity(b) / s_x if s_x > 0 else float("inf")


def selectivity(b: np.ndarray, s_target: np.ndarray) -> float:
    """选择性：目标组分净信号在被模型利用的总信号中的占比。"""
    num = float(np.linalg.norm(b * s_target))
    den = float(np.linalg.norm(b))
    return num / den if den > 0 else 0.0


def spectral_noise(X_blank: np.ndarray) -> float:
    """空白样本光谱噪声 s_x：逐条光谱去趋势后的残差标准差。"""
    X = np.asarray(X_blank, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    resid = X - X.mean(axis=1, keepdims=True)
    return float(np.sqrt(np.mean(resid ** 2)))


def figures_of_merit(
    model,
    X_blank: np.ndarray,
    X_cal: Optional[np.ndarray] = None,
    s_target: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """一次性算出全部指标。

    Args:
        model:    拟合好的模型（fit/predict 接口）
        X_blank:  空白样本光谱 (n_blank, p)，即与样品基质一致、不含待测物的
                  溶剂（UV 水质场景即纯净水/去离子水；地表水取不到绝对零浓度时，
                  可近似为近检出限的低浓度水样，但优先纯水/去离子水）；
                  必须保留原始测量噪声、且未被替为 LOD/2；
                  **必须与模型训练时处于同一输入空间**（即已过预处理链）；
                  **建议同条件独立重复 ≥10 组**（最少 ≥2，n_blank=1 时 s_0 静默退化为 0）
        X_cal:    校正集光谱（可选，用于估计工作点）
        s_target: 目标组分的净信号光谱（可选，用于算选择性）

    Returns:
        dict(SEN, gamma, s_x, s_0, LOD, LOQ, SEL)
    """
    X_blank = np.asarray(X_blank, dtype=np.float64)
    if X_blank.ndim == 1:
        X_blank = X_blank[None, :]
    x0 = X_cal.mean(axis=0) if X_cal is not None else X_blank.mean(axis=0)

    b = regression_vector(model, x0)
    s_x = spectral_noise(X_blank)
    preds = np.asarray(model.predict(X_blank), dtype=np.float64).ravel()
    if len(preds) > 1 and np.allclose(preds, preds[0]):
        warnings.warn(
            "空白样本的预测值全部相同，回归向量将被估计为 0（SEN/LOD 退化）。"
            "常见原因：X_blank 未经预处理，或预测值落在检出限以下被替为 LOD/2。"
        )
    s_0 = float(np.std(preds, ddof=1)) if len(preds) > 1 else float(np.std(preds))

    sen = sensitivity(b)
    sel = selectivity(b, s_target) if s_target is not None else float("nan")
    return {
        "SEN": sen,
        "gamma": analytical_sensitivity(b, s_x),
        "s_x": s_x,
        "s_0": s_0,
        "LOD": 3.3 * s_0 / sen if np.isfinite(sen) and sen > 0 else float("inf"),
        "LOQ": 10.0 * s_0 / sen if np.isfinite(sen) and sen > 0 else float("inf"),
        "SEL": sel,
    }
