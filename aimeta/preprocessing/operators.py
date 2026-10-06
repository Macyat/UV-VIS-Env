"""预处理算子（全部注册到 ``PREPROC``）。

- ``deriv_gram``  MATLAB ``DERIV`` 的移植实现（Gram 多项式），
                  与 ``scipy.signal.savgol_filter`` 数值等价，由回归测试锁死
- ``savgol``      scipy Savitzky-Golay 平滑/求导
- ``wavelet``     sym 小波软阈值去噪（edge 端委托执行，需 PyWavelets）
- ``snv``         标准正态变换，**严格逐条光谱**
- ``msc``         多元散射校正
- ``mean_center`` / ``column_scale``  列缩放

使用约定：
    - 小波去噪输出长度与输入一致（不做截断）。
    - ``snv`` 只提供「逐条光谱」语义；若需要对整个矩阵做列标准化，
      请用 ``column_scale``，避免同一个名字出现两种含义。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pywt
from scipy.signal import savgol_filter

from ..core.registry import PREPROC
from .base import Transformer

# --------------------------------------------------------------------------
# Gram 多项式（移植自 ai-meta-main/preprocessing.py）
# --------------------------------------------------------------------------


def genfact(a: int, b: int) -> int:
    """广义阶乘：a*(a-1)*...*(a-b+1)。"""
    gf = 1
    for i in range(a - b + 1, a + 1):
        gf *= i
    return gf


def grampoly(i: int, m: int, k: int, s: int) -> float:
    """递归计算 Gram 多项式。"""
    if k > 0:
        r1 = grampoly(i, m, k - 1, s)
        r2 = grampoly(i, m, k - 1, s - 1)
        r3 = grampoly(i, m, k - 2, s)
        return (
            ((4 * k - 2) / (k * (2 * m - k + 1))) * (i * r1 + s * r2)
            - (((k - 1) * (2 * m + k)) / (k * (2 * m - k + 1))) * r3
        )
    return 1.0 if (k == 0 and s == 0) else 0.0


def weight(i: int, t: int, m: int, n: int, s: int) -> float:
    """MATLAB DERIV 中的 weight 函数。"""
    total = 0.0
    for k in range(n + 1):
        total += (
            (2 * k + 1)
            * (genfact(2 * m, k) / genfact(2 * m + k + 1, k + 1))
            * grampoly(i, m, k, 0)
            * grampoly(t, m, k, s)
        )
    return total


def DERIV(x: np.ndarray, der: int, window: Optional[int] = None, order: int = 2) -> np.ndarray:
    """Gram 多项式平滑/求导，与 ``savgol_filter(window, order, deriv=der)`` 等价。

    老实现里窗口中心用 ``round(window/2)``，遇 Python 银行家舍入（如 window=11）
    会偏移一位导致崩溃；此处固定为 ``window // 2`` 并强制奇数窗口。
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("DERIV expects a 2-D (n_samples, n_wavelengths) array")
    _, nc = x.shape
    if window is None:
        window = min(17, nc // 2)
        if window % 2 == 0:
            window -= 1
    if window % 2 == 0:
        raise ValueError(f"window must be odd, got {window}")
    if window > nc:
        raise ValueError(f"window ({window}) must not exceed n_wavelengths ({nc})")

    m = window // 2
    p = m
    w = np.zeros((window, window))
    for i in range(window):
        for j in range(window):
            w[i, j] = weight(i - p, j - p, m, order, der)

    yr = np.zeros_like(x)
    yr[:, :m] = x[:, :window] @ w[:, :m]
    for i in range(nc - 2 * m):
        yr[:, i + m] = x[:, i : (i + 2 * m + 1)] @ w[:, p]
    a = nc - 2 * m - 1
    yr[:, (nc - m) : nc] = x[:, a:nc] @ w[:, (p + 1) : window]
    return yr


def savgol(x: np.ndarray, window: int = 15, polyorder: int = 3, deriv: int = 0) -> np.ndarray:
    return savgol_filter(x, window, polyorder, deriv=deriv, axis=1)


def wavelet_denoise(x_row: np.ndarray, wavelet: str = "sym4", level: int = 2) -> np.ndarray:
    """小波软阈值去噪（单条光谱），输出长度与输入一致。"""
    coeff = pywt.wavedec(x_row, wavelet, mode="smooth", level=level)
    sigma = np.median(np.abs(coeff[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(x_row)))
    coeff[1:] = [pywt.threshold(c, value=uthresh, mode="soft") for c in coeff[1:]]
    out = pywt.waverec(coeff, wavelet, mode="smooth")
    return _fit_length(out, len(x_row))


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n]
    return np.pad(x, (0, n - len(x)), mode="edge")


def snv_transform(X: np.ndarray) -> np.ndarray:
    """逐条光谱 SNV：(x - mean_i) / std_i。"""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def msc_transform(X: np.ndarray, reference: Optional[np.ndarray] = None) -> np.ndarray:
    """多元散射校正：对每条光谱做 x ≈ a·ref + b 的一元回归后校正。"""
    X = np.asarray(X, dtype=np.float64)
    ref = X.mean(axis=0) if reference is None else np.asarray(reference, dtype=np.float64)
    out = np.empty_like(X)
    A = np.vstack([ref, np.ones_like(ref)]).T
    for i in range(X.shape[0]):
        sol, *_ = np.linalg.lstsq(A, X[i], rcond=None)
        a, b = sol
        if abs(a) < 1e-12:
            a = 1e-12
        out[i] = (X[i] - b) / a
    return out


# --------------------------------------------------------------------------
# 注册为 Transformer
# --------------------------------------------------------------------------


@PREPROC.register("deriv_gram", family="smooth", compilable=True)
class DerivGram(Transformer):
    """Gram 多项式平滑/求导（与 scipy savgol 等价）。"""

    op = "deriv_gram"

    def __init__(self, der: int = 0, window: Optional[int] = None, order: int = 2):
        super().__init__(der=der, window=window, order=order)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return DERIV(X, der=self.params["der"], window=self.params["window"],
                     order=self.params["order"])


@PREPROC.register("savgol", family="smooth", compilable=True)
class SavGol(Transformer):
    op = "savgol"

    def __init__(self, window: int = 15, polyorder: int = 3, deriv: int = 0):
        super().__init__(window=window, polyorder=polyorder, deriv=deriv)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return savgol(X, **self.params)


@PREPROC.register("wavelet", family="denoise", compilable=False)
class Wavelet(Transformer):
    op = "wavelet"

    def __init__(self, wavelet: str = "sym4", level: int = 2):
        super().__init__(wavelet=wavelet, level=level)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.vstack([wavelet_denoise(r, self.params["wavelet"], self.params["level"])
                          for r in X])


@PREPROC.register("snv", family="scatter", compilable=True)
class SNV(Transformer):
    op = "snv"

    def transform(self, X: np.ndarray) -> np.ndarray:
        return snv_transform(X)


@PREPROC.register("msc", family="scatter", compilable=True)
class MSC(Transformer):
    """MSC：fit 阶段记住参考光谱，transform 阶段复用（部署端需要 fit 态）。"""

    op = "msc"

    def __init__(self):
        super().__init__()
        self.reference_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y=None) -> "MSC":
        self.reference_ = np.asarray(X, dtype=np.float64).mean(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return msc_transform(X, self.reference_)


@PREPROC.register("mean_center", family="scaling", compilable=True)
class MeanCenter(Transformer):
    """列中心化；均值由 fit 记住，供部署端复用。"""

    op = "mean_center"

    def __init__(self):
        super().__init__()
        self.mean_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y=None) -> "MeanCenter":
        self.mean_ = np.asarray(X, dtype=np.float64).mean(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) - self.mean_


@PREPROC.register("column_scale", family="scaling", compilable=True)
class ColumnScale(Transformer):
    """列标准化（自标度）。注意：这是**列方向**缩放，不是 SNV。"""

    op = "column_scale"

    def __init__(self):
        super().__init__()
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray, y=None) -> "ColumnScale":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_
