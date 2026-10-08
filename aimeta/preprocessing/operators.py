"""预处理算子（全部注册到 ``PREPROC``）。

- ``deriv_gram``  MATLAB ``DERIV`` 的移植实现（Gram 多项式），
                  与 ``scipy.signal.savgol_filter`` 数值等价，由回归测试锁死
- ``savgol``      scipy Savitzky-Golay 平滑/求导
- ``wavelet``     sym 小波软阈值去噪（edge 端委托执行，需 PyWavelets）
- ``snv``         标准正态变换，**严格逐条光谱**
- ``msc``         多元散射校正
- ``whittaker``   Whittaker 平滑（Eilers 2003，PLS_Toolbox ``wsmooth`` 同源）
- ``baseline``    非对称最小二乘(ALS)基线扣除（Eilers & Boelens 2005）
- ``wlsbaseline`` 加权最小二乘基线扣除（ALS 底层入口，对齐 PLS_Toolbox 命名）
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
from scipy.sparse import diags, eye as sparse_eye
from scipy.sparse.linalg import spsolve

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
# 基线校正（Eilers 方法：Whittaker 平滑 + 非对称最小二乘 ASLS）
# 注意：ASLS（Asymmetric，基线）与 MCR-ALS 的交替最小二乘（Alternating）是两回事。
# 与 PLS_Toolbox 的 wsmooth / baseline / wlsbaseline 同源，独立实现。
# --------------------------------------------------------------------------


def _diff_matrix(n: int, d: int):
    """d 阶差分矩阵 D（shape (n-d, n)，CSC 稀疏）。D_d = D_1 连续应用 d 次。"""
    D = sparse_eye(n, format="csc")
    for _ in range(d):
        D = D[1:, :] - D[:-1, :]
    return D.tocsc()


def _whittaker_row(x: np.ndarray, lam: float, DtD, n: int) -> np.ndarray:
    """单条光谱 Whittaker 平滑：z = (I + λ DᵀD)⁻¹ x。"""
    return spsolve(sparse_eye(n, format="csc") + lam * DtD, x)


def _asls_row(x: np.ndarray, lam: float, p: float, DtD, n: int, n_iter: int) -> np.ndarray:
    """单条光谱非对称最小二乘(ALS)基线：迭代 (W + λ DᵀD) z = W x。

    每轮把残差为正（信号峰，位于基线之上）的点权重压到 ``p``，
    残差非正（基线）的点权重放到 ``1-p``，从而只拟合峰下方的基线。
    """
    w = np.ones(n)
    z = x.copy()
    for _ in range(n_iter):
        z = spsolve((diags(w, 0, format="csc") + lam * DtD).tocsc(), w * x)
        w = np.where(x > z, p, 1.0 - p)
    return z


def whittaker_smooth(X: np.ndarray, lam: float = 1e4, d: int = 2) -> np.ndarray:
    """Whittaker 平滑（逐条光谱）。返回平滑后的信号（保留峰形与基线）。"""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("whittaker_smooth expects a 2-D (n_samples, n_wavelengths) array")
    if lam <= 0:
        raise ValueError(f"lam 必须为正，收到 {lam}")
    if d < 1:
        raise ValueError(f"d 必须 ≥ 1，收到 {d}")
    n = X.shape[1]
    D = _diff_matrix(n, d)
    DtD = (D.T @ D).tocsc()
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        out[i] = _whittaker_row(X[i], lam, DtD, n)
    return out


def asls_baseline(X: np.ndarray, lam: float = 1e6, p: float = 1e-3,
                 d: int = 2, n_iter: int = 10) -> np.ndarray:
    """非对称最小二乘(ASLS)基线扣除（逐条光谱）。返回 X - baseline。"""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("asls_baseline expects a 2-D (n_samples, n_wavelengths) array")
    if lam <= 0:
        raise ValueError(f"lam 必须为正，收到 {lam}")
    if not (0.0 < p < 1.0):
        raise ValueError(f"p 必须在 (0,1)，收到 {p}")
    if d < 1:
        raise ValueError(f"d 必须 ≥ 1，收到 {d}")
    n = X.shape[1]
    D = _diff_matrix(n, d)
    DtD = (D.T @ D).tocsc()
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        out[i] = X[i] - _asls_row(X[i], lam, p, DtD, n, n_iter)
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


@PREPROC.register("osc", family="orthogonal", compilable=False)
class OSC(Transformer):
    """正交信号校正（OSC，§3.11）：去除 X 中与 y 正交（无关）的变异。

    用 Xᵀy 方向提取得分，再对 y 正交化后从 X 减去，逐成分迭代。用于剔除温度、
    基体等与浓度无关的变异。``fit`` 需要 ``y``（监督式）。
    """

    op = "osc"

    def __init__(self, n_components: int = 1, tol: float = 1e-6):
        super().__init__(n_components=n_components, tol=tol)
        self.W_: Optional[np.ndarray] = None   # (p, k) 权重
        self.P_: Optional[np.ndarray] = None   # (p, k) 载荷
        self.mean_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y=None) -> "OSC":
        if y is None:
            raise ValueError("OSC 需要 y（监督式正交信号校正）")
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        yc = y - y.mean()

        W: list = []
        P: list = []
        for _ in range(self.params["n_components"]):
            w = Xc.T @ yc
            nw = float(np.linalg.norm(w))
            if nw < 1e-12:
                break
            w = w / nw
            t = Xc @ w
            # 得分对 y 正交化
            t_orth = t - yc * float(yc @ t) / float(yc @ yc)
            nt = float(np.linalg.norm(t_orth))
            if nt < self.params["tol"]:
                break
            p = Xc.T @ t_orth / (t_orth @ t_orth)
            W.append(w)
            P.append(p)
            Xc = Xc - np.outer(t_orth, p)

        self.W_ = np.array(W).T if W else np.zeros((X.shape[1], 0))
        self.P_ = np.array(P).T if P else np.zeros((X.shape[1], 0))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xt = np.asarray(X, dtype=np.float64) - self.mean_
        for k in range(self.W_.shape[1]):
            t = Xt @ self.W_[:, k]
            Xt = Xt - np.outer(t, self.P_[:, k])
        return Xt


@PREPROC.register("whittaker", family="smooth", compilable=True)
class Whittaker(Transformer):
    """Whittaker 平滑（Eilers 2003，与 PLS_Toolbox ``wsmooth`` 同源）。"""

    op = "whittaker"

    def __init__(self, lam: float = 1e4, d: int = 2):
        super().__init__(lam=lam, d=d)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return whittaker_smooth(X, lam=self.params["lam"], d=self.params["d"])


@PREPROC.register("baseline", family="baseline", compilable=True)
class Baseline(Transformer):
    """非对称最小二乘(ASLS)基线扣除（Eilers & Boelens 2005；PLS_Toolbox ``baseline``）。

    用非对称权重把「信号峰」压低、只拟合峰下方的基线，再减去。用于剔除
    浊度散射导致的基线抬升/弯曲。参数名对齐 PLS_Toolbox：``order``=差分阶、
    ``itermax``=迭代次数、``p``=非对称权重（0<p<1，越小越贴底部）。
    """

    op = "baseline"

    def __init__(self, order: int = 2, lam: float = 1e6, itermax: int = 10, p: float = 1e-3):
        super().__init__(order=order, lam=lam, itermax=itermax, p=p)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return asls_baseline(X, lam=self.params["lam"], p=self.params["p"],
                            d=self.params["order"], n_iter=self.params["itermax"])


@PREPROC.register("wlsbaseline", family="baseline", compilable=True)
class WLSBaseline(Transformer):
    """加权最小二乘基线扣除（PLS_Toolbox ``wlsbaseline`` 同源，ASLS 底层入口）。

    与 ``baseline`` 同算法，但直接暴露差分阶 ``d`` 与迭代次数 ``n_iter``，
    便于按 PLS_Toolbox 手册的 ``wlsbaseline(x, lam, p, d)`` 一一对照调参。
    """

    op = "wlsbaseline"

    def __init__(self, lam: float = 1e6, p: float = 1e-3, d: int = 2, n_iter: int = 10):
        super().__init__(lam=lam, p=p, d=d, n_iter=n_iter)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return asls_baseline(X, lam=self.params["lam"], p=self.params["p"],
                            d=self.params["d"], n_iter=self.params["n_iter"])
