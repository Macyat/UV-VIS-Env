"""常用图：光谱、诊断、潜变量、贡献图、控制图。

每张图都返回 Figure，由调用方决定 save_fig 还是继续叠加。
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from .theme import COLORS, palette


def _ax(ax=None):
    return ax if ax is not None else plt.subplots()[1]


# ---------------------------------------------------------------- 光谱
def plot_spectra(wavelengths, X, ax=None, alpha: float = 0.5,
                 color: str | None = None, title: str = "光谱"):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3.4))
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    for i in range(X.shape[0]):
        ax.plot(wavelengths, X[i], color=color or COLORS["spectra"],
                alpha=alpha, lw=1.0)
    ax.set_xlabel("波长 / nm"); ax.set_ylabel("吸光度")
    ax.set_title(title)
    return fig or ax.figure


def plot_derivative(wavelengths, X, der: int = 1, ax=None, title: str | None = None):
    from ..preprocessing.operators import DERIV
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3.4))
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    D = DERIV(X, der=der)
    for i in range(D.shape[0]):
        ax.plot(wavelengths, D[i], alpha=0.6, lw=1.0)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("波长 / nm")
    ax.set_ylabel(f"{der} 阶导数")
    ax.set_title(title or f"{der} 阶导数光谱")
    return fig or ax.figure


def plot_difference(wavelengths, a, b, ax=None, labels=("master", "slave"),
                    title: str = "差谱"):
    """差谱：看两台仪器 / 两个批次的系统性差异。"""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3.2))
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    ax.plot(wavelengths, a, label=labels[0])
    ax.plot(wavelengths, b, label=labels[1])
    ax.fill_between(wavelengths, a, b, color=COLORS["accent"], alpha=0.25,
                    label="差异")
    ax.legend(); ax.set_xlabel("波长 / nm"); ax.set_title(title)
    return fig or ax.figure


# ---------------------------------------------------------------- 诊断
def plot_pred_vs_actual(y_true, y_pred, ax=None, label: str = "",
                        unit: str = "", ranges: Optional[Sequence[float]] = None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    ax.scatter(yt, yp, s=18, alpha=0.7, color=COLORS["primary"])
    lo = float(min(yt.min(), yp.min())); hi = float(max(yt.max(), yp.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    if ranges:
        for r in ranges:
            ax.axvline(r, color=COLORS["accent"], lw=0.8, ls=":")
            ax.axhline(r, color=COLORS["accent"], lw=0.8, ls=":")
    ax.set_xlabel(f"实测 {unit}"); ax.set_ylabel(f"预测 {unit}")
    ax.set_title(f"{label} 预测 vs 实测")
    return fig or ax.figure


def plot_residuals(y_true, y_pred, ax=None, label: str = ""):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3.0))
    r = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(r, marker="o", ms=3, lw=0, color=COLORS["bad"])
    ax.set_xlabel("样本序号"); ax.set_ylabel("残差")
    ax.set_title(f"{label} 残差序列")
    return fig or ax.figure


# ------------------------------------------------------------ 潜变量
def plot_explained_variance(X, max_components: int = 20, ax=None):
    """碎石图：用来定 PLS / PCA 的主成分数。"""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
    Xc = np.asarray(X, dtype=np.float64) - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False)
    var = s ** 2 / np.sum(s ** 2)
    k = min(max_components, len(var))
    ax.plot(np.arange(1, k + 1), var[:k] * 100, marker="o", ms=4,
            color=COLORS["primary"])
    ax.set_xlabel("主成分数"); ax.set_ylabel("解释方差 / %")
    ax.set_title("碎石图")
    return fig or ax.figure


# ------------------------------------------------------------ 贡献图
def plot_contribution(wavelengths, contribution, ax=None,
                      title: str = "SPE 贡献图"):
    """把异常拆回波长：哪段波长最该怀疑。"""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3.2))
    c = np.asarray(contribution, dtype=np.float64).ravel()
    ax.plot(wavelengths, c, color=COLORS["bad"], lw=1.2)
    ax.fill_between(wavelengths, 0, c, color=COLORS["bad"], alpha=0.2)
    ax.set_xlabel("波长 / nm"); ax.set_ylabel("贡献")
    ax.set_title(title)
    return fig or ax.figure


# ------------------------------------------------------------ 控制图
def plot_control_chart(values, limits=None, alarms=None, ax=None,
                       title: str = "控制图"):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3.2))
    v = np.asarray(values, dtype=np.float64)
    ax.plot(v, marker="o", ms=3, lw=1, color=COLORS["primary"])
    if limits:
        cl, lcl, ucl = limits
        ax.axhline(cl, color="k", lw=0.8)
        ax.axhline(ucl, color=COLORS["bad"], ls="--", lw=1)
        ax.axhline(lcl, color=COLORS["bad"], ls="--", lw=1)
    if alarms is not None:
        a = np.asarray(alarms, dtype=bool)
        ax.scatter(np.where(a)[0], v[a], color=COLORS["bad"], s=28, zorder=5)
    ax.set_xlabel("序号"); ax.set_title(title)
    return fig or ax.figure
