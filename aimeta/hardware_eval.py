"""光谱仪硬件评价（出厂 / 现场检验）。

与 `metrics.figures_of_merit`（**分析方法**品质因数）区分：本模块评价**光谱仪硬件本身**，
按紫外-可见分光光度计的计量检验口径实现可计算项，主要依据：

- JJG 178《紫外、可见、近红外分光光度计》检定规程
- ASTM E275 系列（紫外-可见-近红外分光光度计性能评价：波长准确度、
  光度准确度、杂散光、基线平直度、分辨率、噪声）

本模块只提供**从实测光谱计算指标**的函数；具体的标准物质 / 滤光片
（钬玻璃、镨钕、中性密度片、截止滤光片、汞灯 / 氩灯等）由使用方按检定规程准备。

约定：吸光度用 A，透射比用 %T；波长单位 nm。所有光谱矩阵**行=重复测量，列=波长**。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# 钬玻璃 / 钬氧化物典型特征峰（nm），用于波长准确度检验
DEFAULT_HOLMIUM_PEAKS = (279.4, 287.5, 333.7, 360.9, 418.5, 453.2, 536.2, 637.5)

# 典型合格判据（示例阈值，按仪器等级 / 检定规程调整）
DEFAULT_LIMITS = {
    "wavelength_accuracy_nm": 0.5,   # 波长示值误差 ≤ 0.5 nm（JJG 178 常见要求）
    "photometric_accuracy": 0.002,   # 光度准确度 ≤ 0.002 A（或 0.3 %T）
    "stray_light_Tpct": 0.05,        # 杂散光 ≤ 0.05 %T（NaI / 截止滤光片）
    "dark_noise": 0.0005,            # 暗噪声示例阈值（A 或 %T）
    "baseline_repeatability": 0.001,  # 基线重复性示例阈值
    "resolution_nm": 2.0,            # 分辨率（FWHM）≤ 2 nm（1 nm 狭缝典型）
}


def _parabolic_center(x: np.ndarray, y: np.ndarray, i: int) -> Tuple[float, float]:
    """对 (x[i-1], x[i], x[i+1]) 做抛物线插值，返回中心波长 x* 与峰值 y*。"""
    if i <= 0 or i >= len(x) - 1:
        return float(x[i]), float(y[i])
    x0, x1, x2 = x[i - 1], x[i], x[i + 1]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(denom) < 1e-30:
        return float(x[i]), float(y[i])
    a = ((y2 - y1) / (x2 - x1)) - ((y1 - y0) / (x1 - x0))
    a = a / (x2 - x0)
    b = (y2 - y1) / (x2 - x1) - a * (x2 + x1)
    xc = -b / (2.0 * a) if a != 0 else float(x[i])
    yc = a * xc ** 2 + b * xc + (y0 - a * x0 ** 2 - b * x0)
    if not (x[i - 1] <= xc <= x[i + 1]):  # 抛物线外推失效时回退
        return float(x[i]), float(y[i])
    return float(xc), float(yc)


def find_peak_wavelength(wavelengths: np.ndarray, spectrum: np.ndarray,
                         around: float, window_nm: float = 5.0) -> float:
    """在 around±window_nm 窗口内找最强峰的中心波长（抛物线细化）。"""
    wl = np.asarray(wavelengths, dtype=float)
    sp = np.asarray(spectrum, dtype=float).ravel()
    mask = (wl >= around - window_nm) & (wl <= around + window_nm)
    if not np.any(mask):
        return float("nan")
    idx = np.arange(len(wl))[mask]
    i = idx[np.argmax(sp[mask])]
    xc, _ = _parabolic_center(wl, sp, i)
    return xc


def wavelength_accuracy(wavelengths: np.ndarray, spectrum: np.ndarray,
                        ref_peaks: Sequence[float],
                        window_nm: float = 5.0) -> Dict[str, float]:
    """波长准确度：用已知峰位（如钬玻璃）的参考物质，比较实测峰位，
    返回各峰误差（实测-参考）与最大绝对误差（nm）。"""
    wl = np.asarray(wavelengths, dtype=float)
    sp = np.asarray(spectrum, dtype=float).ravel()
    errs: Dict[str, float] = {}
    for rp in ref_peaks:
        meas = find_peak_wavelength(wl, sp, around=float(rp), window_nm=window_nm)
        errs[f"{rp:g}nm"] = meas - float(rp)
    max_abs = max(abs(v) for v in errs.values()) if errs else float("nan")
    return {"errors_nm": errs, "max_abs_error_nm": max_abs}


def photometric_accuracy(wavelengths: np.ndarray, spectrum: np.ndarray,
                         certified: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    """光度准确度：在已知吸光度的标准（中性密度片 / 重铬酸钾等）波长点比较
    实测值，返回各点误差与最大绝对误差（与 spectrum 同单位，A 或 %T）。"""
    wl = np.asarray(wavelengths, dtype=float)
    sp = np.asarray(spectrum, dtype=float).ravel()
    errs: Dict[str, float] = {}
    for cwl, cval in certified:
        meas = float(np.interp(cwl, wl, sp))
        errs[f"{cwl:g}nm"] = meas - float(cval)
    max_abs = max(abs(v) for v in errs.values()) if errs else float("nan")
    return {"errors": errs, "max_abs_error": max_abs}


def dark_noise(dark_repeats: np.ndarray) -> Dict[str, float]:
    """暗噪声：入射光遮挡（0%T 线）下多次重复测量的标准差。
    返回逐波长标准差的最大值与均值（与输入同单位）。"""
    d = np.asarray(dark_repeats, dtype=float)
    if d.ndim == 1:
        d = d[None, :]
    if d.shape[0] < 2:
        return {"std_max": float("nan"), "std_mean": float("nan")}
    sd = d.std(axis=0, ddof=1)
    return {"std_max": float(np.max(sd)), "std_mean": float(np.mean(sd))}


def baseline_flatness(baseline_repeats: np.ndarray,
                      ideal: Optional[float] = None) -> Dict[str, float]:
    """基线平直度：100%T 参考线（或空白）多次重复的散布，及相对理想值的偏离。
    - repeatability_max：逐波长跨重复 std 的最大值（基线重复性）
    - deviation_max：平均基线相对 ideal 的最大绝对偏离（未给 ideal 则返回 nan）"""
    b = np.asarray(baseline_repeats, dtype=float)
    if b.ndim == 1:
        b = b[None, :]
    out = {"repeatability_max": float("nan"), "deviation_max": float("nan")}
    if b.shape[0] >= 2:
        sd = b.std(axis=0, ddof=1)
        out["repeatability_max"] = float(np.max(sd))
    if ideal is not None:
        out["deviation_max"] = float(np.max(np.abs(b.mean(axis=0) - float(ideal))))
    return out


def stray_light(wavelengths: np.ndarray, transmittance: np.ndarray,
               check_wl: Sequence[float]) -> Dict[str, float]:
    """杂散光：在滤光片完全截止的波长处实测到的透射比即为杂散光水平。
    例如 NaI 滤光片在 340 nm 处标称 ~0%T，实测 T% 即杂散光。
    返回各检查波长处的实测透射比与最大值（%T）。"""
    wl = np.asarray(wavelengths, dtype=float)
    t = np.asarray(transmittance, dtype=float).ravel()
    vals = {f"{c:g}nm": float(np.interp(c, wl, t)) for c in check_wl}
    mx = max(vals.values()) if vals else float("nan")
    return {"stray_light_Tpct": vals, "max_Tpct": mx}


def signal_to_noise(signal_repeats: np.ndarray,
                    at_wl: Optional[float] = None,
                    wavelengths: Optional[np.ndarray] = None) -> float:
    """信噪比：在指定波长（或全波段取均值）上，信号均值 / 重复测量标准差。
    若原始为 %T，则 SNR = 均值 / 标准差；若已转 A，含义类似。"""
    s = np.asarray(signal_repeats, dtype=float)
    if s.ndim == 1:
        s = s[None, :]
    if at_wl is not None and wavelengths is not None:
        wl = np.asarray(wavelengths, dtype=float)
        col = np.interp(at_wl, wl, s.mean(axis=0))
        noise = np.interp(at_wl, wl, s.std(axis=0, ddof=1))
    else:
        col = s.mean(axis=0)
        noise = s.std(axis=0, ddof=1)
    noise = float(np.mean(noise))
    sig = float(np.mean(col))
    if noise <= 0:
        return float("inf")
    return sig / noise


def resolution(wavelengths: np.ndarray, line_spectrum: np.ndarray,
               peak_wl: float, window_nm: float = 5.0) -> float:
    """分辨率：用窄发射线（汞灯 / 氩灯等）的半高全宽 FWHM（nm）衡量。"""
    wl = np.asarray(wavelengths, dtype=float)
    sp = np.asarray(line_spectrum, dtype=float).ravel()
    mask = (wl >= peak_wl - window_nm) & (wl <= peak_wl + window_nm)
    if not np.any(mask):
        return float("nan")
    seg = sp[mask]
    xl = wl[mask]
    idx_local = int(np.argmax(seg))
    _, yc = _parabolic_center(wl[mask], seg, idx_local)
    half = yc / 2.0
    # 左半高
    left = float(xl[0])
    for k in range(len(seg)):
        if seg[k] >= half:
            if k == 0:
                left = float(xl[0])
            else:
                left = float(xl[k - 1] + (xl[k] - xl[k - 1]) *
                             (half - seg[k - 1]) / (seg[k] - seg[k - 1] + 1e-12))
            break
    # 右半高
    right = float(xl[-1])
    for k in range(len(seg) - 1, -1, -1):
        if seg[k] >= half:
            if k == len(seg) - 1:
                right = float(xl[-1])
            else:
                right = float(xl[k] + (xl[k + 1] - xl[k]) *
                             (half - seg[k]) / (seg[k + 1] - seg[k] + 1e-12))
            break
    return float(right - left)


@dataclass
class InstrumentTestResult:
    """硬件检验结果聚合。"""
    metrics: Dict[str, float] = field(default_factory=dict)
    pass_criteria: Dict[str, float] = field(default_factory=dict)
    passed: Dict[str, bool] = field(default_factory=dict)

    def report(self) -> Dict[str, object]:
        return asdict(self)


def evaluate_instrument(
    *,
    wavelengths: Optional[np.ndarray] = None,
    dark_repeats: Optional[np.ndarray] = None,
    baseline_repeats: Optional[np.ndarray] = None,
    baseline_ideal: Optional[float] = None,
    holmium_spectrum: Optional[np.ndarray] = None,
    ref_peaks: Optional[Sequence[float]] = None,
    neutral_spectrum: Optional[np.ndarray] = None,
    certified: Optional[Sequence[Tuple[float, float]]] = None,
    transmittance: Optional[np.ndarray] = None,
    check_wl: Optional[Sequence[float]] = None,
    signal_repeats: Optional[np.ndarray] = None,
    signal_at_wl: Optional[float] = None,
    line_spectrum: Optional[np.ndarray] = None,
    line_peak_wl: Optional[float] = None,
    limits: Optional[Dict[str, float]] = None,
) -> InstrumentTestResult:
    """汇总各项硬件检验（有数据才算，缺项跳过）。

    Args 均为可选；传入对应测量数据即计算该项指标，并与 `limits` 比较判合格。
    返回 `InstrumentTestResult`（metrics / pass_criteria / passed）。
    """
    limits = dict(DEFAULT_LIMITS if limits is None else limits)
    metrics: Dict[str, float] = {}

    if dark_repeats is not None:
        dn = dark_noise(dark_repeats)
        metrics["dark_noise_std_max"] = dn["std_max"]
        metrics["dark_noise_std_mean"] = dn["std_mean"]

    if baseline_repeats is not None:
        bf = baseline_flatness(baseline_repeats, ideal=baseline_ideal)
        metrics["baseline_repeatability_max"] = bf["repeatability_max"]
        if baseline_ideal is not None:
            metrics["baseline_deviation_max"] = bf["deviation_max"]

    if wavelengths is not None and holmium_spectrum is not None:
        wa = wavelength_accuracy(
            wavelengths, holmium_spectrum,
            ref_peaks if ref_peaks is not None else DEFAULT_HOLMIUM_PEAKS)
        metrics["wavelength_accuracy_max_abs_nm"] = wa["max_abs_error_nm"]

    if wavelengths is not None and neutral_spectrum is not None and certified is not None:
        pa = photometric_accuracy(wavelengths, neutral_spectrum, certified)
        metrics["photometric_accuracy_max_abs"] = pa["max_abs_error"]

    if wavelengths is not None and transmittance is not None and check_wl is not None:
        sl = stray_light(wavelengths, transmittance, check_wl)
        metrics["stray_light_max_Tpct"] = sl["max_Tpct"]

    if signal_repeats is not None:
        metrics["snr"] = signal_to_noise(
            signal_repeats, at_wl=signal_at_wl, wavelengths=wavelengths)

    if wavelengths is not None and line_spectrum is not None and line_peak_wl is not None:
        metrics["resolution_fwhm_nm"] = resolution(wavelengths, line_spectrum, line_peak_wl)

    mapping = {
        "wavelength_accuracy_max_abs_nm": "wavelength_accuracy_nm",
        "photometric_accuracy_max_abs": "photometric_accuracy",
        "stray_light_max_Tpct": "stray_light_Tpct",
        "dark_noise_std_max": "dark_noise",
        "dark_noise_std_mean": "dark_noise",
        "baseline_repeatability_max": "baseline_repeatability",
        "resolution_fwhm_nm": "resolution_nm",
    }
    passed: Dict[str, bool] = {}
    for mk, lim_key in mapping.items():
        if mk in metrics and np.isfinite(metrics[mk]) and lim_key in limits:
            passed[mk] = bool(metrics[mk] <= limits[lim_key])
    return InstrumentTestResult(metrics=metrics, pass_criteria=limits, passed=passed)
