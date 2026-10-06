"""光谱仪硬件评价模块测试。"""
import numpy as np

from aimeta.hardware_eval import (
    dark_noise, baseline_flatness, wavelength_accuracy, photometric_accuracy,
    stray_light, signal_to_noise, resolution, evaluate_instrument,
)


def _gaussian(wl, center, amp=1.0, fwhm=2.0):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return amp * np.exp(-0.5 * ((wl - center) / sigma) ** 2)


def test_dark_noise():
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 0.001, size=(20, 50))
    res = dark_noise(base)
    assert abs(res["std_mean"] - 0.001) < 3e-4
    assert np.isfinite(res["std_max"])
    # 单条不足以估计 -> nan
    assert np.isnan(dark_noise(base[0])["std_max"])


def test_baseline_flatness():
    rng = np.random.default_rng(1)
    b = 0.5 + rng.normal(0.0, 0.0008, size=(10, 30))
    res = baseline_flatness(b, ideal=0.5)
    assert res["repeatability_max"] < 5e-3
    assert res["deviation_max"] < 5e-3


def test_wavelength_accuracy_recovers_shift():
    wl = np.linspace(200, 700, 500)
    true_center = 360.9
    meas = _gaussian(wl, center=true_center + 0.3, fwhm=3.0)
    res = wavelength_accuracy(wl, meas, ref_peaks=[360.9])
    assert abs(res["max_abs_error_nm"] - 0.3) < 0.05


def test_photometric_accuracy_interp():
    wl = np.linspace(200, 800, 600)
    sp = np.full_like(wl, 0.500)
    res = photometric_accuracy(wl, sp, certified=[(400.0, 0.500), (600.0, 0.500)])
    assert abs(res["max_abs_error"]) < 1e-6


def test_stray_light_interp():
    wl = np.linspace(200, 800, 600)
    t = np.zeros_like(wl)
    t[wl >= 400] = 100.0
    res = stray_light(wl, t, check_wl=[340.0, 350.0])
    assert abs(res["stray_light_Tpct"]["340nm"] - 0.0) < 1e-6
    assert abs(res["max_Tpct"] - 0.0) < 1e-6


def test_signal_to_noise():
    rng = np.random.default_rng(2)
    s = 1.0 + rng.normal(0.0, 0.01, size=(50, 1))
    snr = signal_to_noise(s)
    assert abs(snr - 100.0) < 10.0


def test_resolution_fwhm():
    wl = np.linspace(200, 800, 1200)
    line = _gaussian(wl, center=546.1, fwhm=1.5)
    fwhm = resolution(wl, line, peak_wl=546.1, window_nm=10.0)
    assert abs(fwhm - 1.5) < 0.15


def test_evaluate_instrument_aggregates_and_passes():
    rng = np.random.default_rng(3)
    wl = np.linspace(200, 800, 600)
    dark = rng.normal(0.0, 0.0002, size=(20, 600))
    holmium = _gaussian(wl, center=360.9 + 0.1, fwhm=3.0)
    transmittance = np.zeros_like(wl)
    transmittance[wl >= 400] = 100.0
    line = _gaussian(wl, center=546.1, fwhm=1.2)
    res = evaluate_instrument(
        wavelengths=wl,
        dark_repeats=dark,
        holmium_spectrum=holmium,
        ref_peaks=[360.9],
        transmittance=transmittance,
        check_wl=[340.0],
        line_spectrum=line,
        line_peak_wl=546.1,
    )
    assert res.metrics["wavelength_accuracy_max_abs_nm"] < 0.5
    assert res.passed["wavelength_accuracy_max_abs_nm"] is True
    assert res.passed["resolution_fwhm_nm"] is True
    assert res.passed["stray_light_max_Tpct"] is True
