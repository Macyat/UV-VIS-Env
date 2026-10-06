"""仪器迁移：合成两台仪器，验证 DS/PDS 真的能把模型搬过去。

场景设计（贴近现场）：
    slave 相对 master 有
      - 逐波长的增益差异（光源/探测器响应不同）
      - 常数偏置
      - 微小波长漂移

判据：用 master 的模型直接预测 slave 光谱 → 误差大；
      PDS/DS 标准化后再预测 → 误差显著下降。
"""
import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression

from aimeta.transfer.standardization import (
    DirectStandardization, PiecewiseDirectStandardization,
    SlopeBiasCorrection, MeanVarianceAlign, GLSW, kennard_stone,
)
from aimeta.transfer.diagnose import (
    estimate_wavelength_shift, estimate_gain_offset,
    residual_spectrum, recommend_method,
)


@pytest.fixture
def two_instruments():
    """master 干净；slave 有逐波长增益 + 偏置 + 1 像素波长漂移 + 噪声。"""
    rng = np.random.default_rng(7)
    n, p = 80, 251                       # 波长间隔 ≈ 2 nm，1 像素漂移 ≈ 2 nm
    wl = np.linspace(200, 700, p)
    S = np.vstack([
        np.exp(-((wl - 260) ** 2) / (2 * 25 ** 2)),
        np.exp(-((wl - 420) ** 2) / (2 * 35 ** 2)),
    ])
    C = rng.random((n, 2))
    X_master = C @ S + rng.normal(0, 1e-4, (n, p))

    gain = 1.15 + 0.25 * np.sin(np.linspace(0, 3, p))
    offset = 0.02 + 0.01 * np.cos(np.linspace(0, 2, p))
    shift_px = 1
    X_slave = np.roll(X_master * gain + offset, shift_px, axis=1)
    X_slave += rng.normal(0, 3e-4, (n, p))
    y = C[:, 0] * 2.0
    return wl, X_master, X_slave, y, C, S


def _pls_rmse(Xtr, ytr, Xte, yte, k=3):
    m = PLSRegression(n_components=k)
    m.fit(Xtr, ytr)
    return float(np.sqrt(np.mean((yte - np.asarray(m.predict(Xte)).ravel()) ** 2)))


def test_no_transfer_is_bad(two_instruments):
    """不迁移时，主仪器模型直接用从机光谱预测，误差应显著偏大。"""
    wl, Xm, Xs, y, C, S = two_instruments
    n = len(y) // 2
    err = _pls_rmse(Xm[:n], y[:n], Xs[n:], y[n:])
    assert err > 0.05


def test_pds_transfer_reduces_error(two_instruments):
    """PDS 标准化后，slave 光谱应能被 master 模型正确预测。"""
    wl, Xm, Xs, y, C, S = two_instruments
    n = len(y) // 2
    idx = kennard_stone(Xs[:n], 30)
    pds = PiecewiseDirectStandardization(window=5).fit(Xs[idx], Xm[idx])
    Xs_corr = pds.transform(Xs)

    before = _pls_rmse(Xm[:n], y[:n], Xs[n:], y[n:])
    after = _pls_rmse(Xm[:n], y[:n], Xs_corr[n:], y[n:])
    assert after < before * 0.5, f"PDS 未显著降低误差: {before:.4f} -> {after:.4f}"


def test_ds_transfer_also_works(two_instruments):
    wl, Xm, Xs, y, C, S = two_instruments
    n = len(y) // 2
    idx = kennard_stone(Xs[:n], n)       # DS 需要样本数 >= 波长数才稳
    ds = DirectStandardization(ridge=1e-6).fit(Xs[idx], Xm[idx])
    Xs_corr = ds.transform(Xs)
    before = _pls_rmse(Xm[:n], y[:n], Xs[n:], y[n:])
    after = _pls_rmse(Xm[:n], y[:n], Xs_corr[n:], y[n:])
    assert after < before


def test_sbc_cleans_up_residual_slope(two_instruments):
    """SBC 只用预测值对：y' = a·y + b。"""
    wl, Xm, Xs, y, C, S = two_instruments
    rng = np.random.default_rng(1)
    y_pred = y * (1 + rng.normal(0, 0.01, len(y))) + 0.35   # 有偏置的预测
    sbc = SlopeBiasCorrection().fit(y_pred, y)
    fixed = sbc.transform(y_pred)
    assert np.sqrt(np.mean((fixed - y) ** 2)) < np.sqrt(np.mean((y_pred - y) ** 2))
    assert abs(sbc.slope_ - 1.0) < 0.1


def test_meanvar_and_glsw_run(two_instruments):
    wl, Xm, Xs, y, C, S = two_instruments
    n = len(y) // 2
    mv = MeanVarianceAlign().fit(Xs[:n], Xm[:n])
    assert mv.transform(Xs).shape == Xs.shape
    g = GLSW(lam=1e-3).fit(Xs[:n], Xm[:n])
    assert g.transform(Xs).shape == Xs.shape


def test_diagnose_detects_shift(two_instruments):
    wl, Xm, Xs, y, C, S = two_instruments
    shift = estimate_wavelength_shift(wl, Xs, Xm)
    assert abs(shift) > 0.0            # 确实存在漂移
    go = estimate_gain_offset(Xs, Xm)
    assert go["gain"].shape == (Xs.shape[1],)
    assert np.std(go["gain"]) > 0.01   # 增益是逐波长变化的 ⇒ 该用 PDS 而非 DS


def test_recommend_method_downgrades_with_few_samples(two_instruments):
    wl, Xm, Xs, y, C, S = two_instruments
    rich = recommend_method(wl, Xs, Xm, n_std_samples=50)
    poor = recommend_method(wl, Xs, Xm, n_std_samples=3)
    assert rich["method"] in {"ds", "pds", "align_then_pds"}
    assert poor["method"] == "sbc"     # 样本太少只能做响应校正


def test_kennard_stone_spreads_samples():
    rng = np.random.default_rng(3)
    X = rng.random((50, 5))
    idx = kennard_stone(X, 10)
    assert len(idx) == 10 and len(set(idx.tolist())) == 10
