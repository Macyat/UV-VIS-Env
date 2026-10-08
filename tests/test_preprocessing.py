"""预处理：数值等价性 + 链可复现 + SNV 语义。

重点是第一条：DERIV 是从 ai-meta-main 移植的 MATLAB Gram 多项式实现，
必须与 scipy.signal.savgol_filter 逐元素一致，否则历史模型全部失效。
"""
import numpy as np
import pytest
from scipy.signal import savgol_filter

from aimeta.preprocessing.operators import (
    DERIV, snv_transform, msc_transform, whittaker_smooth, asls_baseline,
)
from aimeta.preprocessing.base import Pipeline


@pytest.fixture
def spectra():
    rng = np.random.default_rng(0)
    return rng.random((8, 101)) * 0.8 + 0.1


@pytest.mark.parametrize("window", [11, 17])
@pytest.mark.parametrize("der", [0, 1, 2])
def test_deriv_matches_scipy(spectra, window, der):
    """回归测试：DERIV 与 savgol_filter 数值等价。"""
    a = DERIV(spectra, der=der, window=window, order=2)
    b = savgol_filter(spectra, window, 2, deriv=der)
    assert np.allclose(a, b, atol=1e-8)


def test_deriv_rejects_even_window(spectra):
    """历史 bug：window 为偶数（含银行家舍入导致的偏移）必须显式报错。"""
    with pytest.raises(ValueError):
        DERIV(spectra, der=0, window=10)


def test_snv_is_per_spectrum(spectra):
    """SNV 必须是逐条光谱，不能是全矩阵全局标准化。

    老 river_inference.py 的 pca 变体做成了全局标准化，导致同一条光谱
    在不同批次里得到不同结果 —— 这个测试锁死正确语义。
    """
    out = snv_transform(spectra)
    for i in range(len(spectra)):
        assert abs(out[i].mean()) < 1e-10
        assert abs(out[i].std(ddof=0) - 1.0) < 1e-10
    # 逐条处理 ⇒ 删掉其他样本不影响本条结果
    assert np.allclose(out[0], snv_transform(spectra[:1])[0])


def test_msc_reduces_scatter():
    """MSC 应消除乘性/加性散射。

    注意：必须用「有共同形状」的光谱。若各条光谱互不相关（纯随机），
    MSC 的回归系数 a 会趋近 0，除以 a 反而放大差异 —— 这不是 MSC 的缺陷，
    而是「没有共同信号可对齐」的必然结果。
    """
    p = 101
    base = np.exp(-((np.arange(p) - 50) ** 2) / (2 * 20 ** 2)) + 0.5
    clean = np.tile(base, (8, 1))
    scaled = clean * np.linspace(0.5, 1.5, 8)[:, None] + np.linspace(0.0, 0.3, 8)[:, None]

    out = msc_transform(scaled)
    before = np.std(scaled - scaled.mean(axis=0))
    after = np.std(out - out.mean(axis=0))
    assert after < before * 0.1


def test_pipeline_roundtrip_is_reproducible(spectra):
    """to_config → from_config 必须回到同一条链（指纹稳定的前提）。

    to_config 会把算子的默认参数补全（如 savgol 的 deriv=0），
    所以 YAML 是「简写」，card 里存的是「规范形式」—— 规范形式必须幂等。
    """
    cfg = [{"op": "savgol", "window": 11, "polyorder": 3, "deriv": 0}, {"op": "snv"}]
    p1 = Pipeline.from_config(cfg)
    assert p1.to_config() == cfg
    p2 = Pipeline.from_config(p1.to_config())
    assert p2.to_config() == cfg
    assert np.allclose(p1.fit_transform(spectra), p2.fit_transform(spectra))


def test_pipeline_canonicalises_partial_config(spectra):
    """YAML 里省略默认参数也能工作，并补全为规范形式。"""
    p = Pipeline.from_config([{"op": "savgol", "window": 11, "polyorder": 3}])
    assert p.to_config()[0]["deriv"] == 0
    assert p.transform(spectra).shape == spectra.shape


def test_pipeline_fit_state_is_reused(spectra):
    """MSC / mean_center 这类有状态的算子，必须记住 fit 时的参考光谱。"""
    p = Pipeline.from_config([{"op": "msc"}])
    p.fit(spectra)
    ref_first = p.steps[0].reference_
    assert ref_first is not None
    out = p.transform(spectra[:3])
    assert out.shape == (3, spectra.shape[1])
    assert np.allclose(p.steps[0].reference_, ref_first)


def test_wavelet_preserves_length(spectra):
    """老 utils.wavelet_denoising 返回 [1:] 会少一个点，这里必须长度不变。"""
    p = Pipeline.from_config([{"op": "wavelet", "wavelet": "sym4", "level": 2}])
    assert p.transform(spectra).shape == spectra.shape


# ---------------------------------------------------------------------------
# 基线校正（Whittaker 平滑 + ALS 非对称最小二乘，PLS_Toolbox 同源）
# ---------------------------------------------------------------------------


def test_whittaker_reduces_noise():
    """Whittaker 平滑应让含噪信号更接近干净信号。"""
    rng = np.random.default_rng(0)
    x = np.arange(201)
    clean = np.exp(-((x - 100) ** 2) / (2 * 20 ** 2))
    noisy = clean + rng.normal(0, 0.05, 201)
    smooth = whittaker_smooth(noisy[None, :], lam=1e4, d=2)[0]
    assert np.mean((smooth - clean) ** 2) < np.mean((noisy - clean) ** 2)
    assert smooth.shape == (201,)


def test_asls_baseline_removes_linear_baseline():
    """ALS 应扣掉线性基线，同时保留信号峰。

    二阶差分（d=2）对线性基线零惩罚，故基线能被精确拟合；峰因残差为正
    而被压低权重、不被基线吸收。
    """
    x = np.arange(201)
    baseline = 0.2 + 0.005 * x
    peak = 2.0 * np.exp(-((x - 100) ** 2) / (2 * 8 ** 2))
    signal = baseline + peak
    corrected = asls_baseline(signal[None, :], lam=1e3, p=0.01, d=2, n_iter=30)[0]
    # 远离峰的纯基线区被扣到接近 0
    assert abs(corrected[20]) < 0.05
    # 峰高大部分保留（原高 2.0）
    assert corrected[100] > 1.5


def test_baseline_operators_registered(spectra):
    """三个基线算子都注册进 PREPROC，且输出形状不变。"""
    for op in ["whittaker", "baseline", "wlsbaseline"]:
        p = Pipeline.from_config([{"op": op}])
        assert p.transform(spectra).shape == spectra.shape


def test_asls_baseline_rejects_bad_params(spectra):
    with pytest.raises(ValueError):
        asls_baseline(spectra, lam=0)
    with pytest.raises(ValueError):
        asls_baseline(spectra, p=1.5)
    with pytest.raises(ValueError):
        asls_baseline(spectra, d=0)
