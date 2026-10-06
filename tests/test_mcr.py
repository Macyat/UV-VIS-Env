"""MCR-ALS：能否把混合光谱拆回组分，以及解有多不唯一。"""
import numpy as np

from aimeta.models.curve_resolution import MCRALS, simplisma, orthogonal_projection


def _mixture(seed=0, n=60, p=80):
    rng = np.random.default_rng(seed)
    wl = np.linspace(220, 700, p)
    S = np.vstack([
        np.exp(-((wl - 280) ** 2) / (2 * 30 ** 2)),
        np.exp(-((wl - 400) ** 2) / (2 * 40 ** 2)),
        np.exp(-((wl - 580) ** 2) / (2 * 50 ** 2)),
    ])
    C = rng.random((n, 3))
    return C @ S, C, S, wl


def _cosine_match(A, B):
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return np.abs(A @ B.T).max(axis=1)


def test_simplisma_finds_pure_regions():
    X, C, S, wl = _mixture()
    idx_peaks = [int(np.argmax(s)) for s in S]
    S0 = simplisma(X, 3)
    found = [int(np.argmax(s)) for s in S0]
    for f in found:
        assert min(abs(f - p) for p in idx_peaks) < 15


def test_opa_runs():
    X, C, S, wl = _mixture()
    S0 = orthogonal_projection(X, 2)
    assert S0.shape == (2, X.shape[1])


def test_mcr_recovers_components():
    X, C, S, wl = _mixture()
    mcr = MCRALS(n_components=3, non_negative=True, max_iter=200).fit(X)
    match = _cosine_match(S, mcr.components_)
    assert float(np.min(match)) > 0.95          # 三个组分都找回来了
    rel = np.linalg.norm(X - mcr.reconstruct()) / np.linalg.norm(X)
    assert rel < 0.05


def test_mcr_respects_closure():
    X, C, S, wl = _mixture()
    mcr = MCRALS(n_components=3, non_negative=True, closure=True,
                 max_iter=100).fit(X)
    assert np.allclose(mcr.concentrations_.sum(axis=1), 1.0, atol=1e-8)


def test_ambiguity_is_reported():
    """模糊性估计应给出一个 0~1 的标量，用于提示「这个解有多可信」."""
    X, C, S, wl = _mixture()
    mcr = MCRALS(n_components=3, non_negative=True, max_iter=100).fit(X)
    amb = mcr.ambiguity_estimate(X, n_trials=5, seed=0)
    assert 0.0 <= amb <= 1.0
