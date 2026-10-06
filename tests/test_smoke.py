"""端到端冒烟：合成数据 → 训练 → 导出 edge → train/serve 一致性 → 监控 → 变量选择。

这一条测试就是 P0 的验收标准：
    训练端与部署端跑同一条链，预测必须逐元素一致。
"""
import numpy as np
import pytest

from aimeta.core.spectra import SpectrumSet
from aimeta.io.config import load_params, load_chain
from aimeta.pipelines.train import train_model, sweep
from aimeta.pipelines.infer import predict, predict_with_guard
from aimeta.monitoring import MSPC, shewhart_limits, cusum, ewma
from aimeta.monitoring.control_charts import effective_sample_size, arl0
from aimeta.selection import CARS, permutation_test
from aimeta.models.zoo import glm_family
from edge.runtime import (
    export_card, load_edge_model, verify_against_card, compile_chain, CompileError,
)
compile_error = CompileError


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n, p = 120, 101
    wl = np.linspace(200, 700, p)
    S = np.vstack([
        np.exp(-((wl - 260) ** 2) / (2 * 20 ** 2)),
        np.exp(-((wl - 340) ** 2) / (2 * 30 ** 2)),
        np.exp(-((wl - 520) ** 2) / (2 * 40 ** 2)),
    ])
    C = rng.random((n, 3)) * np.array([1.0, 0.5, 0.3])
    X = C @ S + rng.normal(0, 1e-3, (n, p))
    y = C[:, 0] * 2.0 + 0.1
    return wl, X, y


def test_train_produces_card(data):
    wl, X, y = data
    params = load_params()
    card = train_model(SpectrumSet(X=X, wavelengths=wl, y=y), "AN", "pls",
                       chain=[{"op": "savgol", "window": 11, "polyorder": 3}],
                       model_params={"n_components": 5},
                       param_def=params["AN"], instrument_id="ai14")
    assert card.fingerprint()
    # 合成数据是严格线性的，RMSECV 应远小于标签标准差
    assert card.fom["rmse_cv"] < 0.05 * float(np.std(y))
    assert np.isfinite(predict(card, X)).all()


def test_sweep_is_sorted_by_rmse(data):
    wl, X, y = data
    cards = sweep(SpectrumSet(X=X, wavelengths=wl, y=y), "AN",
                  model_keys=("pls", "ridge", "ols"),
                  chain=[{"op": "snv"}], folds=3)
    r = [c.fom["rmse_cv"] for c in cards]
    assert r == sorted(r)
    assert len(cards) == 3


def test_train_serve_parity(data, tmp_path):
    """核心护栏：edge 端（numpy-only）与训练端的预测必须逐元素一致。"""
    wl, X, y = data
    card = train_model(SpectrumSet(X=X, wavelengths=wl, y=y), "AN", "pls",
                       chain=[{"op": "savgol", "window": 11, "polyorder": 3},
                              {"op": "snv"}],
                       model_params={"n_components": 5}, instrument_id="ai14")
    d = export_card(card, tmp_path / "edge_model", X_reference=X)
    em = load_edge_model(d)
    ok, diff = verify_against_card(em, card, X)
    assert ok, f"train/serve mismatch: {diff:.3e}"


def test_non_compilable_chain_is_rejected():
    """wiener / wavelet 无法编译成 numpy 算子，必须显式报错而不是悄悄算错。"""
    with pytest.raises(compile_error):
        compile_chain([{"op": "wiener"}], n_wavelengths=50)
    with pytest.raises(compile_error):
        compile_chain([{"op": "wavelet"}], n_wavelengths=50)


def test_edge_chain_from_config_compiles():
    chain = load_chain(name="preprocessing/dayu_edge.yaml")
    ops = compile_chain(chain, n_wavelengths=101)
    assert ops and all("kind" in o for o in ops)


def test_guard_bounds_and_flags(data):
    wl, X, y = data
    params = load_params()
    card = train_model(SpectrumSet(X=X, wavelengths=wl, y=y), "AN", "pls",
                       chain=[{"op": "snv"}], model_params={"n_components": 4},
                       param_def=params["AN"])
    out = predict_with_guard(card, X, param_def=params["AN"])
    lo, hi = params["AN"].lower_bound, params["AN"].upper_bound
    # 低于检出限替为 LOD/2，不会低于 LOD/2；高于上限夹到上限
    assert out["prediction"].min() >= lo / 2.0 - 1e-12
    assert out["prediction"].max() <= hi + 1e-12
    assert "below_lod" in out and "above_upper" in out
    assert out["below_lod"].dtype == bool and out["above_upper"].dtype == bool


def test_apply_bounds_substitutes_lod_half():
    from aimeta.metrics.water_standards import apply_bounds
    pred = np.array([-1.0, 0.0, 0.03, 2.0, 100.0])
    lo, hi = 0.025, 3.0
    out, below, above = apply_bounds(pred, lo, hi)
    # 低于检出限 -> LOD/2
    assert out[0] == lo / 2.0 and out[1] == lo / 2.0
    assert below[0] and below[1] and not below[2]
    # 高于上限 -> 夹到上限
    assert out[-1] == hi and above[-1]
    # 中间值不变
    assert out[2] == 0.03 and out[3] == 2.0
    # 无边界时不处理
    out2, b2, a2 = apply_bounds(pred, None, None)
    assert np.array_equal(out2, pred) and not b2.any() and not a2.any()


def test_mspc_flags_contaminated_probe(data):
    wl, X, y = data
    m = MSPC(n_components=3).fit(X)
    clean = m.monitor(X)
    # 控制限按 alpha=0.05 设定，正常样本报警率本来就应在 5% 量级
    assert clean["spe_alarm"].mean() < 0.15

    bad = X.copy()
    bad[0] += 0.05                       # 模拟探头污染
    r = m.monitor(bad)
    assert r["spe_alarm"][0]
    assert r["SPE"][0] > r["SPE"][1:].max()   # 异常样本的残差明显更大
    assert r["contribution"].shape == X.shape


def test_control_charts(data):
    wl, X, y = data
    cl, lcl, ucl = shewhart_limits(y, k=3.0)
    assert lcl < cl < ucl
    c = cusum(y, k=0.5, h=5.0)
    assert c["cusum_pos"].shape == y.shape
    e = ewma(y, lam=0.2)
    assert np.all(e["ucl"] > e["lcl"])
    assert effective_sample_size(y) > 0
    assert arl0(3.0) > 100               # 3σ 下 ARL0 ≈ 370


def test_cars_selects_wavelengths(data):
    wl, X, y = data
    sel = CARS(n_iter=8, n_components=4, cv_folds=3, random_state=0).fit(X, y)
    assert 0 < len(sel.selected_) <= X.shape[1]
    assert sel.transform(X).shape[1] == len(sel.selected_)


def test_permutation_test_detects_real_signal(data):
    """置换检验：真实数据的 RMSECV 应显著优于打乱标签后的分布。"""
    wl, X, y = data

    def factory():
        return CARS(n_iter=6, n_components=4, cv_folds=3, random_state=0)

    def score(Xs, ys, idx):
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=3, shuffle=True, random_state=0)
        errs = []
        for tr, te in kf.split(Xs):
            m = PLSRegression(n_components=2).fit(Xs[tr], ys[tr])
            errs.append(np.sqrt(np.mean((ys[te] - np.asarray(m.predict(Xs[te])).ravel()) ** 2)))
        return float(np.mean(errs))

    res = permutation_test(factory, score, X, y, n_permutations=5)
    assert res["p_value"] < 0.5


def test_glm_family_recommendation():
    rng = np.random.default_rng(0)
    # Gamma(shape, scale)：var/mean = scale，scale > 1 ⇒ 过度离散 ⇒ 选 gamma
    over = rng.gamma(shape=2.0, scale=3.0, size=500)
    assert glm_family(over) == "gamma"
