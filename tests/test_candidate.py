"""候选池 → 打分表 → TOP K → 精度加权融合 → 每模型图册 的端到端测试。

对应你定的口径：
    - 候选池：跨预处理链 × 多模型全量枚举（sweep_grid）
    - 融合权重：1/RMSE² 误差方差反比（退化等权）
    - 打分表指标：验收指标全集
"""
import matplotlib
matplotlib.use("Agg")  # 必须在 import pyplot/viz 之前

import numpy as np
import pytest

from aimeta.core.spectra import SpectrumSet
from aimeta.metrics.water_standards import WaterParam
from aimeta.pipelines.train import train_model, sweep_grid
from aimeta.pipelines.scoring import (
    SCORING_COLUMNS, build_score_row, score_cards, select_top_k, _raw_predict,
)
from aimeta.models.ensembles import fuse_predict
from aimeta.viz.report import export_model_figures, export_top_k_report


@pytest.fixture(scope="module")
def synth():
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
    day_idx = (np.arange(n) // 12)  # ~10 天
    param = WaterParam(
        name="TEST", ranges=[0.1, 0.3, 0.5, 1.0, 2.0], lower_bound=0.02,
        upper_bound=3.0, mape_bound=0.15,
    )
    return wl, X, y, day_idx, param


def _cards(synth, chains, models):
    wl, X, y, _, param = synth
    return sweep_grid(
        SpectrumSet(X=X, wavelengths=wl, y=y), "TEST",
        model_keys=models, chains=chains,
        model_params={"pls": {"n_components": 3}},
        param_def=param, folds=3, instrument_id="ai14",
    )


def test_sweep_grid_enumerates_chains_times_models(synth):
    cards = _cards(synth, chains=[[{"op": "snv"}], [{"op": "savgol", "window": 11, "polyorder": 3}]],
                   models=("pls", "ridge", "ols"))
    # 2 链 × 3 模型 = 6 候选
    assert len(cards) == 6
    fps = {c.fingerprint() for c in cards}
    assert len(fps) == 6  # 每张卡指纹唯一


def test_cv_day_based_no_future_leakage():
    # 按日 CV：expanding / rolling 只能用过去的天，绝不拿未来预测过去
    from aimeta.pipelines.train import _cv_splits
    day_idx = np.array([0, 0, 1, 1, 2, 2, 3, 3])  # 4 天，每天 2 样本

    for tr, te in _cv_splits(8, day_idx, "expanding", folds=5):
        assert (day_idx[tr] < day_idx[te][0]).all()

    for tr, te in _cv_splits(8, day_idx, "rolling", folds=5, window=1):
        assert (day_idx[tr] < day_idx[te][0]).all()
        assert set(day_idx[tr]) == {day_idx[te][0] - 1}

    for tr, te in _cv_splits(8, day_idx, "logo", folds=5):
        assert (day_idx[tr] != day_idx[te][0]).all()


def test_build_score_row_has_full_set(synth):
    wl, X, y, day_idx, param = synth
    card = train_model(SpectrumSet(X=X, wavelengths=wl, y=y), "TEST", "pls",
                       chain=[{"op": "snv"}], model_params={"n_components": 3},
                       param_def=param, folds=3, instrument_id="ai14")
    yp = _raw_predict(card, X)
    row = build_score_row(y, yp, param, day_idx)
    expected = set(SCORING_COLUMNS) - {"model_type", "rank"}
    assert expected.issubset(row.keys())
    # 关键验收列都应有限
    for k in ("rmse", "r2_score", "mape", "rate of reaching the standard",
              "alarm_acc", "daily_r2_score"):
        assert np.isfinite(row[k]), k
    # alarm_acc + alarm_err ≈ 1
    assert abs(row["alarm_acc"] + row["alarm_err"] - 1.0) < 1e-9


def test_score_cards_ranks_and_unique(synth):
    cards = _cards(synth, chains=[[{"op": "snv"}]], models=("pls", "ridge", "ols"))
    ranked = score_cards(cards, _X(synth), _y(synth), _param(synth), _day(synth))
    assert len(ranked) == len(cards)
    assert len({r["model_type"] for r in ranked}) == len(cards)
    ranks = [r["rank"] for r in ranked]
    assert ranks == sorted(ranks)


def test_select_top_k_returns_subset(synth):
    cards = _cards(synth, chains=[[{"op": "snv"}], [{"op": "savgol", "window": 11, "polyorder": 3}]],
                   models=("pls", "ridge", "lasso", "ols"))
    top = select_top_k(cards, 3, _X(synth), _y(synth), _param(synth), _day(synth))
    assert len(top) == 3
    top_fps = {c.fingerprint() for c in top}
    all_fps = {c.fingerprint() for c in cards}
    assert top_fps.issubset(all_fps)
    # TOP1 应当就是全局打分最优的那张卡（按 fingerprint 比对）
    ranked_all = score_cards(cards, _X(synth), _y(synth), _param(synth), _day(synth))
    best_key = min(ranked_all, key=lambda r: r["rank"])["model_type"]
    best_fp = best_key.split("::", 1)[1]
    assert top[0].fingerprint() == best_fp


def test_fuse_predict_inv_rmse2_weights(synth):
    cards = _cards(synth, chains=[[{"op": "snv"}]], models=("pls", "ridge"))
    # 手动给定可预期的 rmse_cv，验证权重 = 1/RMSE² 归一化
    cards[0].fom["rmse_cv"] = 0.2
    cards[1].fom["rmse_cv"] = 0.4
    fused, info = fuse_predict(cards, _X(synth), scheme="inv_rmse2")
    assert fused.shape == (_X(synth).shape[0],)
    assert info["scheme"] == "inv_rmse2"
    w = np.array(info["weights"])
    assert np.allclose(w.sum(), 1.0)
    # 1/0.2² : 1/0.4² = 25 : 6.25 = 4 : 1
    assert abs(w[0] / w[1] - 4.0) < 1e-9


def test_fuse_predict_equal_fallback(synth):
    cards = _cards(synth, chains=[[{"op": "snv"}]], models=("pls", "ridge", "ols"))
    fused, info = fuse_predict(cards, _X(synth), scheme="equal")
    assert info["scheme"] == "equal"
    assert np.allclose(info["weights"], 1 / 3)
    assert fused.shape == (_X(synth).shape[0],)


def test_fuse_predict_falls_back_when_rmse_invalid(synth):
    cards = _cards(synth, chains=[[{"op": "snv"}]], models=("pls", "ridge"))
    cards[0].fom["rmse_cv"] = 0.0  # 非法 → 必须退化为等权
    _, info = fuse_predict(cards, _X(synth), scheme="inv_rmse2")
    assert info["scheme"] == "equal"


def test_export_model_figures_writes_pngs(synth, tmp_path):
    wl, X, y, _, param = synth
    card = train_model(SpectrumSet(X=X, wavelengths=wl, y=y), "TEST", "pls",
                       chain=[{"op": "snv"}], model_params={"n_components": 3},
                       param_def=param, folds=3, instrument_id="ai14")
    paths = export_model_figures(card, X, y, param, tmp_path / "card")
    assert paths
    for p in paths:
        assert p.exists() and p.suffix == ".png"


def test_export_top_k_report(synth, tmp_path):
    cards = _cards(synth, chains=[[{"op": "snv"}], [{"op": "savgol", "window": 11, "polyorder": 3}]],
                   models=("pls", "ridge", "ols"))
    summary = export_top_k_report(
        cards, _X(synth), _y(synth), _param(synth), tmp_path, k=3, day_idx=_day(synth))
    assert len(summary["top_k"]) == 3
    assert (tmp_path / "scoring.csv").exists()
    for tag, plist in summary["figures"].items():
        assert plist
        for p in plist:
            assert __import__("pathlib").Path(p).exists()


# ---- 小工具：从 fixture 取各分量（避免重复解包）----
def _X(synth): return synth[1]
def _y(synth): return synth[2]
def _day(synth): return synth[3]
def _param(synth): return synth[4]
