"""MSPC（T² / SPE）与异常样本标记测试。"""
import numpy as np

from aimeta.monitoring import MSPC, flag_outliers


def _synthetic(n=40, p=20, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(p, 3))
    S = rng.normal(size=(n, 3))
    X = S @ W.T + rng.normal(0, 0.01, size=(n, p))
    return X, W


def _orthogonal(W, rng):
    """生成一个与 W 的列空间正交的单位向量（真正模型解释不了的方向）。"""
    v = rng.normal(size=W.shape[0])
    v = v - W @ (np.linalg.pinv(W) @ v)
    return v / np.linalg.norm(v)


def test_mspc_reports_t2_and_spe():
    X, _ = _synthetic()
    m = MSPC(n_components=3).fit(X)
    r = m.monitor(X)
    for key in ("T2", "SPE", "t2_alarm", "spe_alarm",
                "t2_limit", "spe_limit", "contribution"):
        assert key in r


def test_flag_outliers_ranks_orthogonal_outlier_first():
    X, W = _synthetic()
    rng = np.random.default_rng(1)
    v = _orthogonal(W, rng)
    X[0] += 5.0 * v   # 与干净子空间正交的扰动 → 真正的 SPE 异常
    res = flag_outliers(X, n_components=3, alpha=0.05, by="spe")
    assert res["idx"][0] == 0                 # 异常样本排在候选最前
    assert set(res["idx"]) == set(range(len(X)))  # 只排序、不删除
