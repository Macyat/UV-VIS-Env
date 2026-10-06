"""计量验收：品质因数与 GB3838 判定。"""
import numpy as np
import pytest

from aimeta.metrics.figures_of_merit import figures_of_merit, regression_vector, sensitivity
from aimeta.metrics.water_standards import (
    WaterParam, classify, misclassification_matrix, acceptance_rate,
    daily_r2, alarm_accuracy, acceptance_report,
)
from aimeta.io.config import load_params


@pytest.fixture
def linear_setup():
    """构造一个已知回归向量的线性模型。"""
    rng = np.random.default_rng(0)
    p = 40
    coef = np.zeros(p)
    coef[10:20] = 0.5                      # 只有一段波长有信号
    X = rng.random((60, p))

    class Lin:
        def __init__(self):
            self.coef_ = coef
            self.intercept_ = 0.1

        def predict(self, X):
            return np.asarray(X) @ self.coef_ + self.intercept_

    return Lin(), X, coef


def test_regression_vector_and_sensitivity(linear_setup):
    model, X, coef = linear_setup
    b = regression_vector(model, X.mean(axis=0))
    assert np.allclose(b, coef)
    assert abs(sensitivity(b) - 1 / np.linalg.norm(coef)) < 1e-12


def test_fom_lod_is_finite_and_positive(linear_setup):
    model, X, coef = linear_setup
    rng = np.random.default_rng(2)
    blank = rng.normal(0, 1e-3, (8, X.shape[1]))     # 空白必须有噪声，否则 s_0 = 0
    fom = figures_of_merit(model, blank, X_cal=X)
    assert fom["LOD"] > 0 and np.isfinite(fom["LOD"])
    assert fom["LOQ"] > fom["LOD"]
    assert fom["SEN"] > 0 and fom["gamma"] > 0


def test_weaker_signal_gives_worse_lod():
    """吸光能力越弱 ⇒ 灵敏度越低 ⇒ LOD 越差。

    这正是 LOD 该被算出来而不是写死的原因：它随模型与噪声变化。

    注意符号：模型是 c = X·b，所以 b ≈ 1/ε（ε 为吸光系数）。
    ε 大（强吸收）⇒ ||b|| 小 ⇒ SEN = 1/||b|| 大 ⇒ LOD 小。
    """
    rng = np.random.default_rng(1)
    p = 40
    blank = rng.normal(0, 1e-3, (8, p))

    def make(b_value):
        coef = np.zeros(p); coef[10:20] = b_value

        class L:
            def __init__(self): self.coef_ = coef; self.intercept_ = 0.0
            def predict(self, X): return np.asarray(X) @ self.coef_
        return L()

    high_eps = figures_of_merit(make(0.1), blank)   # b 小 ⇒ 强吸收 ⇒ 高灵敏
    low_eps = figures_of_merit(make(1.0), blank)    # b 大 ⇒ 弱吸收 ⇒ 低灵敏
    assert high_eps["SEN"] > low_eps["SEN"]
    assert high_eps["LOD"] < low_eps["LOD"]


def test_classify_and_confusion():
    p = WaterParam(name="TN", ranges=[0.2, 0.5, 1.0, 1.5, 2.0],
                   lower_bound=0.0, upper_bound=15.0)
    assert list(classify([0.1, 0.3, 1.2, 3.0], p)) == [0, 1, 3, 5]
    M = misclassification_matrix([0.1, 0.3, 1.2], [0.1, 0.6, 1.2], p)
    assert M.shape == (p.n_classes, p.n_classes)
    assert M.sum() == 3


def test_acceptance_rate_penalises_cross_class_errors():
    p = WaterParam(name="AN", ranges=[0.15, 0.5, 1.0, 1.5, 2.0],
                   lower_bound=0.0, upper_bound=3.0, abs_error_bound=0.2)
    good = acceptance_rate(np.array([0.1, 1.0, 2.5]),
                           np.array([0.11, 1.05, 2.4]), p)
    bad = acceptance_rate(np.array([0.1, 1.0, 2.5]),
                          np.array([1.9, 1.9, 0.05]), p)
    assert good > bad


def test_daily_r2_and_alarm():
    day = np.array([1, 1, 1, 2, 2, 2])
    yt = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    r2 = daily_r2(day, yt, yt + 0.01)
    assert set(r2) == {"1", "2"} and all(v > 0.99 for v in r2.values())

    a = alarm_accuracy(np.array([0.1, 0.2, 2.0, 3.0]),
                       np.array([0.1, 2.0, 2.1, 0.3]), threshold=1.0)
    assert a["fp"] == 1 and a["fn"] == 1 and 0 < a["accuracy"] < 1


def test_params_yaml_loads_all_parameters():
    params = load_params()
    for k in ["KMNO", "COD", "TN", "TP", "AN", "TUR"]:
        assert k in params and len(params[k].ranges) >= 3


def test_acceptance_report_runs_for_turbidity():
    """浊度只有 3 个分界，老代码在这里会索引越界 —— 这里必须不崩。"""
    params = load_params()
    tur = params["TUR"]
    yt = np.array([5.0, 40.0, 120.0])
    yp = np.array([6.0, 44.0, 130.0])
    rep = acceptance_report(yt, yp, tur)
    assert 0.0 <= rep["acceptance_rate"] <= 1.0
