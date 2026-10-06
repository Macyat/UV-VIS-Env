"""候选模型集合。

按 family 组织：
    linear   OLS / Ridge / Lasso / ElasticNet / OMP / Lars / Huber / TheilSen
    glm      Gamma / Poisson / Tweedie
    latent   PCR / PLS
    kernel   GPR / SVR / KRR
    trees    RandomForest / LGBM
    nn       MLP

可选依赖（lightgbm / torch）缺失时静默跳过，保证核心链路可运行。

另见 ``glm_family``：按过度离散程度自动推荐 GLM 的 family。
"""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (
    ElasticNet, HuberRegressor, Lasso, Lars, LinearRegression,
    OrthogonalMatchingPursuit, PoissonRegressor, QuantileRegressor,
    Ridge, SGDRegressor, TheilSenRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RationalQuadratic, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline as sk_make_pipeline
from sklearn.svm import SVR

from ..core.registry import MODELS

# ---------------------------------------------------------------- linear
MODELS.register("ols", family="linear")(LinearRegression)
MODELS.register("ridge", family="linear")(Ridge)
MODELS.register("lasso", family="linear")(Lasso)
MODELS.register("elasticnet", family="linear")(ElasticNet)
MODELS.register("omp", family="linear")(OrthogonalMatchingPursuit)
MODELS.register("lars", family="linear")(Lars)
MODELS.register("huber", family="linear")(HuberRegressor)
MODELS.register("theilsen", family="linear")(TheilSenRegressor)
MODELS.register("quantile", family="linear")(QuantileRegressor)
MODELS.register("sgd", family="linear")(SGDRegressor)

# ------------------------------------------------------------------- glm
try:  # statsmodels 非必需，sklearn 自带 Gamma/Tweedie
    from sklearn.linear_model import GammaRegressor, TweedieRegressor

    MODELS.register("gamma", family="glm")(GammaRegressor)
    MODELS.register("tweedie", family="glm")(TweedieRegressor)
except ImportError:  # pragma: no cover
    pass
MODELS.register("poisson", family="glm")(PoissonRegressor)

# ---------------------------------------------------------------- latent
MODELS.register("pls", family="latent")(PLSRegression)


class PCR:
    """主成分回归：PCA + OLS。"""

    def __init__(self, n_components: int = 10, scale: bool = True):
        self.pipe = sk_make_pipeline(
            PCA(n_components=n_components), LinearRegression()
        )

    def fit(self, X, y):
        self.pipe.fit(X, np.asarray(y).ravel())
        return self

    def predict(self, X):
        return self.pipe.predict(X)

    @property
    def coef_(self):
        pca, lr = self.pipe.steps[0][1], self.pipe.steps[1][1]
        return pca.components_.T @ lr.coef_


MODELS.register("pcr", family="latent")(PCR)

# ---------------------------------------------------------------- kernel


def _wq_kernel(c: float = 1.0) -> object:
    return c * RationalQuadratic(length_scale=1.0, alpha=1.0) + WhiteKernel(1e-1)


class GPR:
    """高斯过程回归，核 = RationalQuadratic + White + Constant。"""

    def __init__(self, constant_value: float = 1.0, n_restarts: int = 0, random_state: int = 0):
        self.constant_value = constant_value
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.est: GaussianProcessRegressor | None = None

    def fit(self, X, y):
        y = np.asarray(y, dtype=np.float64).ravel()
        kernel = _wq_kernel() + ConstantKernel(constant_value=float(np.mean(y)))
        self.est = GaussianProcessRegressor(
            kernel=kernel, random_state=self.random_state,
            n_restarts_optimizer=self.n_restarts, normalize_y=True,
        )
        self.est.fit(X, y)
        return self

    def predict(self, X):
        return self.est.predict(X)


MODELS.register("gpr", family="kernel")(GPR)
MODELS.register("svr", family="kernel")(SVR)
MODELS.register("krr", family="kernel")(KernelRidge)

# ----------------------------------------------------------------- trees
MODELS.register("rf", family="trees")(RandomForestRegressor)
try:
    from lightgbm import LGBMRegressor

    MODELS.register("lgbm", family="trees")(LGBMRegressor)
except ImportError:  # pragma: no cover
    pass

# -------------------------------------------------------------------- nn
MODELS.register("mlp", family="nn")(MLPRegressor)


# ------------------------------------------------------------ GLM 自动选型
def glm_family(y: np.ndarray) -> str:
    """按过度离散程度推荐 GLM family。

    依据 / 3.12：方差显著大于均值 → gamma；近似相等 → poisson；
    欠离散 → 负二项（此处退化为 tweedie）。
    """
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    mu = float(np.mean(y))
    if mu <= 0:
        return "gamma"
    r = float(np.var(y)) / mu
    if r > 1.2:
        return "gamma"
    if r < 0.8:
        return "tweedie"
    return "poisson"
