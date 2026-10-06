"""变量选择基类与「选出来的波长是不是碰巧」的检验。 专门讲变量选择的局限：在 p >> n 的光谱数据上，
任何选择算法都能选出一组看起来极好的波长。唯一可靠的护栏是
**置换检验** —— 把 y 打乱重跑同样的选择流程，比较性能分布。
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np


class SelectorBase:
    """波长选择器的统一接口。

    子类实现 ``fit(X, y)`` 并写出 ``selected_``（波长下标数组）；
    ``transform`` 供下游取子集。
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SelectorBase":
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X)[:, self.selected_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    @property
    def selected_(self) -> np.ndarray:
        if not hasattr(self, "_selected_"):
            raise RuntimeError("selector is not fitted")
        return self._selected_

    @selected_.setter
    def selected_(self, value: Sequence[int]) -> None:
        self._selected_ = np.asarray(value, dtype=int)


def permutation_test(
    selector_factory: Callable[[], SelectorBase],
    score_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    X: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 30,
    random_state: int = 0,
) -> dict:
    """对一次波长选择做置换检验。

    Args:
        selector_factory: 每次调用返回一个**新的**选择器实例
        score_fn: (X_subset, y_true, wavelengths_idx) -> score，越小越好（如 RMSECV）
        X, y: 原始数据
        n_permutations: 置换次数

    Returns:
        dict(score=真实分数, null_scores=置换分布, p_value=真实分数不优于随机的概率)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()

    sel = selector_factory().fit(X, y)
    real = float(score_fn(sel.transform(X), y, sel.selected_))

    rng = np.random.default_rng(random_state)
    null = []
    for _ in range(n_permutations):
        yp = rng.permutation(y)
        s = selector_factory().fit(X, yp)
        null.append(float(score_fn(s.transform(X), yp, s.selected_)))
    null = np.asarray(null, dtype=np.float64)
    p = float(np.mean(null <= real))
    return {
        "score": real,
        "null_mean": float(np.mean(null)),
        "null_std": float(np.std(null)),
        "null_scores": null,
        "p_value": p,
    }


def selection_stability(
    selector_factory: Callable[[], SelectorBase],
    X: np.ndarray,
    y: np.ndarray,
    n_runs: int = 20,
    fraction: float = 0.8,
    random_state: int = 0,
) -> np.ndarray:
    """稳定性选择：多次子采样，统计每个波长被选中的频率。"""
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    rng = np.random.default_rng(random_state)
    freq = np.zeros(p)
    m = max(2, int(fraction * n))
    for _ in range(n_runs):
        idx = rng.choice(n, size=m, replace=False)
        sel = selector_factory().fit(X[idx], np.asarray(y).ravel()[idx])
        freq[sel.selected_] += 1.0
    return freq / n_runs
