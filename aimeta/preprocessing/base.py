"""预处理算子基类与 Pipeline 链。

一条链 = 一串 ``{"op": 名字, ...参数}`` 字典，可来自 YAML，可被序列化回去，
因此训练端和部署端**不可能跑出两条不同的链**。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from ..core.registry import PREPROC


class Transformer:
    """所有预处理算子的基类。

    子类只需实现 ``fit`` / ``transform``；``params`` 决定其 YAML 表示与指纹，
    因此**必须只包含可 JSON 序列化的标量**（不要塞数组）。
    """

    #: 算子的注册名，子类必须覆盖
    op: str = ""

    def __init__(self, **params: Any) -> None:
        self.params: Dict[str, Any] = dict(params)

    # ---- 需子类实现 ----
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Transformer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # ---- 通用 ----
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def to_dict(self) -> Dict[str, Any]:
        d = {"op": self.op}
        d.update(self.params)
        return d

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        kv = ",".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.op}({kv})" if kv else f"{self.op}()"


class Pipeline:
    """有序算子链，等价于 sklearn Pipeline，但 YAML 可复现。

    Example::

        chain = [{"op": "savgol", "window": 15, "polyorder": 3}, {"op": "snv"}]
        pipe = Pipeline.from_config(chain)
        Xt = pipe.fit_transform(X)
        pipe.to_config() == chain      # True
    """

    def __init__(self, steps: Iterable[Transformer]) -> None:
        self.steps: List[Transformer] = list(steps)

    # ---- 构造 ----
    @classmethod
    def from_config(cls, config: Iterable[Dict[str, Any]]) -> "Pipeline":
        steps = []
        for item in config:
            item = dict(item)
            op = item.pop("op", None)
            if op is None:
                raise ValueError(f"chain item without 'op': {item}")
            steps.append(PREPROC.build(op, **item))
        return cls(steps)

    def to_config(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.steps]

    # ---- 运行 ----
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Pipeline":
        Xt = np.asarray(X, dtype=np.float64)
        for s in self.steps:
            s.fit(Xt, y)
            Xt = s.transform(Xt)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xt = np.asarray(X, dtype=np.float64)
        for s in self.steps:
            Xt = s.transform(Xt)
        return Xt

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def append(self, step: Transformer) -> "Pipeline":
        self.steps.append(step)
        return self

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return " -> ".join(repr(s) for s in self.steps)


def build_chain(config: Iterable[Dict[str, Any]]) -> Pipeline:
    """``Pipeline.from_config`` 的别名，供 YAML 直接喂进来。"""
    return Pipeline.from_config(config)


class compile_error(Exception):  # noqa: N801 - 保持小写以贴近 "compile" 语义
    """链含未注册算子、或部署端缺少对应依赖时抛出。

    纯 numpy 可编译算子（SG 导数 / SNV / MSC / 中心化 / 标准化）总是可用；
    wavelet 等需外部库（PyWavelets）的算子，只有在目标机装好对应依赖时才可部署。
    """
