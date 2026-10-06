"""工控机部署运行时（依赖与 aimeta 一致：numpy / scipy / PyWavelets）。

设计
----
* 只支持**线性**模型：PLS / Ridge / Lasso / OLS / PCR，本质 ``y = X@coef + b``
* 预处理算子分两类：
  - **纯 numpy 可编译**：``savgol`` / ``deriv_gram``（编译成权重矩阵）、
    ``snv`` ``msc`` ``mean_center`` ``column_scale``（闭式计算），部署端零额外依赖；
  - **需要外部库的算子**（如 ``wavelet`` 需 PyWavelets）：编译为"委托"算子，
    部署端运行时调用注册算子执行，因此需要目标机安装对应依赖
    （aimeta 已将 scipy / PyWavelets 列为主依赖，``pip install aimeta`` 即满足）。

纯 numpy 链（如 dayu_edge.yaml）仍可完全脱离 scipy/pywt 运行；
只有用了 wavelet 这类算子的链才需要对应库。


产物目录
--------
```
model_dir/
├── manifest.json   指纹、波长轴、上下限
└── arrays.npz      编译后的算子参数 + coef/intercept
```
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_COMPILABLE = {"savgol", "deriv_gram", "snv", "msc", "mean_center", "column_scale"}


class CompileError(Exception):
    """预处理链含未注册算子、或部署端缺少对应依赖时抛出。

    纯 numpy 可编译算子（savgol / snv / msc / …）总是可用；
    wavelet 等需外部库的算子只有在目标机装好对应依赖（scipy / PyWavelets）时才可部署。
    """

    pass


# 兼容旧名（早期草稿里用过小写别名）
compile_error = CompileError


def _is_delegable(op: str) -> bool:
    """op 是否为已注册、且需要外部库（scipy / pywt 等）的预处理算子。

    纯 numpy 可编译算子走 ``_COMPILABLE`` 快路径；其余已注册算子（如 wavelet）
    在部署端委托给注册算子执行，需目标机安装对应依赖。aimeta 未安装时返回
    False，使纯 numpy 链仍可脱离 aimeta 独立运行。
    """
    try:
        from aimeta.preprocessing.operators import PREPROC
    except Exception:
        return False
    return op in PREPROC


def _apply_op(op_name: str, params: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """调用注册算子执行（用于需要外部库、且未在内置原生路径覆盖的预处理算子）。

    注意：``wavelet`` 已在 ``_WAVELET_NATIVE`` 原生支持（仅依赖 PyWavelets，
    无需安装完整 aimeta）；此处仅作其它潜在委托算子的兜底。
    """
    from aimeta.preprocessing.operators import PREPROC
    return PREPROC.build(op_name, **params).transform(X)


def _wavelet_denoise(x_row: np.ndarray, wavelet: str = "sym4", level: int = 2) -> np.ndarray:
    """小波软阈值去噪（单条光谱），输出长度与输入一致。

    直接依赖 PyWavelets（``pywt``），使 ``wavelet`` 算子可在**不安装完整 aimeta**
    的工控机 / 小电脑上部署；与 ``aimeta.preprocessing.operators.wavelet_denoise``
    数值一致（由回归测试锁定）。
    """
    import pywt
    x = np.asarray(x_row, dtype=float)
    coeff = pywt.wavedec(x, wavelet, mode="smooth", level=level)
    sigma = np.median(np.abs(coeff[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = [pywt.threshold(c, value=uthresh, mode="soft") for c in coeff[1:]]
    out = pywt.waverec(coeff, wavelet, mode="smooth")
    n = len(x)
    if len(out) == n:
        return out
    if len(out) > n:
        return out[:n]
    return np.pad(out, (0, n - len(out)), mode="edge")


# ------------------------------------------------------------ Savitzky-Golay
def sg_weights(window: int, polyorder: int, deriv: int, offset: int) -> np.ndarray:
    """单个窗口内的 SG 权重（纯 numpy，与 scipy 的 Gram 多项式结果等价）。

    Args:
        window: 奇数窗口
        polyorder: 多项式阶数
        deriv: 导数阶数（0 = 平滑）
        offset: 求值点相对窗口中心的偏移（边界处非 0）
    """
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    V = np.vander(x, polyorder + 1, increasing=True)      # (window, k)
    pinv = np.linalg.pinv(V)                              # (k, window)
    k = np.arange(polyorder + 1)
    a = np.zeros(polyorder + 1)
    for kk in k[k >= deriv]:
        a[kk] = (math.factorial(int(kk)) / math.factorial(int(kk) - deriv)) \
            * float(offset) ** (int(kk) - deriv)
    return a @ pinv


def sg_matrix(n_wavelengths: int, window: int, polyorder: int,
              deriv: int = 0) -> np.ndarray:
    """把 SG 平滑/求导编译成一个 (p, p) 矩阵，部署端只需 X @ W。"""
    if window % 2 == 0:
        raise ValueError("window must be odd")
    p = n_wavelengths
    half = window // 2
    W = np.zeros((p, p))
    for j in range(p):
        lo = int(np.clip(j - half, 0, max(p - window, 0)))
        offset = j - lo - half
        W[lo:lo + window, j] = sg_weights(window, polyorder, deriv, offset)
    return W


# ----------------------------------------------------------------- 编译链
def compile_chain(chain: List[Dict[str, Any]], n_wavelengths: int) -> List[Dict[str, Any]]:
    """把训练端的预处理链编译成 numpy 可执行形式。"""
    compiled: List[Dict[str, Any]] = []
    for raw in chain:
        item = dict(raw)
        op = item.pop("op")
        if op in _COMPILABLE:
            if op in ("savgol", "deriv_gram"):
                W = sg_matrix(
                    n_wavelengths,
                    window=int(item.get("window", 15)),
                    polyorder=int(item.get("polyorder", item.get("order", 3))),
                    deriv=int(item.get("deriv", item.get("der", 0))),
                )
                compiled.append({"kind": "matmul", "W": W})
            elif op == "snv":
                compiled.append({"kind": "snv"})
            elif op == "msc":
                compiled.append({"kind": "msc"})
            else:   # mean_center / column_scale，具体数值在 fit 后填入
                compiled.append({"kind": "affine",
                                 "scale": np.ones(n_wavelengths),
                                 "shift": np.zeros(n_wavelengths)})
        elif _is_delegable(op):
            # 需要 scipy / pywt 等外部库的算子：运行时委托给注册算子执行。
            compiled.append({"kind": "op", "op": op, "params": item})
        else:
            raise CompileError(
                f"op {op!r} is not a registered preprocessing operator "
                f"(compilable: {sorted(_COMPILABLE)})."
            )
    return compiled


# --------------------------------------------------------------- 模型本体
class LinearEdgeModel:
    """工控机部署用线性软测量模型（部署端依赖与 aimeta 一致）。

    Example::

        m = load_edge_model("deploy/TN_ai17")
        y = m.predict(X)          # X: (n, p) 原始吸光度
    """

    def __init__(
        self,
        ops: List[Dict[str, Any]],
        coef: np.ndarray,
        intercept: float,
        wavelengths: np.ndarray,
        fingerprint: str = "",
        label: str = "",
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        dead_bound: Optional[float] = None,
    ) -> None:
        self.ops = ops
        self.coef = np.asarray(coef, dtype=np.float64).ravel()
        self.intercept = float(intercept)
        self.wavelengths = np.asarray(wavelengths, dtype=np.float64)
        self.fingerprint = fingerprint
        self.label = label
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dead_bound = dead_bound

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.coef.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} wavelengths, model expects {self.coef.shape[0]}")
        for op in self.ops:
            k = op["kind"]
            if k == "matmul":
                X = X @ op["W"]
            elif k == "affine":
                X = X * op["scale"] + op["shift"]
            elif k == "snv":
                mu = X.mean(axis=1, keepdims=True)
                sd = X.std(axis=1, keepdims=True)
                sd[sd == 0] = 1.0
                X = (X - mu) / sd
            elif k == "msc":
                ref = op.get("ref")
                if ref is None:
                    ref = X.mean(axis=0)
                A = np.vstack([ref, np.ones_like(ref)]).T
                X = np.vstack([_msc_row(r, A) for r in X])
            elif k == "op":
                op_name = op["op"]
                params = op.get("params", {})
                if op_name == "wavelet":
                    wav = str(params.get("wavelet", "sym4"))
                    lvl = int(params.get("level", 2))
                    X = np.vstack([_wavelet_denoise(r, wav, lvl) for r in X])
                else:
                    X = _apply_op(op_name, params, X)
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        y = self.transform(X) @ self.coef + self.intercept
        y = np.asarray(y, dtype=np.float64).ravel()
        if self.lower_bound is not None:
            below = y < self.lower_bound
            y = np.where(below, self.lower_bound / 2.0, y)   # 未检出替为 LOD/2
        # 复核区 (upper, dead]：保留原值（交给人工复核），不夹断
        if self.dead_bound is not None:
            y = np.where(y > self.dead_bound, np.nan, y)      # 死区不显示
        return y

    # ---- 序列化 ----
    def save(self, model_dir: str | Path) -> Path:
        d = Path(model_dir)
        d.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, np.ndarray] = {"coef": self.coef,
                                         "wavelengths": self.wavelengths}
        ops_meta = []
        for i, op in enumerate(self.ops):
            k = op["kind"]
            if k == "matmul":
                arrays[f"W{i}"] = op["W"]
                ops_meta.append({"kind": k, "key": f"W{i}"})
            elif k == "affine":
                arrays[f"s{i}"] = op["scale"]
                arrays[f"t{i}"] = op["shift"]
                ops_meta.append({"kind": k, "scale": f"s{i}", "shift": f"t{i}"})
            elif k == "msc":
                ref = op.get("ref")
                if ref is not None:
                    arrays[f"r{i}"] = np.asarray(ref, dtype=np.float64)
                    ops_meta.append({"kind": k, "ref": f"r{i}"})
                else:
                    ops_meta.append({"kind": k})
            elif k == "op":
                ops_meta.append({"kind": k, "op": op["op"], "params": op["params"]})
            else:
                ops_meta.append({"kind": k})
        np.savez(d / "arrays.npz", **arrays)
        (d / "manifest.json").write_text(json.dumps({
            "label": self.label,
            "fingerprint": self.fingerprint,
            "intercept": self.intercept,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "dead_bound": self.dead_bound,
            "ops": ops_meta,
            "version": 1,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return d

    @staticmethod
    def load(model_dir: str | Path) -> "LinearEdgeModel":
        d = Path(model_dir)
        mf = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        z = np.load(d / "arrays.npz")
        ops = []
        for m in mf["ops"]:
            k = m["kind"]
            if k == "matmul":
                ops.append({"kind": k, "W": z[m["key"]]})
            elif k == "affine":
                ops.append({"kind": k, "scale": z[m["scale"]],
                            "shift": z[m["shift"]]})
            elif k == "msc":
                ops.append({"kind": k,
                            "ref": z[m["ref"]] if "ref" in m else None})
            elif k == "op":
                ops.append({"kind": k, "op": m["op"], "params": m["params"]})
            else:
                ops.append({"kind": k})
        return LinearEdgeModel(
            ops=ops, coef=z["coef"], intercept=mf["intercept"],
            wavelengths=z["wavelengths"], fingerprint=mf["fingerprint"],
            label=mf["label"], lower_bound=mf["lower_bound"],
            upper_bound=mf["upper_bound"],
            dead_bound=mf.get("dead_bound", None),
        )


def _msc_row(row: np.ndarray, A: np.ndarray) -> np.ndarray:
    sol, *_ = np.linalg.lstsq(A, row, rcond=None)
    a, b = sol
    if abs(a) < 1e-12:
        a = 1e-12
    return (row - b) / a


def load_edge_model(model_dir: str | Path) -> LinearEdgeModel:
    return LinearEdgeModel.load(model_dir)


# ------------------------------------------------------------------ 导出
def export_card(card, model_dir: str | Path,
                X_reference: Optional[np.ndarray] = None) -> Path:
    """把训练端的 ModelCard 导出为 edge 产物（在开发机上执行，会 import aimeta）。"""
    from aimeta.preprocessing.base import Pipeline   # 惰性导入，保持 edge 独立

    wm = card.estimator
    if not (hasattr(wm, "linear_coef") and hasattr(wm, "x_mean_")):
        raise TypeError("only WaterQualityModel-wrapped estimators can be exported")
    coef = wm.linear_coef()
    intercept = wm.linear_intercept()
    if coef is None or intercept is None:
        raise TypeError("estimator is not linear; cannot export to edge runtime")

    ops = compile_chain(card.preproc_chain, len(card.wavelengths))

    # 把 fit 得到的均值 / 方差 / 参考光谱写进算子
    pipe = Pipeline.from_config(card.preproc_chain)
    if X_reference is not None:
        pipe.fit(np.asarray(X_reference, dtype=np.float64))
    for op, tf in zip(ops, pipe.steps):
        if op["kind"] == "affine" and getattr(tf, "mean_", None) is not None:
            scale = getattr(tf, "scale_", None)
            scale = np.ones_like(tf.mean_) if scale is None else np.asarray(scale)
            scale[scale == 0] = 1.0
            op["scale"] = 1.0 / scale
            op["shift"] = -np.asarray(tf.mean_) / scale
        if op["kind"] == "msc" and getattr(tf, "reference_", None) is not None:
            op["ref"] = np.asarray(tf.reference_, dtype=np.float64)

    m = LinearEdgeModel(
        ops=ops, coef=coef, intercept=intercept,
        wavelengths=np.asarray(card.wavelengths, dtype=np.float64),
        fingerprint=card.fingerprint(), label=card.label,
        lower_bound=getattr(wm, "lower_bound", None),
        upper_bound=getattr(wm, "upper_bound", None),
        dead_bound=getattr(wm, "dead_bound", None),
    )
    return m.save(model_dir)


def verify_against_card(edge_model: LinearEdgeModel, card, X: np.ndarray,
                        atol: float = 1e-8) -> Tuple[bool, float]:
    """train/serve 一致性校验：edge 预测 vs 训练端预测应逐元素一致。

    死区内的预测在两端都是 NaN，按"相等"处理，故只比较两端都有限的样本。
    """
    from aimeta.pipelines.infer import predict
    a = predict(card, X)
    b = edge_model.predict(X)
    finite = np.isfinite(a) & np.isfinite(b)
    diff = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
    return bool(diff <= atol), diff
