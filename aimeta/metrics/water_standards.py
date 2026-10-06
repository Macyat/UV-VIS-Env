"""GB3838 水质类别判定与验收指标。

各参数的量程、类别分界、检出限与误差限定义在 ``configs/params.yaml``，
这里是其判定逻辑。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class WaterParam:
    """一个水质参数的计量定义。"""

    name: str
    ranges: List[float]          # 类别分界（GB3838 I/II/III/IV/V）
    lower_bound: float           # 检出限
    upper_bound: float           # 量程上限
    abs_error_bound: float = 0.0  # 低浓度允许绝对误差
    mape_bound: float = 0.15      # 高浓度允许相对误差
    unit: str = "mg/L"
    standard: str = ""            # 检出限的依据标准（溯源用）

    @classmethod
    def from_dict(cls, name: str, d: Dict) -> "WaterParam":
        return cls(name=name, **{k: v for k, v in d.items()
                                 if k in {"ranges", "lower_bound", "upper_bound",
                                          "abs_error_bound", "mape_bound", "unit",
                                          "standard"}})

    @property
    def n_classes(self) -> int:
        return len(self.ranges) + 1


def classify(values: np.ndarray, param: WaterParam) -> np.ndarray:
    """把浓度映射到类别序号（0 = 优于 I 类）。"""
    v = np.asarray(values, dtype=np.float64)
    return np.searchsorted(np.asarray(param.ranges, dtype=np.float64), v, side="right")


def misclassification_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                             param: WaterParam) -> np.ndarray:
    """类别混淆矩阵：行 = 真值类别，列 = 预测类别。"""
    ct, cp = classify(y_true, param), classify(y_pred, param)
    n = param.n_classes
    M = np.zeros((n, n), dtype=int)
    for a, b in zip(ct, cp):
        M[a, b] += 1
    return M


def _acceptable(y_true: float, y_pred: float, param: WaterParam) -> bool:
    """单个样本是否「判得过去」。

    规则（按 GB3838 类别）：
        真值与预测同属 I/II 类      → 通过
        低浓度（<= r[1]）           → 相对误差 <= 40%
        中浓度（r[1], r[3]]        → 相对误差 <= 30%
        高浓度（> 最差类别界）      → 相对误差 <= 20%
        其余（如浊度这类类别数少的）→ 退化为绝对误差限
    """
    r = param.ranges
    rel = abs(y_true - y_pred) / max(abs(y_true), 1e-12)
    if len(r) >= 2:
        if y_true <= r[1] and y_pred <= r[1]:
            return True
        if y_true <= r[1]:
            return rel <= 0.4
        if len(r) >= 4 and r[1] < y_true <= r[3]:
            return rel <= 0.3
        if y_true > r[-1]:
            return rel <= 0.2
    return abs(y_true - y_pred) <= param.abs_error_bound


def acceptance_rate(y_true: np.ndarray, y_pred: np.ndarray,
                    param: WaterParam) -> float:
    """按「跨类别不误判 + 相对误差达标」的口径计算合格率。"""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if len(yt) == 0:
        return float("nan")
    ok = [_acceptable(a, b, param) for a, b in zip(yt, yp)]
    return float(np.mean(ok))


def daily_r2(day_idx: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """按天分组的 R²。

    为什么看 daily R²：整体 R² 会被浓度跨度撑得很好看，
    而现场关心的是「同一天、浓度变化不大时能不能分辨」。
    """
    day_idx = np.asarray(day_idx)
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    out: Dict[str, float] = {}
    for d in np.unique(day_idx):
        m = day_idx == d
        if m.sum() < 2:
            continue
        ss_res = float(np.sum((yt[m] - yp[m]) ** 2))
        ss_tot = float(np.sum((yt[m] - yt[m].mean()) ** 2))
        out[str(d)] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return out


def alarm_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float
                   ) -> Dict[str, float]:
    """超标报警的准确率 / 误报率 / 漏报率。"""
    yt = np.asarray(y_true, dtype=np.float64) >= threshold
    yp = np.asarray(y_pred, dtype=np.float64) >= threshold
    tp = int(np.sum(yt & yp)); fp = int(np.sum(~yt & yp))
    fn = int(np.sum(yt & ~yp)); tn = int(np.sum(~yt & ~yp))
    return {
        "accuracy": (tp + tn) / len(yt) if len(yt) else float("nan"),
        "false_alarm_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "miss_rate": fn / (fn + tp) if (fn + tp) else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def acceptance_report(y_true: np.ndarray, y_pred: np.ndarray, param: WaterParam,
                      day_idx: Optional[np.ndarray] = None) -> Dict[str, object]:
    """一份参数一份验收结论，供 ``viz`` 渲染成报告。"""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    err = yp - yt
    rep: Dict[str, object] = {
        "param": param.name,
        "n": len(yt),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "r2": float(1 - np.sum(err ** 2) / np.sum((yt - yt.mean()) ** 2))
        if np.sum((yt - yt.mean()) ** 2) > 0 else float("nan"),
        "acceptance_rate": acceptance_rate(yt, yp, param),
        "confusion": misclassification_matrix(yt, yp, param).tolist(),
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.abs(err / np.where(yt == 0, np.nan, yt))
    rep["mape"] = float(np.nanmean(mape))
    if day_idx is not None:
        rep["daily_r2"] = daily_r2(day_idx, yt, yp)
    if param.upper_bound:
        rep["alarm"] = alarm_accuracy(yt, yp, param.ranges[-1])
    return rep
