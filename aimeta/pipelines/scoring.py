"""模型打分表（scoring table）：复刻 ``Train.py`` 的 ``metrics.csv`` 排名逻辑。

一张打分表 = 一组候选模型（不同预处理链 / 模型）的指标行。给定这些指标后，
按每个指标分别排名、求和得到 ``rank``（越小越优），并写出 CSV，供用户挑选
TOP K 与最终模型（结合各模型自己的图）。

排名规则（在 ``Train.py`` 基础上增加 R² 杠杆）：
  - 越大越好（倒序）：alarm_acc / r2_score / daily_r2_score /
    daily_pearson_r_score / rate of reaching the standard
  - 越小越好（正序）：alarm_err / mape / rmse / bad grouped ratio /
    |durbin_watson − 2|
  - mape / rmse / rate / bad grouped / durbin 用「竞赛排名」（并列取最小序）；
    其余用 argsort 序位排名。最终 ``rank`` = 十个分量排名之和。
  - R² 杠杆：``r2_score`` / ``daily_r2_score`` 为负时按其幅度放大惩罚
    （见 ``_r2_leveraged_ranks``），避免 R²=-10 与 R²=-0.1 只差一档。
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# (指标键, 方向, 排名方法)
#   direction: desc=越大越好, asc=越小越好, asc_abs2=|x-2| 越小越好
#   method:    ordinal=序位排名, competition=并列取最小序, r2=序位+负R²杠杆
RANK_COMPONENTS: List[Tuple[str, str, str]] = [
    ("alarm_acc", "desc", "ordinal"),
    ("alarm_err", "asc", "ordinal"),
    ("mape", "asc", "competition"),
    ("r2_score", "desc", "r2"),
    ("daily_r2_score", "desc", "r2"),
    ("daily_pearson_r_score", "desc", "ordinal"),
    ("rmse", "asc", "competition"),
    ("rate of reaching the standard", "desc", "competition"),
    ("bad grouped ratio", "asc", "competition"),
    ("durbin_watson_value", "asc_abs2", "competition"),
]

# 打分表 CSV 列顺序（与 Train.py metrics.csv 对齐）。
SCORING_COLUMNS: List[str] = [
    "model_type",
    "alarm_acc", "alarm_err", "mape", "r2_score", "pearson_r_score",
    "daily_r2_score", "daily_pearson_r_score", "rmse",
    "rate of reaching the standard", "bad grouped ratio",
    "durbin_watson_value", "Kurtosis", "skew", "LM_p", "F_p",
    "rank",
]


def _ordinal_ranks(values: Sequence[float], descending: bool) -> np.ndarray:
    """argsort 序位排名：0 = 最优。"""
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr)[::-1] if descending else np.argsort(arr)
    ranks = np.empty(len(arr), dtype=float)
    ranks[order] = np.arange(len(arr))
    return ranks


def _competition_ranks(values: Sequence[float], descending: bool) -> np.ndarray:
    """竞赛排名（并列取最小序，0 = 最优）。"""
    arr = [float(v) for v in values]
    sorted_vals = sorted(arr, reverse=descending)
    return np.array([sorted_vals.index(v) for v in arr], dtype=float)


# 负 R² 杠杆系数：惩罚 = R2_LEVERAGE * (n-1) * |R²| / (1 + |R²|)。
# 默认 1.0（极端负 R² 最多多扣 n-1 个序位，即该分量直接垫底）；调大则对负 R² 更严厉。
R2_LEVERAGE: float = 1.0


def _r2_leveraged_ranks(values: Sequence[float]) -> np.ndarray:
    """R² 杠杆排名：序位基础上，负 R² 按其幅度放大惩罚。

    负 R² 表示模型劣于「用均值预测」的基线，幅度越负越应重罚；纯序位会把
    R²=-10 与 R²=-0.1 只差一档，掩盖质的差异。惩罚 = R2_LEVERAGE * (n-1) * |R²| / (1 + |R²|)，
    随 |R²| 增大饱和于 R2_LEVERAGE * (n-1)（等价于该分量直接垫底），避免单个极端负 R² 无限放大。
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    order = np.argsort(arr)[::-1]
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n)
    neg = arr < 0
    ranks[neg] += R2_LEVERAGE * (n - 1) * (-arr[neg]) / (1.0 - arr[neg])
    return ranks


def _component_rank(values: Sequence[float], direction: str,
                    method: str) -> np.ndarray:
    if method == "r2":
        return _r2_leveraged_ranks(values)
    if direction == "asc_abs2":
        key = np.abs(np.asarray([float(v) for v in values], dtype=float) - 2.0)
        return _competition_ranks(key, descending=False) if method == "competition" \
            else _ordinal_ranks(key, descending=False)
    if direction == "desc":
        return _ordinal_ranks(values, True) if method == "ordinal" \
            else _competition_ranks(values, True)
    if direction == "asc":
        return _ordinal_ranks(values, False) if method == "ordinal" \
            else _competition_ranks(values, False)
    raise ValueError(f"unknown direction: {direction}")


def rank_models(rows: List[Dict], components: Optional[List[Tuple[str, str, str]]] = None
                ) -> List[Dict]:
    """对候选模型指标行排名，返回带 ``rank`` 且按 rank 升序的新列表（不修改入参）。

    每个指标单独排名后求和，和越小越优。
    """
    components = components or RANK_COMPONENTS
    data = [dict(r) for r in rows]
    n = len(data)
    if n == 0:
        return data
    total = np.zeros(n, dtype=float)
    for key, direction, method in components:
        vals = [float(r.get(key, np.nan)) for r in data]
        total += _component_rank(vals, direction, method)
    for r, t in zip(data, total):
        r["rank"] = float(t)
    data.sort(key=lambda r: r["rank"])
    return data


def _merge_with_existing(rows: List[Dict], path: Path) -> List[Dict]:
    """复刻 Train.py：同 ``model_type`` 已存在时，仅当 mape 更优才更新。"""
    existing: Dict[str, Dict] = {}
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            existing[row["model_type"]] = row
    for r in rows:
        key = str(r.get("model_type", ""))
        if key in existing:
            old = existing[key]
            try:
                if float(r.get("mape", np.inf)) < float(old.get("mape", np.inf)):
                    existing[key].update({k: v for k, v in r.items()
                                          if k != "model_type"})
            except (TypeError, ValueError):
                existing[key].update({k: v for k, v in r.items()
                                      if k != "model_type"})
        else:
            existing[key] = dict(r)
    return [dict(v) for v in existing.values()]


def write_scoring_csv(rows: List[Dict], path: str | Path,
                      columns: Optional[List[str]] = None,
                      merge: bool = False) -> None:
    """写打分表 CSV（含 rank，按 rank 升序）。``merge=True`` 时与已有文件按
    ``model_type`` 合并（mape 更优才覆盖），与 Train.py 行为一致。
    """
    path = Path(path)
    columns = columns or SCORING_COLUMNS
    if merge and path.exists():
        rows = _merge_with_existing(rows, path)
    rows = rank_models(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in columns})


def read_scoring_csv(path: str | Path) -> List[Dict]:
    """读回打分表；数值列尽量转 float，rank 一并保留。"""
    path = Path(path)
    out: List[Dict] = []
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            rec = dict(row)
            for k, v in rec.items():
                if k == "model_type":
                    continue
                try:
                    rec[k] = float(v)
                except (TypeError, ValueError):
                    pass
            out.append(rec)
    return out


# ----------------------------------------------------------------------------
# 打分表行：从验证集算「验收指标全集」
#
# 下面这些列（pearson_r_score / daily_pearson_r_score / bad grouped ratio /
# durbin_watson_value / Kurtosis / skew / LM_p / F_p）此前只在 SCORING_COLUMNS
# 占位，没有计算逻辑；这里补上，使打分表能真正被填满。
# ----------------------------------------------------------------------------
import scipy.stats as st  # noqa: E402  (文件尾增补，保持顶部 import 干净)

from ..preprocessing.base import Pipeline  # noqa: E402
from ..core.artifact import ModelCard  # noqa: E402  (仅类型标注)
from ..metrics.water_standards import (  # noqa: E402
    acceptance_rate, alarm_accuracy, daily_r2,
)


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson 相关系数（单值/常量时返回 nan）。"""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _durbin_watson(resid: Sequence[float]) -> float:
    """Durbin-Watson 统计量：2=无自相关，趋近 0/4=正/负自相关。"""
    e = np.asarray(resid, dtype=float).ravel()
    if len(e) < 2:
        return float("nan")
    num = float(np.sum(np.diff(e) ** 2))
    den = float(np.sum(e ** 2))
    return num / den if den > 0 else float("nan")


def _bad_grouped_ratio(resid: Sequence[float]) -> float:
    """残差同号游程过长比例：反映误差是否成片聚集（系统性偏差）。

    ratio = 1 − 实际游程数 / 期望游程数；随机时≈0，长片聚集时→1。
    """
    e = np.asarray(resid, dtype=float).ravel()
    n = len(e)
    if n < 2:
        return 0.0
    sign = np.sign(e)
    runs = 1
    for i in range(1, n):
        if sign[i] != sign[i - 1]:
            runs += 1
    expected = (n + 1) / 2.0
    return float(max(0.0, min(1.0, 1.0 - runs / expected)))


def _slope_f_p(y: Sequence[float], x: Sequence[float]) -> float:
    """一元回归斜率=0 的 F 检验 p 值（缺自由度/常量时返回 nan）。"""
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    n = len(y)
    if n < 3:
        return float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    sxx = float(np.sum(xc ** 2))
    if sxx <= 0:
        return float("nan")
    b1 = float(np.sum(xc * yc)) / sxx
    yhat = y.mean() + b1 * xc
    ssr = float(np.sum((yhat - y.mean()) ** 2))
    sse = float(np.sum((y - yhat) ** 2))
    if sse <= 0:
        return 1.0 if ssr <= 0 else 0.0
    F = (ssr / 1.0) / (sse / (n - 2))
    return float(st.f.sf(F, 1, n - 2))


def _breusch_pagan_p(resid: Sequence[float], fitted: Sequence[float]) -> float:
    """Breusch-Pagan 异方差检验 p 值（以拟合值为唯一外生变量）。

    LM_p 越小越说明残差方差随浓度变化（需加权 / 变换）。
    """
    e = np.asarray(resid, dtype=float).ravel()
    f = np.asarray(fitted, dtype=float).ravel()
    return _slope_f_p(e ** 2, f)


def build_score_row(y_true: np.ndarray, y_pred: np.ndarray,
                    param: "WaterParam", day_idx: Optional[np.ndarray] = None
                    ) -> Dict[str, float]:
    """从一组 (真值, 预测) 算出整行验收指标（不含 model_type / rank）。

    覆盖 SCORING_COLUMNS 除 model_type / rank 外的全部列：
    误差类（mape / rmse / r2_score / pearson_r_score）、
    日级（daily_r2_score / daily_pearson_r_score）、
    报警（alarm_acc / alarm_err）、合格率（rate of reaching the standard）、
    残差诊断（bad grouped ratio / durbin_watson_value / Kurtosis / skew /
    LM_p / F_p）。
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    n = len(yt)
    resid = yp - yt

    rmse = float(np.sqrt(np.mean(resid ** 2))) if n else float("nan")
    denom = float(np.sum((yt - yt.mean()) ** 2)) if n else 0.0
    r2 = float(1.0 - np.sum(resid ** 2) / denom) if denom > 0 else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.nanmean(np.abs(resid) / np.where(yt == 0, np.nan, yt))) \
            if n else float("nan")

    rate = acceptance_rate(yt, yp, param)
    thr = param.ranges[-1] if getattr(param, "ranges", None) else 0.0
    al = alarm_accuracy(yt, yp, thr)
    alarm_acc = float(al["accuracy"])
    alarm_err = 1.0 - alarm_acc

    daily = daily_r2(day_idx, yt, yp) if day_idx is not None else {}
    daily_r2_score = float(np.mean(list(daily.values()))) if daily else float("nan")
    daily_pearson: Dict[str, float] = {}
    if day_idx is not None:
        for d in np.unique(np.asarray(day_idx)):
            m = np.asarray(day_idx) == d
            if m.sum() >= 2:
                daily_pearson[str(d)] = _pearson(yt[m], yp[m])
    daily_pearson_r_score = (float(np.nanmean(list(daily_pearson.values())))
                             if daily_pearson else float("nan"))

    return {
        "alarm_acc": alarm_acc,
        "alarm_err": alarm_err,
        "mape": mape,
        "r2_score": r2,
        "pearson_r_score": _pearson(yt, yp),
        "daily_r2_score": daily_r2_score,
        "daily_pearson_r_score": daily_pearson_r_score,
        "rmse": rmse,
        "rate of reaching the standard": rate,
        "bad grouped ratio": _bad_grouped_ratio(resid),
        "durbin_watson_value": _durbin_watson(resid),
        "Kurtosis": float(st.kurtosis(resid, fisher=True)) if n >= 4 else float("nan"),
        "skew": float(st.skew(resid)) if n >= 3 else float("nan"),
        "LM_p": _breusch_pagan_p(resid, yp),
        "F_p": _slope_f_p(yt, yp),
    }


def _raw_predict(card: "ModelCard", X: np.ndarray) -> np.ndarray:
    """用卡片自己的预处理链做 raw 预测（不经过检出限/量程裁剪）。

    评分/融合用 raw 值，避免 LOD/2 替值扭曲残差分布与排名。
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    pipe = Pipeline.from_config(card.preproc_chain)
    Xt = pipe.transform(X)
    return np.asarray(card.estimator.predict_raw(Xt), dtype=float).ravel()


def score_cards(cards: Sequence["ModelCard"], X_val: np.ndarray, y_val: np.ndarray,
                param: "WaterParam", day_idx: Optional[np.ndarray] = None
                ) -> List[Dict]:
    """对每个候选卡在验证集上打分并排名，返回带 ``model_type`` 与 ``rank`` 的行。

    ``model_type`` 取 ``"{model_key}::{fingerprint}"``，唯一且可用于 CSV 合并。
    """
    rows: List[Dict] = []
    for c in cards:
        yp = _raw_predict(c, X_val)
        row = build_score_row(y_val, yp, param, day_idx)
        row["model_type"] = f"{c.model_key}::{c.fingerprint()}"
        rows.append(row)
    return rank_models(rows)


def select_top_k(cards: Sequence["ModelCard"], k: int, X_val: np.ndarray,
                 y_val: np.ndarray, param: "WaterParam",
                 day_idx: Optional[np.ndarray] = None) -> List["ModelCard"]:
    """按打分表 rank 选 TOP K 候选卡（K 超过候选数时返回全部）。"""
    scored = score_cards(cards, X_val, y_val, param, day_idx)
    ordered = sorted(scored, key=lambda r: r["rank"])
    by_key = {f"{c.model_key}::{c.fingerprint()}": c for c in cards}
    return [by_key[r["model_type"]] for r in ordered[:k]]
