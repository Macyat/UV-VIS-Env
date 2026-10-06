"""候选模型图册导出：每个候选（TOP K）一套图，便于人工挑最终模型。

一张图册 = 每个候选卡一个子目录，内含：
    - <tag>_pred_vs_actual.png   预测 vs 实测（带 GB3838 类别线）
    - <tag>_residuals.png        残差序列（看漂移 / 聚集）
    - <tag>_scatter_resid.png    残差 vs 拟合值（看异方差）

若候选是潜变量模型（PLS/PCR），额外导出碎石图。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ..core.artifact import ModelCard
from ..preprocessing.base import Pipeline
from . import plots
from .theme import apply_theme, save_fig


def _raw_predict(card: ModelCard, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    pipe = Pipeline.from_config(card.preproc_chain)
    Xt = pipe.transform(X)
    return np.asarray(card.estimator.predict_raw(Xt), dtype=np.float64).ravel()


def export_model_figures(
    card: ModelCard,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param: Optional[object] = None,
    out_dir: str | Path = ".",
) -> List[Path]:
    """为一个候选卡导出一套图，返回生成的文件路径列表。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_theme()

    yt = np.asarray(y_val, dtype=np.float64).ravel()
    yp = _raw_predict(card, X_val)
    tag = f"{card.model_key}_{card.fingerprint()}"
    ranges = getattr(param, "ranges", None)
    unit = getattr(param, "unit", "")
    paths: List[Path] = []

    fig = plots.plot_pred_vs_actual(yt, yp, ranges=ranges, label=tag, unit=unit)
    p = save_fig(fig, out_dir / f"{tag}_pred_vs_actual.png")
    paths.append(p)

    fig = plots.plot_residuals(yt, yp, label=tag)
    paths.append(save_fig(fig, out_dir / f"{tag}_residuals.png"))

    resid = yp - yt
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.scatter(yp, resid, s=18, alpha=0.7, color="#b3352c")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(f"预测 {unit}"); ax.set_ylabel("残差")
    ax.set_title(f"{tag} 残差 vs 拟合值")
    paths.append(save_fig(fig, out_dir / f"{tag}_scatter_resid.png"))

    # 潜变量模型：碎石图（用训练集变换后的 X 解释方差）
    if card.model_key in {"pls", "pcr"}:
        try:
            pipe = Pipeline.from_config(card.preproc_chain)
            Xt = pipe.fit_transform(np.asarray(X_val, dtype=np.float64), yt)
            fig = plots.plot_explained_variance(Xt, max_components=20)
            paths.append(save_fig(fig, out_dir / f"{tag}_scree.png"))
        except Exception:  # pragma: no cover - 非关键诊断图
            pass

    return paths


def export_top_k_report(
    cards: Sequence[ModelCard],
    X_val: np.ndarray,
    y_val: np.ndarray,
    param: object,
    out_dir: str | Path,
    k: int = 5,
    day_idx: Optional[np.ndarray] = None,
) -> dict:
    """选 TOP K 候选，导出每模型一套图 + 打分表 CSV，返回汇总信息。

    产物：
        <out_dir>/figures/<tag>/*.png   每模型一套图
        <out_dir>/scoring.csv           带 rank 的打分表
    返回 {"top_k": [...model_type], "scoring_csv": Path, "figures": {tag: [paths]}}
    """
    from ..pipelines.scoring import score_cards, write_scoring_csv  # 局部，明确依赖

    out_dir = Path(out_dir)
    scored = score_cards(cards, X_val, y_val, param, day_idx)  # type: ignore[arg-type]
    ordered = sorted(scored, key=lambda r: r["rank"])
    top = ordered[:k]

    fig_root = out_dir / "figures"
    fig_map: dict = {}
    by_key = {f"{c.model_key}::{c.fingerprint()}": c for c in cards}
    for row in top:
        c = by_key[row["model_type"]]
        paths = export_model_figures(c, X_val, y_val, param, fig_root)
        fig_map[row["model_type"]] = [str(p) for p in paths]

    csv_path = out_dir / "scoring.csv"
    write_scoring_csv(scored, csv_path)

    return {
        "top_k": [r["model_type"] for r in top],
        "scoring_csv": str(csv_path),
        "figures": fig_map,
    }
