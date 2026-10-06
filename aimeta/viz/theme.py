"""统一可视化主题：一处定义，全局一致。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

COLORS = {
    "primary": "#1a3a5c",
    "accent": "#c8a415",
    "ok": "#2e7d5b",
    "warn": "#c86a15",
    "bad": "#b3352c",
    "grid": "#d7dee5",
    "spectra": "#4a7fb5",
}

_PALETTE = ["#1a3a5c", "#c8a415", "#2e7d5b", "#b3352c", "#6a5acd",
            "#008b8b", "#a0522d", "#4a7fb5"]


def apply_theme() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.edgecolor": "#8896a4",
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.4,
        "legend.frameon": False,
        "axes.prop_cycle": plt.cycler("color", _PALETTE),
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })


def save_fig(fig, path: str | Path, close: bool = True) -> Path:
    """统一导出：自动建目录，避免每个脚本各写一遍 savefig。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path


def palette(n: int) -> list[str]:
    return [_PALETTE[i % len(_PALETTE)] for i in range(n)]
