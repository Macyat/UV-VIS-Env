"""SpectrumSet：显式的数据契约。

波长轴是一等公民：所有数据都显式携带 wavelengths，
避免按列号取数导致的错位，也是波长对齐与仪器迁移的前提。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class SpectrumSet:
    """一组光谱 + 其波长轴 + 元数据。

    Args:
        X: (n_samples, n_wavelengths) 吸光度矩阵
        wavelengths: (n_wavelengths,) 波长轴（nm）
        meta: 与样本一一对应的元数据，至少含 instrument_id / timestamp / site
        y: 可选标签，形状 (n_samples,) 或 (n_samples, n_targets)
        y_names: 标签名（多目标时）
    """

    X: np.ndarray
    wavelengths: np.ndarray
    meta: Optional[pd.DataFrame] = None
    y: Optional[np.ndarray] = None
    y_names: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {self.X.shape}")
        self.wavelengths = np.asarray(self.wavelengths, dtype=np.float64).ravel()
        if self.X.shape[1] != self.wavelengths.shape[0]:
            raise ValueError(
                f"X has {self.X.shape[1]} wavelengths but axis has {self.wavelengths.shape[0]}"
            )
        if self.meta is not None:
            if len(self.meta) != len(self.X):
                raise ValueError("meta length must match n_samples")
            self.meta = self.meta.reset_index(drop=True)
        if self.y is not None:
            self.y = np.asarray(self.y, dtype=np.float64)
            if len(self.y) != len(self.X):
                raise ValueError("y length must match n_samples")

    # ---- 基本属性 ----
    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_wavelengths(self) -> int:
        return self.X.shape[1]

    @property
    def step(self) -> float:
        """波长采样间隔（nm），用于把「样本点数偏移」换算成 nm。"""
        return float(np.median(np.diff(self.wavelengths)))

    def __len__(self) -> int:
        return self.n_samples

    # ---- 选择 / 变换 ----
    def select(self, mask) -> "SpectrumSet":
        """按布尔掩码或索引数组取子集（X / meta / y 同步）。"""
        mask = np.asarray(mask)
        meta = self.meta.iloc[mask] if self.meta is not None else None
        y = self.y[mask] if self.y is not None else None
        return SpectrumSet(self.X[mask], self.wavelengths, meta, y, self.y_names)

    def clip_wavelengths(self, lo: float, hi: float) -> "SpectrumSet":
        keep = (self.wavelengths >= lo) & (self.wavelengths <= hi)
        return replace(self, X=self.X[:, keep], wavelengths=self.wavelengths[keep])

    def with_X(self, X: np.ndarray) -> "SpectrumSet":
        return replace(self, X=np.asarray(X, dtype=np.float64))

    def sort_by(self, column: str) -> "SpectrumSet":
        if self.meta is None or column not in self.meta.columns:
            raise KeyError(f"meta has no column {column!r}")
        order = np.argsort(self.meta[column].values)
        meta = self.meta.iloc[order].reset_index(drop=True)
        y = self.y[order] if self.y is not None else None
        return SpectrumSet(self.X[order], self.wavelengths, meta, y, self.y_names)

    def instrument_ids(self):
        if self.meta is None or "instrument_id" not in self.meta.columns:
            return []
        return sorted(set(self.meta["instrument_id"].astype(str)))

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        rng = f"{self.wavelengths.min():.0f}-{self.wavelengths.max():.0f}nm"
        return f"<SpectrumSet {self.n_samples}x{self.n_wavelengths} @ {rng}>"
