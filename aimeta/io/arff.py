"""ARFF 光谱读取（工控机侧格式）→ SpectrumSet。

自动识别「哪些属性是波长」，输出带显式波长轴的 SpectrumSet。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.io import arff

from ..core.spectra import SpectrumSet

_NUM = re.compile(r"^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$")


def _is_wavelength(name: str) -> bool:
    """判断属性名是否为波长（纯数字，如 200.0 / 205）。"""
    return bool(_NUM.match(str(name).strip()))


def read_arff(path: str | Path, instrument_id: Optional[str] = None,
              timestamp_col: Optional[str] = None) -> SpectrumSet:
    """读取 ARFF 文件并构造 SpectrumSet。

    数值型且名字是纯数字的属性被视为波长；其余列进 meta。
    """
    path = Path(path)
    data, meta = arff.loadarff(str(path))
    df = pd.DataFrame(data)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v.decode() if isinstance(v, bytes) else v)

    wl_cols = [c for c in df.columns if _is_wavelength(c)]
    if not wl_cols:
        raise ValueError(f"no wavelength-like columns found in {path.name}")
    wl_cols = sorted(wl_cols, key=lambda c: float(c))
    X = df[wl_cols].to_numpy(dtype=np.float64)
    wavelengths = np.array([float(c) for c in wl_cols], dtype=np.float64)

    rest = df[[c for c in df.columns if c not in wl_cols]].copy()
    if instrument_id is not None:
        rest["instrument_id"] = instrument_id
    if timestamp_col is not None and timestamp_col in rest.columns:
        rest["timestamp"] = pd.to_datetime(rest[timestamp_col], errors="coerce")
    return SpectrumSet(X=X, wavelengths=wavelengths, meta=rest)
