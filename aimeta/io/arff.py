"""ARFF 光谱读取（工控机侧格式）→ SpectrumSet。

工控机导出的标准 ARFF 用 ``wavelength190`` / ``wavelength_200.5`` 这类前缀命名每条
波长通道，波长（nm）直接写在属性名里，无需按列序或仪器参数猜测。解析器据此提取
波长轴；其余列（ID / 浓度标签等）进 meta。
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
# 工控机标准 ARFF：wavelength190 / wavelength_200.5（大小写不敏感）
_WL = re.compile(r"^wavelength\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$", re.IGNORECASE)


def _wavelength_value(name: str) -> Optional[float]:
    """从 ARFF 属性名解析波长值（nm）；非波长列返回 ``None``。

    支持两种命名：
      - 显式波长前缀：``wavelength190`` / ``wavelength_200.5``（工控机标准格式）
      - 纯数字列名：``205`` / ``200.0``（兼容旧格式）
    """
    s = str(name).strip()
    m = _WL.match(s)
    if m:
        return float(m.group(1))
    if _NUM.match(s):
        return float(s)
    return None


def _is_wavelength(name: str) -> bool:
    """属性名是否代表一条波长通道。"""
    return _wavelength_value(name) is not None


def read_arff(path: str | Path, instrument_id: Optional[str] = None,
              timestamp_col: Optional[str] = None) -> SpectrumSet:
    """读取 ARFF 文件并构造 SpectrumSet。

    工控机标准 ARFF 用 ``wavelengthXXX`` 前缀命名波长通道，波长直接从列名解析；
    其余列（ID / 浓度标签等）进 meta。
    """
    path = Path(path)
    data, meta = arff.loadarff(str(path))
    df = pd.DataFrame(data)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v.decode() if isinstance(v, bytes) else v)

    wl = [(c, _wavelength_value(c)) for c in df.columns
          if _wavelength_value(c) is not None]
    if not wl:
        raise ValueError(f"no wavelength-like columns found in {path.name}")
    wl.sort(key=lambda t: t[1])
    wl_cols = [c for c, _ in wl]
    X = df[wl_cols].to_numpy(dtype=np.float64)
    wavelengths = np.array([v for _, v in wl], dtype=np.float64)

    rest = df[[c for c in df.columns if c not in wl_cols]].copy()
    if instrument_id is not None:
        rest["instrument_id"] = instrument_id
    if timestamp_col is not None and timestamp_col in rest.columns:
        rest["timestamp"] = pd.to_datetime(rest[timestamp_col], errors="coerce")
    return SpectrumSet(X=X, wavelengths=wavelengths, meta=rest)
