"""把 merge_data_xikeng.csv 整理成 ai-meta 示例数据。

用法：python Data/Example1/make_example1.py
输出到同目录：spectra.npy / wavelengths.npy / tn.npy / meta.json

- 光谱：wavelength_1 .. wavelength_611（非消化，190..800 nm，1 nm 步进，共 611 点）
- 标签：TN 列
- 源 CSV 中另有 wavelength_digested_1..611（消化后光谱），本示例未采用；
  如要用消化光谱预测 TN，改下面的 SPECTRA_PREFIX 即可。
"""
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path

SRC = r"D:\Enveda CASMI 2026\ai-meta-main\input_data\merge_data_xikeng.csv"
OUT = Path(__file__).resolve().parent
SPECTRA_PREFIX = "wavelength_"   # 非消化；消化光谱请用 "wavelength_digested_"

df = pd.read_csv(SRC)
print("csv shape:", df.shape)

wl_cols = [c for c in df.columns if re.fullmatch(rf"{SPECTRA_PREFIX}\d+", c)]
wl_cols.sort(key=lambda c: int(c.split("_")[-1]))
assert len(wl_cols) == 611, f"期望 611 个光谱列，实际 {len(wl_cols)}"
print("n wavelength cols:", len(wl_cols))

X = df[wl_cols].to_numpy(dtype=np.float64)
wl = np.arange(190, 801, dtype=np.float64)   # 190..800 nm，1 nm 步进，611 点
assert wl.shape[0] == X.shape[1], (wl.shape[0], X.shape[1])

tn = df["TN"].to_numpy(dtype=np.float64)

print("X", X.shape, "NaN in X:", int(np.isnan(X).sum()))
print("TN", tn.shape, "NaN in TN:", int(np.isnan(tn).sum()))

np.save(OUT / "spectra.npy", X)
np.save(OUT / "wavelengths.npy", wl)
np.save(OUT / "tn.npy", tn)

meta = {
    "source": SRC,
    "source_rows": int(df.shape[0]),
    "n_wavelengths": int(X.shape[1]),
    "wavelength_range_nm": [190, 800],
    "wavelength_step_nm": 1,
    "spectra_columns": f"{SPECTRA_PREFIX}1 .. {SPECTRA_PREFIX}611",
    "target": "TN",
    "nan_in_TN": int(np.isnan(tn).sum()),
    "note": "TN 缺失的 9 行需在训练前剔除（~np.isnan(tn) 掩码）。",
}
(OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
print("saved to", OUT)
