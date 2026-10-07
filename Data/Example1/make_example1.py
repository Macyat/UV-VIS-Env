"""通用光谱表读取器：把含「光谱列 + 采样日列 + 标签列」的 CSV 整理成 ai-meta 数据。

不绑定某一张表——源文件、光谱列前缀、波长起止、采样日列、标签列都可命令行指定。

示例：
    python make_example1.py path/to/data.csv \
        --spectra-prefix wavelength_ --wl-start 190 --wl-step 1 \
        --day-idx DAY_idx --labels "TN,AN,TP,COD,CODMn=KMNO,DO1,TUR1,DO2,TUR2"

输出到 --out（默认：源 CSV 同目录）：
    spectra.npy / wavelengths.npy / spectra_digest.npy / [day_idx.npy] / [labels.npy] / meta.json

规则：
- 光谱列：匹配「<spectra-prefix><整数>」的列（如 wavelength_1），按整数升序排列，
  波长 = wl_start + (整数 - 1) * wl_step。
- 消化光谱：--digested-prefix 指定其列前缀（默认 wavelength_digested_）；匹配到列则
  输出 spectra_digest.npy，匹配不到则输出与主光谱同形状的全 NaN 占位。
- 采样日列：--day-idx 指定的列；该列不存在则跳过（不导出 day_idx.npy）。
- 标签列：--labels 逗号分隔；「输出名=源列名」可重命名（如 CODMn=KMNO），
  未写「=」则输出名与源列同名；省略 --labels 则不导出 labels.npy。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _match_spectra_cols(columns: Sequence[str], prefix: str) -> List[Tuple[str, int]]:
    """返回 [(列名, 整数序号)]，按整数升序排列。"""
    pat = re.compile(rf"{re.escape(prefix)}(\d+)")
    found = []
    for c in columns:
        m = pat.fullmatch(c)
        if m:
            found.append((c, int(m.group(1))))
    if not found:
        raise ValueError(f"未找到以 {prefix!r} 开头且带整数序号的光谱列")
    found.sort(key=lambda t: t[1])
    return found


def _parse_labels(specs: Sequence[str]) -> List[Tuple[str, str]]:
    """'OUT=SRC' -> (OUT, SRC)；'NAME' -> (NAME, NAME)。"""
    pairs = []
    for s in specs:
        s = s.strip()
        if not s:
            continue
        if "=" in s:
            out, src = s.split("=", 1)
            pairs.append((out.strip(), src.strip()))
        else:
            pairs.append((s, s))
    return pairs


def prepare_dataset(
    csv_path: str,
    out_dir: str,
    spectra_prefix: str = "wavelength_",
    wavelength_start: float = 190.0,
    wavelength_step: float = 1.0,
    day_idx_col: Optional[str] = "DAY_idx",
    labels: Optional[Sequence[str]] = None,
    digested_prefix: Optional[str] = "wavelength_digested_",
) -> dict:
    """读取光谱表，输出 npy + meta.json，返回 meta 字典。"""
    csv_path = str(csv_path)
    df = pd.read_csv(csv_path)

    # 光谱
    wl_cols = _match_spectra_cols(df.columns, spectra_prefix)
    wl_nums = np.array([n for _, n in wl_cols], dtype=np.float64)
    X = df[[c for c, _ in wl_cols]].to_numpy(dtype=np.float64)
    wl = wavelength_start + (wl_nums - 1.0) * wavelength_step

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "spectra.npy", X)
    np.save(out / "wavelengths.npy", wl)

    # 采样日
    day = None
    if day_idx_col:
        if day_idx_col in df.columns:
            day = df[day_idx_col].to_numpy(dtype=np.int64)
            np.save(out / "day_idx.npy", day)
        else:
            print(f"[skip] 采样日列 {day_idx_col!r} 不存在，不导出 day_idx.npy")

    # 标签
    label_names: List[str] = []
    lab = None
    if labels:
        pairs = _parse_labels(labels)
        label_names = [o for o, _ in pairs]
        src_names = [s for _, s in pairs]
        missing = [s for s in src_names if s not in df.columns]
        if missing:
            raise ValueError(f"标签源列不存在：{missing}")
        lab = df[src_names].to_numpy(dtype=np.float64)
        np.save(out / "labels.npy", lab)

    # 消化光谱（可选；源表无该前缀列时输出与主光谱同形状的全 NaN 占位）
    Xd = None
    digest_wl_file = "wavelengths.npy"
    if digested_prefix:
        try:
            d_cols = _match_spectra_cols(df.columns, digested_prefix)
            d_nums = np.array([n for _, n in d_cols], dtype=np.float64)
            Xd = df[[c for c, _ in d_cols]].to_numpy(dtype=np.float64)
            wl_d = wavelength_start + (d_nums - 1.0) * wavelength_step
        except ValueError:
            Xd = np.full(X.shape, np.nan)
            wl_d = wl
        np.save(out / "spectra_digest.npy", Xd)
        if not np.array_equal(wl_d, wl):
            np.save(out / "wavelengths_digest.npy", wl_d)
            digest_wl_file = "wavelengths_digest.npy"

    meta = {
        "source": csv_path,
        "source_rows": int(df.shape[0]),
        "n_wavelengths": int(X.shape[1]),
        "wavelength_range_nm": [float(wl.min()), float(wl.max())],
        "wavelength_step_nm": float(wavelength_step),
        "spectra_prefix": spectra_prefix,
        "nan_in_spectra": int(np.isnan(X).sum()),
    }
    if lab is not None:
        meta["labels"] = {
            "columns": label_names,
            "shape": list(lab.shape),
            "nan_per_column": {c: int(np.isnan(lab[:, i]).sum())
                               for i, c in enumerate(label_names)},
        }
    if day is not None:
        meta["day_idx"] = {
            "source_column": day_idx_col,
            "unique_days": int(np.unique(day).size),
            "range": [int(day.min()), int(day.max())],
            "nan": int(np.isnan(day).sum()),
        }
    if Xd is not None:
        meta["digested"] = {
            "spectra_file": "spectra_digest.npy",
            "wavelengths_file": digest_wl_file,
            "prefix": digested_prefix,
            "shape": list(Xd.shape),
            "all_nan": bool(np.isnan(Xd).all()),
        }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print("source:", csv_path)
    print("spectra:", X.shape, "wl", wl.min(), "-", wl.max(), "nm")
    if day is not None:
        print("day_idx:", day.shape, "unique days:", int(np.unique(day).size))
    if lab is not None:
        print("labels:", lab.shape, "columns:", label_names)
        print("labels NaN per col:",
              {c: int(np.isnan(lab[:, i]).sum()) for i, c in enumerate(label_names)})
    if Xd is not None:
        print("digested:", Xd.shape, "all_nan=" + str(bool(np.isnan(Xd).all())))
    print("saved to", out)
    return meta


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="通用光谱表 → ai-meta 数据（npy + meta.json）")
    p.add_argument("csv", help="源 CSV 路径")
    p.add_argument("-o", "--out", default=None, help="输出目录（默认：源 CSV 同目录）")
    p.add_argument("--spectra-prefix", default="wavelength_", help="光谱列前缀")
    p.add_argument("--wl-start", type=float, default=190.0, help="起始波长 nm")
    p.add_argument("--wl-step", type=float, default=1.0, help="波长步长 nm")
    p.add_argument("--day-idx", default="DAY_idx", help="采样日列名（列不存在则跳过）")
    p.add_argument("--labels", default=None,
                   help="逗号分隔的标签列，支持 输出名=源列名（如 CODMn=KMNO）")
    p.add_argument("--digested-prefix", default="wavelength_digested_",
                   help="消化光谱列前缀（默认 wavelength_digested_；空串则不输出）")
    args = p.parse_args(argv)

    out_dir = args.out if args.out else str(Path(args.csv).resolve().parent)
    labels = args.labels.split(",") if args.labels else None
    digested_prefix = args.digested_prefix or None
    prepare_dataset(
        csv_path=args.csv,
        out_dir=out_dir,
        spectra_prefix=args.spectra_prefix,
        wavelength_start=args.wl_start,
        wavelength_step=args.wl_step,
        day_idx_col=args.day_idx,
        labels=labels,
        digested_prefix=digested_prefix,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
