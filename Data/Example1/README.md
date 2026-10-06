# Example1 — 西坑站示例数据

来源：`ai-meta-main/input_data/merge_data_xikeng.csv`（437 行 × 1237 列）。

## 提取方式

由 `make_example1.py` 从源 CSV 整理得到（可重跑）：

| 文件 | 内容 | 形状 |
|---|---|---|
| `spectra.npy` | 吸光度矩阵，取自 `wavelength_1 .. wavelength_611`（非消化光谱） | (437, 611) |
| `wavelengths.npy` | 波长轴，190–800 nm，1 nm 步进，共 611 点 | (611,) |
| `tn.npy` | 标签，TN 列 | (437,) |

`meta.json` 记录了来源、行列数、波长范围与 TN 缺失数。

## 注意事项

- **TN 有 9 个缺失值**，训练前需剔除：`mask = ~np.isnan(tn); X, tn = X[mask], tn[mask]`。
- 源 CSV 还包含 `wavelength_digested_1..611`（消化后光谱）。TN 的物理测定通常
  需过硫酸盐消解后再测紫外吸收，因此若要用光谱预测 TN，**消化后光谱可能更合适**。
  本示例默认使用非消化光谱；要切换，把 `make_example1.py` 里的
  `SPECTRA_PREFIX` 改为 `"wavelength_digested_"` 重跑即可。
- 该数据集仅作库的上手 / 冒烟示例，不代表建模精度达标。
