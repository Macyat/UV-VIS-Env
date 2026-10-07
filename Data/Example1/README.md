# Example1 — 西坑站示例数据

来源：`ai-meta-main/input_data/merge_data_xikeng.csv`（437 行 × 1237 列）。

## 提取方式

由通用脚本 `make_example1.py` 从源 CSV 整理得到（可重跑，换表只改参数）：

```bash
python Data/Example1/make_example1.py "D:\Enveda CASMI 2026\ai-meta-main\input_data\merge_data_xikeng.csv" -o Data/Example1 --spectra-prefix wavelength_ --wl-start 190 --wl-step 1 --day-idx DAY_idx --labels "TN,AN,TP,COD,CODMn=KMNO,DO1,TUR1,DO2,TUR2"
```

通用参数：

| 参数 | 作用 | 本例取值 |
|---|---|---|
| `csv` | 源 CSV 路径 | `merge_data_xikeng.csv` |
| `-o/--out` | 输出目录（默认源 CSV 同目录） | `Data/Example1` |
| `--spectra-prefix` | 光谱列前缀（自动匹配 `<前缀><整数>`） | `wavelength_` |
| `--wl-start` / `--wl-step` | 起始波长 / 步长（nm） | `190` / `1` |
| `--day-idx` | 采样日列名（列不存在则跳过导出） | `DAY_idx` |
| `--labels` | 逗号分隔标签列，`输出名=源列名` 可重命名 | 见上 |
| `--digested-prefix` | 消化光谱列前缀（匹配不到则输出全 NaN 占位） | `wavelength_digested_` |

| 文件 | 内容 | 形状 |
|---|---|---|
| `spectra.npy` | 吸光度矩阵，取自 `wavelength_1 .. wavelength_611`（非消化光谱） | (437, 611) |
| `wavelengths.npy` | 波长轴，190–800 nm，1 nm 步进，共 611 点 | (611,) |
| `spectra_digest.npy` | 消化后光谱（`wavelength_digested_1..611`；本表该列**全为 NaN**） | (437, 611) |
| `day_idx.npy` | 采样日索引，取自源 CSV 的 `DAY_idx` 列（1–81，共 81 天，无缺失） | (437,) int64 |
| `labels.npy` | 全部水质标签（含 TN），列序见下表 | (437, 9) |

`labels.npy` 列序（第 0 列起）：

| 列 | 名称 | 缺失 |
|---|---|---|
| 0 | TN | 9 |
| 1 | AN | 1 |
| 2 | TP | 14 |
| 3 | COD | 437（全空） |
| 4 | CODMn（源列 `KMNO`） | 0 |
| 5 | DO1 | 437（全空） |
| 6 | TUR1 | 0 |
| 7 | DO2 | 437（全空） |
| 8 | TUR2 | 437（全空） |

`meta.json` 记录了来源、行列数、波长范围，以及 `labels` 各列名与缺失统计。

### 加载示例

```python
import numpy as np
X     = np.load("spectra.npy")      # (437, 611)
wl    = np.load("wavelengths.npy")  # (611,)
day   = np.load("day_idx.npy")      # (437,) 采样日，供按日 CV 使用
lab   = np.load("labels.npy")       # (437, 9) 列序 [TN, AN, TP, COD, CODMn, DO1, TUR1, DO2, TUR2]

TN    = lab[:, 0]   # 缺 9 行
AN    = lab[:, 1]   # 缺 1 行
TP    = lab[:, 2]   # 缺 14 行
COD   = lab[:, 3]   # 全空
CODMn = lab[:, 4]   # 完整
DO1   = lab[:, 5]   # 全空
TUR1  = lab[:, 6]   # 完整
DO2   = lab[:, 7]   # 全空
TUR2  = lab[:, 8]   # 全空
```

## 注意事项

- 训练某个标签前，用该列缺失掩码剔除：
  `mask = ~np.isnan(col); X2, y2 = X[mask], col[mask]`。
- **DO1、COD、DO2、TUR2 在本源 CSV 中全部为 NaN（437/437）**，暂不可用于建模；
  TN 缺 9、AN 缺 1、TP 缺 14；TUR1、CODMn 完整。
- 源 CSV 中 `KMNO` 列对外命名为 `CODMn`（高锰酸盐指数，同一指标常见别名）。
- 源 CSV 还包含 `wavelength_digested_1..611`（消化后光谱）。TN 的物理测定通常
  需过硫酸盐消解后再测紫外吸收，因此若要用光谱预测 TN，**消化后光谱可能更合适**。
  本示例默认使用非消化光谱；要切换，把命令里的 `--spectra-prefix` 改为
  `wavelength_digested_` 重跑即可。
- 该数据集仅作库的上手 / 冒烟示例，不代表建模精度达标。
