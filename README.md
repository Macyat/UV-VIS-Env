# UV-VIS-Env

UV-Vis 光谱水质在线监测算法库。覆盖从硬件评价、光谱预处理、建模、波长选择、仪器间模型迁移，
到计量验收与在线监控的完整链路，可同时用于实验室研究、离线建模与工控机在线部署。

支持的水质参数：CODMn、COD、TN、TP、氨氮（AN）、浊度（TUR）。

> **术语约定**：本库大量使用光谱计量与国标专有名词（FOM、空白、死限、复核限、达标率、日级 R² 等）。
> 阅读代码或配置前，请先浏览 [`docs/术语表.md`](docs/术语表.md)，避免口径混淆。
> 质量评价分两层——**分析方法品质因数（FOM）**与**光谱仪硬件评价**，详见「质量评价：分析方法与仪器硬件」一节。

---

## 安装

```bash
cd ai-meta
pip install -e .
```

可选依赖（按需安装）：

```bash
pip install lightgbm torch        # LGBM / 神经网络等扩展模型
pip install pytest                # 运行测试
```

环境要求：Python >= 3.10，核心依赖为 numpy / scipy / scikit-learn / pandas / PyWavelets / PyYAML / matplotlib。

---

## 5 分钟上手

```python
import numpy as np
from aimeta.core.spectra import SpectrumSet
from aimeta.io.config import load_params, load_chain
from aimeta.pipelines.train import train_model
from aimeta.pipelines.infer import predict

# 1. 准备数据：吸光度矩阵 + 波长轴 + 标签
#    示例数据见 Data/Example1/（西坑站：437 样本 × 611 波长 190–800 nm，目标 TN）
X = np.load("Data/Example1/spectra.npy")        # (n_samples, n_wavelengths)
wl = np.load("Data/Example1/wavelengths.npy")   # (n_wavelengths,) = 190..800 nm
tn = np.load("Data/Example1/tn.npy")            # (n_samples,)
mask = ~np.isnan(tn)                            # 剔除参考值缺失的 9 个样本
X, tn = X[mask], tn[mask]
spectra = SpectrumSet(X=X, wavelengths=wl, y=tn)

# 2. 取参数定义与预处理链（configs/）
params = load_params()
chain = load_chain(name="preprocessing/dayu_edge.yaml")

# 3. 训练
#    方式 A（推荐）：不传 n_components，按 PLS_toolbox routine CV 遍历 1..20 取误差最小的维数
card = train_model(
    spectra, label="TN", model_key="pls",
    chain=chain,
    param_def=params["TN"],
    instrument_id="ai14",
)
#    方式 B（手动指定）：显式给定维数，以手填为准（覆盖自动选维）
#    card = train_model(
#        spectra, label="TN", model_key="pls",
#        chain=chain, model_params={"n_components": 8},
#        param_def=params["TN"], instrument_id="ai14",
#    )
print(card)          # <ModelCard TN/pls @ai14 ... n_components=11 rmse_cv=0.11>

# 4. 推理（自动套用卡内记录的预处理链）
pred = predict(card, X)
```
备注：此例子仅为简易建模示例，未对按数据分布对数据进行选取，亦未含验证集。鲁棒建模需要避免仅选取高相似数据，例如实验室样本需要考虑样本代表性（e.g.SPXY），河流样本需要按时间序列处理方式交叉验证（Walk-forward validation：rolling/expanding）。

---

## 目录结构

```
ai-meta/
├── aimeta/
│   ├── core/            数据契约、注册表、模型产物（ModelCard）
│   ├── preprocessing/   预处理算子与链（SG、DERIV、小波、SNV、MSC、缩放）
│   ├── models/          模型库（线性 / GLM / 潜变量 / 核 / 树 / 神经网络）+ MCR 曲线分辨
│   ├── selection/       波长选择（CARS、稳定性与置换检验）
│   ├── transfer/        仪器间迁移（DS / PDS / SBC / GLSW）+ 硬件差异诊断（波长漂移 / 增益）
│   ├── stats/           假设检验、置信区间、容忍区间
│   ├── metrics/         分析方法品质因数（FOM：LOD / LOQ / 灵敏度 / 选择性）+ GB3838 类别判定
│   ├── hardware_eval.py 光谱仪硬件评价 / 出厂检验（暗噪声 / 基线 / 波长准确度 / 光度准确度 / 杂散光 / SNR / 分辨率）
│   ├── monitoring/      控制图（Shewhart / CUSUM / EWMA）与 MSPC
│   ├── viz/             可视化
│   ├── io/              配置加载、ARFF 读取
│   ├── pipelines/       训练 / 模型筛选 / 推理
│   └── cli.py           命令行入口
├── configs/             参数定义、预处理链、仪器档案
├── edge/                工控机运行时（依赖 numpy / scipy / PyWavelets）
├── tests/               单元测试
└── docs/                设计文档
```

---

## 模块说明

| 模块 | 用途 | 主要接口 |
|---|---|---|
| `core` | 数据契约与模型产物 | `SpectrumSet`、`ModelCard`、`Registry` |
| `preprocessing` | 光谱预处理 | `Pipeline.from_config`、`DERIV`、`snv_transform` |
| `models` | 建模 | `build_model`、`list_models`、`WaterQualityModel`、`MCRALS` |
| `selection` | 波长选择 | `CARS`、`permutation_test`、`selection_stability` |
| `transfer` | 跨仪器迁移 / 硬件差异诊断 | `PiecewiseDirectStandardization`、`DirectStandardization`、`recommend_method`、`estimate_wavelength_shift` |
| `stats` | 统计检验 | `consistency_report`、`tolerance_interval`、`confidence_interval_mean` |
| `metrics` | 分析方法品质因数（FOM）+ 计量验收 | `figures_of_merit`、`acceptance_report`、`daily_r2`、`alarm_accuracy` |
| `hardware_eval` | 光谱仪硬件评价 / 出厂检验 | `evaluate_instrument`、`wavelength_accuracy`、`photometric_accuracy`、`stray_light`、`resolution`、`dark_noise` |
| `monitoring` | 在线监控 | `MSPC`、`shewhart_limits`、`cusum`、`ewma` |
| `viz` | 可视化 | `plots.plot_spectra`、`plots.plot_pred_vs_actual`、`save_fig` |
| `io` | 配置与数据 | `load_params`、`load_chain`、`load_instrument`、`read_arff` |
| `pipelines` | 流程编排 | `train_model`、`sweep`、`predict` |

---

## 用法

### 数据契约

所有流程以 `SpectrumSet` 为单位，强制携带波长轴，避免按列号取数带来的错位。

```python
import pandas as pd
from aimeta.core.spectra import SpectrumSet

spectra = SpectrumSet(
    X=X,
    wavelengths=wl,
    meta=pd.DataFrame({"instrument_id": ["ai14"] * len(X), "site": ["xikeng"] * len(X)}),
    y=y,
)
spectra.clip_wavelengths(220, 700)      # 截取波段
spectra.sort_by("timestamp")            # 按时间排序
```

### 预处理链

链用配置描述，可被序列化回配置，训练与部署共用同一份定义。

```python
from aimeta.preprocessing.base import Pipeline

pipe = Pipeline.from_config([
    {"op": "savgol", "window": 15, "polyorder": 3},
    {"op": "snv"},
])
Xt = pipe.fit_transform(X)
pipe.to_config()        # [{'op': 'savgol', 'window': 15, 'polyorder': 3, 'deriv': 0}, {'op': 'snv'}]
```

可用算子：

| 算子 | 说明 | 可部署到工控机 |
|---|---|---|
| `savgol` | Savitzky-Golay 平滑 / 求导 | 是 |
| `deriv_gram` | Gram 多项式平滑 / 求导（与 `savgol` 数值等价） | 是 |
| `snv` | 标准正态变换（逐条光谱） | 是 |
| `msc` | 多元散射校正 | 是 |
| `mean_center` / `column_scale` | 列中心化 / 列标准化 | 是 |
| `wavelet` | 小波软阈值去噪（`sym4`） | 否 |

### 建模与模型筛选

```python
from aimeta.models.registry import list_models
from aimeta.pipelines.train import sweep

print(list_models())               # ['ols', 'ridge', 'lasso', 'pls', 'gpr', 'lgbm', ...]

cards = sweep(spectra, "TN",
              model_keys=("pls", "ridge", "lasso", "gpr"),
              chain=chain,
              param_def=params["TN"],
              folds=5)
best = cards[0]                    # 已按 rmse_cv 升序排列
```

新增模型只需注册，无需改动其他文件：

```python
from aimeta.core.registry import MODELS

@MODELS.register("my_model", family="linear")
class MyModel:
    ...
```

### 候选池、打分表与 TOP K 融合

不止单链 `sweep`：跨「预处理链 × 模型」全量枚举候选池，按验收指标全集打分排名，
挑 TOP K 做精度加权（1/RMSE²）融合，并给每个候选导出一套诊断图。

```python
from aimeta.pipelines.train import sweep_grid
from aimeta.pipelines.scoring import score_cards, select_top_k
from aimeta.models.ensembles import fuse_predict
from aimeta.viz.report import export_top_k_report

# 1) 候选池：2 条链 × 4 个模型 = 8 个候选
cards = sweep_grid(spectra, "TN",
                   model_keys=("pls", "ridge", "lasso", "ols"),
                   chains=[[{"op": "snv"}],
                           [{"op": "savgol", "window": 11, "polyorder": 3}]],
                   param_def=params["TN"], folds=5)

# 2) 打分表（验收指标全集）+ 排名，写出 scoring.csv
ranked = score_cards(cards, X_val, y_val, params["TN"], day_idx)

# 3) TOP K 候选（按打分 rank）
top = select_top_k(cards, 3, X_val, y_val, params["TN"], day_idx)

# 4) 精度加权融合：w_i = 1/RMSE_CV_i²（任一张 rmse_cv 缺失/≤0 自动退化为等权）
y_fused, info = fuse_predict(top, X_new, scheme="inv_rmse2", param=params["TN"])
print(info["scheme"], info["weights"])

# 5) 每个候选一套图 + 打分表 CSV，落到 out_dir/figures 与 out_dir/scoring.csv
export_top_k_report(cards, X_val, y_val, params["TN"], out_dir, k=3, day_idx=day_idx)
```

### 仪器间模型迁移

把 A 机器上的模型用到 B 机器，避免每台设备重新训练。
前提是一批**在两台仪器上都测过**的样本（可用同一套标液获得）。

```python
from aimeta.transfer.standardization import (
    PiecewiseDirectStandardization, kennard_stone,
)
from aimeta.transfer.diagnose import recommend_method

# 先看差在哪，再决定用什么方法
print(recommend_method(wl, X_slave, X_master, n_std_samples=len(X_slave)))
# {'method': 'pds', 'window': 5, 'shift_nm': 0.0, 'reason': '样本 40 < 波长数 251，用窗口回归的 PDS'}

idx = kennard_stone(X_slave, 40)                       # 挑代表性样本做标准化
pds = PiecewiseDirectStandardization(window=5).fit(X_slave[idx], X_master[idx])
X_slave_corr = pds.transform(X_slave)                  # 之后即可套用主仪器模型
```

| 方法 | 需要样本量 | 适用情形 |
|---|---|---|
| `SlopeBiasCorrection` | >= 2 | 只有两台机的预测值，无配对光谱 |
| `PiecewiseDirectStandardization` | >= 5 | 通用；能处理随波长变化的差异 |
| `DirectStandardization` | >= 波长数 | 差异是全局线性的，参数更省 |
| `GLSW` | >= 5 | 只想压掉仪器差异子空间 |
| `MeanVarianceAlign` | >= 2 | 极简兜底 |

备注：大多数测河水场景不具备迁移可行性，不同河流配置不同设备，河流基体存在差异，此场景需要同时迁移测量系统和被测系统，一维光谱无法提供足够信息表征。

### 计量验收

```python
from aimeta.metrics.figures_of_merit import figures_of_merit
from aimeta.metrics.water_standards import acceptance_report
from aimeta.preprocessing.base import Pipeline

# 注意：指标要在「模型的输入空间」上算，即空白样本也须先过预处理链
pipe = Pipeline.from_config(chain).fit(X_train)
X_train_p = pipe.transform(X_train)
X_blank_p = pipe.transform(X_blank)

metric = figures_of_merit(card.estimator, X_blank_p, X_cal=X_train_p)
print(metric["SEN"], metric["LOD"], metric["LOQ"])

# 按 GB3838 类别口径出验收结论
rep = acceptance_report(y_true, y_pred, params["TN"], day_idx=days)
print(rep["rmse"], rep["acceptance_rate"], rep["daily_r2"])
```

### 在线监控

```python
from aimeta.monitoring import MSPC, shewhart_limits, cusum

m = MSPC(n_components=5).fit(X_normal)      # 用正常时期的光谱建模
r = m.monitor(X_new)
r["spe_alarm"]      # 是否出现异常（探头污染、气泡、异物常体现在这里）
r["contribution"]   # 各波长对异常的贡献，用于定位问题波段

cl, lcl, ucl = shewhart_limits(pred_series)  # 预测值序列的控制限（含自相关修正）
cusum(pred_series, k=0.5, h=5.0)             # 慢漂移检测
```

### 曲线分辨（MCR）

从混合光谱中分离出各组分的浓度剖面与光谱：

```python
from aimeta.models.curve_resolution import MCRALS

mcr = MCRALS(n_components=3, non_negative=True, closure=True).fit(X)
mcr.components_          # (k, p) 各组分光谱
mcr.concentrations_      # (n, k) 各组分浓度剖面
mcr.transform(X_new)     # 新样本的浓度剖面
mcr.ambiguity_estimate(X)  # 0~1，越大说明解的不确定性越高
```

### 波长选择

```python
from aimeta.selection import CARS, permutation_test

sel = CARS(n_iter=30, n_components=10).fit(X, y)
X_sel = sel.transform(X)

# 检验选出的波长是不是碰巧（打乱标签重跑，比较分数分布）
res = permutation_test(lambda: CARS(n_iter=10, n_components=5), score_fn, X, y)
print(res["score"], res["p_value"])
```

### 可视化

```python
from aimeta.viz import apply_theme, save_fig
from aimeta.viz import plots

apply_theme()
fig = plots.plot_pred_vs_actual(y_true, y_pred, label="TN", unit="mg/L",
                                ranges=params["TN"].ranges)
save_fig(fig, "figs/TN_pred.png")
```

---

## 命令行

```bash
python -m aimeta.cli list-models [--family latent]   # 列出可用模型
python -m aimeta.cli list-ops                        # 列出预处理算子及是否可部署
python -m aimeta.cli smoke                           # 端到端自检（合成数据）
python -m aimeta.cli transfer --slave ai17           # 查看某台仪器的迁移配置
```

`smoke` 输出示例：

```
[train] <ModelCard AN/pls @ai14 fp=a1bd620a4660a3ac rmse_cv=0.3296>
[infer] rmse = 0.24679
[edge ] parity=OK max_diff=4.00e-14
[mspc ] 异常样本 SPE=0.1012 vs 限=0.0001 -> 报警
```

---

## 配置

| 文件 | 内容 |
|---|---|
| `configs/params.yaml` | 各水质参数的量程、类别分界、检出限、误差限 |
| `configs/preprocessing/dayu_v1.yaml` | 训练用链：`savgol → wavelet → snv` |
| `configs/preprocessing/dayu_edge.yaml` | 部署用链：`savgol → snv` |
| `configs/instruments/*.yaml` | 每台仪器的波长轴、光程、迁移参数 |

修改阈值、增减预处理步骤、登记新仪器，都只改 YAML，不动代码。

`params.yaml` 中的 `lower_bound` 取国标实验室方法的检出限；预测值低于它时
替换为 `lower_bound / 2` 作为替代值（避免把未检出一律报成检出限值本身而系统性
高估），并在 `standard` 字段标注依据标准：

| 参数 | 检出限 | 依据标准 |
|---|---|---|
| CODMn | 0.5 mg/L | GB 11892-89（测定范围下限） |
| COD | 4 mg/L | HJ 828-2017（取样 10.0 mL） |
| TN | 0.05 mg/L | HJ 636-2012（样品 10 mL） |
| TP | 0.01 mg/L | GB 11893-89（25 mL 试料） |
| 氨氮 | 0.025 mg/L | HJ 535-2009（50 mL、20 mm 比色皿） |
| 浊度 | 0.3 NTU | HJ 1075-2019 |

需要注意：这是**实验室参考方法**的检出限，不是光谱模型自身的检出限。
后者与仪器噪声、预处理和回归向量有关，通常远大于前者，
严格来讲应用 `metrics.figures_of_merit` 从数据计算（见下节「质量评价：分析方法与仪器硬件」）。

---

## 质量评价：分析方法与仪器硬件

本库把"好不好"拆成**两层**，请分别看待、不要混为一谈：

- **方法层（分析方法品质因数，FOM）**：评价你建的多元校正**模型**本身——灵不灵敏、能检多低、抗不抗干扰。
- **硬件层（光谱仪评价）**：评价**仪器**本身的状态与跨机一致性——波长准不准、增益稳不稳、噪声大不大。

评价顺序建议：**先确认硬件 / 跨机一致 → 再算方法 FOM**（硬件噪声是方法 `s_x` 的来源之一；波长漂移不修会直接污染建模）。

---

### 1. 分析方法品质因数（FOM）测定方案

#### 概念与口径
FOM（Figures of Merit，品质因数）评价**分析方法 / 校正模型**本身，走 Olivieri 净分析信号（NAS）多元口径，**不能**套用单变量 3σ/slope（会系统性低估 LOD）。

输出七项：`SEN`（灵敏度）、`γ`（分析灵敏度）、`s_x`（光谱噪声）、`s_0`（空白预测标准差）、`LOD`、`LOQ`、`SEL`（选择性）。

#### 需要的输入
- **已拟合模型**：来自校正集（多种浓度标液 / 水样建校正曲线）。
- **空白样本 `X_blank`**：与样品基质一致、不含被测物的溶剂（纯净水 / 去离子水）。
  同条件**独立重复 ≥10 组**（最少 ≥2，否则 `s_0` 静默退化为 0 → `LOD=0`）。
  必须**保留原始噪声、未被替为 LOD/2**，且**已过同一预处理链**。
- **`X_cal`（可选）**：校正集光谱，用于取工作点（默认取 `X_blank` 均值）。
- **`s_target`（可选）**：目标组分纯组分光谱，用于算 `SEL`。

#### 计算公式
| 量 | 公式 | 含义 |
|---|---|---|
| 灵敏度 `SEN` | `1 / ‖b‖` | `b`=回归向量（线性取 `coef_`，非线性数值微分）；越大越灵敏 |
| 分析灵敏度 `γ` | `SEN / s_x` | 单位浓度对应的信噪比 |
| 光谱噪声 `s_x` | 空白光谱去趋势残差标准差 | 含硬件 + 环境 + 预处理的总噪声 |
| 空白预测标准差 `s_0` | 空白预测值 std（ddof=1） | 空白预测散布 |
| 检出限 `LOD` | `3.3 · s_0 / SEN` | 多元检出限 |
| 定量限 `LOQ` | `10 · s_0 / SEN` | 多元定量限 |
| 选择性 `SEL` | `‖b ⊙ s_target‖ / ‖b‖` | 净信号占比，0~1，越近 1 越不受干扰 |

#### 测定步骤
1. 建校正模型（校正集覆盖量程）。
2. 采集空白：纯水 / 去离子水，同条件独立重复 ≥10 次，**保留噪声**。
3. 空白过**同一**预处理链 → `X_blank_p`。
4. 调用 `figures_of_merit(card.estimator, X_blank_p, X_cal=X_train_p)`。
5. 读取 `SEN / γ / LOD / LOQ / SEL`。

#### 坑与判据
- `n_blank` 必须 **≥2**（≥10 推荐）；`n_blank=1` 时 `s_0` 静默退化为 0，`LOD=0` 坏掉。
- 空白必须过预处理链且保留噪声；喂原始光谱或 `LOD/2` 替值 → 预测全相同 →
  `SEN/LOD` 退化 `inf`（代码 `figures_of_merit.py` 第 99–103 行告警）。
- `SEN` **尺度相关**（受 y 标准化、校正集浓度范围影响），仅用于候选模型间横向比较，
  不是绝对物理灵敏度。
- 跨天漂移别混进空白噪声（归 `drift.py`）。

#### 参考文献
- Lorber, A.; Faber, N. M.; Kowalski, B. R. *Net analyte signal calculation in multivariate calibration.* **Anal. Chem.** 1997, 69(9), 1620–1626.（净分析信号 NAS 框架，FOM 的多元基础）
- Olivieri, A. C.; Faber, N. M.; Ferre, J.; Boque, R.; Kalivas, J. H.; Mark, H. *Guidelines for calibration in analytical chemistry. Part 2. Multivariable calibration.* **Pure Appl. Chem.** 2006, 78(3), 633–664.（IUPAC 多元校正指南，含 FOM 定义）
- IUPAC. *Nomenclature in evaluation of analytical methods including detection and quantification capabilities (IUPAC Recommendations 1995).* **Pure Appl. Chem.** 1995, 67(10), 1699–1723.（LOD/LOQ 术语与 3.3 / 10 系数来源）
- ISO 11843 系列 *Capability of detection*（检出限 / 定量限能力评估，一元口径，多元在其上扩展）。
- IUPAC *Compendium of Chemical Terminology*（Gold Book）：sensitivity、limit of detection 等术语定义。
- 国标语境：GB 3838-2002《地表水环境质量标准》（水质类别与 V 类界）；HJ 915.3-2024（紫外吸收水质在线监测仪 第 3 部分）；各参数实验室方法见上表（GB 11892-89 / HJ 828-2017 / HJ 636-2012 / GB 11893-89 / HJ 535-2009 / HJ 1075-2019）——这些给的是**实验室参考方法**检出限，非模型 FOM。

---

### 2. 光谱仪硬件评价

#### 概念与口径
硬件评价针对**光谱仪本身**，与"分析方法 FOM"是两个层面。UV 水质在线监测里要盯的硬件维度：

- **波长准确度 / 漂移**：峰位是否偏移（直接影响 PLS 建模与跨机复用）。
- **光度 / 增益一致性**：逐波长增益、偏置是否稳定。
- **信噪比（SNR）、暗噪声、基线稳定性**：决定你实测到的 `s_x`（方法噪声里含硬件噪声）。
- **杂散光（stray light）**：抬高基线、压低吸光度上限。
- **分辨率**：能否分辨相邻吸收峰。

#### 本库当前实现：仪器间差异诊断
本库不是做逐项出厂检验，而是提供"两台机差在哪、该不该 / 怎么迁移"的**门控诊断**（`aimeta/transfer/diagnose.py`）：

- `estimate_wavelength_shift`：用平均光谱互相关估计 slave 相对 master 的波长偏移（nm）。
- `estimate_gain_offset`：逐波长增益 / 偏置（master ≈ gain·slave + offset）；增益随波长变化明显 → 差异是"逐波长"的，PDS 比 DS 合适。
- `residual_spectrum`：主从平均光谱之差，看差异是全局倾斜还是局部结构。
- `recommend_method`：根据偏移 / 样本数自动选 DS / PDS / SBC；偏移 > 0.5 nm 时先对齐波长轴。

#### 仪器档案（规格与策略）
`configs/instruments/*.yaml` 记录每台机的光学规格与迁移策略，是硬件评价的基线档案：

- `optics`：波长起止、步长、光程（path length）。
- `transfer`：标准化样本集、重校周期、回退方法。

#### 参考实现
```python
from aimeta.transfer.diagnose import (
    estimate_wavelength_shift, estimate_gain_offset,
    residual_spectrum, recommend_method,
)

shift = estimate_wavelength_shift(wl, X_slave, X_master)
gain = estimate_gain_offset(X_slave, X_master)["gain"]
diag = recommend_method(wl, X_slave, X_master, n_std_samples=len(X_slave))
print(diag["method"], diag["shift_nm"], diag["reason"])
```

#### 坑与判据
- 波长漂移 > 0.5 nm 必须先做波长轴对齐，否则 DS / PDS 在学一个错位映射。
- SNR、杂散光、分辨率等**出厂规格**请对照仪器 datasheet / 检定规程核验；本库不直接测这些，
  只从实测光谱**诊断跨机差异**。要做逐项出厂检验需另接标准物质与测试流程。

#### 参考文献
- 仪器间校准 / 迁移与差异诊断（DS / PDS / SBC / GLSW）：见 `aimeta/transfer/` 与 `docs/重构方案.md`。
- 紫外-可见分光光度计性能评价（波长准确度、光度准确度、杂散光、基线平直度、分辨率、噪声）：ASTM E275 系列；国内计量溯源见 **JJG 178《紫外、可见、近红外分光光度计》检定规程**。
- 检测实验室能力通用要求：**ISO/IEC 17025**（溯源与量值传递的整体框架）。

---

### 3. 光谱仪硬件出厂检验方案

#### 概念与口径
在把仪器搬去现场 / 复用模型之前，按紫外-可见分光光度计的计量检验口径做**逐项出厂检验**。
本库 `aimeta/hardware_eval.py` 提供从实测光谱**计算指标**的函数；标准物质 / 滤光片由使用方按检定规程准备。
主要依据：**JJG 178《紫外、可见、近红外分光光度计》检定规程**、**ASTM E275** 系列。

#### 检验项目与所需标准物质
| 项目 | 计算方法 | 所需标准物质 / 滤光片 | 典型合格判据 |
|---|---|---|---|
| 波长准确度 | 测已知峰位，比较实测峰位（`wavelength_accuracy`） | 钬玻璃 / 钬氧化物（279.4 / 287.5 / 333.7 / 360.9 / 418.5 / 453.2 / 536.2 / 637.5 nm） | 误差 ≤ 0.5 nm |
| 光度准确度 | 在已知吸光度点比较（`photometric_accuracy`） | 中性密度片 / 重铬酸钾标准溶液 | ≤ 0.002 A（或 0.3 %T） |
| 杂散光 | 截止滤光片完全吸收处测残余透射（`stray_light`） | NaI / Corning 截止滤光片（如 340 nm） | ≤ 0.05 %T |
| 暗噪声 | 遮光下多次重复 std（`dark_noise`） | 无（shutter 关闭） | ≤ 0.0005（示例） |
| 基线平直度 | 100%T 参考线重复性与偏离（`baseline_flatness`） | 空气 / 空白 | 重复性 ≤ 0.001（示例） |
| 信噪比 | 稳定光源重复测量的信号 / 噪声（`signal_to_noise`） | 稳定光源 / 纯水 | 越高越好 |
| 分辨率 | 窄发射线半高全宽（`resolution`） | 汞灯 / 氩灯 | FWHM ≤ 2 nm（1 nm 狭缝典型） |

#### 参考实现
```python
from aimeta.hardware_eval import evaluate_instrument

res = evaluate_instrument(
    wavelengths=wl,
    dark_repeats=dark,                 # (n, p) 遮光重复
    holmium_spectrum=holmium,           # 钬玻璃实测
    ref_peaks=[360.9, 418.5, 536.2],    # 选用峰
    transmittance=t_pct,                # %T 谱
    check_wl=[340.0],                   # 杂散光检查波长
    line_spectrum=mercury,              # 汞灯线
    line_peak_wl=546.1,
)
print(res.report())     # metrics / pass_criteria / passed
```

#### 坑与判据
- 波长准确度要选**足够尖锐且分离**的参考峰；峰位用抛物线细化（`find_peak_wavelength`）比直接取 argmax 更准。
- 杂散光必须在滤光片**完全截止**的波长处测；截止不彻底会低估。
- 暗噪声 / 基线平直度都要求**多次重复**（≥10 组），单次测量无意义。
- 这些硬件指标是**方法 `s_x`（FOM 里的光谱噪声）的来源之一**：硬件噪声大，方法的 LOD / LOQ 必然差。

#### 参考文献
- **JJG 178《紫外、可见、近红外分光光度计》检定规程**（国内计量溯源，含波长 / 光度 / 杂散光 / 基线 / 噪声 / 分辨率检验）。
- ASTM E275 系列 *Standard Practices for Describing and Measuring Performance of UV, Visible, and Near-IR Spectrophotometers*（波长准确度、光度准确度、杂散光、基线平直度、分辨率、噪声）。
- ISO/IEC 17025（检测实验室能力通用要求，量值传递整体框架）。

---

## 部署到工控机

`edge/` 是独立的极简运行时，部署端依赖与 aimeta 一致（numpy / scipy / PyWavelets）；纯 numpy 可编译算子构成的链仍可脱离 scipy/pywt 单独运行。
仅支持线性模型（PLS / Ridge / Lasso / OLS / PCR）与可编译算子。

```python
from edge.runtime import export_card, load_edge_model, verify_against_card

export_card(card, "deploy/TN_ai17", X_reference=X_train)   # 在开发机上导出
```

```python
import numpy as np
from edge.runtime import load_edge_model

m = load_edge_model("deploy/TN_ai17")    # 工控机侧
y = m.predict(np.load("new_spectra.npy"))
```

导出后建议校验一次一致性：

```python
ok, diff = verify_against_card(m, card, X_test)   # 应 <= 1e-8
```

---

## 测试

```bash
python -m pytest -q
```

| 文件 | 覆盖 |
|---|---|
| `test_registry.py` | 注册表的可插拔性 |
| `test_preprocessing.py` | 导数算子与 scipy 的数值等价、SNV 语义、链的可复现性 |
| `test_transfer.py` | 迁移方法在合成双仪器上的效果、样本选择、差异诊断 |
| `test_metrics.py` | 指标、类别判定、合格率、报警指标 |
| `test_mcr.py` | 曲线分辨的初值、收敛、约束与不确定性 |
| `test_smoke.py` | 端到端：训练 → 导出 → 一致性 → 监控 → 波长选择 |

---

## 注意事项

- **部署端依赖与 aimeta 一致（numpy / scipy / PyWavelets）**。
  纯 numpy 可编译算子（``savgol`` / ``snv`` / ``msc`` / 缩放）仍走零依赖快路径；
  ``wavelet`` 等需外部库的算子会委托注册算子执行，工控机需装好对应依赖
  （aimeta 主依赖已含 scipy / PyWavelets，``pip install aimeta`` 即满足）。
- **`snv` 是逐条光谱标准化**；若需要对整个矩阵做列标准化，请用 `column_scale`。
- **`edge` 只支持线性模型**。非线性模型（GPR、LGBM、MLP）需在工控机上安装完整依赖后
  直接使用 `ModelCard`。
- **迁移需要配对样本**：至少 5 个在两台仪器上均测过的样本；不足时只能用
  `SlopeBiasCorrection` 做响应校正。
- **`ModelCard` 会校验波长轴**：推理时**波长数量**（`X` 的列数）与卡内记录不等会直接报错；
  若调用 `predict` 时还传入了 `wavelengths`（或用带波长轴的 `SpectrumSet`），则进一步比对
  **具体波长值**是否落在训练网格上（`np.allclose`）。只喂 `X` 不传波长时仅验数量，
  波长网格被重采样/错位不会被发现——生产部署建议始终传入波长轴。
- **算指标时，空白样本必须先过预处理链**。若直接喂原始光谱，
  预测值可能落在检出限以下被替为 LOD/2，导致灵敏度与 LOD 退化为 `inf`（此时会给出告警）。
