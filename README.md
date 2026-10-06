# UV-VIS-Env

UV-Vis 光谱水质在线监测算法库。覆盖从硬件评价、光谱预处理、建模、波长选择、仪器间模型迁移，
到计量验收与在线监控的完整链路，可同时用于实验室研究、离线建模与工控机在线部署。

支持的水质参数：CODMn、COD、TN、TP、氨氮（AN）、浊度（TUR）。

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
#    方式 A（推荐）：不传 n_components，按 PLS_toolbox 惯例 CV 遍历 1..20 取误差最小的维数
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
备注：此例子仅为简易建模示例，未对按数据分布对数据进行选取。实验室样本需要考虑样本代表性（可选用xy方法选取样本？），河流样本需要按时间序列处理方式交叉验证（e.g. 滚动更新或expanding）。

---

## 目录结构

```
ai-meta/
├── aimeta/
│   ├── core/            数据契约、注册表、模型产物（ModelCard）
│   ├── preprocessing/   预处理算子与链（SG、DERIV、小波、SNV、MSC、缩放）
│   ├── models/          模型库（线性 / GLM / 潜变量 / 核 / 树 / 神经网络）+ MCR 曲线分辨
│   ├── selection/       波长选择（CARS、稳定性与置换检验）
│   ├── transfer/        仪器间模型迁移（SBC / DS / PDS / GLSW / 均值方差对齐）
│   ├── stats/           假设检验、置信区间、容忍区间
│   ├── metrics/         品质因数（LOD / LOQ / 灵敏度 / 选择性）、GB3838 类别判定
│   ├── monitoring/      控制图（Shewhart / CUSUM / EWMA）与 MSPC
│   ├── viz/             绘图与主题
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
| `transfer` | 跨仪器迁移 | `PiecewiseDirectStandardization`、`DirectStandardization`、`recommend_method` |
| `stats` | 统计检验 | `consistency_report`、`tolerance_interval`、`confidence_interval_mean` |
| `metrics` | 计量验收 | `figures_of_merit`、`acceptance_report`、`daily_r2` |
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

### 仪器间模型迁移

把 A 机器上的模型用到 B 机器，避免每台设备重新训练。
前提是一批**在两台仪器上都测过**的样本（现场可用同一套标液获得）。

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

### 计量验收

```python
from aimeta.metrics.figures_of_merit import figures_of_merit
from aimeta.metrics.water_standards import acceptance_report
from aimeta.preprocessing.base import Pipeline

# 注意：品质因数要在「模型的输入空间」上算，即空白样本也须先过预处理链
pipe = Pipeline.from_config(chain).fit(X_train)
X_train_p = pipe.transform(X_train)
X_blank_p = pipe.transform(X_blank)

fom = figures_of_merit(card.estimator, X_blank_p, X_cal=X_train_p)
print(fom["SEN"], fom["LOD"], fom["LOQ"])

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
应用 `metrics.figures_of_merit` 从数据计算（见下节）。

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
| `test_metrics.py` | 品质因数、类别判定、合格率、报警指标 |
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
- **算品质因数时，空白样本必须先过预处理链**。若直接喂原始光谱，
  预测值可能落在检出限以下被替为 LOD/2，导致灵敏度与 LOD 退化为 `inf`（此时会给出告警）。
