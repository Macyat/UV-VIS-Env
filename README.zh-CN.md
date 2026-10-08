# UV-VIS-Env

> [English](README.md) | 中文

UV-Vis 光谱水质在线监测算法库。覆盖从硬件评价、光谱预处理、建模、波长选择、仪器间模型迁移，
到计量验收与在线监控的完整链路，可同时用于实验室研究、离线建模与工控机在线部署。

支持的水质参数：CODMn、COD、TN、TP、氨氮（AN）、浊度（TUR）。

### 致敬

> **《Comprehensive Chemometrics: Chemical and Biochemical Data Analysis》**——Steven Brown、Romà Tauler、
> Beata Walczak 主编（Elsevier）。它被广泛视为 Massart《Chemometrics: A Textbook》(1988) 与
> 《Handbook of Chemometrics and Qualimetrics》A/B 卷 (1998) 的精神续作，篇幅从 488 页扩展到四卷
> 2958 页（第二版，2020）。
>
> 《Analytical and Bioanalytical Chemistry》："essential for researchers working in the field"；
> 《Spectroscopy Europe》："well worthy of a place in any analytical science library"。

> **术语约定**：本库大量使用光谱计量与国标专有名词（FOM、空白、理论上限、复核限、达标率、日级 R² 等）。
> 阅读代码或配置前，请先浏览 [`docs/术语表.md`](docs/术语表.md)，避免口径混淆。

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

## 快速上手

```python
import numpy as np
from aimeta.core.spectra import SpectrumSet
from aimeta.io.config import load_params, load_chain
from aimeta.pipelines.train import train_model
from aimeta.pipelines.infer import predict

# 1. 准备数据：吸光度矩阵 + 波长轴 + 标签
#    示例数据见 Data/Example1/（437 样本 × 611 波长 190–800 nm）
X = np.load("Data/Example1/spectra.npy")        # (n_samples, n_wavelengths)
wl = np.load("Data/Example1/wavelengths.npy")   # (n_wavelengths,) = 190..800 nm
lab = np.load("Data/Example1/labels.npy")       # (n_samples, 9) 列序 [TN, AN, TP, COD, CODMn, DO1, TUR1, DO2, TUR2]
tn = lab[:, 0]                                  # TN 在 labels 第 0 列（NaN 行由 train_model 内部自动剔除）
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
#    方式 B（手动指定）：显式给定维数，以显式指定为准（覆盖自动选维）
#    card = train_model(
#        spectra, label="TN", model_key="pls",
#        chain=chain, model_params={"n_components": 8},
#        param_def=params["TN"], instrument_id="ai14",
#    )
print(card)          # <ModelCard TN/pls @ai14 ... n_components=11 rmse_cv=0.11>

# 4. 推理（自动套用卡内记录的预处理链）
pred = predict(card, X)
```
备注：该示例仅用于演示建模流程，未按数据分布划分样本，亦未设置验证集。鲁棒建模应避免仅选取高相似样本：实验室样本需考虑代表性抽样（如 SPXY），河流样本应按时间序列进行 Walk-forward 交叉验证（rolling / expanding）。

---

### 按日 Walk-forward 交叉验证（rolling）示例

河流水样是时间序列，随机 K 折会把「未来」样本混进训练集、高估模型性能。应按
采样日做 Walk-forward：用「该天之前最近 `cv_window` 天」训练，预测当天，绝不使用未来数据。
下面以预测 CODMn（高锰酸盐指数）为例，滚动窗口 10 天：

```python
import numpy as np
from aimeta.core.spectra import SpectrumSet
from aimeta.io.config import load_params, load_chain
from aimeta.pipelines.train import train_model

# 1. 数据（rolling/expanding 是时序 CV，样本须按采样时间顺序排列；
#    DAY_idx 已按天递增，直接使用即可）
X = np.load("Data/Example1/spectra.npy")        # (437, 611)
wl = np.load("Data/Example1/wavelengths.npy")   # (611,)
lab = np.load("Data/Example1/labels.npy")       # (437, 9) 列序 [TN, AN, TP, COD, CODMn, DO1, TUR1, DO2, TUR2]
day = np.load("Data/Example1/day_idx.npy")      # (437,) 采样日（1..81）

codmn = lab[:, 4]                               # CODMn 在第 4 列（无缺失）

spectra = SpectrumSet(X=X, wavelengths=wl, y=codmn)

# 2. 配置：CODMn 计量定义 + 预处理链
params = load_params()
chain = load_chain(name="preprocessing/dayu_edge.yaml")

# 3. 训练：rolling 窗口 10 天（每折用「该天之前最近 10 天」训练，预测当天）
card = train_model(
    spectra, label="CODMn", model_key="pls",
    chain=chain,
    param_def=params["CODMn"],
    instrument_id="ai14",
    day_idx=day,           # 每样本所属「第几天」
    cv="rolling",          # 按日滚动窗
    cv_window=10,          # 窗口 = 10 天
)
print(card)                # card.fom["rmse_cv"] 即 rolling CV 的预测 RMSE
```

- `cv="expanding"` 用「该天之前的所有天」扩窗训练；`cv="rolling"` 只用最近 `cv_window` 天，
  更能捕捉近期基体变化。
- 若换参数（如 TN），只需改 `label=`、`param_def=params["TN"]`，并把 `lab[:, 4]` 换成
  对应列即可（缺失值由 `train_model` 内部自动剔除）。

---

## 目录结构

```
ai-meta/
├── aimeta/             算法库主包（内部结构见下）
│   ├── core/            数据契约、注册表、模型产物（ModelCard）
│   ├── preprocessing/   预处理算子与链（SG、DERIV、小波、SNV、MSC、缩放）
│   ├── models/          模型库（线性 / GLM / 潜变量 / 核 / 树 / 神经网络）+ MCR 曲线分辨
│   ├── selection/       波长选择（CARS、稳定性与置换检验）
│   ├── transfer/        仪器间迁移（DS / PDS / SBC / GLSW）+ 硬件差异诊断（波长漂移 / 增益）
│   ├── stats/           假设检验、置信区间、容忍区间
│   ├── metrics/         分析方法品质因数（FOM：LOD / LOQ / 灵敏度 / 选择性）+ GB3838 类别判定
│   ├── hardware_eval.py 光谱仪硬件评价 / 入库检验（暗噪声 / 基线 / 波长准确度 / 光度准确度 / 杂散光 / SNR / 分辨率）
│   ├── monitoring/      控制图（Shewhart / CUSUM / EWMA）与 MSPC
│   ├── viz/             可视化
│   ├── io/              配置加载、ARFF 读取
│   ├── pipelines/       训练 / 模型筛选 / 推理
│   └── cli.py           命令行入口
├── configs/            参数定义、预处理链、仪器档案
├── standards/          金标准与计量标准（原 gold_standard）
│   ├── gold_standards/ 各水质参数检测方法标准（CODMn / TP / AN / TN / COD / TUR，6 份）
│   └── （其余计量与仪器标准：GB 3838 / HJ 915.3 / JJG 178 / ASTM E275）
├── edge/               工控机运行时（依赖 numpy / scipy / PyWavelets）
├── deploy/             部署 SOP 文档
│   ├── 标液模型部署和测试.docx
│   └── 实际水样模型部署和测试.docx
├── Data/               数据（样例 / 参考）
├── docs/               设计文档
├── tests/              单元测试
├── river_inference.py  脱离 aimeta 的独立推理脚本（支持 pca / range 两种 variant）
├── requirements.txt    river_inference.py 配套依赖（joblib / numpy / scikit-learn / scipy / pywavelets）
└── pyproject.toml      打包配置
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
| `hardware_eval` | 光谱仪硬件评价 / 入库检验 | `evaluate_instrument`、`wavelength_accuracy`、`photometric_accuracy`、`stray_light`、`resolution`、`dark_noise` |
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

### 数据清洗（异常样本剔除）

异常样本会抬高控制限、污染建模。`MSPC` 用潜变量把整条光谱压成两个互补的统计量，
用于标记异常样本：

| 统计量 | 含义 | 报警含义 |
|---|---|---|
| T²（Hotelling） | 模型内部变异：样本在正常波动方向上偏离多远 | 沿正常方向偏离过大（如浓度极端） |
| SPE（Q 残差） | 模型解释不了的新变异 | 出现模型未见过的新结构（探头污染、气泡、异物） |

控制限取 `alpha` 分位（T² 用 F 分布，SPE 用 Jackson–Mudholkar 卡方近似）。联合判读：

- T² 正常、SPE 正常 → 受控；
- T² 正常、SPE 高 → 出现模型未见过的新结构（污染 / 异物）；
- T² 高、SPE 正常 → 沿正常方向但偏离极端（如超高浓度）；
- T² 高、SPE 高 → 严重异常。

清洗采用**人工在环**流程——工具只负责标记候选，删除由人拍板：

```python
from aimeta.monitoring import flag_outliers

res = flag_outliers(X, n_components=5, alpha=0.05, by="spe")
res["idx"]          # 候选样本下标，按「最可疑」从高到低排
res["SPE"]          # 对应 Q 残差
res["spe_alarm"]    # 是否超过 SPE 控制限
res["contribution"] # 各波长对 SPE 的贡献，用于人工看异常来源
```

人工逐一看最可疑的几个，结合 `contribution` 贡献图判断异常是否来自真实的
污染 / 异物（而非正常波动），再决定删除哪些：

```python
from aimeta.viz import plots
plots.plot_contribution(wl, res["contribution"][res["idx"][0]])   # 最可疑样本的 SPE 贡献图

drop = [res["idx"][0], res["idx"][2]]   # 人工决定要删的下标
X = np.delete(X, drop, axis=0)
# 删完重跑 flag_outliers，看是否还有要删的，直到满意
```

每一轮「标记 → 人工删 → 重跑」都由人控制，避免自动删除误伤正常样本。

交互式演示（T² vs SPE 散点 + 选删 + 重算重画）见 [`notebooks/mspc_outlier_clean.ipynb`](notebooks/mspc_outlier_clean.ipynb)。

### 预处理链

预处理链以配置描述，可序列化回配置，训练与部署共用同一份定义。

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
| `whittaker` | Whittaker 平滑（Eilers 2003，同 PLS_Toolbox `wsmooth`） | 是 |
| `baseline` | 非对称最小二乘(ALS)基线扣除（Eilers & Boelens 2005） | 是 |
| `wlsbaseline` | 加权最小二乘基线扣除（ALS 底层入口） | 是 |
| `mean_center` / `column_scale` | 列中心化 / 列标准化 | 是 |
| `wavelet` | 小波软阈值去噪（`sym4`） | 是（需安装 PyWavelets） |

### 建模与模型筛选

```python
from aimeta.models.registry import list_models
from aimeta.pipelines.train import sweep
from aimeta.pipelines.scoring import score_cards, select_top_k

print(list_models())               # ['ols', 'ridge', 'lasso', 'pls', 'gpr', 'lgbm', ...]

cards = sweep(spectra, "TN",
              model_keys=("pls", "ridge", "lasso", "gpr"),
              chain=chain,
              param_def=params["TN"],
              folds=5)

# 综合打分：验收指标全集 10 项分量排名求和（alarm_acc/alarm_err/mape/r2_score/
# daily_r2_score/daily_pearson_r_score/rmse/合格率/坏分组比/durbin_watson），
# rank 越小越优——选最优据此综合打分，而非只看 rmse_cv
ranked = score_cards(cards, X_val, y_val, params["TN"], day_idx)
best = select_top_k(cards, 1, X_val, y_val, params["TN"], day_idx)[0]
```

新增模型只需注册，无需改动其他文件：

```python
from aimeta.core.registry import MODELS

@MODELS.register("my_model", family="linear")
class MyModel:
    ...
```

### 候选池、打分表与 TOP K 融合

除单链 `sweep` 外，还支持跨「预处理链 × 模型」全量枚举候选池，按验收指标全集打分排名，
选取 TOP K 做精度加权（1/RMSE²）融合，并为每个候选导出一套诊断图。

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

# 5) 每个候选画一套诊断图（落到 out_dir/figures）；所有候选的打分表汇总到
#    同一个 out_dir/scoring.csv（一起横向比较，而非每个候选各一个 csv）
export_top_k_report(cards, X_val, y_val, params["TN"], out_dir, k=3, day_idx=day_idx)
```

### 仪器间模型迁移

将 A 仪器上的模型迁移至 B 仪器，避免每台设备重复训练。
前提是一批**在两台仪器上都测过**的样本（可用同一套标液获得）。

```python
from aimeta.transfer.standardization import (
    PiecewiseDirectStandardization, kennard_stone,
)
from aimeta.transfer.diagnose import recommend_method

# 先诊断差异，再选择迁移方法
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
| `DirectStandardization` | >= 波长数 | 差异为全局线性，参数更少 |
| `GLSW` | >= 5 | 用于抑制仪器差异子空间 |
| `MeanVarianceAlign` | >= 2 | 最简备选 |

备注：多数河流水样场景不具备迁移可行性——不同河流配置不同设备，且河流基体存在差异，需同时迁移测量系统与被测系统，而一维光谱不足以表征这些差异。

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

在线阶段用正常时期的光谱建参考模型，对**新到样本**做 T²/SPE 监控：

```python
from aimeta.monitoring import MSPC

m = MSPC(n_components=5, alpha=0.05).fit(X_normal)   # 用正常时期的光谱建模
r = m.monitor(X_new)

r["t2_alarm"]      # T² 是否超限
r["spe_alarm"]     # SPE 是否超限
r["contribution"]  # 各波长对 SPE 的贡献，定位「哪个波长段出问题」
```

再配合预测值控制图检测慢漂移：

```python
from aimeta.monitoring import shewhart_limits, cusum

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

# 检验所选波长是否由偶然因素产生（置换标签后重跑，比较分数分布）
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

需注意：这是**实验室参考方法**的检出限，并非光谱模型自身的检出限。
后者取决于仪器噪声、预处理与回归向量，通常显著高于前者，
严格而言应使用 `metrics.figures_of_merit` 从数据计算（见下节「质量评价：分析方法与仪器硬件」）。

---

## 质量评价：分析方法与仪器硬件

本库将质量评价分为两个层次，二者应分别考量：

- **方法层（分析方法品质因数，FOM）**：评价所建多元校正**模型**的性能——灵敏度、检出限与抗干扰能力。
- **硬件层（光谱仪评价）**：评价**仪器**自身状态与跨机一致性——波长准确度、增益稳定性与噪声水平。

建议评价顺序为**先确认硬件与跨机一致性，再计算方法 FOM**（硬件噪声是方法 `s_x` 的来源之一；波长漂移若未校正将直接影响建模质量）。

---

### 1. 分析方法品质因数（FOM）测定方案

#### 概念与口径
FOM（Figures of Merit，品质因数）评价**分析方法 / 校正模型**本身，基于 Olivieri 净分析信号（NAS）多元框架，**不可**套用单变量 3σ/slope 口径（否则会系统性低估 LOD）。

输出七项：`SEN`（灵敏度）、`γ`（分析灵敏度）、`s_x`（光谱噪声）、`s_0`（空白预测标准差）、`LOD`、`LOQ`、`SEL`（选择性）。

#### 需要的输入
- **已拟合模型**：来自校正集（多种浓度标液 / 水样建校正曲线）。
- **空白样本 `X_blank`**：与样品基质一致、不含被测物的溶剂（纯净水 / 去离子水）。
  同条件**独立重复 ≥10 组**（最少 ≥2，否则 `s_0` 静默退化为 0 → `LOD=0`）。
  必须**保留原始噪声**，且**已过同一预处理链**。
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

#### 注意事项
- `n_blank` 必须 **≥2**（≥10 推荐）；`n_blank=1` 时 `s_0` 静默退化为 0，`LOD=0` 失效。
- 空白必须经过预处理链且保留噪声；输入原始光谱或 `LOD/2` 替值将使预测完全相同，
  `SEN/LOD` 退化为 `inf`（代码 `figures_of_merit.py` 第 99–103 行给出告警）。
- `SEN` **尺度相关**（受 y 标准化、校正集浓度范围影响），仅用于候选模型间横向比较，
  不是绝对物理灵敏度。
- 跨天漂移不应计入空白噪声（由 `drift.py` 处理）。

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
硬件评价针对**光谱仪本身**，与"分析方法 FOM"属于不同层面。UV 水质在线监测中需关注的硬件维度：

- **波长准确度 / 漂移**：峰位是否偏移（直接影响 PLS 建模与跨机复用）。
- **光度 / 增益一致性**：逐波长增益、偏置是否稳定。
- **信噪比（SNR）、暗噪声、基线稳定性**：决定实测所得的 `s_x`（方法噪声中包含硬件噪声）。
- **杂散光（stray light）**：抬升基线、降低吸光度上限。
- **分辨率**：能否分辨相邻吸收峰。

> 本节与下方「第 3 节 入库检验」的计算函数**不假设扫描机构**，对**光纤 / 阵列光谱仪（固定光栅、无机械光栅）同样适用**；其失效模式差异见第 3 节「适用说明：光纤 / 阵列光谱仪」。

#### 本库当前实现：仪器间差异诊断
本库不直接执行逐项入库检验，而是提供跨仪器差异与迁移可行性的**门控诊断**（`aimeta/transfer/diagnose.py`）：

- `estimate_wavelength_shift`：用平均光谱互相关估计 slave 相对 master 的波长偏移（nm）。
- `estimate_gain_offset`：逐波长增益 / 偏置（master ≈ gain·slave + offset）；增益随波长变化显著，表明差异具有逐波长结构，此时 PDS 优于 DS。
- `residual_spectrum`：主从平均光谱之差，用于判断差异属于全局倾斜还是局部结构。
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

#### 注意事项
- 波长漂移 > 0.5 nm 时必须先进行波长轴对齐，否则 DS / PDS 将拟合一个错位的波长映射。
- SNR、杂散光、分辨率等硬件指标**本库 `aimeta/hardware_eval.py` 可直接测算**（函数见第 3 节 `signal_to_noise` / `stray_light` / `resolution` 等），是否合格按仪器 datasheet / 检定规程（JJG 178、ASTM E275）的判据核验。
- 本段（`transfer/diagnose`）仅从实测光谱**诊断跨机差异**，不能替代逐项入库检验；执行逐项入库检验需另行准备标准物质与测试流程（见第 3 节）。

#### 参考文献
- 仪器间校准 / 迁移与差异诊断（DS / PDS / SBC / GLSW）：见 `aimeta/transfer/` 与 `docs/重构方案.md`。
- 紫外-可见分光光度计性能评价（波长准确度、光度准确度、杂散光、基线平直度、分辨率、噪声）：ASTM E275 系列；国内计量溯源见 **JJG 178《紫外、可见、近红外分光光度计》检定规程**。
- 检测实验室能力通用要求：**ISO/IEC 17025**（溯源与量值传递的整体框架）。

---

### 3. 光谱仪硬件入库检验方案

#### 概念与口径
在仪器部署到现场或复用既有模型之前，按紫外-可见分光光度计的计量检验口径执行**逐项入库检验**。
本库 `aimeta/hardware_eval.py` 提供从实测光谱**计算指标**的函数；标准物质 / 滤光片由使用方按检定规程准备。
主要依据：**JJG 178《紫外、可见、近红外分光光度计》检定规程**、**ASTM E275** 系列。

#### 检验项目与所需标准物质
| 项目 | 需要的工具 / 标准物质 | 操作过程 / 计算原理 | 典型合格判据 |
|---|---|---|---|
| 波长准确度 | 钬玻璃 / 钬氧化物标准片（特征峰 279.4 / 287.5 / 333.7 / 360.9 / 418.5 / 453.2 / 536.2 / 637.5 nm）；或已知发射线（汞 / 氩灯） | 测标准片光谱，在每个参考峰 ±window 内用抛物线细化（`find_peak_wavelength`）取实测峰位；误差 = 实测 − 标称，取 `max_abs_error_nm`（函数 `wavelength_accuracy`）。 | 误差 ≤ 0.5 nm |
| 光度准确度 | 中性密度片 / 重铬酸钾标准溶液（已知吸光度点） | 测标准片 / 溶液光谱，在认证波长点线性插值取实测值；误差 = 实测 − 认证值，取 `max_abs_error`（函数 `photometric_accuracy`）。 | ≤ 0.002 A（或 0.3 %T，T = 透过率 Transmittance） |
| 杂散光 | NaI / Corning 截止滤光片（在完全截止波长处，如 340 nm） | 用截止滤光片测透射谱，在滤光片标称完全吸收（≈0%T）的波长处线性插值取实测 T%，该残余透射即杂散光，取 `max_Tpct`（函数 `stray_light`）。 | ≤ 0.05 %T（T = 透过率） |
| 暗噪声 | 无（入射光遮挡：shutter 关闭 / 暗室） | 遮光下采集 ≥2 次重复暗谱，逐波长算样本标准差（ddof=1），报告 `std_max` / `std_mean`（函数 `dark_noise`）。 | ≤ 0.0005（示例） |
| 基线平直度 | 空气 / 空白（100%T 参考，T = 透过率） | 多次重复测 100%T 参考线，逐波长跨重复 std 得基线重复性 `repeatability_max`；若给 ideal 值再算平均基线相对 ideal 的最大偏离 `deviation_max`（函数 `baseline_flatness`）。 | 重复性 ≤ 0.001（示例） |
| 信噪比 | 稳定光源 / 纯水（或任意已知稳定信号） | 同条件重复测 ≥10 次，在指定波长（或全波段均值）上 `SNR = 信号均值 / 重复标准差`（ddof=1）；噪声为 0 时返回 `inf`（函数 `signal_to_noise`）。 | 越高越好 |
| 分辨率 | 汞灯 / 氩灯等窄发射线光源 | 测发射线光谱，在峰 ±window 内抛物线细化峰高，线性插值求半高全宽 FWHM（nm）（函数 `resolution`）。 | FWHM ≤ 2 nm（1 nm 狭缝典型） |
| 标液线性度（各波长） | 无 | 测各种物质标液的多浓度梯度光谱，自第三个浓度起对每个波长做线性回归并计算累计 R²（按浓度梯度逐步累加拟合，逐波长用 `sklearn.metrics.r2_score` 或 `np.polyfit` 求解）；输出各波长 R² 曲线。 | 按 R² 曲线定性判断，并结合高 R² 值连续波长区间（e.g. ≥ 0.99）判定线性范围。 |

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

#### 注意事项
- 波长准确度要选**足够尖锐且分离**的参考峰；峰位用抛物线细化（`find_peak_wavelength`）比直接取 argmax 更准。
- 杂散光必须在滤光片**完全截止**的波长处测；截止不彻底会低估。
- 暗噪声 / 基线平直度都要求**多次重复**（≥10 组），单次测量无意义。
- 这些硬件指标是**方法 `s_x`（FOM 里的光谱噪声）的来源之一**：硬件噪声大，方法的 LOD / LOQ 必然差。

#### 适用说明：光纤 / 阵列光谱仪（无机械光栅）
本节方法同样适用于**光纤 / 阵列光谱仪（固定光栅、无机械扫描单色器，如 Ocean Optics / Avantes 类）**——
本库所有硬件指标函数仅依赖「波长数组 + 光谱数组」，不假定扫描机构。差异主要体现在失效模式与检验频次：

| 项目 | 扫描式单色器 | 光纤 / 阵列光谱仪 | 本节做法 |
|---|---|---|---|
| 波长漂移主因 | 丝杠 / 齿轮机械回差 | **温度驱动**的像元↔波长标定漂移（可达数 nm） | 做**周期性波长重标定**（Hg-Ar / Ne 灯或钬 / 镨钕标准片）；在线监测更关键（`drift.py`） |
| 分辨率 | 靠狭缝 / 步长调 | **固定**（狭缝 + 光栅 + 像元），不可调 | 用汞 / 氩灯线 FWHM 核验收货指标；无法通过增大狭缝提高 SNR |
| SNR | 取决于扫描次数 | 取决于**积分时间 + 信号平均** | 必须在**实际运行的积分时间 / 平均次数下**测 SNR / 暗噪声 |
| 杂散光 | 双单色器可有效抑制 | 光纤耦合且无双单色器，**通常更高** | 仍需测量，且应更为重视（NaI / 截止滤光片） |
| 暗 / 读出噪声 | 光电管暗电流 | CCD / CMOS 暗电流、读出噪声（制冷与否差别大） | `dark_noise`（关快门）直接测 |
| JJG 178 扫描条款 | 适用 | 不适用（无扫描机构） | 其余（波长 / 光度 / 杂散光 / 基线 / 噪声 / 分辨率）计算口径**完全一致** |

对阵列光谱仪，计算口径与扫描式单色器一致（JJG 178 相应条款照搬即可），重心是温度漂移标定、积分时间下的 SNR 与杂散光。

#### 参考文献
- **JJG 178《紫外、可见、近红外分光光度计》检定规程**（国内计量溯源，含波长 / 光度 / 杂散光 / 基线 / 噪声 / 分辨率检验）。
- ASTM E275 系列 *Standard Practices for Describing and Measuring Performance of UV, Visible, and Near-IR Spectrophotometers*（波长准确度、光度准确度、杂散光、基线平直度、分辨率、噪声）。
- ISO/IEC 17025（检测实验室能力通用要求，量值传递整体框架）。

---

## 部署到工控机

`edge/` 是独立的精简运行时，部署端依赖与 aimeta 一致（numpy / scipy / PyWavelets）；纯 numpy 可编译算子构成的链仍可脱离 scipy/pywt 单独运行。
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

### 独立推理脚本部署（river_inference.py，无需 aimeta）

除 `edge/` 运行时外，仓库另提供脱离 aimeta 依赖树的独立推理脚本 `river_inference.py`
（与 `requirements.txt` 一同置于仓库根目录；完整部署 SOP 见 `deploy/实际水样模型部署和测试.docx` 与 `deploy/标液模型部署和测试.docx`）。它读取 ARFF 光谱、输出
JSON 结果，支持 `--variant pca`（默认）与 `range`（原 Linux 流水线）。脚本自带全部预处理
（Wiener → SG → 小波 → SNV），模型以 pickle/joblib 保存且已含 `MeanCenterer`/`YAutoScaler` 等 scaler，
故**线性、随机森林、LightGBM、AdaBoost 等任意 sklearn 系模型都能直接加载预测**，不要求「仅线性模型」。

工控机只需 Python 3.12 + 随脚本附带的 `requirements.txt`（joblib / numpy / scipy / scikit-learn /
pywavelets），用 `pip install -r requirements.txt` 安装即可，无需安装 aimeta 完整依赖树。

简要流程：

1. 复制 `.pkl` 模型到 `<部署目录>\data\models\`（第二套模型置于 `models\wider_range\`）；
2. 复制 `river_inference.py` 与 `requirements.txt` 到 `<部署目录>\`；
3. `conda create -n reg_venv python=3.12 && conda activate reg_venv && pip install -r requirements.txt`；
4. 在站点 `strategy_ai???.ini` 的 `[Python]` 段设 `reg_exe = ...\reg_venv\python.exe`；
5. 测试：`python river_inference.py -file_name SpectrumData_****.arff -re_train False`，
   结果见 `result_****.json`。

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
  纯 numpy 可编译算子（``savgol`` / ``snv`` / ``msc`` / 缩放）仍沿零依赖快速路径执行；
  ``wavelet`` 等需外部库的算子会委托注册算子执行，工控机需装好对应依赖
  （aimeta 主依赖已含 scipy / PyWavelets，``pip install aimeta`` 即满足）。
- **`snv` 是逐条光谱标准化**；若需要对整个矩阵做列标准化，请用 `column_scale`。
- **`edge` 只支持线性模型**。非线性模型（GPR、LGBM、MLP）需在工控机上安装完整依赖后
  直接使用 `ModelCard`。
- **迁移需要配对样本**：至少 5 个在两台仪器上均测过的样本；不足时只能用
  `SlopeBiasCorrection` 做响应校正。
- **`ModelCard` 会校验波长轴**：推理时**波长数量**（`X` 的列数）与卡内记录不等会直接报错；
  若调用 `predict` 时还传入了 `wavelengths`（或用带波长轴的 `SpectrumSet`），则进一步比对
  **具体波长值**是否落在训练网格上（`np.allclose`）。仅传入 `X` 而不传波长时仅校验数量，
  波长网格被重采样/错位不会被发现——生产部署建议始终传入波长轴。
- **计算指标时，空白样本必须先经过预处理链**。若直接输入原始光谱，
  预测值可能落在检出限以下并被替换为 LOD/2，导致灵敏度与 LOD 退化为 `inf`（此时会给出告警）。
