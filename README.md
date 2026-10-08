# UV-VIS-Env

> English | [中文](README.zh-CN.md)

An algorithm library for online UV-Vis spectroscopic water-quality monitoring. It covers the full pipeline
from hardware evaluation, analytical-method evaluation, spectral preprocessing, modeling, and inter-instrument model
transfer to metrological acceptance and online monitoring, and can be used for laboratory research, offline
modeling, and online deployment.

Water-quality parameters currently under verification: CODMn, COD, TN, TP, ammonia nitrogen (AN), and turbidity (TUR).

### Acknowledgements

This library is informed by the chemometrics literature; in particular, we acknowledge:

> **Comprehensive Chemometrics: Chemical and Biochemical Data Analysis** — Steven Brown, Romà Tauler &
> Beata Walczak (eds.), Elsevier. Widely regarded as the spiritual successor to Massart's *Chemometrics:
> A Textbook* (1988) and the *Handbook of Chemometrics and Qualimetrics*, Parts A/B (1998), growing from
> 488 pages to 2,958 pages across four volumes in its second edition (2020).
>
> — *Analytical and Bioanalytical Chemistry*: "essential for researchers working in the field."
> — *Spectroscopy Europe*: "well worthy of a place in any analytical science library."

> **Terminology**: this library makes heavy use of spectrometric-metrology and Chinese-national-standard
> terms (FOM, blank, theoretical upper, review limit, acceptance rate, daily R², etc.). Before reading the code or
> configuration, please skim [`docs/glossary.md`](docs/glossary.md) to avoid mixing up definitions.

---

## Installation

```bash
cd ai-meta
pip install -e .
```

Optional dependencies (install as needed):

```bash
pip install lightgbm torch        # extended models such as LGBM / neural networks
pip install pytest                # to run tests
```

Requirements: Python >= 3.10; core dependencies are numpy / scipy / scikit-learn / pandas / PyWavelets / PyYAML / matplotlib.

---

## Quick Start

```python
import numpy as np
from aimeta.core.spectra import SpectrumSet
from aimeta.io.config import load_params, load_chain
from aimeta.pipelines.train import train_model
from aimeta.pipelines.infer import predict

# 1. Prepare data: absorbance matrix + wavelength axis + labels
#    Example data in Data/Example1/ (437 samples x 611 wavelengths, 190-800 nm)
X = np.load("Data/Example1/spectra.npy")        # (n_samples, n_wavelengths)
wl = np.load("Data/Example1/wavelengths.npy")   # (n_wavelengths,) = 190..800 nm
lab = np.load("Data/Example1/labels.npy")       # (n_samples, 9) columns [TN, AN, TP, COD, CODMn, DO1, TUR1, DO2, TUR2]
tn = lab[:, 0]                                  # TN is in column 0 of labels; NaN rows are dropped inside train_model
spectra = SpectrumSet(X=X, wavelengths=wl, y=tn)

# 2. Load parameter definitions and preprocessing chain (configs/)
params = load_params()
chain = load_chain(name="preprocessing/dayu_edge.yaml")

# 3. Train
#    Option A (recommended): do not pass n_components; the PLS_toolbox routine sweeps
#    CV over 1..20 and picks the dimension with the smallest error
card = train_model(
    spectra, label="TN", model_key="pls",
    chain=chain,
    param_def=params["TN"],
    instrument_id="ai14",
)
#    Option B (manual): explicitly specify the dimension; the explicit value wins (overrides auto-selection)
#    card = train_model(
#        spectra, label="TN", model_key="pls",
#        chain=chain, model_params={"n_components": 8},
#        param_def=params["TN"], instrument_id="ai14",
#    )
print(card)          # <ModelCard TN/pls @ai14 ... n_components=11 rmse_cv=0.11>

# 4. Inference (the preprocessing chain recorded in the card is applied automatically)
pred = predict(card, X)
```

Note: this example only demonstrates the modeling flow; it does not split samples by distribution, nor does it
set up a validation set. Robust modeling should avoid selecting only highly similar samples: laboratory samples
need representative sampling (e.g. SPXY), while river samples should use walk-forward cross-validation
(rolling / expanding) along the time series.

---

### Example: daily walk-forward cross-validation (rolling)

River samples form a time series; random K-fold would mix "future" samples into the training set and overestimate
performance. Instead, do walk-forward by sampling day: train on the most recent `cv_window` days before a given day
and predict that day, never using future data. The example below predicts CODMn (permanganate index) with a 10-day
rolling window:

```python
import numpy as np
from aimeta.core.spectra import SpectrumSet
from aimeta.io.config import load_params, load_chain
from aimeta.pipelines.train import train_model

# 1. Data (rolling/expanding are temporal CV; samples must be ordered by sampling time.
#    DAY_idx is already ascending, so use it directly)
X = np.load("Data/Example1/spectra.npy")        # (437, 611)
wl = np.load("Data/Example1/wavelengths.npy")   # (611,)
lab = np.load("Data/Example1/labels.npy")       # (437, 9) columns [TN, AN, TP, COD, CODMn, DO1, TUR1, DO2, TUR2]
day = np.load("Data/Example1/day_idx.npy")      # (437,) sampling day (1..81)

codmn = lab[:, 4]                               # CODMn is in column 4 (no missing values)

spectra = SpectrumSet(X=X, wavelengths=wl, y=codmn)

# 2. Config: CODMn metrology definition + preprocessing chain
params = load_params()
chain = load_chain(name="preprocessing/dayu_edge.yaml")

# 3. Train: 10-day rolling window (each fold trains on the most recent 10 days and predicts the current day)
card = train_model(
    spectra, label="CODMn", model_key="pls",
    chain=chain,
    param_def=params["CODMn"],
    instrument_id="ai14",
    day_idx=day,           # the day index of each sample
    cv="rolling",          # daily rolling window
    cv_window=10,          # window = 10 days
)
print(card)                # card.fom["rmse_cv"] is the rolling-CV prediction RMSE
```

- `cv="expanding"` trains on all days before the current day; `cv="rolling"` uses only the most recent
  `cv_window` days, which better captures recent matrix changes.
- To switch parameters (e.g. TN), just change `label=` and `param_def=params["TN"]`, and replace `lab[:, 4]`
  with the corresponding column (missing values are dropped automatically inside `train_model`).

---

## Directory Layout

```
ai-meta/
├── aimeta/             main package (internal structure below)
│   ├── core/            data contracts, registry, model artifacts (ModelCard)
│   ├── preprocessing/   preprocessing operators and chains (SG, DERIV, wavelet, SNV, MSC, scaling)
│   ├── models/          model zoo (linear / GLM / latent / kernel / tree / neural net) + MCR curve resolution
│   ├── selection/       wavelength selection (CARS, stability & permutation tests)
│   ├── transfer/        inter-instrument transfer (DS / PDS / SBC / GLSW) + hardware difference diagnosis (wavelength shift / gain)
│   ├── stats/           hypothesis tests, confidence intervals, tolerance intervals
│   ├── metrics/         analytical figures of merit (FOM: LOD / LOQ / sensitivity / selectivity) + GB3838 class assignment
│   ├── hardware_eval.py spectrometer hardware evaluation / incoming inspection (dark noise / baseline / wavelength accuracy / photometric accuracy / stray light / SNR / resolution)
│   ├── monitoring/      control charts (Shewhart / CUSUM / EWMA) and MSPC
│   ├── viz/             visualization
│   ├── io/              config loading, ARFF reading
│   ├── pipelines/       training / model selection / inference
│   └── cli.py           CLI entry point
├── configs/            parameter definitions, preprocessing chains, instrument profiles
├── standards/          gold standards and metrology standards (formerly gold_standard)
│   ├── gold_standards/ method standards for each water-quality parameter (CODMn / TP / AN / TN / COD / TUR, 6 files)
│   └── (other metrology & instrument standards: GB 3838 / HJ 915.3 / JJG 178 / ASTM E275)
├── edge/               industrial-PC runtime (depends on numpy / scipy / PyWavelets)
├── deploy/             deployment SOP documents
│   ├── 标液模型部署和测试.docx
│   └── 实际水样模型部署和测试.docx
├── Data/               data (examples / references)
├── docs/               design documents
├── tests/              unit tests
├── river_inference.py  standalone inference script independent of aimeta (supports pca / range variants)
├── requirements.txt    dependencies for river_inference.py (joblib / numpy / scikit-learn / scipy / pywavelets)
└── pyproject.toml      packaging config
```

---

## Module Overview

| Module | Purpose | Key interfaces |
|---|---|---|
| `core` | data contracts & model artifacts | `SpectrumSet`, `ModelCard`, `Registry` |
| `preprocessing` | spectral preprocessing | `Pipeline.from_config`, `DERIV`, `snv_transform` |
| `models` | modeling | `build_model`, `list_models`, `WaterQualityModel`, `MCRALS` |
| `selection` | wavelength selection | `CARS`, `permutation_test`, `selection_stability` |
| `transfer` | cross-instrument transfer / hardware difference diagnosis | `PiecewiseDirectStandardization`, `DirectStandardization`, `recommend_method`, `estimate_wavelength_shift` |
| `stats` | statistical tests | `consistency_report`, `tolerance_interval`, `confidence_interval_mean` |
| `metrics` | analytical figures of merit (FOM) + metrological acceptance | `figures_of_merit`, `acceptance_report`, `daily_r2`, `alarm_accuracy` |
| `hardware_eval` | spectrometer hardware evaluation / incoming inspection | `evaluate_instrument`, `wavelength_accuracy`, `photometric_accuracy`, `stray_light`, `resolution`, `dark_noise` |
| `monitoring` | online monitoring | `MSPC`, `shewhart_limits`, `cusum`, `ewma` |
| `viz` | visualization | `plots.plot_spectra`, `plots.plot_pred_vs_actual`, `save_fig` |
| `io` | config & data | `load_params`, `load_chain`, `load_instrument`, `read_arff` |
| `pipelines` | pipeline orchestration | `train_model`, `sweep`, `predict` |

---

## Usage

### Data Contract

All flows operate on `SpectrumSet`, which always carries the wavelength axis to avoid misalignment from
column-index-based access.

```python
import pandas as pd
from aimeta.core.spectra import SpectrumSet

spectra = SpectrumSet(
    X=X,
    wavelengths=wl,
    meta=pd.DataFrame({"instrument_id": ["ai14"] * len(X), "site": ["xikeng"] * len(X)}),
    y=y,
)
spectra.clip_wavelengths(220, 700)      # clip the band
spectra.sort_by("timestamp")            # sort by time
```

### Data Cleaning (outlier removal)

Outliers inflate control limits and pollute the model. `MSPC` compresses each spectrum into two complementary
statistics via latent variables, used to flag abnormal samples:

| Statistic | Meaning | Alarm meaning |
|---|---|---|
| T² (Hotelling) | in-model variation: how far a sample deviates along the normal-variation direction | too far along the normal direction (e.g. extreme concentration) |
| SPE (Q residual) | new variation not explained by the model | a new structure the model has never seen (probe fouling, bubbles, foreign matter) |

Control limits use the `alpha` quantile (T² uses the F distribution; SPE uses the Jackson–Mudholkar chi-square
approximation). Joint interpretation:

- T² normal, SPE normal → in control;
- T² normal, SPE high → a new structure the model has never seen (contamination / foreign matter);
- T² high, SPE normal → along the normal direction but extreme (e.g. very high concentration);
- T² high, SPE high → severe anomaly.

Cleaning uses a **human-in-the-loop** flow — the tool only flags candidates, while deletion is decided by a human:

```python
from aimeta.monitoring import flag_outliers

res = flag_outliers(X, n_components=5, alpha=0.05, by="spe")
res["idx"]          # candidate sample indices, sorted from most to least suspicious
res["SPE"]          # corresponding Q residuals
res["spe_alarm"]    # whether the SPE control limit is exceeded
res["contribution"] # per-wavelength contribution to SPE, for inspecting the anomaly source
```

A human inspects the most suspicious ones one by one, using the `contribution` plot to judge whether the anomaly
comes from real contamination / foreign matter (rather than normal variation), then decides which to delete:

```python
from aimeta.viz import plots
plots.plot_contribution(wl, res["contribution"][res["idx"][0]])   # SPE contribution plot of the most suspicious sample

drop = [res["idx"][0], res["idx"][2]]   # indices to delete, decided by a human
X = np.delete(X, drop, axis=0)
# re-run flag_outliers after deletion, check whether more should be removed, until satisfied
```

Every "flag -> human-delete -> re-run" round is human-controlled, avoiding accidental deletion of normal samples.

An interactive demo (T² vs SPE scatter + select/delete + recompute/redraw) is in
[`notebooks/mspc_outlier_clean.ipynb`](notebooks/mspc_outlier_clean.ipynb).

### Preprocessing Chain

The preprocessing chain is described by config, can be serialized back to config, and is shared by training and
deployment.

```python
from aimeta.preprocessing.base import Pipeline

pipe = Pipeline.from_config([
    {"op": "savgol", "window": 15, "polyorder": 3},
    {"op": "snv"},
])
Xt = pipe.fit_transform(X)
pipe.to_config()        # [{'op': 'savgol', 'window': 15, 'polyorder': 3, 'deriv': 0}, {'op': 'snv'}]
```

Available operators:

| Operator | Description | Deployable to IPC |
|---|---|---|
| `savgol` | Savitzky-Golay smoothing / derivative | yes |
| `deriv_gram` | Gram-polynomial smoothing / derivative (numerically equivalent to `savgol`) | yes |
| `snv` | standard normal variate (per spectrum) | yes |
| `msc` | multiplicative scatter correction | yes |
| `whittaker` | Whittaker smoothing (Eilers 2003, cf. PLS_Toolbox `wsmooth`) | yes |
| `baseline` | asymmetric least-squares (ALS) baseline removal (Eilers & Boelens 2005) | yes |
| `wlsbaseline` | weighted least-squares baseline removal (ALS, low-level entry) | yes |
| `mean_center` / `column_scale` | column centering / column scaling | yes |
| `wavelet` | wavelet soft-threshold denoising (`sym4`) | yes (requires PyWavelets) |

### Modeling and Model Selection

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

# Composite score: sum of ranks over the full 10 acceptance metrics (alarm_acc/alarm_err/mape/r2_score/
# daily_r2_score/daily_pearson_r_score/rmse/acceptance rate/bad-group ratio/durbin_watson);
# smaller rank is better — select the best by this composite score, not just rmse_cv
ranked = score_cards(cards, X_val, y_val, params["TN"], day_idx)
best = select_top_k(cards, 1, X_val, y_val, params["TN"], day_idx)[0]
```

Adding a model only requires registration, without touching other files:

```python
from aimeta.core.registry import MODELS

@MODELS.register("my_model", family="linear")
class MyModel:
    ...
```

### Candidate Pool, Score Table and TOP-K Fusion

Besides single-chain `sweep`, the library supports enumerating the full candidate pool across
"preprocessing chain x model", scoring/ranking by the full acceptance-metric set, fusing the TOP K by
precision weighting (1/RMSE²), and exporting a set of diagnostic plots for each candidate.

```python
from aimeta.pipelines.train import sweep_grid
from aimeta.pipelines.scoring import score_cards, select_top_k
from aimeta.models.ensembles import fuse_predict
from aimeta.viz.report import export_top_k_report

# 1) Candidate pool: 2 chains x 4 models = 8 candidates
cards = sweep_grid(spectra, "TN",
                   model_keys=("pls", "ridge", "lasso", "ols"),
                   chains=[[{"op": "snv"}],
                           [{"op": "savgol", "window": 11, "polyorder": 3}]],
                   param_def=params["TN"], folds=5)

# 2) Score table (full acceptance metrics) + ranking, written to scoring.csv
ranked = score_cards(cards, X_val, y_val, params["TN"], day_idx)

# 3) TOP K candidates (by score rank)
top = select_top_k(cards, 3, X_val, y_val, params["TN"], day_idx)

# 4) Precision-weighted fusion: w_i = 1/RMSE_CV_i^2 (degenerates to equal weights if any rmse_cv is missing / <= 0)
y_fused, info = fuse_predict(top, X_new, scheme="inv_rmse2", param=params["TN"])
print(info["scheme"], info["weights"])

# 5) Draw a diagnostic figure set per candidate (into out_dir/figures); all candidate score tables are
#    collected into the same out_dir/scoring.csv (compared side by side, not one csv per candidate)
export_top_k_report(cards, X_val, y_val, params["TN"], out_dir, k=3, day_idx=day_idx)
```

### Inter-Instrument Model Transfer

Transfer a model trained on instrument A to instrument B, avoiding retraining on every device. The premise is a
batch of samples **measured on both instruments** (obtainable with the same set of standard solutions).

```python
from aimeta.transfer.standardization import (
    PiecewiseDirectStandardization, kennard_stone,
)
from aimeta.transfer.diagnose import recommend_method

# diagnose the difference first, then choose the transfer method
print(recommend_method(wl, X_slave, X_master, n_std_samples=len(X_slave)))
# {'method': 'pds', 'window': 5, 'shift_nm': 0.0, 'reason': 'samples 40 < wavelengths 251, use windowed-regression PDS'}

idx = kennard_stone(X_slave, 40)                       # pick representative samples for standardization
pds = PiecewiseDirectStandardization(window=5).fit(X_slave[idx], X_master[idx])
X_slave_corr = pds.transform(X_slave)                  # then the master model can be applied
```

| Method | Required sample size | Applicable case |
|---|---|---|
| `SlopeBiasCorrection` | >= 2 | only predicted values on two instruments, no paired spectra |
| `PiecewiseDirectStandardization` | >= 5 | general; handles wavelength-dependent differences |
| `DirectStandardization` | >= #wavelengths | difference is globally linear, fewer parameters |
| `GLSW` | >= 5 | for suppressing the instrument-difference subspace |
| `MeanVarianceAlign` | >= 2 | simplest fallback |

Note: transfer is often infeasible for river samples — different rivers use different devices, and the river matrix
differs, requiring transfer of both the measurement system and the measured system; a 1-D spectrum is insufficient
to characterize these differences.

### Metrological Acceptance

```python
from aimeta.metrics.figures_of_merit import figures_of_merit
from aimeta.metrics.water_standards import acceptance_report
from aimeta.preprocessing.base import Pipeline

# Note: metrics must be computed in the "model input space", i.e. blank samples must pass the preprocessing chain first
pipe = Pipeline.from_config(chain).fit(X_train)
X_train_p = pipe.transform(X_train)
X_blank_p = pipe.transform(X_blank)

metric = figures_of_merit(card.estimator, X_blank_p, X_cal=X_train_p)
print(metric["SEN"], metric["LOD"], metric["LOQ"])

# acceptance conclusion under GB3838 class definitions
rep = acceptance_report(y_true, y_pred, params["TN"], day_idx=days)
print(rep["rmse"], rep["acceptance_rate"], rep["daily_r2"])
```

### Online Monitoring

During online operation, build a reference model from normal-period spectra and monitor **new samples** via
T²/SPE:

```python
from aimeta.monitoring import MSPC

m = MSPC(n_components=5, alpha=0.05).fit(X_normal)   # build model from normal-period spectra
r = m.monitor(X_new)

r["t2_alarm"]      # whether T² exceeds the limit
r["spe_alarm"]     # whether SPE exceeds the limit
r["contribution"]  # per-wavelength contribution to SPE, for locating "which band is problematic"
```

Then use prediction-value control charts to detect slow drift:

```python
from aimeta.monitoring import shewhart_limits, cusum

cl, lcl, ucl = shewhart_limits(pred_series)  # control limits of the prediction series (with autocorrelation correction)
cusum(pred_series, k=0.5, h=5.0)             # slow-drift detection
```

### Curve Resolution (MCR)

Separate the concentration profiles and spectra of individual components from mixed spectra:

```python
from aimeta.models.curve_resolution import MCRALS

mcr = MCRALS(n_components=3, non_negative=True, closure=True).fit(X)
mcr.components_          # (k, p) component spectra
mcr.concentrations_      # (n, k) component concentration profiles
mcr.transform(X_new)     # concentration profiles of new samples
mcr.ambiguity_estimate(X)  # 0~1, larger means higher solution uncertainty
```

### Wavelength Selection

```python
from aimeta.selection import CARS, permutation_test

sel = CARS(n_iter=30, n_components=10).fit(X, y)
X_sel = sel.transform(X)

# test whether the selected wavelengths arise by chance (re-run after label permutation, compare score distributions)
res = permutation_test(lambda: CARS(n_iter=10, n_components=5), score_fn, X, y)
print(res["score"], res["p_value"])
```

### Visualization

```python
from aimeta.viz import apply_theme, save_fig
from aimeta.viz import plots

apply_theme()
fig = plots.plot_pred_vs_actual(y_true, y_pred, label="TN", unit="mg/L",
                                ranges=params["TN"].ranges)
save_fig(fig, "figs/TN_pred.png")
```

---

## Command Line

```bash
python -m aimeta.cli list-models [--family latent]   # list available models
python -m aimeta.cli list-ops                        # list preprocessing operators and deployability
python -m aimeta.cli smoke                           # end-to-end self-check (synthetic data)
python -m aimeta.cli transfer --slave ai17           # inspect the transfer config of an instrument
```

Example `smoke` output:

```
[train] <ModelCard AN/pls @ai14 fp=a1bd620a4660a3ac rmse_cv=0.3296>
[infer] rmse = 0.24679
[edge ] parity=OK max_diff=4.00e-14
[mspc ] anomalous sample SPE=0.1012 vs limit=0.0001 -> alarm
```

---

## Configuration

| File | Content |
|---|---|
| `configs/params.yaml` | range, class boundaries, LOD, error limits of each water-quality parameter |
| `configs/preprocessing/dayu_v1.yaml` | training chain: `savgol → wavelet → snv` |
| `configs/preprocessing/dayu_edge.yaml` | deployment chain: `savgol → snv` |
| `configs/instruments/*.yaml` | wavelength axis, path length, transfer parameters of each instrument |

Changing thresholds, adding/removing preprocessing steps, or registering a new instrument only edits YAML, no code.

The `lower_bound` in `params.yaml` takes the LOD of the national-standard laboratory method; when a prediction falls
below it, the value is replaced by `lower_bound / 2` (to avoid systematically over-reporting every "not detected"
as the LOD value itself), and the `standard` field records the basis standard:

| Parameter | LOD | Basis standard |
|---|---|---|
| CODMn | 0.5 mg/L | GB 11892-89 (lower limit of the determination range) |
| COD | 4 mg/L | HJ 828-2017 (10.0 mL sample) |
| TN | 0.05 mg/L | HJ 636-2012 (10 mL sample) |
| TP | 0.01 mg/L | GB 11893-89 (25 mL test portion) |
| ammonia nitrogen | 0.025 mg/L | HJ 535-2009 (50 mL, 20 mm cuvette) |
| turbidity | 0.3 NTU | HJ 1075-2019 |

Note: these are the LODs of **laboratory reference methods**, not the LOD of the spectral model itself. The latter
depends on instrument noise, preprocessing, and the regression vector, and is usually significantly higher; strictly,
it should be computed from data via `metrics.figures_of_merit` (see the next section
"Quality Evaluation: Analytical Method and Instrument Hardware").

---

## Quality Evaluation: Analytical Method and Instrument Hardware

This library divides quality evaluation into two layers, which should be assessed separately:

- **Method layer (analytical figures of merit, FOM)**: evaluates the performance of the built multivariate
  calibration **model** — sensitivity, LOD, and interference immunity.
- **Hardware layer (spectrometer evaluation)**: evaluates the **instrument** itself and cross-instrument
  consistency — wavelength accuracy, gain stability, and noise level.

The recommended order is **confirm hardware and cross-instrument consistency first, then compute method FOM**
(hardware noise is one source of the method's `s_x`; uncorrected wavelength drift directly degrades modeling).

---

### 1. Analytical Figures of Merit (FOM) Protocol

#### Concept and Definitions
FOM (Figures of Merit) evaluates the **analytical method / calibration model** itself, based on Olivieri's net
analyte signal (NAS) multivariate framework; do **not** apply the univariate 3σ/slope definition (that would
systematically underestimate the LOD).

Seven quantities are output: `SEN` (sensitivity), `γ` (analytical sensitivity), `s_x` (spectral noise), `s_0`
(blank-prediction standard deviation), `LOD`, `LOQ`, `SEL` (selectivity).

#### Required Inputs
- **Fitted model**: from a calibration set (a calibration curve built from multi-concentration standards / water samples).
- **Blank samples `X_blank`**: solvent matrix-matched to the samples, containing no analyte (pure / deionized water).
  **>= 10 independent replicates** under the same conditions (>= 2 minimum; otherwise `s_0` silently degenerates to 0
  → `LOD=0`). The **raw noise must be retained**, and the blanks must have passed **the same preprocessing chain**.
- **`X_cal` (optional)**: calibration-set spectra, for picking the working point (defaults to the mean of `X_blank`).
- **`s_target` (optional)**: pure-component spectrum of the target, for computing `SEL`.

#### Formulas
| Quantity | Formula | Meaning |
|---|---|---|
| sensitivity `SEN` | `1 / ‖b‖` | `b` = regression vector (linear: `coef_`; nonlinear: numerical derivative); larger is more sensitive |
| analytical sensitivity `γ` | `SEN / s_x` | signal-to-noise ratio per unit concentration |
| spectral noise `s_x` | std of blank-spectrum detrended residuals | total noise from hardware + environment + preprocessing |
| blank-prediction std `s_0` | std of blank predictions (ddof=1) | blank-prediction spread |
| LOD | `3.3 · s_0 / SEN` | multivariate limit of detection |
| LOQ | `10 · s_0 / SEN` | multivariate limit of quantification |
| selectivity `SEL` | `‖b ⊙ s_target‖ / ‖b‖` | net-signal fraction, 0~1, closer to 1 = less interfered |

#### Procedure
1. Build the calibration model (calibration set covers the range).
2. Acquire blanks: pure / deionized water, >= 10 independent replicates under the same conditions, **retain noise**.
3. Pass blanks through the **same** preprocessing chain → `X_blank_p`.
4. Call `figures_of_merit(card.estimator, X_blank_p, X_cal=X_train_p)`.
5. Read `SEN / γ / LOD / LOQ / SEL`.

#### Caveats
- `n_blank` must be **>= 2** (>= 10 recommended); with `n_blank=1`, `s_0` silently degenerates to 0 and `LOD=0` fails.
- Blanks must pass the preprocessing chain and retain noise; feeding raw spectra or `LOD/2` substitutes makes the
  predictions identical, and `SEN/LOD` degenerate to `inf` (the code warns at `figures_of_merit.py` lines 99–103).
- `SEN` is **scale-dependent** (affected by y-scaling and the calibration concentration range); it is only for
  horizontal comparison among candidate models, not an absolute physical sensitivity.
- Cross-day drift should not be counted into blank noise (handled by `drift.py`).

#### References
- Lorber, A.; Faber, N. M.; Kowalski, B. R. *Net analyte signal calculation in multivariate calibration.* **Anal. Chem.** 1997, 69(9), 1620–1626. (the NAS framework, multivariate basis of FOM)
- Olivieri, A. C.; Faber, N. M.; Ferre, J.; Boque, R.; Kalivas, J. H.; Mark, H. *Guidelines for calibration in analytical chemistry. Part 2. Multivariable calibration.* **Pure Appl. Chem.** 2006, 78(3), 633–664. (IUPAC multivariate calibration guidelines, with FOM definitions)
- IUPAC. *Nomenclature in evaluation of analytical methods including detection and quantification capabilities (IUPAC Recommendations 1995).* **Pure Appl. Chem.** 1995, 67(10), 1699–1723. (source of LOD/LOQ terminology and the 3.3 / 10 coefficients)
- ISO 11843 series *Capability of detection* (detection/quantification capability assessment, univariate; multivariate extends it).
- IUPAC *Compendium of Chemical Terminology* (Gold Book): definitions of sensitivity, limit of detection, etc.
- National-standard context: GB 3838-2002 Environmental Quality Standards for Surface Water (water classes and Class V boundary); HJ 915.3-2024 (UV-absorption online water monitors, Part 3); per-parameter laboratory methods are in the table above (GB 11892-89 / HJ 828-2017 / HJ 636-2012 / GB 11893-89 / HJ 535-2009 / HJ 1075-2019) — these give **laboratory reference-method** LODs, not model FOM.

---

### 2. Spectrometer Hardware Evaluation

#### Concept and Definitions
Hardware evaluation targets the **spectrometer itself**, a different layer from "analytical-method FOM". Hardware
dimensions of concern in UV water-quality online monitoring:

- **Wavelength accuracy / drift**: whether peak positions shift (directly affects PLS modeling and cross-instrument reuse).
- **Photometric / gain consistency**: whether per-wavelength gain and offset are stable.
- **SNR, dark noise, baseline stability**: determine the measured `s_x` (hardware noise is part of method noise).
- **Stray light**: raises the baseline and lowers the absorbance ceiling.
- **Resolution**: whether adjacent absorption peaks can be resolved.

> The computation functions in this section and in "Section 3 Incoming Inspection" **do not assume a scanning
> mechanism**; they also apply to **fiber-optic / array spectrometers (fixed grating, no mechanical monochromator)**.
> Failure-mode differences are in Section 3 "Applicability: fiber-optic / array spectrometers".

#### Current Implementation: Inter-Instrument Difference Diagnosis
The library does not directly run item-by-item incoming inspection; instead it provides **gating diagnosis** of
cross-instrument difference and transfer feasibility (`aimeta/transfer/diagnose.py`):

- `estimate_wavelength_shift`: estimate the wavelength offset (nm) of slave vs master via cross-correlation of mean spectra.
- `estimate_gain_offset`: per-wavelength gain / offset (master ≈ gain·slave + offset); if gain varies markedly with
  wavelength, the difference has per-wavelength structure, in which case PDS beats DS.
- `residual_spectrum`: difference of master/slave mean spectra, for judging global tilt vs local structure.
- `recommend_method`: auto-select DS / PDS / SBC by offset / sample count; align the wavelength axis first when offset > 0.5 nm.

#### Instrument Profiles (Specs and Strategy)
`configs/instruments/*.yaml` records each instrument's optical specs and transfer strategy — the baseline profile
for hardware evaluation:

- `optics`: wavelength start/end, step, path length.
- `transfer`: standardization sample set, recalibration period, fallback method.

#### Reference Implementation
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

#### Caveats
- Wavelength drift > 0.5 nm must be aligned first, otherwise DS / PDS will fit a misaligned wavelength mapping.
- SNR, stray light, resolution, etc. **can be measured directly by this library's `aimeta/hardware_eval.py`**
  (functions in Section 3: `signal_to_noise` / `stray_light` / `resolution`, etc.); pass/fail is checked against
  the instrument datasheet / verification regulation criteria (JJG 178, ASTM E275).
- This part (`transfer/diagnose`) only **diagnoses cross-instrument differences** from measured spectra; it does not
  replace item-by-item incoming inspection, which requires separately preparing reference materials and procedures
  (see Section 3).

#### References
- Inter-instrument calibration / transfer and difference diagnosis (DS / PDS / SBC / GLSW): see `aimeta/transfer/` and `docs/重构方案.md`.
- UV-Vis spectrophotometer performance evaluation (wavelength accuracy, photometric accuracy, stray light, baseline
  flatness, resolution, noise): ASTM E275 series; domestic metrological traceability: **JJG 178 Verification
  Regulation of Ultraviolet, Visible, Near-Infrared Spectrophotometers**.
- General requirements for the competence of testing laboratories: **ISO/IEC 17025**.

---

### 3. Spectrometer Hardware Incoming-Inspection Protocol

#### Concept and Definitions
Before deploying an instrument to the field or reusing an existing model, run **item-by-item incoming inspection**
per the metrological verification of UV-Vis spectrophotometers. This library's `aimeta/hardware_eval.py` provides
functions that **compute metrics from measured spectra**; reference materials / filters are prepared by the user per
the verification regulation. Main bases: **JJG 178** and **ASTM E275**.

#### Inspection Items and Required Reference Materials
| Item | Required tool / reference material | Procedure / computation | Typical pass criterion |
|---|---|---|---|
| Wavelength accuracy | holmium glass / holmium-oxide standard (characteristic peaks 279.4 / 287.5 / 333.7 / 360.9 / 418.5 / 453.2 / 536.2 / 637.5 nm); or known emission lines (mercury / argon lamp) | measure the standard spectrum; locate each reference peak within ±window via parabolic refinement (`find_peak_wavelength`); error = measured − nominal, take `max_abs_error_nm` (`wavelength_accuracy`). | error ≤ 0.5 nm |
| Photometric accuracy | neutral-density filter / potassium dichromate standard solution (known absorbance points) | measure the standard spectrum; linear-interpolate at certified wavelengths; error = measured − certified, take `max_abs_error` (`photometric_accuracy`). | ≤ 0.002 A (or 0.3 %T) |
| Stray light | NaI / Corning cut-off filter (at the fully-cut wavelength, e.g. 340 nm) | measure the transmittance spectrum with the cut-off filter; interpolate at the wavelength where the filter is nominally fully absorbing (≈0%T); the residual transmittance is the stray light, take `max_Tpct` (`stray_light`). | ≤ 0.05 %T |
| Dark noise | none (block incident light: shutter closed / dark room) | acquire >= 2 repeated dark spectra under light blocking; per-wavelength sample std (ddof=1); report `std_max` / `std_mean` (`dark_noise`). | ≤ 0.0005 (example) |
| Baseline flatness | air / blank (100%T reference) | repeat the 100%T reference line multiple times; per-wavelength std across repeats gives baseline repeatability `repeatability_max`; if `ideal` is given, also compute the max deviation `deviation_max` of the mean baseline from `ideal` (`baseline_flatness`). | repeatability ≤ 0.001 (example) |
| SNR | stable light source / pure water (or any known stable signal) | repeat >= 10 times under the same conditions; at a specified wavelength (or full-band mean), `SNR = signal mean / repeat std` (ddof=1); returns `inf` when noise is 0 (`signal_to_noise`). | higher is better |
| Resolution | narrow emission-line source such as mercury / argon lamp | measure the emission-line spectrum; parabolic-refine the peak height within peak ±window; linear-interpolate the full width at half maximum FWHM (nm) (`resolution`). | FWHM ≤ 2 nm (typical for 1 nm slit) |
| Standard-solution linearity (per wavelength) | none | measure multi-concentration gradient spectra of various standards; from the third concentration onward, do per-wavelength linear regression and compute cumulative R² (progressively accumulate fits along the concentration gradient; per wavelength via `sklearn.metrics.r2_score` or `np.polyfit`); output the per-wavelength R² curve. | judge qualitatively by the R² curve, and use contiguous high-R² bands (e.g. ≥ 0.99) to determine the linear range. |

#### Reference Implementation
```python
from aimeta.hardware_eval import evaluate_instrument

res = evaluate_instrument(
    wavelengths=wl,
    dark_repeats=dark,                 # (n, p) light-blocked repeats
    holmium_spectrum=holmium,           # measured holmium glass
    ref_peaks=[360.9, 418.5, 536.2],    # selected peaks
    transmittance=t_pct,                # %T spectrum
    check_wl=[340.0],                   # stray-light check wavelength
    line_spectrum=mercury,              # mercury line
    line_peak_wl=546.1,
)
print(res.report())     # metrics / pass_criteria / passed
```

#### Caveats
- Wavelength accuracy needs **sufficiently sharp and separated** reference peaks; parabolic refinement
  (`find_peak_wavelength`) is more accurate than a plain argmax.
- Stray light must be measured at the wavelength where the filter is **fully cut**; incomplete cut underestimates it.
- Dark noise / baseline flatness both require **multiple repeats** (>= 10 groups); a single measurement is meaningless.
- These hardware metrics are **one source of the method's `s_x` (the spectral noise in FOM)**: if hardware noise is
  large, the method's LOD / LOQ will inevitably be poor.

#### Applicability: Fiber-Optic / Array Spectrometers (no mechanical grating)
The methods here also apply to **fiber-optic / array spectrometers (fixed grating, no mechanical scanning
monochromator, e.g. Ocean Optics / Avantes)** — all hardware-metric functions in this library depend only on
"wavelength array + spectrum array" and assume no scanning mechanism. Differences are mainly in failure modes and
inspection frequency:

| Item | Scanning monochromator | Fiber-optic / array spectrometer | Approach here |
|---|---|---|---|
| Main cause of wavelength drift | leadscrew / gear backlash | **temperature-driven** pixel↔wavelength calibration drift (up to several nm) | do **periodic wavelength recalibration** (Hg-Ar / Ne lamp or holmium / Pr-Nd standard); more critical for online monitoring (`drift.py`) |
| Resolution | adjusted by slit / step | **fixed** (slit + grating + pixel), not adjustable | verify the acceptance spec via the FWHM of the mercury / argon line; SNR cannot be raised by widening the slit |
| SNR | depends on scan count | depends on **integration time + signal averaging** | measure SNR / dark noise **under the actual operating integration time / averaging count** |
| Stray light | double monochromator can suppress effectively | fiber-coupled and no double monochromator, **usually higher** | still measure, and pay more attention (NaI / cut-off filter) |
| Dark / readout noise | photocell dark current | CCD / CMOS dark current, readout noise (cooling matters) | measure directly with `dark_noise` (shutter closed) |
| JJG 178 scanning clauses | applicable | not applicable (no scanning mechanism) | the remaining items (wavelength / photometric / stray light / baseline / noise / resolution) use **identical** computation |

For array spectrometers, the computation is the same as for scanning monochromators (the corresponding JJG 178
clauses carry over); the focus is temperature-drift calibration, SNR under integration time, and stray light.

#### References
- **JJG 178 Verification Regulation of Ultraviolet, Visible, Near-Infrared Spectrophotometers** (domestic
  metrological traceability; wavelength / photometric / stray light / baseline / noise / resolution).
- ASTM E275 series *Standard Practices for Describing and Measuring Performance of UV, Visible, and Near-IR
  Spectrophotometers* (wavelength accuracy, photometric accuracy, stray light, baseline flatness, resolution, noise).
- ISO/IEC 17025 (general requirements for testing-laboratory competence; overall traceability framework).

---

## Deployment to Industrial PC

`edge/` is an independent minimal runtime; deployment-side dependencies match aimeta (numpy / scipy / PyWavelets);
chains composed solely of pure-numpy compilable operators can still run without scipy/pywt. Only linear models
(PLS / Ridge / Lasso / OLS / PCR) and compilable operators are supported.

```python
from edge.runtime import export_card, load_edge_model, verify_against_card

export_card(card, "deploy/TN_ai17", X_reference=X_train)   # export on the dev machine
```

```python
import numpy as np
from edge.runtime import load_edge_model

m = load_edge_model("deploy/TN_ai17")    # on the industrial PC
y = m.predict(np.load("new_spectra.npy"))
```

After export, verify consistency once:

```python
ok, diff = verify_against_card(m, card, X_test)   # should be <= 1e-8
```

### Standalone Inference-Script Deployment (river_inference.py)

Besides the `edge/` runtime, the repo provides `river_inference.py`, a standalone inference script independent of the
aimeta dependency tree (placed in the repo root together with `requirements.txt`; full deployment SOPs are in
`deploy/实际水样模型部署和测试.docx` and `deploy/标液模型部署和测试.docx`). It reads ARFF spectra and outputs JSON
results, supporting `--variant pca` (default) and `range` (the original Linux pipeline). The script bundles all
preprocessing (Wiener → SG → wavelet → SNV); models are saved as pickle/joblib and already include scalers such as
`MeanCenterer`/`YAutoScaler`, so **any sklearn-family model — linear, random forest, LightGBM, AdaBoost, etc. — can
be loaded and used for prediction directly**, without the "linear-only" restriction.

The industrial PC only needs Python 3.12 + the bundled `requirements.txt` (joblib / numpy / scipy / scikit-learn /
pywavelets), installed via `pip install -r requirements.txt`; the full aimeta dependency tree is not required.

Brief flow:

1. copy the `.pkl` model(s) into `<deploy_dir>\data\models\` (a second model set goes into `models\wider_range\`);
2. copy `river_inference.py` and `requirements.txt` into `<deploy_dir>\`;
3. `conda create -n reg_venv python=3.12 && conda activate reg_venv && pip install -r requirements.txt`;
4. in the site's `strategy_ai???.ini`, under `[Python]`, set `reg_exe = ...\reg_venv\python.exe`;
5. test: `python river_inference.py -file_name SpectrumData_****.arff -re_train False`,
   results in `result_****.json`.

---

## Tests

```bash
python -m pytest -q
```

| File | Coverage |
|---|---|
| `test_registry.py` | registry pluggability |
| `test_preprocessing.py` | derivative operators' numerical equivalence to scipy, SNV semantics, chain reproducibility |
| `test_transfer.py` | transfer methods on synthetic dual instruments, sample selection, difference diagnosis |
| `test_metrics.py` | metrics, class assignment, acceptance rate, alarm metrics |
| `test_mcr.py` | curve-resolution initialization, convergence, constraints, uncertainty |
| `test_smoke.py` | end-to-end: train → export → consistency → monitoring → wavelength selection |

---

## Notes

- **Deployment-side dependencies match aimeta (numpy / scipy / PyWavelets)**. Pure-numpy compilable operators
  (``savgol`` / ``snv`` / ``msc`` / scaling) still run on the zero-dependency fast path; ``wavelet`` and other
  operators requiring external libraries delegate to registered operators, and the industrial PC must have the
  corresponding dependencies installed (aimeta's main dependencies already include scipy / PyWavelets; `pip install
  aimeta` satisfies this).
- **`snv` normalizes spectrum by spectrum**; to column-scale the whole matrix, use `column_scale`.
- **`edge` only supports linear models**. Nonlinear models (GPR, LGBM, MLP) need the full dependency set installed
  on the industrial PC and are used directly via `ModelCard`.
- **Transfer needs paired samples**: at least 5 samples measured on both instruments; otherwise only
  `SlopeBiasCorrection` can be used for response correction.
- **`ModelCard` validates the wavelength axis**: at inference, if the **number of wavelengths** (columns of `X`)
  differs from the card, it errors directly; if you also pass `wavelengths` to `predict` (or use a `SpectrumSet`
  with a wavelength axis), it further compares the **actual wavelength values** against the training grid
  (`np.allclose`). Passing only `X` without wavelengths checks only the count — resampling / misalignment of the
  wavelength grid would go undetected. For production, always pass the wavelength axis.
- **When computing metrics, blank samples must pass the preprocessing chain first**. Feeding raw spectra may put
  predictions below the LOD and replace them with LOD/2, causing sensitivity and LOD to degenerate to `inf`
  (a warning is emitted in that case).
