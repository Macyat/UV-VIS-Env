# Glossary

> English | [中文](术语表.md)

One term, one explanation. Values and code locations are written inside the explanation. Organized by topic:

## 1. Water-quality Parameters and National-standard Classification

| Term | Explanation |
|---|---|
| **Water-quality parameter** | The monitored water-quality indicators: CODMn (permanganate index), COD (chemical oxygen demand), TN (total nitrogen), TP (total phosphorus), AN (ammonia nitrogen), TUR (turbidity). |
| **Water class (Class I–V)** | Surface-water quality classes in GB 3838; Class I is best, Class V worst. The `ranges` in `params.yaml` are the class boundary values. |
| **Below Class V (劣Ⅴ类)** | A reading at or above the Class V limit — worse than the worst class — which triggers an over-limit alarm. |
| **Standard limit `ranges`** | The I–V class boundary concentrations for each parameter, taken from GB 3838 Table 1; the baseline for all thresholds. |
| **GB 3838-2002** | *Environmental Quality Standards for Surface Water*; the source of thresholds. |
| **HJ 915.3-2024** | *Technical specifications for operation and maintenance of automatic surface-water quality monitoring stations*; the procedural basis for range/alarm setting and abnormal-condition handling. |
| **Over-limit alarm** | A reading ≥ the Class V limit (`ranges[-1]`) is judged "below Class V" (over-limit), a water-quality-class alarm; see `alarm_accuracy`. |
| **Gold standard** | Concentration measured by the laboratory chemical method, used as the authoritative reference (true value) for sample concentration labels. |
| **Standard analytical method `standard`** | The chemical method for each parameter: CODMn→GB 11892-89, COD→HJ 828-2017, TN→HJ 636-2012, TP→GB 11893-89, AN→HJ 535-2009, TUR→HJ 1075-2019. |
| **Abnormal-condition handling** | On-site O&M actions prescribed by HJ 915.3-2024; the procedural entry point when readings are abnormal. |

## 2. Bounds and the Three-tier Handling (LOD / Review Limit / Theoretical Upper)

| Term | Explanation |
|---|---|
| **Detection limit `lower_bound`** | The LOD of the national-standard laboratory method; predicted values below it are replaced by LOD/2 and flagged `below_lod`. |
| **Model's own spectral LOD** | Computed from data by `figures_of_merit`; usually much larger than the national-standard method LOD, so the national-standard value must not be used in its place. |
| **Review limit `review_upper`** | = 1.5 × Class V; values above it are still displayed but flagged `above_review` (requires review). |
| **Theoretical upper `theoretical_upper`** | = `min(device theoretical upper, river theoretical upper)`; physically impossible, so not displayed (set to NaN) and flagged `not_display`. |
| **Device theoretical upper `device_theoretical_upper`** | Defaults to 2 × the highest calibration standard (computed at training), can be overridden manually; sensor saturation / out of method range — the device cannot measure it. |
| **River theoretical upper `river_theoretical_upper`** | Defaults to 2 × Class V, can be overridden manually; a concentration physically impossible for this river. |
| **Three-tier handling** | The `apply_bounds` rules: below LOD → LOD/2; the review zone keeps the original value; above theoretical upper → NaN. |
| **Range/alarm upper (HJ 915.3)** | Instrument range upper / QC alarm, taken as 2 × standard limit, corresponding to this system's theoretical upper; distinct from the "over-limit alarm (Class V)". |
| **Review (re-verification)** | After a spectral reading exceeds the review limit, arrange a laboratory chemical-method confirmation. |
| **Calibration standard** | The reference concentrations in the training calibration set; their maximum is used to auto-compute the device theoretical upper (×2). |

## 3. Analytical Figures of Merit (FOM)

| Term | Explanation |
|---|---|
| **Figure of merit (FOM)** | The `{SEN, gamma, s_x, s_0, LOD, LOQ, SEL}` output by `figures_of_merit` in one call. |
| **Regression vector `b`** | The model's coefficient vector along the concentration direction (`coef_` for linear models, numerical differentiation for nonlinear ones). |
| **Sensitivity `SEN`** | `1 / ‖b‖`; larger is more sensitive. |
| **Analytical sensitivity `γ`** | `SEN / s_x`, the signal-to-noise ratio per unit concentration. |
| **Blank (method blank / reagent blank)** | A solvent with the same matrix as samples but without the analyte — for UV water-quality work, pure water / deionized water. Used to estimate `s_x` and `s_0`. Strictly follows this definition; when absolute-zero surface water is unavailable, a "near-LOD low-concentration water sample" may approximate, but pure/deionized water is preferred. Measuring `s_x`/`s_0` requires ≥10 independent replicates under the same conditions (at least ≥2, otherwise `s_0` degenerates). |
| **Spectral noise `s_x`** | The residual standard deviation of blank samples (pure/deionized water — same background as samples, no analyte) after detrending. Requires ≥10 independent blank replicates. |
| **Blank prediction std `s_0`** | The standard deviation of predictions of blank samples (pure/deionized water), i.e. the prediction spread over repeated pure-water measurements; used for LOD/LOQ. Blanks must retain raw noise and not be replaced by LOD/2, otherwise they collapse to a constant and `s_0 = 0`; with `n_blank=1` `s_0` silently degenerates to 0, so ≥2 replicates are needed. |
| **Detection limit `LOD` (multivariate)** | `3.3 · s_0 / SEN`; the univariate 3σ/slope convention must not be used. |
| **Quantitation limit `LOQ`** | `10 · s_0 / SEN`. |
| **Selectivity `SEL`** | The fraction of the target component's net signal in the total signal (0–1). |

## 4. Modeling and Data Contract

| Term | Explanation |
|---|---|
| **Spectral soft sensing** | Using UV-Vis spectra + a regression model to estimate water-quality concentrations, replacing/aiding the chemical method. |
| **Multivariate calibration** | Mapping the whole spectrum (multiple variables) to concentration, rather than single-wavelength Lambert–Beer. |
| **Preprocessing chain `preproc_chain`** | A YAML-reproducible sequence of preprocessing steps (savgol / snv / msc / mean_center, etc.). |
| **Fingerprint `fingerprint`** | A hash of preprocessing chain + wavelength axis + model name, checked at deployment for consistency. |
| **Wavelength axis `wavelengths`** | The wavelength grid that must match point-by-point at deployment. |
| **ModelCard** | The training artifact = estimator + preprocessing chain + fingerprint + metrics + instrument ID. |
| **Estimator `estimator`** | The regressor wrapped by `WaterQualityModel` (PLS/Ridge/Lasso/…). |
| **x/y scaling** | `x_mean_` / `y_mean_` / `y_scale_`, the centering/scaling shared by training and inference. |
| **Data version `data_version`** | A hash of the training data, written into the ModelCard for provenance. |

## 5. Process Monitoring (MSPC and Control Charts)

| Term | Explanation |
|---|---|
| **MSPC** | Multivariate Statistical Process Control, compressing the whole spectrum into two statistics: T² and SPE. |
| **T² (Hotelling)** | Deviation in the direction of normal internal model variation. |
| **SPE (Q residual)** | Variation the model cannot explain; probe fouling / bubbles / foreign matter most often show up here. |
| **Contribution plot `contribution`** | Decomposing SPE back to each wavelength to locate "which wavelength region is problematic". |
| **Control chart** | Shewhart / CUSUM / EWMA, in `control_charts`. |
| **ARL** | Average run length, for assessing control-chart sensitivity (≈ 370 at 3σ). |

## 6. Deployment and Transfer

| Term | Explanation |
|---|---|
| **edge runtime** | The on-IPC deployment runtime, with dependencies matching aimeta (numpy / scipy / PyWavelets); a pure-numpy chain can run with zero extra dependencies. |
| **train/serve consistency** | Training-side and edge-side predictions match element-by-element (`verify_against_card`). |
| **Instrument transfer `transfer`** | DS / PDS / SBC / GLSW etc., making a model usable across instruments. |

## 7. Acceptance Metrics

| Term | Explanation |
|---|---|
| **Alarm accuracy `alarm_accuracy`** | The hit rate with which predictions trigger the "over-limit alarm" consistently with the national-standard true value (including false negatives/positives); one of the acceptance metrics; code `aimeta/metrics/water_standards.py`. Score-sheet column `alarm_acc`. |
| **Alarm error `alarm_err`** | The error rate of the alarm decision = 1 − `alarm_accuracy`; score-sheet column `alarm_err`. |
| **Daily R² `daily_r2`** | Grouped by sampling day (`day_idx`), computes per-day prediction-vs-truth R² (and Pearson r), measuring model stability on samples from "the same day"; `aimeta/metrics/water_standards.py:daily_r2`. Score-sheet columns `daily_r2_score` / `daily_pearson_r_score`. |
| **Acceptance rate `acceptance_rate`** | The fraction of samples whose predicted value vs national-standard true value falls within the allowed error band (absolute `abs_error_bound` and relative `mape_bound`), the qualified fraction for acceptance; code `aimeta/metrics/water_standards.py:acceptance_rate`. Score-sheet column `rate of reaching the standard`. |

## 8. Spectrometer Hardware Evaluation

| Term | Explanation |
|---|---|
| **Spectrometer hardware evaluation** | Evaluating the **instrument itself** and cross-instrument consistency (distinct from "3. FOM", which evaluates the analytical method / model). Functions are in `aimeta/hardware_eval.py`, per JJG 178 / ASTM E275; `evaluate_instrument` summarizes them in one call (computes only what data is provided, skips the rest). |
| **Wavelength accuracy** | The maximum absolute error (nm) between measured and nominal peak positions (holmium glass / mercury-lamp standard peaks), with parabolic refinement within ±window of each reference peak; `wavelength_accuracy` / `find_peak_wavelength`. Typical pass ≤ 0.5 nm. |
| **Photometric accuracy** | The maximum absolute error between measured absorbance and certified values (neutral-density filters / potassium dichromate) at certified wavelength points (interpolated); `photometric_accuracy`. Typical pass ≤ 0.002 A (or 0.3 %T). |
| **Stray light** | The residual transmittance (%T) measured at the fully cut-off wavelength of a cut-off filter; `stray_light`. Typical pass ≤ 0.05 %T. |
| **Dark noise** | The per-wavelength standard deviation (ddof=1) over ≥2 repeated dark spectra under blocked light (shutter closed / darkroom), reported as `std_max`/`std_mean`; `dark_noise`. |
| **Baseline flatness** | The per-wavelength cross-replicate std of repeated 100%T reference scans (`repeatability_max`), and the maximum deviation of the mean baseline from the ideal (`deviation_max`); `baseline_flatness`. |
| **Signal-to-noise ratio `SNR`** | Signal mean / replicate standard deviation (ddof=1), computed at a specified wavelength or averaged over the band; returns `inf` when noise is 0; `signal_to_noise`. |
| **Resolution `FWHM`** | The full width at half maximum (nm) of a narrow emission line (mercury / argon lamp), with parabolic peak-height refinement and linear interpolation for FWHM; `resolution`. Typical pass FWHM ≤ 2 nm. |
| **Standard-solution linearity (per wavelength)** | Multi-concentration standard solutions, per-wavelength cumulative linear regression (absorbance → concentration) with cumulative R², to determine each wavelength's linear range; `standard_solution_linearity` (plot `plot_linearity_r2`). R² closer to 1 and not decreasing with concentration means better linearity. |
| **FWHM** | Full Width at Half Maximum; measures a spectrometer's ability to resolve adjacent absorption peaks. |
| **Reference materials / filters** | For verification: holmium glass / holmium oxide (wavelength), neutral-density filters / potassium dichromate (photometric), NaI / cut-off filters (stray light), mercury / argon lamps (resolution). |
| **JJG 178** | The verification regulation for *UV, visible, near-infrared spectrophotometers*; the domestic metrological-traceability basis (wavelength / photometric / stray light / baseline / noise / resolution). |
| **ASTM E275** | The UV-Vis-NIR spectrophotometer performance-evaluation standard (wavelength accuracy, photometric accuracy, stray light, baseline flatness, resolution, noise). |
