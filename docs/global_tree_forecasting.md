# Phase 2C: global direct multi-lead LightGBM

`GlobalTreeForecaster` in `src/eso/forecasting/ml.py` implements one pooled
LightGBM regressor per forecast origin. The same learner serves every selected
GSP and every lead, with `lead` and categorical `gsp_id` as explicit predictors.
It uses the existing `fit(history)` / `predict(timestamps)` interface and produces
the usual GSP-by-UTC-timestamp DataFrame. XGBoost, weather, deep learning and
prediction intervals remain later milestones.

## Data and leakage policy

`fit(history)` builds direct training examples only inside the history supplied
by the outer chronological backtest. Inner origins use their own prefixes for
lag and rolling features. Their full label horizons must end before the outer
forecast origin. Both origin-relative and guarded target-relative seasonal lags
are retained, along with target calendar features and lead. No validation or final
test labels are passed into fitting, preprocessing, early stopping or encoding.

Training rows with missing targets are excluded; missing predictor values remain
NaN. Zero targets/predictors are observed values. Numeric features are converted
to float32 for memory use, without fitting a scaler. GSP categories are fitted
only from observed training examples. There is no target encoding or global
imputation. GSPs with no observed training targets receive NaN predictions and a
diagnostic, rather than being treated as an unseen-GSP transfer experiment.

The model uses [LightGBM's native missing-value and categorical support](https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html).
It fixes CPU execution, seed, thread count, deterministic mode and column-wise
histogram building. The first learner uses an unweighted squared-error objective,
fixed boosting rounds and explicit hyperparameters; no automatic search or
early stopping is performed. Larger-volume series can dominate that objective,
so inspect per-GSP errors and MASE alongside pooled MAE/RMSE. Exact results across
different package versions/platforms are not guaranteed; package versions are saved.

The initial demand-only model includes clock/calendar features but receives no
holiday calendar: `holiday` is unknown (NaN), rather than silently false. No weather
or other external covariates are joined. A future holiday/calendar study needs
complete explicit regional date flags with appropriate announcement vintages.
The benchmark remains retrospective latest-revision GP9 with zero publication
delay assumed, consistent with the statistical subsystem.

## Run presets

| Preset | Initial history | GSP population | LightGBM fitting history |
| --- | --- | --- | --- |
| `configs/phase2_ml_smoke.json` | 14 elapsed days | 20 deterministic low/medium/high-volume representatives | Latest 672 half-hours |
| `configs/phase2_ml.json` | 180 elapsed days | All initially eligible GSPs | Entire expanding history |

Both presets run persistence, elapsed daily naive, weekly naive and LightGBM on
identical origins/targets/GSPs within each run, for 8- and 48-step horizons. Outer
evaluation origins are weekly (`step=336`); inner training origins are daily
(`lightgbm.training_step=48`). The maximum configured evaluation horizon is also
the LightGBM training horizon. Predictions beyond the fitted lead range are refused.

`selection_min_observed_fraction=0.95` selects eligibility using only the initial
training window. It is fixed for all later origins, and is independent of model
failures or future coverage. All-zero series are retained if sufficiently observed.
`series_selection.csv` records eligibility, reason, coverage, volume strata and
selection for every input GSP. Existing configs default to a zero coverage threshold
to preserve their prior behaviour. Explicit GSP IDs failing an active threshold
are refused rather than silently dropped.

Install the optional ML dependency, then run from the repository root:

```powershell
$env:PYTHONPATH = 'src'
.\venv\python.exe -m pip install -e ".[ml]"

# Runnable smoke validation using the existing 61-day panel; use a fresh output path.
.\venv\python.exe -m eso.forecasting.pipeline --config configs/phase2_ml_smoke.json --input output/forecast_input --output output/my_ml_smoke_validation --test-start 2022-07-15T00:00:00Z
```

For the study preset, prepare a longer observed panel and set its fixed final
test boundary. The following variables must match that panel:

```powershell
$forecastInput = 'output/your_verified_long_history_panel'
$forecastTestStart = 'YOUR_FIXED_TIMEZONE_AWARE_TEST_BOUNDARY'
.\venv\python.exe -m eso.forecasting.pipeline --config configs/phase2_ml.json --input $forecastInput --output output/my_ml_study_validation --test-start $forecastTestStart
```

The study needs 180 days before validation plus complete validation/test horizons;
the existing 61-day panel cannot satisfy it. Prefer longer verified settlement
history where available. Defaults evaluate future periods for known GSPs, not
held-out-GSP spatial transfer. Freeze model choices, settings, selection rules and
the boundary before explicitly adding `--split test`. Earlier observed test values
can enter subsequent test origins, as in the existing walk-forward contract.

SARIMA/ETS remain available through their presets. To compare them head-to-head
with LightGBM, add their names to `models` in a copied config and use the same
GSP subset, panel and origins. Install both extras with `pip install -e ".[ml,statistical]"`.
Do not rank the earlier six-GSP statistical smoke run against the 20-GSP ML smoke
run: their case populations, origins and fitting histories differ. Per-series
SARIMA/ETS fits across all GSPs remain an explicit runtime choice, not an ML default.

## Memory and artifacts

Before constructing training features, the model calculates the number of
inner-origin × GSP × lead rows. Above `max_training_rows` (default five million)
it fails with an explanatory error. Nothing is silently sampled or truncated.
Increase `training_step`, explicitly set a trailing `train_window`, choose a subset
or raise the budget with suitable memory. This guard limits rows, not bytes:
feature construction/concatenation and LightGBM fitting can still use substantial RAM.
`min_training_rows` counts observed training labels; insufficient history/labels
give missing forecasts with diagnostics, without a naive fallback.

Backtesting now creates prediction rows in vectorized blocks, preserving the old
case identity, row order and metric semantics. The complete prediction table and
metric groupings still live in memory. Streaming/partitioned evaluation remains
future work for larger multi-year studies.

Standard per-GSP, pooled, common-case and per-lead metrics are unchanged.
`fit_diagnostics.csv` has per-GSP outcome rows, training-example counts, fitting
range and actual boosting iteration count. These are diagnostics for a **shared
global fit**, not separate GSP models. The presets enable `save_models=true`.
Each origin writes under `models/lightgbm/origin_*/`:

- `model.txt`: native LightGBM booster.
- `schema.json`: feature order/specification, GSP categories, hyperparameters,
  training-example/target ranges and missing-label counts.
- `observed_panel.csv`: the minimal observed history tail needed to reconstruct
  forecast features, without storing the full training table.
- `feature_importance.csv`: split counts and gains, descriptive rather than causal.

`GlobalTreeForecaster.load(directory)` restores the saved feature schema,
category vocabulary, booster and required history for `predict(timestamps)`.
All output directories remain immutable. Input/source hashes, resolved settings,
Python, LightGBM and scikit-learn versions are recorded in `run.json`.

Validation tests cover one pooled fit across GSPs/leads, fold-fitted categories,
missing targets versus predictors, zero/negative values, future perturbation,
purged label boundaries, fixed-history eligibility, deterministic repeat fitting,
budget refusal before feature allocation, native model round trips and full CLI
artifacts. See [the initial ML smoke results](global_tree_results.md).
