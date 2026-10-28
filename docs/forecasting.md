# Phase 2 forecasting contract

Phase 1 is research-complete for now. Battery dispatch simulation and multi-year
validation remain deferred extensions. Its screening labels are research hypotheses;
its profile-completion scores are not forecast accuracy scores.

The initial Phase 2 subsystem lives in `src/eso/forecasting/`. It implements the
observed-data contract, persistence, daily and weekly seasonal naive forecasts,
expanding-window evaluation, per-GSP/pooled/per-lead MAE/RMSE/MASE, common-case
comparisons, and representative-series SARIMA/ETS benchmarks. See the
[statistical forecasting review guide](statistical_forecasting.md) for model
settings, failure policies, historical data and configuration-driven execution.
Global boosted trees, LSTM/GRU and intervals remain later milestones; no model
superiority or operational accuracy is claimed yet.

## Input and timing

Pass a Phase 1 output directory containing `observed_panel.csv`, or both
`cleaned_panel.csv` and `value_provenance.csv`. The fallback discards every value
not labelled observed, observed_import or observed_export. Phase 1 interpolation
and seasonal imputation use future observations and must not be reused.

Rows are unique GSP IDs; columns are strictly increasing, contiguous, timezone-aware
half-hour interval starts, converted to UTC. Missing observations remain NaN.
Negative signed net import and constant series are retained. Units remain source
Meter Volume units; this code does not silently convert energy to MW.
Use contiguous full-calendar panels, not weekday-only or representative profiles.

An origin is the first predicted interval start. History ends one half-hour earlier.
This initial retrospective benchmark assumes immediate observation availability and
uses Phase 1's latest revised values. It prevents future-target and preprocessing
leakage but is NOT an operational as-of benchmark. Before operational claims, add
publication-delay and per-origin revision cutoffs to raw-data ingestion. Weather
must likewise be forecast-vintage data available at the origin.

## Evaluation

Default horizons cover all leads 1..8 (4 hours) and 1..48 (24 hours). All models
receive identical folds and training history. Factories create a fresh model for
each origin with `fit(history)` and `predict(future_timestamps)` methods. Predictions
must preserve the GSP and timestamp axes. Fit transformations inside each fold;
future models must also purge training examples whose labels cross the origin.

Set an explicit timezone-aware final test boundary. Validation windows must end
before it; only complete maximum-horizon windows are evaluated. Validation is the
default command. Run test explicitly after freezing model choices and parameters.
During test walk-forward, earlier observed test values enter subsequent training
histories, as they would in deployment. The framework cannot enforce that a user
never consults test scores during model selection.

Daily/weekly naive models repeat the last 48/336 elapsed UTC values, not local-clock
slots across DST. Missing donors produce missing predictions; there is no silent
fallback. Persistence repeats the last observation. Default origins are 48 steps
apart; use `--step 8` for more frequent operational origins. Overlapping forecasts
are separate cases, so pooled scores may count one target at several origins.

MASE uses each GSP's mean absolute lag-48 differences from observed training pairs
at that origin. A zero or unavailable scale yields undefined MASE, reported with
its valid count. MAE/RMSE retain those series. Scores report observed-target counts,
scored counts and coverage so missing predictions cannot masquerade as accuracy.
Pooled metrics are micro averages over GSP forecast cases, not errors of summed
system demand. Compare coverage and common scored cases before ranking models.
Intervals and summed-demand evaluation are not implemented in this foundation.

## Run

For step-by-step input preparation, validation/test commands and troubleshooting,
see the [forecasting run guide](../README_FORECASTING.md).

From the repository root:

```powershell
$env:PYTHONPATH = 'src'
.\venv\python.exe -m eso.forecasting.pipeline --input output/your_contiguous_run --output output/forecast_validation --test-start 2022-08-15T00:00:00Z
```

The example boundary must be changed to a timestamp in your panel. Defaults need
14 days of initial history, at least one full validation day, and one full test day.
This is a software smoke minimum. The statistical study configuration requires
180 initial days; prefer 6–12 months or more for the eventual modelling study.
The CLI prints `smoke_benchmark` for histories shorter than 180 elapsed days and
records the assessment in `run.json` under `training_history_assessment`. This
also accounts for shorter SARIMA/ETS `train_window` settings. Longer histories
are labelled `long_history_benchmark`; this describes duration only and does not
establish adequate observed coverage, seasonal diversity or model quality.
Prefer 1–2+ years where verified settlement coverage makes it practical.
A five-day Phase 1 run is insufficient. Set `--min-train` and `--step` in half-hours.
Use a new output directory for each run; existing directories are refused.

Artifacts: `predictions.csv` (origin, target timestamp, lead, horizon, model, GSP,
actual, prediction and training MASE scale), `metrics_per_gsp.csv`,
`metrics_pooled.csv`, `metrics_by_lead.csv`, `metrics_common_cases.csv`,
`metrics_common_cases_by_lead.csv`, `series_selection.csv`, `config.json`,
`run_status.json`, and `run.json` (configuration, hashes, versions and limitations).
Statistical runs also produce `fit_diagnostics.csv` and optionally fitted models.
Keep the source Phase 1 run alongside these artifacts for its revision/source audit.

The [feature foundation](forecasting_features.md) now provides origin-relative
lags/rolling statistics, known-future local calendar features and purged training
examples. ML fitting and fold-specific transformations remain to be implemented.

Next implement a global tree model with
GSP identity and origin-available lags/rolling features plus known future calendar
features. Fit encoders/scalers only on training folds. Add a global sequence model,
calibration-only intervals, seasonal/day-type/regime error slices, then join Phase 1
outputs with explicit availability rules.
