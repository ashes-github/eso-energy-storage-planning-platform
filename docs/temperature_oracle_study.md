# Phase 2D: temperature feasibility and controlled oracle ablation

This study asks whether perfectly known future 2 m temperature adds demand
forecasting value to the frozen Phase 2C.1 M2 seasonal model. It is a
retrospective feature-feasibility experiment, not an operational weather forecast.

Run from the repository root:

```powershell
.\venv\python.exe -m eso.forecasting.temperature_study --output output/phase2d_temperature_oracle_2022_validation
```

Use a fresh output directory. The default inputs are:

- Reference: `output/phase2c1_long_history_2022_validation/M2_seasonal`.
- Demand: `output/phase2c1_observed_2021_2023`.
- Weather: `Data/gsp_temperature/temperature_halfhourly_c.parquet`.
- Spatial audit: `Data/gsp_temperature/spatial_coverage.csv`.

The temperature preparation method is documented in [GSP temperature](gsp_temperature.md).

## Frozen comparison

The reference run supplies the exact ordered 363-GSP population, 29 validation
origins every 13 elapsed days during 2022, and 8/48-step (4/24-hour) horizons.
Demand history expands from January 2021, with at least 365 days initially.
Inner training origins are spaced every eight elapsed days, with full horizons
ending before the outer origin. Targets are the observed, original-unit meter
volumes; missing targets are excluded, without filling or scaling demand.

M2 comprises categorical GSP identity, target calendar/lead features, origin
demand lags and guarded seasonal target lags. Its rolling-statistic lists are
empty. The treatment adds exactly one feature:

```text
oracle_temperature_c[gsp, target_timestamp_utc]
```

The unchanged learner uses 200 LightGBM trees, 31 leaves, learning rate 0.05,
L2 objective, regularization 1.0, seed 7 and four CPU threads. No temperature
transformations, interactions, lags, coverage predictors, new row weights,
hyperparameter search or early stopping are introduced.

All 363 selected GSPs remain. Weather is present for 357; `GLAG_P`, `HARG_P`,
`HUER`, `STOR_P`, `TONG1`, and `TOWH` receive native missing temperature.
Shared polygon aliases receive shared temperatures. Spatial coverage is used
for diagnostics, not selection or model input.

The runner verifies the demand-file hashes, package versions and fitting-source
hashes against M2. Reporting-only changes are recorded separately. It replays
each saved M2 model and checks its predictions against the cached CSV. The
oracle fit must have identical training audits, target counts, learner settings
and feature names apart from the single appended temperature. Comparison
merges are one-to-one and fail on missing, duplicate or altered cases.

Saved M2 and persistence/daily/weekly naive forecasts are reused after these
checks, rather than refitted unnecessarily. CSV round-trip differences in
actuals/scales are checked to 1e-12 tolerance before canonicalizing to the
reference values for common-case metrics. Predictions are checked on replay
to relative tolerance 1e-11 / absolute tolerance 1e-10.

## Feasibility EDA

Only 2021-2022 demand/weather is analyzed. The weather lookup table and demand
panel passed to modelling also exclude every timestamp from January 2023 onward.
Final-test target values are never used for fitting, EDA, tuning or scoring.

EDA exports fixed two-degree temperature bins overall and by GSP, local season,
hour band, weekday/weekend, initial-history demand-volume stratum and year.
An additional per-GSP/local-hour table supports all 24 hour-of-day slices.
The time-zone conversion is Europe/London for calendar slices; case timestamps
and joins remain UTC.

Raw demand curves remain available in CSVs. Standardized curves use each GSP's
2021 mean/std only, with both case-weighted and equal-GSP-weighted bin means.
Constant series have undefined standardized demand and correlations, but stay
in the modelling population and raw metrics. Plots suppress bins with fewer
than 200 paired cases or fewer than ten GSPs; underlying counts are retained.
There are no independence-based confidence intervals: these observations are
temporally autocorrelated and shared polygons can have identical temperatures.

Per-GSP feasibility also reports raw Pearson correlation and correlation after
demeaning temperature and demand within season/local-hour/daytype cells using
development data. This is descriptive adjustment, not a model feature or a
claim of a causal temperature effect. Temperature bins may have different GSP
composition; season, solar generation and demand composition remain confounders.

## Evaluation and artifacts

`report.md` in the output directory gives measured results and figure links.
Pooled MAE/RMSE/MASE and per-GSP benefits use identical common observed cases
across all five model labels. Reported horizon-H errors include leads 1 through
H. Undefined zero-scale MASE is excluded explicitly while the GSP stays in
MAE/RMSE. Relative improvements are undefined when baseline MAE is zero.

Additional metrics/benefits cover season, weekday/weekend, volume stratum,
individual lead and spatial-coverage bands. Coverage bands are fixed as no
boundary, zero land overlap, below 50%, 50-90%, and at least 90%.
Per-GSP win/loss/tie counts and macro MAE complement pooled scores. No claim
of statistical significance is made from this validation comparison.

Important artifacts:

- `protocol.json`: population, origins, settings, input/source hashes, versions.
- `origin_audit.csv`: verified replay, identical training targets and rows.
- `predictions/`: one paired Parquet per origin, with all GSPs retained.
- `eda/`: binned tables, GSP feasibility audit and static figures.
- `metrics_*.csv`, `benefits_*.csv`, `gsp_benefit_summary.csv`.
- `importance_aggregated.csv`: descriptive feature gain, not causal attribution.
- `models/`: native oracle boosters, schemas, demand tails and the weather
  required to reproduce that origin's predictions.

Reload oracle models with `OracleTemperatureForecaster.load(directory)` from
`eso.forecasting.oracle_weather`; a demand-only loader does not attach weather.
The stored prediction weather covers the fitted forecast horizon only. Model
retraining needs the original historical covariate dataset.

The runner records progress/failure in `study_status.json`. Existing output
directories are refused, preventing accidental overwrites. A failed run keeps
its audit artifacts and is not treated as a completed result.

## Interpretation

Positive temperature benefit would support evaluation using archived weather
forecasts issued before each demand origin. It would not establish operational
gain: realized reanalysis is future knowledge, and `:30` interpolation uses the
following hour. The 2023 final holdout remains reserved until the modelling
protocol is frozen for an explicitly authorized final-test evaluation.

Supplementary descriptive diagnostics are documented in
[Temperature suitability diagnostics](temperature_suitability_diagnostics.md).
They use the saved models and 2021-2022 data without modifying this protocol.
