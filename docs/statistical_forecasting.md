# Phase 2B: statistical forecasting review

SARIMA and ETS now use the same expanding-window evaluator as persistence,
elapsed-UTC daily seasonal naive and weekly seasonal naive. This milestone
implements representative per-series statistical benchmarks. Global ML, weather,
deep learning and intervals remain later work.

## Models and policies

The starting SARIMA specification is `(1,0,0)(1,0,0,48)`, with no deterministic
trend. This is a modest daily-seasonal starting benchmark, not an automatically
selected or tuned specification. Orders are explicit configuration values.
ETS uses additive errors and additive daily seasonality, with no trend by default
(ETS(A,N,A)). An additive or damped additive trend can be configured. Additive
components accept signed net imports; no log transform or conversion to MW occurs.

The implementations use statsmodels [SARIMAX](https://www.statsmodels.org/v0.14.6/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)
and [ETSModel](https://www.statsmodels.org/v0.14.6/generated/statsmodels.tsa.exponential_smoothing.ets.ETSModel.html).
At each origin, each GSP is fitted independently from past observations only.
Numerical scaling uses the maximum absolute observed value in that fitting window;
predictions are restored to source settlement-period Meter Volume units.
Daily seasonality is **48 elapsed UTC half-hours**, with the same DST limitation
as the daily naive baseline. The models do not jointly model daily/weekly seasons.

SARIMA retains missing observations at their original elapsed positions and uses
state-space handling. ETS refuses any fitting window containing a missing value.
Neither drops internal time steps nor imputes gaps. Both require enough observed
history, at least two seasonal cycles, and optimizer convergence. Failed fits or
nonfinite forecasts produce NaN predictions, with no fallback to a baseline.
`fit_diagnostics.csv` records the GSP, origin, fitting range, observation count,
convergence, AIC and warning/failure details. Compare model specifications on
validation errors; AIC across these differently formulated models is not a ranking
criterion. A complete run can contain failed fits: inspect diagnostics and coverage.

The fit/predict interface is unchanged. `train_window=null` fits the entire expanding
history; an explicit positive half-hour count limits a model to a trailing window.
That changes the model's training policy and is recorded in configuration and
per-fit diagnostics. Baselines and MASE retain the full available history.

## Reproducible representative benchmark

Two explicit run-purpose presets are available:
`configs/phase2_pipeline_check.json` checks all five models with 14 initial days
and three GSPs; `configs/phase2_forecasting_comparison.json` compares the same
models with 180 initial days and six GSPs. Both use weekly forecast origins.
The check caps SARIMA/ETS fitting windows at 14 days; the comparison fits their
full expanding history. See [run-purpose commands](../README_FORECASTING.md#choose-the-run-purpose).

`configs/phase2_statistical.json` requires **180 elapsed days** before validation.
The legacy 14-day default and short overrides are software smoke benchmarks,
not recommended histories for a seasonal forecasting study. Prefer 6–12 months,
and 1–2 years where actual settlement coverage permits.

The statistical config selects six GSPs from low/medium/high strata defined by
mean absolute Meter Volume in the initial training window. It chooses deterministic
evenly spaced ranks within each stratum, breaking ties by GSP ID. Selection never
reads validation/test values and does not use future coverage to exclude hard cases.
`series_selection.csv` records all candidates, initial coverage, strata and selection.
This sample spans volume magnitude, not Phase 1 recurrence or spatial diversity;
extend the sampling design before claiming general performance across GSPs.
For a curated subset, replace `representative_count` with `null` and set
`series_ids` to a list of string GSP IDs. All five models evaluate the same subset.

```powershell
$env:PYTHONPATH = 'src'
.\venv\python.exe -m pip install -e ".[statistical]"
.\venv\python.exe -m eso.forecasting.pipeline --config configs/phase2_statistical.json --input output/multiyear_forecast_input --output output/statistical_validation --test-start YOUR_FIXED_TIMEZONE_AWARE_BOUNDARY
```

Choose a boundary present in the panel after at least 180 training days and one
full validation day, leaving at least one full test day. Config defaults evaluate
origins every 336 half-hours to keep per-series fitting practical; use `--step 48`
for daily origins. CLI `--min-train` and `--step` override their config values.
Model options are fixed within the run; there is no automatic order search.
Select options on validation only. Run `--split test` explicitly after freezing
the configuration, sample and boundary. Test observations can enter later test
origins' histories, as in the existing framework; there is no operational as-of
revision/publication reconstruction.

### Historical local inputs

Additional source batches are available at `Data/GP9_2020/*.csv` (September–December
publication files), `Data/GP9_2021/*.csv` and `Data/GP9_2022/*.csv` (12 publication
files each). ZIP copies of these are also local. These are publication years;
they **do not establish settlement coverage**. The January 2022 sample includes
November 2020 settlement observations. Use `Settlement Date` filters and the
existing latest-revision policy across all relevant source batches, rather than
concatenating independently prepared yearly panels.

For the observed-panel preparation example in `README_FORECASTING.md`, replace
the `paths` assignment with:

```python
paths = sorted({
    path
    for folder in [Path('Data') / f'GP9_{year}' for year in (2020, 2021, 2022, 2023)]
                  + [Path('GP9_2023')]
    for path in folder.glob('*.csv')
})
```

Choose `start` and `end` from verified settlement coverage. `load_records` reads
files in chunks but retains date-filtered records for revision resolution, so a
long multi-year all-GSP panel can still require substantial memory. Check the
resulting observed fraction and uninterrupted half-hour axis before modelling.
The short existing 2022 forecasting input is suitable for a smoke review, not the
180-day study. No additional raw-data downloads are required for this milestone.

## Review artifacts

Every run refuses an existing output directory and writes completion/failure status.
Along with the existing prediction, per-GSP and pooled CSVs, it writes:

- `metrics_by_lead.csv`: separate metrics for every lead within each horizon.
- `metrics_common_cases.csv`: pooled errors on observed cases for which **all
  configured models** generated forecasts.
- `metrics_common_cases_by_lead.csv`: the same intersection at each lead.
- `fit_diagnostics.csv`: statistical fit outcomes, including unsuccessful fits.
- `series_selection.csv`, `config.json`, `run.json`, `run_status.json`.

Common-case outputs retain the original observed denominator. Their coverage is
the fraction shared by every model. If one model fails for a GSP/origin, that case
is excluded from errors for every model, while remaining visible in coverage.
An empty intersection gives undefined errors, not a zero error. Unrestricted
pooled metrics remain useful for coverage, but should not alone rank models.
The 8- and 48-step horizons remain separate; targets appearing at multiple origins
are distinct forecast cases. MASE denominators use only observed training pairs.

Metadata includes resolved settings, selected GSPs, initial training and evaluated
target ranges, final test boundary, input SHA-256, forecasting source hashes, Git
commit/worktree status, Python/package versions and research assumptions.
Set `save_models=true` to write fitted statsmodels objects with per-GSP scaling
manifests under `models/`; this can be large. Only successful fits are saved.
These are local pickle artifacts; use only trusted model files.

The reviewed smoke run is `output/statistical_review_validation_v2/`.
Its explicit short training override verifies implementation and produces reviewable
results. It is not a final test evaluation or evidence of multi-year performance.
See [the initial review results](statistical_forecasting_results.md) for the
validation comparison and limitations.
