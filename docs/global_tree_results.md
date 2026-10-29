# Phase 2C: initial global LightGBM smoke results

The full **65-test** suite passed on October 29, 2026, including actual LightGBM
fits, future-perturbation isolation, native missingness, one pooled fit, purged
labels, training-only eligibility, row-budget checks and native model round trips.

## Run

Artifacts are in `output/phase2c_lightgbm_smoke_validation_v2/`, generated with:

```powershell
$env:PYTHONPATH = 'src'
.\venv\python.exe -m eso.forecasting.pipeline --config configs/phase2_ml_smoke.json --input output/forecast_input --output output/your_new_ml_smoke_validation --test-start 2022-07-15T00:00:00Z
```

The existing 61-day panel contains 364 GSPs. The preset selects 20 deterministic
low/medium/high-volume GSPs with at least 95% initial observed coverage, using only
the first 14 days. All 20 selected GSPs are fully observed in this run. Validation
origins are June 14, June 21, June 28, July 5 and July 12, all at 23:00 UTC.
Every validation target precedes the fixed July 15, 2022 00:00 UTC test boundary.
**No final test evaluation was performed.**

Each origin fits one shared LightGBM model using the latest 672 half-hours. Each
fit pools 6,720 observed training rows across 20 GSPs, seven daily inner origins
and 48 leads. Training targets end before the outer origin. Parameters are fixed:
100 trees, 31 leaves, learning rate 0.05, seed 7 and four CPU threads. No search or
early stopping is performed. The 180-day all-eligible-GSP study has not been run.

## Results on identical cases

All models have **100% observed-target forecast coverage**: 800 cases for the
4-hour horizon and 4,800 for day-ahead. Pooled and common-case scores coincide.
The run contains 22,400 forecast rows across four models and both horizons.
MAE/RMSE retain source settlement-period Meter Volume units; MASE is dimensionless.
Scores include all leads within each horizon.

| Model | 4-hour MAE | 4-hour RMSE | 4-hour MASE | Day-ahead MAE | Day-ahead RMSE | Day-ahead MASE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence | 4.098 | 6.285 | 0.865 | 10.677 | 23.780 | 1.899 |
| Daily naive | 3.089 | 5.303 | **0.549** | 5.335 | 9.260 | **0.882** |
| Weekly naive | 3.761 | 6.636 | 0.672 | 5.853 | 9.886 | 1.007 |
| Global LightGBM | **2.927** | **4.700** | 0.637 | **4.793** | **7.723** | 0.917 |

LightGBM reduces pooled MAE against daily naive by about **5.2% at 4 hours** and
**10.1% day-ahead**, and also has lower pooled RMSE. Daily naive remains better
on MASE at both horizons. MASE scales errors by each GSP's observed training
seasonal differences; pooled absolute errors emphasize larger quantities.
Inspect per-GSP metrics before claiming benefits across the population.

MASE is defined for 760/800 and 4,560/4,800 cases respectively, identically for
all models. Zero/unavailable seasonal scales are excluded from MASE, while those
cases still enter MAE/RMSE and coverage. Constant/zero series were retained.

## Artifacts and limits

The five native boosters and schemas/history tails are under
`models/lightgbm/origin_*/`. A saved-model reload reproduced the first origin's
recorded forecasts. `fit_diagnostics.csv` contains 100 successful GSP/origin rows
for the **five shared fits**, not 100 individual models. Schemas record training
target cutoffs, categories, row counts and feature settings. Metadata includes
source/input hashes and LightGBM 4.7.0. First-fit feature gains emphasize recent
demand and target-aligned daily/weekly lags; gains are descriptive, not causal.

Review `metrics_per_gsp.csv`, `metrics_common_cases_by_lead.csv` and the local
`common_case_mae_by_lead.png`. The earlier six-GSP SARIMA/ETS experiment used
different cases/origins and cannot be ranked directly against this run. Use a
matched subset, panel, model list and origins for that comparison.

This is an implementation smoke benchmark, not evidence of multi-year superiority,
unseen-GSP transfer, operational as-of availability or weather response. Weather
and an actual holiday calendar are absent. The next experiment is verified
longer-history validation on all eligible GSPs, followed by a separate weather
ablation. See the [global model guide](global_tree_forecasting.md) for commands.
