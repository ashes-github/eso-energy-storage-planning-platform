# Statistical forecasting: initial review

The original Phase 2B milestone recorded **42 passing tests**. After the
October 29, 2026 review fixes, the current full regression suite passes **56 tests**,
including restored output-directory protection and target-aligned lag leakage tests.
The tests include statistical forecast isolation from future perturbations, signed
quantities, missing-data policies, nonconvergence, training-only GSP selection,
common-case metrics, output protection and fitted-model round-trip forecasts.

The local review run is `output/statistical_review_validation_v2/`. It uses the
existing complete `output/forecast_input/observed_panel.csv` with 364 GSPs and
2,928 contiguous half-hours (June 1–July 31, 2022 in Europe/London).
This is a **smoke benchmark**, with a 30-elapsed-day initial-history override.
It does not meet the 180-day research study setting.

```powershell
$env:PYTHONPATH = 'src'
.\venv\python.exe -m eso.forecasting.pipeline --config configs/phase2_statistical.json --input output/forecast_input --output output/your_new_statistical_review --min-train 1440 --test-start 2022-07-15T00:00:00Z
```

The initial selection history ends at June 30, 2022 22:30 UTC. Six series span
low/medium/high initial-history mean absolute Meter Volume: `FERRB1`, `BAGA`,
`BERB_P`, `WBUR_1`, `HUTT_1`, `CITR_1`. The sample includes an all-zero series;
it was not removed based on model performance. Two validation origins are
June 30 at 23:00 UTC and July 7 at 23:00 UTC. Every validation target precedes
the fixed July 15 at 00:00 UTC final test boundary. **No final test run was made.**

## Results on common observed cases

SARIMA and ETS each converged for five GSPs at each origin (10 of 12 fits),
and failed to converge on the all-zero `FERRB1` series at both origins.
Both failures remain visible as missing forecasts in raw predictions and diagnostics.
Baselines generated forecasts for every case. All-model common-case coverage is
83.3%: 80 of 96 4-hour cases and 480 of 576 day-ahead cases per model.
The table uses this identical case intersection; MAE/RMSE are in source
settlement-period Meter Volume units, and MASE is dimensionless.

| Model | 4-hour MAE | 4-hour RMSE | 4-hour MASE | Day-ahead MAE | Day-ahead RMSE | Day-ahead MASE |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 6.103 | 8.717 | 0.447 | 18.837 | 36.060 | 1.288 |
| Daily seasonal naive | 3.661 | 5.366 | 0.320 | **6.554** | **8.988** | 0.570 |
| Weekly seasonal naive | 6.588 | 8.253 | 0.571 | 10.985 | 14.362 | 0.938 |
| SARIMA | **2.663** | **3.930** | **0.242** | 7.007 | 10.070 | 0.639 |
| ETS | 2.929 | 4.189 | 0.245 | 7.018 | 10.092 | **0.554** |

SARIMA has the lowest 4-hour errors in this sample. Daily seasonal naive has
the lowest day-ahead MAE and RMSE; ETS has the lowest day-ahead MASE. These
metrics weight series differently, so the rankings need not agree. Pooled MAE
can be dominated by larger quantities. Six volume-stratified series and two
origins are insufficient for a general conclusion or a final model selection.
This comparison provides no evidence about weather, multi-year drift, or DST
transition performance. Longer validation and verified historical settlement
coverage are needed before a model study.

Review `fit_diagnostics.csv` first, then `metrics_common_cases.csv` and
`metrics_common_cases_by_lead.csv`. The unrestricted `metrics_pooled.csv`
reports different coverage across models; its raw errors should not alone rank
them. The local `common_case_mae_by_lead.png` visualizes how errors change by lead.
Optional fitted-model saving is implemented and tested, but disabled for this run
to keep artifacts small.

See [the implementation and configuration guide](statistical_forecasting.md)
for fitting policies, historical 2020–2022 sources and how to run the 180-day study.
Global ML and external covariates remain deferred for review after this milestone.
