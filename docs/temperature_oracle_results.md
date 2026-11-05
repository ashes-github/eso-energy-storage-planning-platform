# Phase 2D temperature feasibility: 2022 validation results

Adding one oracle target-time temperature feature gives a small MAE/RMSE
improvement over frozen M2, with heterogeneous benefits and worse pooled MASE.
This supports modest incremental weather signal under the tested specification;
it does not establish an operational improvement or statistical significance.

The completed experiment used the exact 363 GSPs, 29 validation origins,
training schedule, targets, LightGBM settings and naive references from Phase
2C.1 M2. All origins passed saved-baseline replay and identical-training checks.
All 589,512 paired cases were retained across the two cumulative horizons.
January-March 2023 was not scored or used in EDA/model development.

| Metric | M2, 4 h | M2 + temperature, 4 h | M2, 24 h | M2 + temperature, 24 h |
|---|---:|---:|---:|---:|
| MAE | 3.1183 | 3.0899 | 5.1574 | 5.0908 |
| RMSE | 6.9440 | 6.9038 | 10.7876 | 10.6295 |
| MASE | 24.8500 | 28.7815 | 47.1314 | 51.2264 |
| MAE improvement | | 0.91% | | 1.29% |
| GSPs with lower MAE | | 220 / 363 | | 266 / 363 |
| Median per-GSP MAE improvement | | 1.07% | | 1.35% |

MAE/RMSE use source Meter Volume units. Each horizon includes all leads through
its endpoint, not only the last lead. Forecast coverage is 100% of observed
validation targets for every model. Macro MAE equals pooled MAE here because
each GSP has the same number of scored cases.

## What the feasibility analysis shows

The median within-GSP temperature/demand Pearson correlation is -0.163 raw
and -0.265 after descriptive demeaning within season/local-hour/daytype cells.
Binned curves vary substantially by season and time of day; correlation alone
would miss that structure. The two-degree bins and per-GSP normalization were
fixed before evaluating the oracle model, with scaling from 2021 only.

357 selected GSPs have temperature. The other six (`GLAG_P`, `HARG_P`, `HUER`,
`STOR_P`, `TONG1`, `TOWH`) retain native missing weather and are included in
every metric. Shared polygon aliases use shared regional temperatures. The
lowest nonzero spatial coverage is `SHRU`, approximately 4.2%; coverage is an
audit diagnostic, not an extra predictor or an exclusion rule.

![Development temperature-demand curves](../output/phase2d_temperature_oracle_2022_validation/eda/temperature_demand_slices.png)

## Benefits differ across slices

Positive values below indicate lower MAE with temperature:

| Season | 4-hour MAE improvement | 24-hour MAE improvement |
|---|---:|---:|
| Winter | -0.80% | 2.13% |
| Spring | 4.20% | 2.12% |
| Summer | -1.01% | -0.20% |
| Autumn | 1.46% | 0.42% |

Four-hour weekday MAE improves 2.28%, while weekend MAE worsens 1.66%.
At 24 hours both improve: 1.23% weekdays, 1.41% weekends. At four hours,
high-volume GSPs are slightly worse (-0.11%), while low/medium improve
2.20%/1.87%. At 24 hours all three volume strata improve, with the largest
gain in the high-volume stratum (1.84%). These slices are descriptive and
were not used to tune or select different models.

![GSP temperature benefits and spatial coverage](../output/phase2d_temperature_oracle_2022_validation/gsp_temperature_benefits.png)

Temperature accounts for about 0.10% of mean LightGBM gain importance; demand
lags dominate. Gain importance is descriptive, depends on correlated features,
and does not quantify temperature's causal effect or its forecast benefit.

## Why MASE worsens

`GLAG_P` has a median training-only daily-difference scale of approximately
0.00002511. Its relatively small absolute errors therefore have enormous scaled
errors. It accounts for 3.8829 of the total 3.9315 four-hour pooled-MASE increase,
and 4.0980 of the total 4.0950 24-hour increase.

The full-population MASE above remains the primary result. An explicitly
post-hoc sensitivity excluding only this largest contributor yields four-hour
MASE 0.7439 to 0.7926 (still worse), and 24-hour MASE 1.2276 to 1.2246 (slightly
better). This sensitivity does not replace the primary scores, justify dropping
the GSP, or establish a broad improvement in scaled error.

Although `GLAG_P` has missing temperature, its predictions can change because
the global model's shared trees change when temperature is added elsewhere.

## Reproduce and inspect

See [the protocol and commands](temperature_oracle_study.md). The complete
[generated report](../output/phase2d_temperature_oracle_2022_validation/report.md)
links the EDA and benefit figures. Its directory also contains frozen hashes,
the per-origin control audit, paired predictions, reloadable native models,
all slice metrics, and per-GSP scaled-error attribution.

```powershell
.\venv\python.exe -m eso.forecasting.temperature_scaled_audit output/phase2d_temperature_oracle_2022_validation
```

The experiment completed in approximately 8.3 minutes in the current environment,
including development EDA and 29 oracle fits. Baselines were reused after
verification; no hyperparameter search or final-test evaluation was performed.

The measured gain is small even with perfectly realized target weather. An
operational claim requires a separate comparison using archived weather forecasts
issued before each demand origin. January-March 2023 remains reserved.

## Supplementary suitability diagnostics

The [supplementary Pearson/Spearman, mutual-information, spline and saved-model
permutation study](temperature_suitability_diagnostics.md) fills the remaining
feasibility diagnostics without changing or refitting this ablation. It confirms
nonlinear, season/hour-dependent association and positive fitted-model reliance
on temperature, while preserving the same oracle and non-causal limitations.
