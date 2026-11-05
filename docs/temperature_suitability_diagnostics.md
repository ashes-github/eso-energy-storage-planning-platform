# Supplementary temperature suitability diagnostics

The completed Phase 2D oracle ablation showed a small incremental MAE benefit.
This supplement fills the remaining feasibility diagnostics without refitting
that experiment or accessing January-March 2023.

The analysis uses all 363 frozen GSPs and 2021-2022 development data. Six GSPs
without temperature remain in every audit with undefined association measures.
Within-GSP Pearson/Spearman coefficients use the full half-hourly paired data.

| Per-GSP association | Median | Interquartile range | Negative GSPs | \(|r|>0.3\) |
|---|---:|---:|---:|---:|
| Pearson | -0.163 | -0.279 to -0.019 | 78.4% | 23.5% |
| Spearman | -0.155 | -0.261 to -0.030 | 78.4% | 23.0% |
| Calendar-adjusted Pearson | -0.265 | -0.436 to -0.126 | 89.4% | 43.1% |
| Calendar-adjusted Spearman | -0.268 | -0.431 to -0.122 | 87.4% | 45.7% |

Calendar adjustment descriptively demeans demand and temperature within local
season, hour and weekday/weekend cells. It is not a fitted forecast feature or
a causal adjustment.

Association varies materially by context. Median Spearman is -0.098 in winter,
-0.027 in autumn, 0.014 in spring, and 0.304 in summer. The strongest negative
hourly association occurs around 17:00-19:00 Europe/London, with the median near
-0.52 to -0.56. Weekend median Spearman is -0.193 versus -0.158 on weekdays.
Between 2021 and 2022, the association sign agrees for about 86.8% of GSPs by
Pearson and 87.1% by Spearman. This is descriptive two-year stability, not an
out-of-period guarantee.

![Association distributions, MI and hour slices](../output/phase2d_temperature_supplementary_diagnostics/association_diagnostics.png)

## Nonlinear dependence and response shape

Mutual information uses `mutual_info_regression` with five neighbours, a fixed
seed, and up to 2,000 exact-hour observations per GSP. Exact hours avoid treating
interpolated `:30` temperatures as independent information. Median per-GSP MI is
0.137 nats (interquartile range 0.084-0.196), compared with a median 0.004 nats
under three randomized-temperature references. Those references diagnose the
estimator floor; they are not p-values and do not correct for season, calendar
or temporal autocorrelation.

The pooled 30,000-row fixed sample has MI 0.015 nats for raw demand and 0.037
nats for GSP-standardized demand. Cross-sectional mixing makes these pooled
estimates less informative than the per-GSP distribution.

The smooth response uses a fixed six-knot cubic B-spline followed by Ridge
regression with alpha 1.0. Each GSP receives equal total weight, demand is
standardized with its 2021 mean/standard deviation, and curves stop at the 1st
and 99th temperature percentiles. It is a visualization only; it was not used
for feature construction, model selection or tuning.

![Descriptive smooth response](../output/phase2d_temperature_supplementary_diagnostics/smooth_temperature_response.png)

The overall curve declines through moderate temperatures and turns upward at
warmer temperatures. Seasonal curves differ sharply, reinforcing that the
temperature-demand relationship is nonlinear and conditional on season and
time. Composition, solar generation and other omitted weather variables can
still confound these curves. Per-GSP two-degree-bin medians are exported for a
less model-dependent view.

## Saved-model permutation importance

Permutation importance does not refit LightGBM. For each of the 29 validation
origins, it first replays the saved oracle prediction. It then substitutes a
whole 48-step weather profile from a different origin in the same season. The
same donor origin applies to every GSP, preserving the cross-GSP weather field,
UTC diurnal ordering and missingness. Demand, lags, calendar features, model
parameters and weights remain unchanged.

| Horizon | Original MAE | Mean permuted MAE | Increase | Repeat SD | GSPs with positive mean importance |
|---|---:|---:|---:|---:|---:|
| 4 hours | 3.0899 | 3.1468 | 0.0570 | 0.0146 | 71.9% |
| 24 hours | 5.0908 | 5.1718 | 0.0810 | 0.0398 | 83.5% |

![Same-season profile permutation](../output/phase2d_temperature_supplementary_diagnostics/permutation_importance.png)

All five fixed derangements increase pooled MAE. The spread across repeats
reflects donor choice and is not a confidence interval. This measures model
reliance under the specified intervention; it is not formal conditional mutual
information, causal importance, or a substitute for the M2-versus-M2+temperature
walk-forward ablation.

## Reproduce and inspect

```powershell
.\venv\python.exe -m eso.forecasting.temperature_diagnostics `
  --output output/phase2d_temperature_supplementary_diagnostics
```

Use a new directory. The runner verifies the original demand/weather hashes,
package versions and saved-model feature code, refuses 2023 data, replays every
saved prediction, and verifies that original ablation artifacts remain unchanged.

The generated [supplement report](../output/phase2d_temperature_supplementary_diagnostics/report.md)
documents the fixed sampling and limits. Detailed outputs include:

- correlations for every GSP, season, local hour, daytype and year;
- correlation distribution and year-to-year stability summaries;
- the persisted 714,000-row exact-hour sample and MI estimates;
- per-GSP bin medians and fixed spline curves;
- every permutation donor, replay audit, GSP/origin/repeat error and summary.

These results strengthen the conclusion that instantaneous temperature contains
nonlinear, context-dependent signal and that the fitted oracle model uses it.
They do not change the decision to defer HDD/CDD, temperature lags/rolling means,
additional weather variables and operational archived-weather evaluation.

Methods follow the official scikit-learn documentation for
[mutual information](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html),
[spline bases](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html),
and [permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html).
