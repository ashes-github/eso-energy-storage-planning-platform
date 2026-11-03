# Phase 2C.1: long-history model study results

The resource-bounded study is complete (November 3, 2026): **363 eligible GSPs,
365 days of initial history, 29 validation origins across 2022, and four frozen
feature experiments**. It provides substantially broader evidence than the earlier
20-GSP smoke run, while exposing important weaknesses. No final-test forecasts
or scores were generated. All **69 regression tests pass**.

The study protocol, source preparation and exact commands are documented in
[the study guide](long_history_study.md). Local results are under
`output/phase2c1_long_history_2022_validation/`; static figures and additional
audits are under its `report_v2/` directory.

## Population, timeline and counts

Source inventory used the extracted GP9 2020–2023 publication batches. The chosen
settlement panel is January 2021–March 2023, with 367 candidate IDs and 39,358
contiguous UTC half-hours. Eligibility used only the 2021 initial year: at least
95% observed coverage. Four insufficient-history/later IDs were excluded;
the retained 363 have complete published observations in initial training and
validation. Constant/zero series were retained.

- Initial training: January–December 2021; subsequently expanding to 729 days.
- Validation: 29 daily forecast windows, every 13 days through calendar 2022.
- Reserved final test: January–March 2023; not forecast, fitted on or scored.
- Each origin predicts 48 half-hours; the 8-step metric uses its first eight leads.
- Each model/horizon uses 84,216 four-hour cases and 505,296 day-ahead cases.
- Every model has 100% forecast coverage on those observed cases; unrestricted,
  all-model-common and pairwise-common scores therefore coincide.

All four experiments use the same population, origins, target cases, seed and
learner settings. Baseline metrics were verified identical across experiments.
There are 116 saved global boosters: 29 origins × four feature experiments.
The 9,432,192 exported rows include all four models in each experiment and the
nested 8/48 horizon outputs; they are not independent statistical trials.

## Predeclared full model versus baselines

M3 is the original full feature set and the predeclared reference. Errors remain
in source settlement-period Meter Volume units; MASE is dimensionless. These
are pooled errors over GSP/origin/lead cases, not averages of individual GSP RMSE
and not errors of summed system demand.

| Model | 4-hour MAE | 4-hour RMSE | 4-hour MASE | Day-ahead MAE | Day-ahead RMSE | Day-ahead MASE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence | 3.243 | **6.656** | 0.695 | 10.449 | 18.961 | 2.251 |
| Daily naive | 4.685 | 11.653 | **0.670** | 6.379 | 13.782 | **1.031** |
| Weekly naive | 6.259 | 14.839 | 0.981 | 7.657 | 16.199 | 1.253 |
| Full LightGBM (M3) | **3.184** | 6.942 | 25.535 | **5.174** | **10.788** | 52.868 |

M3 reduces pooled MAE against daily naive by **32.0% at four hours** and **18.9%
day-ahead**. However, persistence is the stronger four-hour baseline: M3's MAE
advantage over it is only **1.8%**, while persistence has lower RMSE. A claim that
LightGBM uniformly wins across models, horizons or metrics is unsupported.

This study's origins are at midnight UTC. Four-hour performance here describes
those overnight windows, not four-hour forecasts issued at every time of day.
The earlier smoke experiment used another population, fitting history and origin
schedule; differences between the two studies cannot be attributed solely to
longer history. The earlier six-GSP SARIMA/ETS scores are also not directly
comparable: no matched long-history statistical run was performed here.

## Who benefits?

For each GSP, delta MAE is daily-naive MAE minus M3 MAE on identical cases.

| Quantity | 4 hours | Day-ahead |
| --- | ---: | ---: |
| GSPs with lower MAE than daily naive | 234 / 363 (64.5%) | 296 / 363 (81.5%) |
| Median relative MAE improvement | 20.0% | 17.3% |
| Relative improvement interquartile range | −22.7% to 36.0% | 7.3% to 24.0% |
| GSPs with lower MAE than persistence | 45.5% | 89.5% |

Relative percentages are undefined when baseline MAE is zero. `FERRB1`, `GLAG_P`
and `TOWH` remain in absolute-error scores and win/loss counts, but are excluded
from relative medians, leaving 360 defined relative comparisons. No GSP was
removed because the model performed poorly.

Initial-history mean absolute volume defines three strata of 121 GSPs each:

| Volume stratum | 4-hour GSP win rate | 4-hour median improvement | Day-ahead GSP win rate | Day-ahead median improvement |
| --- | ---: | ---: | ---: | ---: |
| Low | 52.9% | 6.4% | 59.5% | 7.4% |
| Medium | 62.8% | 22.6% | 87.6% | 18.7% |
| High | 77.7% | 23.7% | 97.5% | 23.1% |

The gain is not confined to high-volume GSPs, but it is clearly less reliable
among low-volume series. Day-ahead improvements are broad in medium/high strata;
four-hour improvements over daily naive coexist with a majority of GSPs losing
to persistence at that horizon.

## Day type, season and lead

| Target day type | Daily / M3 4-hour MAE | Daily / M3 day-ahead MAE | Day-ahead GSP win rate |
| --- | ---: | ---: | ---: |
| Weekday | 4.207 / 3.027 | 5.858 / 4.969 | 73.8% |
| Weekend | 5.745 / 3.533 | 7.529 / 5.627 | 87.1% |

Day types are assigned by Europe/London target timestamps, including windows that
cross local midnight. Improvements relative to daily naive are stronger on weekends.

| Season | Daily / M3 4-hour MAE | Daily / M3 day-ahead MAE |
| --- | ---: | ---: |
| Winter | 5.556 / 3.971 | 7.308 / 6.161 |
| Spring | 3.912 / 3.145 | 6.330 / 5.411 |
| Summer | 3.654 / 2.402 | 5.006 / 4.052 |
| Autumn | 5.491 / 3.106 | 6.738 / 4.932 |

M3 has lower pooled MAE than daily naive in all four meteorological seasons
among these sampled windows. Winter absolute errors are highest. This is a
descriptive slice, not evidence attributing errors to weather, which is absent.
Per-lead metrics and `lead_errors.png` show the error curve rather than hiding
all leads inside the horizon average. Input/training data include DST transitions,
but the 29 evaluation windows do not directly score the clock-change days.

## Why pooled MASE is poor

The unfavorable MASE results above are retained without clipping, denominator
floors or retrospective exclusions. The day-ahead scaled-error total is **97.6%
attributable to `GLAG_P`**. Its training seasonal-difference scale ranges from
roughly 0.000019 to 0.000036 across origins. M3's absolute MAE of about 0.455
becomes a per-GSP MASE of about 18,668. `FERRB1` is another nearly-flat failure,
with absolute MAE about 0.456 and per-GSP MASE about 125.

This exposes a real weakness: the pooled global model predicts small nonzero
quantities for some zero/near-flat series where daily naive is effectively exact.
The unweighted objective does not protect these series against large relative
errors. It also demonstrates the extreme sensitivity of arithmetic-mean MASE to
tiny positive denominators. Neither observation justifies deleting those cases.

Median individual-GSP MASE is 0.549/0.839 for M3 versus 0.639/1.014 for daily
naive at four hours/day-ahead. These secondary distribution diagnostics explain
the concentration; they do not replace the unfavorable pooled metric or establish
uniform scale-balanced superiority. MASE is defined for 83,960/84,216 and
503,760/505,296 cases; undefined scales remain visible in the exported counts.

See `report_v2/mase_contribution_audit.csv`, `median_gsp_mase.csv`, and M3's
`mase_scale_outlier_audit.csv` for the numerical evidence. A predeclared
constant-series treatment or scale-aware objective experiment is justified as a
next study, rather than an ad hoc modification to make these scores look better.

## Controlled feature ablation

All variants retain GSP identity and lead, along with calendar columns. No actual
holiday flags are supplied. The same 200-tree learner is refitted on exactly the
same target rows and inner origins in each variant.

| Variant | 4-hour MAE | 4-hour RMSE | Day-ahead MAE | Day-ahead RMSE | Day-ahead GSP win rate vs daily |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 calendar/GSP/lead | 7.125 | 13.765 | 8.347 | 15.550 | 17.4% |
| M1 + origin lags | **3.118** | **6.716** | 5.645 | 11.365 | 62.5% |
| M2 + target seasonal lags | 3.118 | 6.944 | **5.157** | **10.788** | **83.2%** |
| M3 + rolling means/stds | 3.184 | 6.942 | 5.174 | 10.788 | 81.5% |

Origin history adds substantial predictive value over calendar-only inputs.
Adding target-aligned seasonal lags reduces day-ahead MAE by about **8.6%** versus
M1. It leaves four-hour pooled MAE almost unchanged while improving the proportion
of GSPs beating daily naive. M1 has the lowest four-hour MAE among these variants,
although M1/M2 differ by only about 0.0004; that is not a meaningful general ranking
without further evidence. Persistence still has lower four-hour RMSE than every
tree variant.

Rolling features do not improve aggregate MAE here: M3 is about **0.3% worse** than
M2 day-ahead and about **2.1% worse** at four hours. The day-ahead RMSE difference
is negligible. This supports keeping M2 as a simpler candidate for the next
evaluation; it does not prove rolling statistics are intrinsically harmful.
The original M3 default was not silently replaced after inspecting outcomes.

Across M3's 29 origins, average within-origin feature gain is about 53.9% for
`target_lag_48`, 9.7% for `target_lag_96` and 9.1% for `target_lag_336`.
Rolling features also receive substantial split gain, despite adding no aggregate
accuracy in the ablation. Correlated predictors can exchange split gain; gain
importance is descriptive and cannot establish independent or causal usefulness.

## Scope, checks and next experiments

The study is deliberately resource-bounded: training target origins are sampled
every eight days and validation uses 29 windows, not every historical/validation
interval. The same GSP population is forecast through time; unseen-GSP transfer
is not tested. The observed-source policy includes publisher estimates, and latest
revisions do not reconstruct what was available operationally at each origin.
No actual holiday calendar or weather is included.

The full test suite passed 69 tests. The accompanying verification checks frozen
core source/config/protocol/input hashes, all 116 training-label cutoffs and saved
schemas, independent NumPy pooled-error aggregation, and native forecast round trips
for the first, middle and last full-model origins. Baseline metrics match across
all ablations. See the local root `verification.json`.

The results are validation evidence from one sampled calendar year, not final-test
confirmation or a statistical-significance claim. Next experiments should isolate
small/constant-series treatment and optional scale-aware training, validate whether
M2's simpler feature set holds up, and consider denser origin sampling. Complete
regional holiday calendars and then origin-available weather are separate
extensions. Final-test scoring should follow a frozen model/objective decision;
the 2023 reserved period remains unscored.
