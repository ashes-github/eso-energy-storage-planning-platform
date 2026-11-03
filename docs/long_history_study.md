# Phase 2C.1: resource-bounded long-history validation

The frozen protocol is `configs/phase2_long_history_protocol.json`. This study
keeps the Phase 2C learner, squared-error objective and hyperparameters unchanged.
It adds verified historical data, matched-case benefit analysis and four controlled
feature experiments. Weather, actual holiday flags, target normalization, new
algorithms, sequence models and final-test scoring are outside this protocol.

## Data and fixed timeline

The inventory scans the 32 extracted GP9 publication batches under `Data/GP9_2020`
through `Data/GP9_2023`. Their actual settlement dates range from May 2018 to
April 2023. The ZIP contents match the extracted file lists; the four root
`GP9_2023` CSVs were verified to match the inventoried Data copies by SHA-256.
They are not double-counted. Filenames are publication months, not settlement coverage.

The prepared panel contains January 1, 2021–March 31, 2023: **367 GSP IDs and
39,358 contiguous UTC half-hours**. Every settlement date is present. Preparation
stages raw records into bounded daily partitions and applies the existing
`align_panel` revision/flow policy to all source observations for each day.
It preserves signed import/export quantities, zeros, missing values and UK DST
46/50-period days. It performs no interpolation or model-driven GSP exclusion.

The 74,106,648 date-filtered source rows resolve to 14,336,486 unique GSP/settlement
observations. The daily audit records 213,169 publisher-estimated rows across the
full prepared panel, including 79,893 in calendar 2022 across all candidate GSPs.
These retain the existing observed-source policy: published estimates are not
pipeline-imputed values and are not separately excluded here. Forecast accuracy
is measured against published GP9 quantities, not independently verified physical
meter truth. Completeness does not establish metering accuracy.

The following settings were fixed before any new model outcome was examined:

| Role | Period / rule |
| --- | --- |
| Initial fitting history | January 1–December 31, 2021: 365 elapsed days |
| Validation | January 1–December 31, 2022 |
| Reserved final test | January 1–March 31, 2023; not forecast or scored |
| Eligibility | At least 95% observed initial-year coverage; constants retained |
| Eligible GSPs | 363; all fully observed in initial training and validation |
| Excluded IDs | `AXMI_H`, `ROTI3`, `SAFO_1`, `WESA_H`, based solely on initial coverage |
| Outer validation schedule | Every 13 elapsed days: 29 complete daily windows, cycling all weekdays and all seasons |
| Inner training schedule | Every 8 elapsed days, cycling weekdays; all 48 direct leads at each inner origin |
| Fitting window | Full expanding history: 365 days at the first origin, 729 at the last |
| Learner | Existing unweighted L2 LightGBM, 200 trees, 31 leaves, learning rate 0.05, seed 7, four CPU threads |

The available RAM at design time was about 2.3 GiB. Fixed training-origin thinning
limits the largest table to **1,585,584 rows**, below the declared two-million-row
budget. Each GSP contributes 45 training target days initially and 91 at the last
origin. Long historical context does **not** mean every historical interval is
used as a supervised label: target origins are sampled every eight days, while
lag/rolling inputs use their required observations from the complete history.
A denser daily-origin training study remains a separate resource/accuracy experiment.

Likewise, validation covers 29 sampled daily forecast windows across 2022, rather
than forecasting every day of the year. Forecast origins remain at 00:00 UTC;
target day-type and season slices use Europe/London target timestamps. This allows
weekday/weekend variation and avoids the single-weekday sampling of weekly origins.

This is a retrospective latest-revision benchmark with zero publication delay
assumed. The history is long and target leakage is guarded, but publication-time
revision availability is not reconstructed. No operational as-of claim follows.

## Controlled feature experiments

| Experiment | Configuration | Inputs beyond GSP identity and lead |
| --- | --- | --- |
| M0 calendar | `phase2_long_history_calendar.json` | Target calendar features |
| M1 origin | `phase2_long_history_origin.json` | M0 + origin-relative demand lags |
| M2 seasonal | `phase2_long_history_seasonal.json` | M1 + target-aligned daily/two-day/weekly lags |
| M3 full | `phase2_long_history_full.json` | M2 + origin-relative rolling means/stds |

All experiments retain categorical GSP identity, lead and the same calendar
columns. `holiday` remains unknown in all experiments because no calendar is
supplied; it contributes no observed holiday signal. The runner snapshots resolved
configs before fitting and requires identical population, folds, training schedule
and learner options across experiments. Only the feature specifications differ.
M3 is run first, followed by the predeclared M0/M1/M2 runs. Outcomes do not change
the experiment list or settings. Each run includes the same persistence and daily/
weekly naive baselines. Statistical benchmarks are unchanged and are not refitted
across all GSPs in this global-ML study.

## Bounded-memory execution and reproducibility

`prepare.py` inventories settlement dates and source hashes, then prepares a panel
without holding all raw years in memory. The local prepared directory retains
daily staging records, source metadata, daily audits, observed monthly counts and
the panel hash. `study.py` calls the existing backtester one origin at a time and
appends forecast CSV blocks, releasing the block before the next fit. It does not
change the model fit/predict implementation. Source/input hashes, protocol/config
hashes, versions, selected GSPs and per-origin model artifacts are saved.

The output root is immutable; use fresh directories when reproducing:

```powershell
$env:PYTHONPATH = 'src'
.\venv\python.exe -m pip install -e ".[ml]"

.\venv\python.exe -m eso.forecasting.prepare --output output/my_history_inventory
.\venv\python.exe -m eso.forecasting.prepare --inventory output/my_history_inventory --start 2021-01-01 --end 2023-03-31 --output output/my_history_panel
.\venv\python.exe -m eso.forecasting.study --protocol configs/phase2_long_history_protocol.json --input output/my_history_panel --output output/my_long_history_study
.\venv\python.exe -m eso.forecasting.study_analysis --study output/my_long_history_study
.\venv\python.exe -m eso.forecasting.study_report --study output/my_long_history_study --output output/my_long_history_study/report
```

The completed local inventory/prepared-panel paths are
`output/phase2c1_history_inventory` and `output/phase2c1_observed_2021_2023`.
The study output is `output/phase2c1_long_history_2022_validation`, with a root
`study_status.json` for live progress and one directory per experiment. Metrics
are generated by the separate analysis command after forecasts complete.

## Outcome analysis

The analysis reads bounded CSV chunks and carries incomplete forecast origins
across chunk boundaries. It forms model intersections only after a complete
origin is available. Additive absolute/squared/scaled error sums and counts are
combined across origins. RMSE is the square root of combined squared error/count;
individual GSP RMSE or per-origin roots are never averaged into the pooled RMSE.

Alongside the existing per-GSP, pooled, common-case and lead metrics, each run
writes daily-naive versus LightGBM comparisons on **identical paired cases**:

- `gsp_benefits.csv`: daily/LightGBM MAE, delta, relative improvement, paired-case
  count, initial-volume stratum and win/loss/tie classification for each GSP/horizon.
- `gsp_benefit_summary.csv`: fraction of comparable GSPs improved, median/IQR
  relative improvement and macro per-GSP MAE. Zero-baseline MAE has undefined
  relative improvement; those GSPs remain in absolute-error comparisons and win counts.
- `gsp_benefit_summary_volume.csv`: the same benefit distribution in initial-history
  low/medium/high volume strata, without using validation demand to define strata.
- Day-type and season benefit tables, plus matched pooled metrics by volume,
  weekday/weekend, meteorological season and lead.
- `importance_aggregated.csv`: per-origin gain fractions averaged with equal origin
  weight, plus dispersion and split counts. Gains are descriptive, not causal.
- Root `ablation_metrics.csv` and `ablation_gsp_benefits.csv`: the controlled
  feature experiments on the same evaluation cases.

Each horizon includes all leads 1..H. Per-lead tables isolate individual leads.
All quantities remain source settlement-period Meter Volume, without MW conversion.
Unrestricted scores retain coverage; pairwise scores prevent missing predictions
from making one model look artificially better. Undefined MASE counts stay visible.

The regression suite checks chunk-boundary integrity, pooled squared-error
aggregation, zero-baseline handling, direct equivalence of partitioned preparation
to the existing revision policy, and a streamed model/analysis integration run.
Results are reported in [the long-history results note](long_history_results.md).
