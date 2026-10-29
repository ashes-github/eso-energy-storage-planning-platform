# Forecasting feature foundation

`src/eso/forecasting/features.py` provides origin-available features for future
direct multi-lead ML models. It implements no model fitting, encoding, scaling or
imputation. Existing baseline/statistical forecasts remain unchanged. The ML
preset now uses these features in [GlobalTreeForecaster](global_tree_forecasting.md),
with categorical GSP identity fitted inside each training fold.

## Time convention and schema

The forecast origin is the **first predicted interval start**, matching the
backtester. Lead 1 is that interval, lead 2 is 30 minutes later, and lead 48 is
23.5 hours after the first interval start. The horizon covers 24 hours of intervals.
For an origin `o`, target lead `h` starts at `o + (h - 1) * 30 minutes`.
Only history strictly before `o` enters demand-derived features.

| Features | Definition |
| --- | --- |
| `lead` | One-based half-hour forecast lead |
| `origin_lag_1`, `origin_lag_2`, `origin_lag_48`, `origin_lag_96`, `origin_lag_336` | Observations 1, 2, 48, 96, 336 half-hours before the origin; fixed across leads |
| `target_lag_48`, `target_lag_96`, `target_lag_336` | Observations 48, 96, 336 elapsed half-hours before each target, only when the donor precedes the origin |
| `mean_6`, `mean_12`, `mean_48`, `mean_336` | Mean of the last complete observed window ending immediately before the origin |
| `std_48`, `std_336` | Sample standard deviation (`ddof=1`) of the same history windows |
| `settlement_period` | One-based elapsed half-hour since target's Europe/London local midnight |
| `half_hour_of_day` | Local clock slot 0–47; repeated clock slots repeat on the autumn DST transition |
| `hour`, `day_of_week`, `weekend`, `month` | Target's Europe/London calendar; Monday is 0, weekend is Saturday/Sunday |
| `holiday` | Explicit supplied local-date flag: 1 holiday, 0 known non-holiday, NaN unknown |
| `time_of_day_sin`, `time_of_day_cos` | Cyclic local clock-slot encoding with period 48 |
| `day_of_week_sin`, `day_of_week_cos` | Cyclic weekday encoding with period 7 |

Origin-relative lags and rolling statistics repeat across all leads at an origin.
For example, at a 10:00 origin, `origin_lag_1` is the observed 09:30 quantity even
for the 24th or 48th lead. Target-relative seasonal lags follow the target's elapsed
time instead, providing the corresponding daily/two-day/weekly historical slots.
This is a direct multi-lead design; no recursive predictions are used as donors.

Let `p` be the origin position in a contiguous UTC panel and `h` the one-based
lead. The target position is `p + h - 1`. For `target_lag_k`, the donor position is
`p + h - 1 - k`. It is read only if `0 <= donor < p`, with an explicit assertion
before indexing. Thus daily donors at leads 1–48 are `p-48` through `p-1`;
at lead 49 the donor reaches `p` and is unavailable, so the feature is NaN.
Insufficient early history also gives NaN, never negative-index wraparound.
These are elapsed-UTC seasonal lags, not local-clock matches across DST.

**Feature schema update:** the original `lag_k` output columns are now named
`origin_lag_k` to distinguish the two families. Update any consumers referencing
the old names. `FeatureSpec.lags` continues to configure origin-relative lags;
the new `FeatureSpec.target_lags` defaults to `(48, 96, 336)`. Existing statistical
models do not consume this feature matrix, so their forecasts are unchanged.

Missing lag donors remain NaN. A rolling feature is NaN if its entire window is
not present and observed, even if a partial mean could be computed. Standard
deviations of fully observed constant windows are zero. Signed quantities and
leading-zero GSP identifiers are preserved. `FeatureSpec` can customize lag and
rolling windows, and can set `std_ddof=0` for population standard deviation.

The UTC half-hour axis remains contiguous through DST. Settlement periods run
1–46 or 1–50 on the spring/autumn transition days; local clock slots skip or repeat.
Both columns are retained to distinguish settlement numbering from clock time.

## Build features at one origin

```python
from eso.forecasting import ForecastDataset, build_features

panel = ForecastDataset.from_phase1('output/forecast_input').panel
position = 672
history = panel.iloc[:, :position]
future = panel.columns[position:position + 48]
features = build_features(history, future)
```

The result has numeric feature columns and an index `(gsp, origin, timestamp)`.
GSP identity is an index field available to a future fold-fitted encoder. The
function accepts only contiguous future half-hours immediately after history,
and uses the same observed-panel validation as the forecasting framework.

To supply known holiday flags, pass a mapping such as
`holiday_calendar={'2022-06-02': True, '2022-06-03': True, '2022-06-06': False}`.
Include explicit false values for dates known to be non-holidays. Missing dates
remain unknown; weekends are never automatically treated as holidays. Provide
the correct regional calendar and the version known at the origin. Group GSPs
by holiday region or call the builder separately when their calendars differ.
The builder does not fetch holidays or establish announcement/publication vintages.

## Prepare training examples within an outer fit

```python
from eso.forecasting import build_training_examples

# Inside a future model.fit(history), using only that fold's supplied history:
X, y = build_training_examples(history, horizon=48, min_history=336, step=48)
observed_targets = y.notna()
# A future learner fits transformations and its model on training examples only.
```

For each historical inner origin, X is computed from its prefix only. An inner
origin is included only when its entire target horizon is contained in the outer
fit history. Targets at or after the outer forecast origin are therefore purged.
Features and targets have exactly matching indexes. Missing labels remain NaN
for explicit exclusion by the future learner. Missing feature values also remain
NaN; any encoding, scaling or filling must be fitted within the outer training fold.
Training examples accumulate in memory; choose origin spacing and GSP sampling
deliberately for multi-year/global-model training.

Raw input must still respect the observed-data contract. This builder does not
make retrospective revisions operationally available or validate raw publication
delays. It introduces no future-target reads or Phase 1 imputation.

The regression suite checks exact lag/rolling values, signed and constant series,
missingness, calendar encodings, 46/50-period DST days, holiday unknowns, matching
target axes, purged final horizons, and forecast-feature invariance when future
observations are perturbed. A feature-backed test model also runs through the
existing fit/predict walk-forward interface.
Target-lag tests check exact agreement with daily/weekly naive donors through
48 leads and reject unavailable donors at/after every inner origin, including
longer horizons and insufficient-history boundaries.
