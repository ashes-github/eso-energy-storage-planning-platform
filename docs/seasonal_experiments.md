# Multi-week seasonal Phase 1 experiments

The original June 20–24 configuration and saved results remain available. The
separate `eso.seasonal` workflow evaluates June 1–August 31, 2022 (92 days) and
December 1, 2022–February 28, 2023 (90 days). Each season is split into weekdays
and weekends. Weekdays mean Monday–Friday, including bank holidays; holidays
and weather are not yet controlled covariates. One summer and one winter do not
establish year-to-year generalisation.

```powershell
.\venv\python.exe -m eso.seasonal --output output/my_seasonal_run
.\venv\python.exe -m eso.seasonal_report --run output/my_seasonal_run
```

Use a new output directory. `--skip-models` produces the complete daily/weekly/
seasonal screening experiment without fitting latent models. Edit
`configs/phase1_seasonal.json` to change date ranges or training budgets.

## What counts as a qualifying day

The unit is GSP × local calendar day. Signed import/export observations are
resolved using the existing revision policy. Negative exports are observed
values, not missing data. For each focal day:

1. Compute p10 and p90 from that day's demand. The low threshold is
   `p10 + 0.35*(p90-p10)`.
2. Find uninterrupted low-demand runs using the previous, focal and next day.
   Attribute an event to the **local date its low run starts**, even when it
   crosses midnight. A bounded run must end before its following peak window.
3. Require a subsequent **positive** demand peak within eight hours of the run's
   end, at least `p10 + 0.75*(p90-p10)`.
4. Record separately whether qualifying runs last **at least six hours** and
   **at least eight hours**. These are minimum-duration sensitivity cases, not
   an upper-bounded six-to-eight-hour interval.

The complete three-day context must be available, and relevant censored runs
make a day unknown. This conservative rule can invalidate a focal day even if
its own 48 periods are present. Constant profiles are valid nonqualifying days.
Clock changes use elapsed UTC half-hours; local days can have 46 or 50 periods.
The 48-clock-slot representative profile averages repeated autumn slots and
leaves unobserved spring slots missing before across-day aggregation.

There is **no imputation in this recurrence experiment's primary denominator**:
qualifying days / valid source-observed days. GSPs with gaps remain in the
daily and seasonal tables, including GSPs absent for an entire season. Actual
source estimates are retained as source observations; audit metadata counts
them. They are not equivalent to independently verified measurements.
The existing bounded-imputation workflow remains available for short-window
experiments. Avoiding imputed outcomes here prevents manufactured recurrence.

Evidence status requires at least four valid days and 80% valid-day coverage
within the relevant group. Four days is an eligibility guard, not proof of
statistical precision; inspect counts and intervals. Unknown frequencies are
blank, never zero. Weekly partial-calendar bins retain their actual denominator.

## Spectral information and uncertainty

Each valid day has a Hann-window periodogram, sampled at two observations/hour.
Its frequency spacing is approximately 1/24 cycles/hour. Integrating the 6–8h
and 20–28h bands with interpolated edges does **not** increase that resolution.
These daily spectral fractions are coarse descriptive covariates, not evidence
of a well-resolved daily periodicity and not a condition for qualification.
Weekends are never concatenated into an artificial continuous signal.
See the [SciPy periodogram documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html)
and [spectral analysis discussion](https://docs.scipy.org/doc/scipy/tutorial/signal.html#spectral-analysis).

Seasonal tables include descriptive 95% intervals from 1,000 resamples of
calendar-week blocks, using the ratio of summed qualifying and valid counts.
At least four observed weeks are required. Resampling whole weeks preserves
within-week dependence but does not solve longer dependence, missing-not-at-
random bias, weather confounding or interannual uncertainty. All-zero/all-one
observations can produce degenerate intervals; these are not certainty about
future behaviour or calibrated probabilities of battery suitability.

No new weighted suitability score or permanent binary GSP classification is
introduced. Battery capacity, losses, dispatch economics and network constraints
still require separate physical/economic validation.

## Earlier versus later weeks and latent models

The first approximately two-thirds of each season form development data; the
last third provides later targets. Two focal dates straddling the split are
purged because their three-day contexts would overlap the opposite phase.
All-season recurrence still includes these dates. The split date and per-day
phase are saved, making the chronology auditable.

Compare two GSP representations, computed **only from development days**:

- Median 48-clock-slot profile.
- Median 48 + interquartile range 48 + five median daily behaviours: bounded
  low-run duration, maximum upward ramp, p90–p10, and the two spectral fractions.

Recurrence outcomes and coverage are excluded from model inputs. A common GSP
cohort must have sufficient development and holdout evidence and finite input/
target profiles. Other GSPs retain their screening results. Eligibility based
on later availability makes this an evaluable-cohort benchmark; it does not
demonstrate performance on entirely unobserved GSPs.

Within each regime, a fixed 80/20 GSP split is reused for both representations.
The scaler and models fit only training-GSP development representations.
Held-out GSP coordinates are inferred from their development representations;
**no later-day values enter fitting or latent inference**. The decoded first
48 dimensions are compared with held-out GSPs' later median profiles in original
Meter Volume units. The persistence baseline predicts the earlier median
unchanged. This is temporal transfer of a typical profile, not a chronological
half-hour forecast. Development reconstruction MSE is also saved; its values
across different representation widths should not be directly ranked.

Four regimes × two representations × (one PCA + three autoencoders + three
independent-prior GPLVMs) gives **56 fitted models**, plus four persistence
baselines. GPLVM uses 32 k-means inducing points, prior scale 1, latent dimension
2, 300 steps; AE uses 500 steps and both use 200 latent-inference steps. These
are fixed comparison budgets, not a convergence claim. Saved loss curves and
rotation-invariant pairwise-distance correlations expose optimisation and seed
stability. Latent plot colour is descriptive all-season recurrence, unused in
training; it must not be presented as independent classification accuracy.

The historic 240-dimensional experiment has a different target and cohort, so
its MSE cannot be ranked directly against this 48-point temporal-transfer task.
`reference_week_comparison.csv` instead re-evaluates June 20–24 under the same
new daily rule and compares its recurrence with the season and with the other
summer weekdays. This isolates window representativeness. The new rule uses
daily thresholds and extra midnight context, so it is not a literal reproduction
of the older window-wide screening scores.

## Saved results

- `daily_features.csv` and `daily_profiles.npz`: identical row order, every GSP/day.
- `weekly_recurrence.csv`, `seasonal_recurrence.csv`: counts, coverage, frequencies,
  evidence status and seasonal week-bootstrap intervals.
- Observed seasonal panels, configuration, revision audit, source file metadata,
  package versions, code hashes, chronological splits and completion marker.
- `models/`: fixed GSP splits, scaled inputs, scalers, fitted artifacts, latent
  coordinates, losses, earlier medians, later targets and predictions.
- `model_metrics.csv`, `model_eligibility.csv`: actual model cohort and errors.
- Report command adds `regime_summary.csv`, `model_summary.csv`,
  `reference_week_comparison.csv`, `training_diagnostics.csv`, loss/latent plots
  and `RESULTS.md`. Main command creates weekly heatmaps and regime boxplots.

Each seasonal GPLVM `gplvm_<seed>.npz` includes `latent_posterior_sd`, aligned
with its training latent means in `latent` and `train_gsps` in the regime's
`split.json`. The adjacent `gplvm_<seed>_latent_posterior_sd.csv` labels rows
with training GSP IDs and columns with `z1_sd`, `z2_sd`, etc. These are
variational posterior standard deviations of the **training latent variables**,
in latent-coordinate units. Held-out GSP coordinates are optimized point
estimates from `infer_decoder()`; no held-out latent posterior SDs are produced.
The main `eso.pipeline` output `latent_posterior_sd.csv` has the same training-only
interpretation. Existing saved seasonal runs require rerunning the models to
produce these additional exports.

Future extensions should add more years, spring/autumn, holiday/weather strata,
masked-gap recurrence sensitivity and physical storage/dispatch validation.
Threshold calibration needs separate evidence; selecting thresholds because
they produce attractive seasonal results would overfit this experiment.
