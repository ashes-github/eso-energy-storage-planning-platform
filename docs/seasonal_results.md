# Completed seasonal Phase 1 experiment

Run: `output/phase1_seasonal_2022_23`, using
`configs/phase1_seasonal.json`. Both screening and model fitting completed.
The original June 20–24 experiment remains available separately.

**The five-day window is not representative enough for a permanent GSP label.**
Using the same daily rule for both windows, 115 of 364 comparable GSPs differ
by at least 20 percentage points between June 20–24 and the **other** summer
weekdays. The median absolute difference is 11.68 percentage points. This
comparison excludes the reference days from the longer comparison window.

## Coverage and recurrence

The experiment covers 367 distinct GSP identities and 182 calendar dates:
66,794 GSP-days, of which 63,959 (95.8%) are evaluable. Missing context accounts
for 729 unknown GSP-days; censored low-demand runs account for another 2,106.
All identities remain in the results, including those without sufficient data.

| Regime | Days | GSPs with sufficient seasonal evidence | Median frequency, ≥6h | Median frequency, ≥8h |
| --- | ---: | ---: | ---: | ---: |
| Summer weekdays | 66 | 363 / 367 | 57.9% | 16.9% |
| Summer weekends | 26 | 360 / 367 | 65.4% | 27.4% |
| Winter weekdays | 64 | 365 / 367 | 55.6% | 33.3% |
| Winter weekends | 26 | 366 / 367 | 69.2% | 50.0% |

The frequencies are medians across GSPs with a nonempty valid-day denominator;
they are not the percentage of GSPs suitable for a battery. The sufficient-
evidence column applies the separate coverage/count guard. Six and eight hours
are minimum low-run durations, each followed by a positive demand peak.
The substantial duration sensitivity makes it inappropriate to hide these
assumptions inside a single suitability score.

Earlier/later recurrence rank correlations are 0.876 (summer weekdays), 0.820
(summer weekends), 0.922 (winter weekdays) and 0.883 (winter weekends). Median
absolute earlier/later changes are 8.3, 11.8, 7.1 and 11.3 percentage points,
respectively. These descriptive results suggest some persistent ordering across
GSPs, alongside changes in individual frequencies. They are not independent
physical validation or evidence of year-to-year stability.

![Weekly recurrence](../output/phase1_seasonal_2022_23/weekly_recurrence.png)

Rows are independently ordered within each panel, so a row number does not
identify the same GSP across panels. Blank cells have no valid days. Calendar
week labels can precede a season's first day; only dates inside the season
contribute. Consult the CSV for GSP identifiers and exact denominators.

## Representative-profile model comparison

Completed 56 fits: PCA, three AE seeds and three independent-prior GPLVM seeds
for each of two representations and four regimes. Four persistence baselines
bring the metrics table to 60 rows. The later median target is identical across
representations within a regime; all prediction errors are finite.

| Regime | Model-eligible GSPs | Persistence RMSE | Lowest mean fitted-model RMSE | Fitted model / representation |
| --- | ---: | ---: | ---: | --- |
| Summer weekdays | 353 | 5.009 | 6.021 | AE / median48 |
| Summer weekends | 352 | 13.275 | 9.748 | GPLVM / median48 + IQR48 + behaviour5 |
| Winter weekdays | 363 | 9.078 | 9.898 | PCA / median48 |
| Winter weekends | 351 | 8.162 | 8.357 | PCA / median48 |

RMSE is in the source Meter Volume units, computed against later held-out-GSP
median profiles. AE/GPLVM entries average three seeds; PCA has one fit. Selecting
the lowest result in this table is descriptive, not a validated model-selection
procedure. Different regimes have different targets/cohorts, so their absolute
RMSEs should not be used to rank which season is easier.

Persistence is better than every fitted-model mean in three regimes. The richer
GPLVM representation improves summer-weekend performance but worsens the other
three regimes compared with GPLVM median48. Therefore, the results do **not**
support universally replacing median profiles with the richer representation,
or claiming GPLVM is always best.

Seed-to-seed latent-distance rank correlations range from 0.939 to 0.999 across
AE/GPLVM comparisons, but training objectives still change materially near the
end of several runs. In particular, some AE losses rise between the last two
50-step blocks. These are completed fixed-budget experiments, not a claim of
converged optimisation. Inspect saved loss curves before selecting models.

## Verification and interpretation

All 26 automated tests passed, including signed-flow handling, missing-data
policies, cross-midnight attribution, clock changes, unknown-day denominators
and isolation of holdout-day values from development representations. Saved
source hashes matched the executed implementation at verification.

The methodology, commands, uncertainty assumptions and output definitions are
in [seasonal_experiments.md](seasonal_experiments.md). Detailed results are in
`regime_summary.csv`, `seasonal_recurrence.csv`, `reference_week_comparison.csv`,
`model_summary.csv` and `training_diagnostics.csv` under the run directory.

This extends evidence across weeks and two seasons within one annual cycle.
It does not establish permanent battery suitability. Additional years and
physical battery/dispatch validation remain necessary for broader inference.
