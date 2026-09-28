# Initial implementation results - 13 September 2026

Historical import-only results. The direction filter was corrected on 15 September:
the missing periods were exports. See [the correction](missing_data_handling.md)
and `output/phase1_signed_summer_weekdays/` for the 364-GSP run. Do not use these
232-GSP counts as the current data coverage.

The Phase 1 pipeline was run against the four local GP9 publication CSV files.
These results validate the implementation and provide an initial research comparison;
they do not establish engineering feasibility or a final model ranking.

## Summer weekdays: full configured grid

Window: 20-24 June 2022. The default configuration retained **232 GSPs x 240 periods**,
excluded 120 observed GSPs, and resolved one superseded/duplicate import record.
The training/test split contains 185/47 GSPs. Profiles without complete observations
are not padded. Two excluded profiles were also constant among their observed values.
Of the retained GSPs, 37 identifiers did not match the configured boundary file.

All **40 model runs completed**: one PCA, three autoencoder seeds and 36 GPLVM
combinations. GPLVM budgets were 300 steps per run, autoencoders 500 steps, and
test latent inference 200 steps. The last 25% of each test GSP profile was withheld
from latent inference. Standardization was fit on training GSPs only.

| Model family | Runs | Mean withheld standardized MSE | Range |
| --- | ---: | ---: | ---: |
| PCA | 1 | 0.020206 | 0.020206 |
| Small autoencoder | 3 | 0.020324 | 0.020129-0.020692 |
| GPLVM | 36 | 0.022085 | 0.021471-0.024589 |

This grid does not show a GPLVM reconstruction advantage. The difference between
PCA and the best autoencoder seed is small; selecting the best seed using this test
set would be model selection on the holdout. No significance claim is made. GPLVM
losses were still decreasing at the end of the inspected run, so convergence is not
established. The GPLVM family average mixes prior scales and inducing strategies;
inspect individual rows before drawing configuration-specific conclusions.

Seed stability uses pairwise-distance rank correlation against the first seed within
each configuration, avoiding arbitrary latent-axis orientation. The observed range
over reported model groups is approximately 0.973-0.994. This describes geometric
stability, not uncertainty calibration or physical suitability.

## Screening sensitivity

Default uncalibrated rules flag **209/232** retained GSPs. Holding other defaults
fixed, moving from a six-hour to an eight-hour minimum trough flags **66/232**.
This changes 143 labels. Across the 18 threshold/duration settings tested, candidate
counts range from 5 to 228. Independent targets and physical constraints are needed
before these rules can be calibrated or interpreted as installation recommendations.

## Additional seasonal checks

Both configurations completed the quick workflow (80 sampled GSPs, one seed,
40 GPLVM steps, 80 autoencoder steps). Features and audits cover the full retained
population; only model fitting is subsampled.

| Window | Retained complete GSPs | Periods | Default candidates |
| --- | ---: | ---: | ---: |
| Summer weekend, 18-19 June 2022 | 241 | 96 | 206 |
| Winter weekdays, 12-16 December 2022 | 262 | 240 | 219 |

These short runs are execution checks, not seasonal model comparisons. Two days
provide only one default Welch segment, and temporal boundary censoring is more
influential for that short window.

## Verification and artifact locations

Nine unit tests passed, covering revision conflicts, numeric revision ordering,
missing middle GSPs, zero exclusions, clock changes, invalid periods, constant
profiles, physical spectral bands, cross-midnight troughs, temporal gating, model
reproducibility, withheld-value isolation and rotation-invariant stability.
Generated maps, latent plots and a GPLVM loss curve were visually inspected.
The reporting notebook's JSON and code-cell syntax were checked.

- `output/phase1_summer_weekdays/`: full 40-run comparison and threshold sensitivity.
- `output/phase1_quick/`: initial summer execution check.
- `output/phase1_summer_weekend_quick/`: weekend execution check.
- `output/phase1_winter_weekdays_quick/`: winter execution check.

Each output contains the actual configuration, package versions, data audit,
split, individual model metrics, artifacts and a completion marker. The full grid
was run before source-hash recording was added; later seasonal runs also contain
source hashes. All output directories are local generated files and are gitignored.
