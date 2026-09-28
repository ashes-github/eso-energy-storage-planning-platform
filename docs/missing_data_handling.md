# Flow correction and missing-data handling - 15 September 2026

## Why the 120 GSPs looked incomplete

The previous loader selected import (`I`) rows before assessing completeness.
For 20-24 June 2022, every apparently missing period had an export (`E`) observation.
Imputing those periods as demand would discard real direction information.

Across the four configured publication CSV files:

| Quantity | Count |
| --- | ---: |
| Selected raw rows, both directions | 87,840 |
| Unique resolved GSP/date/period observations | 87,360 |
| Import observations after revision/duplicate resolution | 75,070 |
| Export observations after revision/duplicate resolution | 12,290 |
| GSPs with 240 observed periods | 364 |
| Genuine missing periods | 0 |
| Previously retained import-only GSPs | 232 |
| Previously excluded mixed-direction GSPs restored | 120 |
| Additional export-only GSPs restored | 12 |

Repeated identical reports are deduplicated, not summed. Elexon's interface description
states that Meter Volume is unsigned and the I/E flag supplies direction. It also
explains that a GSP feeding multiple groups may be repeated with its total volume.
The pipeline now uses the explicit analysis convention **import positive, export
negative**; it does not assume the raw volumes were already signed.

[Elexon interface definition, section 5.13, page 51](https://assets.elexon.co.uk/wp-content/uploads/sites/11/2019/02/28155938/NETA_IDD_Part_2_v39.0_redlined_P344_P359.pdf)

Signed conversion precedes conflict checks. Opposite-direction observations at the
same latest revision raise an error. Later revisions can change direction. Raw negative
magnitudes are rejected to prevent double-sign conversion. Raw missing volumes remain
missing. Revision precedence otherwise remains the declared aggregation/flow/run policy.

## Current screening results

All **364** GSPs enter the feature and screening calculations, including four constant
profiles. With the current heuristic settings:

- 258 are candidates; 106 are not flagged.
- Of the previously excluded 120, 49 are candidates and 71 are not flagged.
- No values are imputed in this summer window. These are restored observations.

The 9,410 previously missing import-only slots across the excluded 120 match their
observed export slots exactly. `restored_gsps.csv` in the new summer run lists each
of those GSPs, its previous missing count, recovered exports and updated score.

Negative signed values can be low-demand/export intervals. A qualifying low interval
must be followed by positive demand within the configured horizon, as well as meeting
the peak-amplitude condition. An always-exporting GSP therefore does not satisfy this
particular demand-shifting screen. `not_flagged` is not a finding of battery infeasibility;
other uses, including export absorption, need a separate physical/dispatch criterion.

## Genuine-gap policy

The default configuration now contains `direction: signed_net` and:

```json
"missing": {
  "method": "hybrid",
  "max_missing_fraction": 0.2,
  "max_gap_periods": 8,
  "linear_max_gap_periods": 4,
  "min_seasonal_donors": 2
}
```

1. Keep every observed reading exactly. Mark where observations are absent.
2. For a GSP with at least 80% coverage and no missing run longer than four hours,
   interpolate bounded gaps up to two hours between their observed neighbours.
3. For remaining gaps, use the median at the same local half-hour on other observed
   days of the same type (weekday/weekend), requiring at least two distinct donor days.
   Donors always come from original readings, never previous imputations. Repeated
   local slots on a clock-change day count as one donor day.
4. Do not extrapolate, borrow another GSP's shape, replace missing values with zero,
   or synthesize a whole six-hour trough. Sparse/long-gap profiles remain in outputs
   as `insufficient_evidence` with a blank/nullable candidate label.

All limits are configurable research defaults, not confidence bounds. Even short-gap
filling can alter troughs and ramps. Successfully imputed profiles carry an `_imputed`
screening status, with their coverage and value-level provenance alongside the score.
Fully observed constants remain calculable and are not automatically removed.

Set `missing.method` to `none` to retain gaps without filling them. Direction-only
legacy runs (`I` or `E`) require this setting: imputation after dropping a known
direction would recreate the original mistake. The legacy `clean_panel` helper remains
available for historical complete-case tests; the current pipeline uses `align_panel`
and `impute_panel`.

This is retrospective analysis. Same-GSP donor days and right-hand interpolation
neighbours may be later than the gap. Forecasting needs an origin-specific causal
imputer and as-of revisions; this implementation does not claim to supply those.

## Validation and limits

Tests exercise sign conversion, direction revisions/conflicts, genuine missing values,
nullable CSV data, short interpolation, seasonal donors, weekday/weekend isolation,
sparse profiles, constant profiles and observed-value preservation. An end-to-end
fixture verifies that an imputed GSP enters training and a sparse GSP retains an
unknown screening result. Test GSPs are chosen only from originally fully observed
profiles, so imputed values are never treated as ground truth or used to leak their
future observations into latent-inference inputs.

Artificial masking uses 30 fully observed nonconstant GSPs, seed 17. It compares the
hybrid with permissive linear interpolation under random 5%/20% missingness and
2/4/6-hour blocks. The two-hour interpolation limit was chosen after an initial
diagnostic found same-clock medians less accurate for those short blocks. This is
development-set evidence, not an untouched external validation dataset.

For the final default settings in the saved masking experiment:

| Missing pattern | Hybrid filled fraction | Mean RMSE / original-profile SD | Screening label changes |
| --- | ---: | ---: | ---: |
| Random 5% | 100% | 0.168 | 0/30 |
| Random 20% | 100% | 0.217 | 0/30 |
| Two-hour block | 100% | 0.203 | 0/30 |
| Four-hour block | 100% | 0.451 | 0/30 |
| Six-hour block | 0% | Not evaluated | Unknown, not labelled |

The hybrid is not uniformly the lowest-RMSE method. The linear comparator has lower
four-hour RMSE (0.338) but changes one of 30 screening labels; it also fills all six-hour
blocks and changes one label there. These small-sample results do not prove the hybrid
is superior, nor that real missingness is random. Four-hour reconstruction errors can
still be substantial. Each filled value remains flagged, and larger gaps stay unknown.

## Outputs and commands

`observed_panel.csv` contains signed observations and genuine blanks.
`cleaned_panel.csv` contains all GSPs after permitted fills, with unresolved blanks.
`data_quality.csv` records original coverage, longest gap, imputation count, constant
status, model eligibility and observed export/import counts. `value_provenance.csv`
distinguishes observed import/export, linear, seasonal_median and missing values.
`missing_mask.csv` records original gaps. `screening.csv` includes all GSPs, including
unknown results. `excluded_gsps.csv` is a compatibility export of model-ineligible
rows, not a removal of those identities from the screening report.

```powershell
.\venv\python.exe -m eso.pipeline --output output/phase1_signed_new_run
.\venv\python.exe -m eso.validate_imputation --run output/phase1_signed_new_run
.\venv\python.exe -m unittest discover -s tests -v
```

The completed-data model protocol remains PCA/AE/GPLVM profile completion. Imputed
profiles may enter training; only originally fully observed GSPs enter the holdout.
There is no scored validation of latent models on imputed test GSPs yet. New error
values cannot be compared directly with the historical 232-GSP benchmark because
the population and test split changed.

Current summer results are in `output/phase1_signed_summer_weekdays/`; the earlier
`output/phase1_summer_weekdays/` is retained as a historical import-only result.

## Completed execution checks

All 19 tests pass, including the end-to-end incomplete-CSV fixture. The full summer
grid completed 40 models on 364 GSPs (291 training, 73 test). Mean standardized
withheld MSE is 0.051217 for PCA, 0.053315 across three autoencoders, and 0.072502
across 36 GPLVM variants. These are fixed-budget exploratory results, not converged
model rankings. Two further quick runs completed: summer weekend retained 364 GSPs
and winter weekdays retained 366. These also had no genuine gaps to impute.

Final preprocessing was checked against the full summer run: its numerical matrix
matches exactly, all 364 screening labels match, and all original 232 retained GSPs
keep their previous screening labels. The final nullable-gap conversion and legacy
mode guard were added after the all-observed full run began; this equivalence check
is saved in `post_update_verification.json`. The seasonal quick runs use those fixes.

The new map has 77 identifiers without boundary matches, saved separately; they
remain in calculation outputs. The map was visually checked. Latent posterior SD
does not incorporate imputation uncertainty: filled training values are currently
treated as point estimates. The `_imputed` status must accompany any interpretation
until multiple-imputation or a missing-observation likelihood is implemented.
