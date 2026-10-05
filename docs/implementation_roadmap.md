# Research upgrade roadmap

## Implemented Phase 1

1. Chunked source reading and a declared revision policy.
2. Complete UTC-aligned GSP panels with explicit exclusions and clock-change support.
3. Welch spectral features in physical units and contiguous temporal checks.
4. Configurable, uncalibrated screening labels with negative-volume review status.
5. PCA and autoencoder baselines, independent and label-informed GPLVM priors.
6. Seed, prior-scale and inducing-initialization comparisons.
7. Held-out GSP profile completion, descriptive latent metrics and uncertainty output.
8. Configurations, versions, splits, data hashes, fitted artifacts, loss plots and maps.
9. Synthetic regression tests for data integrity, signal meaning and holdout isolation.
10. Multi-week summer/winter × weekday/weekend recurrence, with all GSP identities
    retained, six/eight-hour sensitivity, coverage and calendar-week bootstrap intervals.
11. Chronological representative-profile evaluation: median48 versus median48 plus
    variability/behaviour, PCA/AE/GPLVM and a persistence baseline. See
    [seasonal experiments](seasonal_experiments.md) for the separate evaluation contract.

## Before interpreting screening as battery feasibility

Obtain independent suitability assessments or specify a physical dispatch model.
Required inputs include charging power, battery energy, efficiency, local capacity
constraints and a definition of available charging energy. Calibrate screening weights
and thresholds on development data; validate on untouched regions and date windows.
Demand alone cannot supply these independent targets.

Compare segment lengths, low-demand thresholds and six-hour versus eight-hour rules.
Investigate excluded regions, estimated values and unmatched boundaries rather than
silently dropping their audit records. Extend mapping with an explicit, reviewed GSP
identifier crosswalk where needed.

## Further model studies

Check optimization convergence and increase training budgets where required. Compare
inducing counts independently of latent dimensions, and benchmark an exact GP if the
data scale permits. Kernel PCA and UMAP can extend the baseline suite; t-SNE is a
visualization-only comparison. Fit richer autoencoders only if the small baseline's
errors justify added complexity. Hyperparameter selection needs a validation split
separate from the final holdout; current grid results are exploratory comparisons.

## Phase 1 freeze and Phase 2

Phase 1 is research-complete for now. Dispatch simulation and multi-year validation
remain deferred Phase 1 extensions, not blockers for forecasting.

The [Phase 2 foundation](forecasting.md) implements the observed-data contract,
persistence/daily/weekly naive baselines, expanding-window validation, a reserved
final test period, and MAE/RMSE/MASE with coverage reporting for 8/48-step forecasts.
Retrospective Phase 1 fills are excluded. Latest-revision panels remain retrospective;
operational evaluation still requires per-origin revision and availability cutoffs.

Next: representative-series SARIMA/ETS, global tree-based forecasting, global
LSTM/GRU benchmarks, calibrated intervals, error slicing, and Phase 1 integration.
The Phase 1 held-out completion metric is not a forecast accuracy claim.
