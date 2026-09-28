# Guide-to-implementation cross-check - 13 September 2026

Historical audit snapshot. Data completeness, export treatment and exclusion policy
were subsequently corrected; see [missing-data handling](missing_data_handling.md).

Verdict: the runnable core Phase 1 workflow is implemented and verified. The entire
browser recommendation set is **not yet complete**. In particular, calibrated
suitability, the full proposed baseline suite and justification of the sparse GP
choice remain open. Earlier completion statements apply to the delivered core
implementation, not to every recommendation in the guide.

This audit reread the current modules and configuration, reran the nine unit tests,
checked existing artifacts and recalculated saved reconstruction metrics. It did
not retrain the full grid or change model/preprocessing behaviour.

| Guide recommendation | Status | Current evidence and remaining work |
| --- | --- | --- |
| Clarify supervision and label use | Addressed with correction | `models.py:gplvm_model` separates independent and label-informed priors. The original report/notebooks used labels in the prior, so the guide's colouring-only description was incorrect. |
| Correct physical frequency scaling | Implemented | `features.py:spectral_features` uses `fs=2.0`, giving cycles/hour for half-hourly data. |
| Standardize PSD, units and integration | Implemented, sensitivity pending | Explicit Welch/Hann/constant detrending/50% overlap/density scaling; trapezoidal band integration with interpolated endpoints. Segment-length comparison is not implemented as a study. |
| Replace heuristic thresholds with empirical calibration | Partial | Transparent configurable score and an 18-setting sensitivity analysis exist. No independent labels, fitted calibration or external validation exist. Sensitivity is not calibration. |
| Combine temporal and spectral evidence | Implemented with narrower feature use | Contiguous low runs, following-peak condition, peak/trough ratio and upward ramp are computed. Score uses qualifying-day fraction and daily-band power. The 6-8-hour spectral band, ramp and ratio are diagnostic outputs, not score inputs. Explicit time-to-next-peak and charge-energy feasibility are not implemented. |
| Compare prior scales 0.1, 0.5 and 1.0 | Implemented | All three appear in the completed 36-run GPLVM grid, crossed with independent/label-informed priors. Results are saved per configuration; convergence and external validity remain unestablished. |
| PCA initialization, multiple seeds and stability | Implemented | PCA plus seeded perturbation, three seeds and pairwise-distance rank stability. Prior mean is separate from guide initialization. |
| Improve inducing-point strategy | Partial | Random and k-means initialization are compared; inducing coordinates are trainable. Count is configurable but all full-grid runs use 32. No inducing-count sweep, exact-GP comparison or runtime/memory justification exists. |
| Quantitative latent evaluation | Implemented core metrics | Silhouette, Davies-Bouldin, Calinski-Harabasz, neighbour label agreement and reconstruction errors exist. Label-based metrics are descriptive and circular for informed priors. No held-out likelihood or independent feasibility accuracy exists. |
| Interpret posterior variance cautiously | Implemented interpretation; limited uncertainty evaluation | Per-coordinate latent posterior standard deviations are saved and documented as model uncertainty. No coverage/calibration assessment, predictive uncertainty propagation or uncertainty plots are implemented. |
| Add PCA, Kernel PCA, UMAP, AE and possibly t-SNE | Partial | PCA and a small autoencoder are implemented and run. Kernel PCA and UMAP are absent; optional t-SNE is also absent. |
| Reproducibility and experiment tracking | Substantially implemented | Seeds, configurations, versions, splits, cleaned-data hashes, model files, predictions, losses and completion markers are saved. Tracking is filesystem-based. No dependency lockfile or restore-and-predict round-trip test exists. Early full-grid artifacts predate source-hash recording. |
| Extract code from notebooks | Implemented for the new workflow | `src/eso` provides focused modules, with configs, tests and a reporting notebook. It uses a compact package instead of the suggested nested folder tree. Historical `experiments` notebooks remain unchanged and are not migrated to thin wrappers. |
| Geospatial visualization | Implemented with coverage limitation | Maps and unmatched-ID exports exist. The summer full run has 37 retained IDs without boundary matches. No reviewed identifier crosswalk exists. |
| Share preprocessing with Phase 2 | Foundation only | UTC-indexed cleaned panels are reusable. Forecasting, as-of revision selection and chronological backtesting are not implemented. |

## Verification performed again

- All nine unit tests passed on the current installed package. They cover data
  integrity, clock changes, synthetic spectral/temporal behaviour, reproducibility
  and withholding test values from inference.
- All four saved run directories have complete status markers and the expected
  model counts: 40 in the summer-weekday grid and four in each of three quick runs.
- Expected model files, prediction arrays, latent tables, metrics and latent plots
  are present for every listed model. This is an existence check, not a model-load
  round-trip check.
- All four cleaned-panel SHA-256 values match their recorded audit hashes.
- The two seasonal quick runs have source hashes matching the current recorded
  source files. The earlier full grid and initial quick run lack source hashes.
- Training MSE, withheld-period MSE and raw-unit withheld RMSE were independently
  recalculated from saved predictions, splits, scaler and data for all 40 full-grid
  models. All matched recorded values within numerical tolerance.
- The stored screening sensitivity table contains 18 settings, with candidate
  counts ranging from 5 to 228 out of 232 retained GSPs.

No full retraining was necessary to perform these checks. Passing unit tests and
artifact consistency do not establish scientific calibration or optimization convergence.

## Important interpretation limits

1. The default score is `0.8 * qualifying_day_fraction + 0.2 * fraction_20_28h`,
   with temporal eligibility and negative-volume review rules. It is still a
   hand-designed score, albeit more explicit than the historical PSD threshold.
2. The test is held-out **GSP profile completion**. Training GSPs expose every time
   column; test GSPs expose their first 75% of periods for coordinate inference.
   It is not an out-of-time forecast or an unseen-season test.
3. The 40 configurations use a shared test split without a separate tuning set.
   Descriptive comparison is valid, but choosing a winner on it requires a fresh
   final holdout for an unbiased performance estimate.
4. Only summer weekdays received the complete grid. Weekend/winter results are
   short execution checks with 80 sampled GSPs and cannot establish seasonal robustness.
5. The default 96-sample Welch segment gives 1/48 cycles/hour spacing; the daily
   band is coarse. A two-day window has only one segment. Resolution and window
   sensitivity remain necessary.
6. Low runs touching the window start or lacking the full following window are
   censored. The qualifying-day denominator still includes all days. This is a
   documented conservative choice that affects short-window comparisons.
7. Latent SD is variational uncertainty. Predictions plug in latent means, and
   there is no uncertainty calibration. The code does not turn SD into suitability.

## Suggested next implementation sequence

1. Add a validation split, training/inference convergence diagnostics and model
   save/load round-trip checks before expanding claims about model rankings.
2. Add Kernel PCA and UMAP under clearly stated evaluation protocols; retain t-SNE
   as optional visualization. Compare inducing counts and exact/sparse costs.
3. Compare spectral segment lengths, quantify boundary-censoring effects, and run
   the full comparison across multiple seasonal windows.
4. Obtain independent engineering labels or a specified dispatch model, then
   calibrate screening and validate it on untouched data. This needs information
   beyond the current demand-only implementation.
5. Build Phase 2 with as-of revisions and chronological forecast backtests using
   the same cleaned-data contract.

See [research decisions](research_decisions.md) for methodology and source references,
and [initial results](initial_results.md) for the observed model errors.
