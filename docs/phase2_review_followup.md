# Phase 2 review follow-up — October 29, 2026

The review's two immediate pre-ML changes are implemented:

1. Forecast runs refuse an existing output directory before reading inputs.
   Directory creation also uses `exist_ok=False` to prevent a path created during
   preparation from being reused. A regression verifies prior artifact bytes
   remain unchanged on refusal. VS Code output prompts now describe this policy.
2. The feature builder distinguishes `origin_lag_k` from `target_lag_k` and adds
   daily/two-day/weekly target-aligned seasonal donors. Donors are read only when
   strictly before the origin. Unavailable donors remain NaN, including daily
   donors beyond lead 48. Tests cover exact values, matching naive donors, short
   histories, missing observations, future perturbation and inner-origin guards.

The earlier `lag_k` feature columns were renamed `origin_lag_k`. The feature guide
documents migration and the one-based-lead formula. Statistical fitting and the
recorded smoke results are unchanged. Existing experiment directories were not
rerun or overwritten.

Validation: **all 56 tests pass** using
`python -m unittest discover -s tests -v` with `PYTHONPATH=src` and the statistical
extra installed. The original 42-test milestone is now explicitly identified as
historical in the statistical results note.

## Remaining recommendations

- **Training-only coverage eligibility:** add an optional, audited observed-fraction
  threshold for representative-series sampling. Current sampling reports coverage
  without imposing that threshold. Keep constants/zeros and avoid excluding GSPs
  based on model performance.
- **Holiday calendar:** provide every local date explicitly, including non-holidays,
  with the correct regional/vintage rules before using this feature in ML.
- **Global-scale execution:** vectorize or stream prediction construction and review
  training-example memory use before multi-year all-GSP runs.
- **Repository polish:** expand package description, add an `eso-forecast` entry point
  and CI coverage for core and statistical tests.
- **Long-history study:** verify actual settlement coverage in the local historical
  GP9 batches and prepare a longer observed panel. The existing review run remains
  a short-history smoke experiment, not a completed 180-day modelling study.

Retain unrestricted and common-case/per-lead metrics, existing SARIMA/ETS settings,
and explicit missingness/failure policies. ML models, fold-fitted identity encoders
and preprocessing remain a subsequent implementation milestone; no ML model is
activated by these review fixes.
