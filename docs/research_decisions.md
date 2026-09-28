# Phase 1 research decisions

## Evidence from the report and notebooks

The source report is *Nonlinear Gaussian Process Latent Variable Model to identify
latent structure from high dimensional data space*, Ashes Roy Chowdhury, Bath, 2023.
Page references below are the printed page numbers (PDF page = printed page + 1).

- Page 6 assumes a 6-8-hour low-demand window followed by a higher-demand period.
  This motivates temporal checks; it does not establish battery capacity, charging
  power, network headroom, efficiency, available generation or economics.
- Pages 30-31 describe normalized FFT power and heuristic thresholds (0.0034 and
  0.0075). These are not thresholds for the new Welch estimator.
- Page 32 explicitly sets the first latent prior coordinate from Fourier labels.
  The browser review missed this. The notebooks implement log2(label), mapping
  labels 1 and 2 to prior means 0 and 1, with standard deviation 0.1.
- Page 33 chooses as many inducing points as latent dimensions. The Fourier
  notebooks actually use 2 (or 3), despite some comments saying 32. These two
  quantities do not need to be equal.
- Page 45 proposes future forecasting. Shared UTC-indexed cleaned data supports
  a later forecasting pipeline; this implementation is not a forecast model.
- Existing notebooks already call `pyro.set_rng_seed(1)`.
- A prior review incorrectly inferred date coverage from publication filenames.
  Direct inspection shows GP9_202301.csv contains settlement dates 2020-09-07
  through 2023-01-26, including the report's summer 2022 window.

The original notebooks remain historical experiments. The new modules make
deliberate methodological changes and do not claim to reproduce their figures.

## Data contract

The default window reproduces the report's June 20-24, 2022 selection. All files
matching the configured glob are scanned in chunks, selecting settlement dates.
The default `signed_net` mode retains both directions and explicitly represents
imports as positive and exports as negative. Direction is resolved with revisions;
export records are observations, not gaps. See [missing-data handling](missing_data_handling.md)
for the correction to the earlier import-only implementation and its validation.
Revision precedence is lexicographically latest Date of Aggregation, Flow Run Date,
then numeric CDCA Run Number. Conflicting volumes at an identical latest revision
raise an error. This is an explicit research policy requiring domain review before
production use; it is not an asserted settlement-run hierarchy.

Every observed GSP remains in screening outputs. Short genuine gaps can be imputed
under the configured limits; unresolved profiles get a nullable candidate label and
`insufficient_evidence` status. Constant observed profiles are retained. Model matrices
use fully observed or successfully imputed rows. No hard-coded GSP count is used.
London calendar boundaries define 46/48/50-period days, converted
to a continuous half-hourly UTC grid. A region absent for the entire window cannot
be discovered from those rows; supply a separate registry for a population audit.
Estimated values are retained and counted. Source file sizes, modification times,
resolved configuration, package versions and a SHA-256 of the cleaned matrix are saved.

The current revision policy is retrospective. A forecasting backtest must additionally
filter revisions by information available at each forecast origin to avoid look-ahead.
Meter Volume is kept in source units; no undocumented conversion to MW is performed.

## Features and screening

Welch uses fs=2 samples/hour, Hann windows, constant detrending, 50% overlap and
density scaling. The default segment is 96 samples (48 hours), with spacing
1/48 cycles/hour. Integrals use trapezoidal integration with interpolated band
endpoints. Bands are 1/8 to 1/6 cycles/hour and 1/28 to 1/20 cycles/hour.
Density units are squared Meter Volume units per cycles/hour; integrals have squared
Meter Volume units. Ratios to total integrated power are dimensionless.

The daily band is coarse at this resolution. Two-day datasets contain only one
default Welch segment and cannot benefit from averaging multiple segments. Longer
windows and segment-length sensitivity are necessary for reliable spectral estimates.
Power at a 6-8-hour cycle does not measure a 6-8-hour trough duration.

Low demand is at or below p10 + 0.35*(p90-p10), calculated per GSP over the analysis
window. A run qualifies when it lasts at least six hours and a subsequent value
within eight hours reaches p10 + 0.75*(p90-p10) and is positive (later import demand).
Midnight-crossing runs are supported.
Runs touching the start or lacking a complete following window are censored. The
qualifying-day fraction counts distinct local start dates divided by all window days;
boundary censoring makes this conservative, especially for short windows.

The default score is 0.8*qualifying-day fraction + 0.2*daily-band power fraction.
Candidate screening requires a qualifying-day fraction of at least 0.2 and a score
of at least 0.25. Negative signed values are valid export observations, not an
automatic disqualification. Always-exporting profiles do not meet the later-positive-
demand criterion, but that does not establish general battery non-suitability.
These weights and thresholds are explicit, **uncalibrated research defaults**.
The 6-8-hour band, ramp, duration and peak/trough ratio are diagnostic features.
None of the labels are ground-truth battery feasibility decisions.

To calibrate, obtain independent engineering assessments or a dispatch simulation
with battery power/energy, charging constraints, efficiency and network capacity.
Choose thresholds using training regions/windows and freeze them before external
validation. Agreement with the same heuristic labels is not such validation.

## Models and evaluation

PCA and a small tanh autoencoder are the initial baselines. GPLVM uses an RBF
SparseGPRegression with diagonal Normal latent guide, PCA initialization with
seeded small perturbations, and a separately specified prior mean. The independent
prior is zero; the label-informed prior assigns 0/1 from the new screening labels
to its first coordinate. The guide initialization is the same across both modes.
This isolates the prior effect but is not an exact historical initialization replay.

The full configuration compares three seeds, scales 0.1/0.5/1.0 and random/k-means
initial inducing locations. Inducing coordinates are optimized during training.
The count is configurable (default 32); sparse versus exact GP is not benchmarked.
The grid contains 36 GPLVM runs, three autoencoders and one PCA. The step counts
are experiment budgets, not proof of convergence; inspect saved loss curves.

The split holds out 20% of originally fully observed GSPs with a fixed seed. Any
successfully imputed GSPs enter training, not test ground truth. Standardization is fit only on
training GSPs, per time column, preserving relative GSP scale. All models see the
same matrix. Test latent coordinates use only the first 75% of periods. Reconstruction
error is measured on the remaining periods, with both standardized MSE and raw-unit
RMSE. GPLVM and autoencoder test coordinates are optimized against the frozen
decoder, while PCA uses least squares. This is a held-out **profile completion**
task, not a chronological forecast: training GSPs expose all time columns.

Silhouette, Davies-Bouldin, Calinski-Harabasz and nearest-neighbour label agreement
describe training embeddings relative to heuristic labels. Label-informed runs are
circular evidence for separation and must not be ranked as independent classifiers.
Undefined single-class metrics are null. Pairwise-distance rank correlation measures
seed stability without relying on axis orientation. Latent posterior standard
deviation is model uncertainty, not battery feasibility; it is not predictive
variance or a calibrated confidence interval. Decoding plugs in latent means and
does not integrate latent uncertainty. No held-out likelihood is claimed.

Additional Kernel PCA/UMAP/autoencoder architectures, exact-GP comparisons and
external calibration can extend this common protocol. They should follow evidence
from these initial baselines rather than be prerequisites for every pipeline run.

## References checked during implementation

- [SciPy Welch API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html)
- [SciPy spectral analysis and resolution tradeoffs](https://docs.scipy.org/doc/scipy/tutorial/signal.html)
- [Pyro GPLVM tutorial](https://pyro.ai/examples/gplvm.html)
- [Elexon settlement-period definition](https://www.elexon.co.uk/bsc/glossary/settlement-period/)
