# ESO demand-profile research

A reproducible Phase 1 pipeline for Grid Supply Point (GSP) demand-pattern screening
and GPLVM evaluation, developed from Ashes Roy Chowdhury's 2023 MSc report.

The pipeline combines explicit low-demand windows with spectral features, then
compares PCA, an autoencoder and GPLVM variants. Screening labels are uncalibrated
research hypotheses. They do not establish battery installation feasibility.

The default now retains **both import and export observations** as signed flows
(import positive, export negative). For June 20-24, 2022 this restores all 364 GSPs;
the previous 232-GSP result came from filtering out exports. See
[flow correction and missing-data handling](docs/missing_data_handling.md).

## Run

From the project root using the existing Windows environment:

```powershell
$env:PYTHONPATH = 'src'
.\venv\python.exe -m eso.pipeline --quick --output output/my_quick_run
.\venv\python.exe -m eso.pipeline --output output/my_full_run
.\venv\python.exe -m unittest discover -s tests -v
.\venv\python.exe -m eso.sensitivity --run output/my_full_run
.\venv\python.exe -m eso.validate_imputation --run output/my_full_run
```

Use a new output directory for each run to keep comparisons separate.
The quick run fits 80 sampled GSPs with short optimization budgets to verify the
workflow. The full default run uses all retained GSPs and fits 40 models: PCA,
three autoencoders and 36 GPLVM variants. Inspect losses before judging convergence.

For a fresh environment, install the package with `python -m pip install -e .`;
then use `eso-phase1 --config configs/phase1.json --output output/my_run`.
The Python package requires SciPy >=1.14 and does not use the removed `simps` API.

Edit `configs/phase1.json` for date windows, screening rules, seeds, prior scales,
inducing count and training budgets. Paths in the configuration resolve relative
to the project root. Input filenames describe publication batches; the loader
filters their actual settlement dates. The default is June 20-24, 2022, as in the report.
Seasonal alternatives are `configs/phase1_summer_weekend.json` and
`configs/phase1_winter_weekdays.json`; select them with `--config`.

For the expanded **92-day summer / 90-day winter** study, including separate
weekdays and weekends, daily recurrence and later-week model evaluation:

```powershell
.\venv\python.exe -m eso.seasonal --output output/my_seasonal_run
.\venv\python.exe -m eso.seasonal_report --run output/my_seasonal_run
```

See [seasonal experiment definitions and limits](docs/seasonal_experiments.md).
This workflow retains every GSP in screening and reports observed pattern
frequencies with coverage, rather than assigning permanent suitability labels.
The completed summer/winter study is summarized in
[seasonal results](docs/seasonal_results.md).

## Local data

Downloaded datasets are excluded from Git. Before running the pipeline, supply
the GP9 CSV files under `GP9_2023/` and the GSP boundary shapefile with its companion
files under `Data/gsp_regions_20220314/`, or update the input paths in your chosen
configuration. See that configuration's `input_glob` and `shapefile` settings for
the exact paths; historical notebooks may require additional files under `Data/`.
The historical exports `experiments/small_data.csv` and
`experiments/nonzero_gsps.csv` are also kept local.

Generated results in `output/`, temporary files in `tmp/`, Python environments,
and local `.env` credentials are excluded as well. Source code, configurations,
tests, documentation, and research notebooks remain eligible for version control.
Notebook cell outputs are part of the notebook file; review them before publishing.

## Outputs

Each run saves:

- `observed_panel.csv`: signed observations before any imputation, including real gaps.
- `cleaned_panel.csv`: all GSP rows after permitted imputation; unresolved gaps remain blank.
- `data_quality.csv`, `missing_mask.csv`, `value_provenance.csv`: coverage, original gaps and value origins.
- `data_audit.json`: revision/flow policy, source metadata and recovery counts.
- `excluded_gsps.csv`: compatibility export of model-ineligible rows; these GSPs remain in screening as insufficient evidence.
- `screening.csv`: temporal/spectral features, heuristic scores and review status.
- `screening_map.png`, `unmatched_gsps.csv`: spatial view and unmatched identifiers.
- `comparison.csv`: reconstruction errors and descriptive latent-space metrics.
- `split.json`, `config.json`, `versions.json`, `scaler.joblib`: reproducibility records.
- Per-model artifacts, latent coordinates, predictions, losses and plots.
- `run_status.json`: completion marker; incomplete runs lack a complete status.

The shared evaluation holds out GSPs and withholds their final 25% of periods during
latent inference. This is profile completion, not forecasting. Label separation in
label-informed GPLVM runs is not independent validation because labels enter the prior.
Only originally fully observed GSPs enter the test split; recovered imputed profiles
may enter training. Metrics are never scored against imputed ground truth.

## Project layout

```text
src/eso/                    data, features, models, evaluation, plots, pipeline
configs/                    runnable research settings
tests/                      scientific and data-integrity regressions
notebooks/exploration/      reporting from saved pipeline outputs
experiments/                original historical research notebooks
docs/                       assumptions, research decisions and next studies
output/                     local generated experiment artifacts (gitignored)
```

See [research decisions](docs/research_decisions.md) for the report-to-code audit,
feature definitions, units, uncertainty interpretation and evaluation limitations.
See the [roadmap](docs/implementation_roadmap.md) for external calibration and Phase 2.
See [initial results](docs/initial_results.md) for the completed 40-model comparison,
seasonal execution checks and screening-threshold sensitivity.
The [original README](docs/original_research_readme.md) is preserved as historical context.
