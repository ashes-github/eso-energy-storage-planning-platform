# Running the forecasting pipeline

The Phase 2 pipeline runs **persistence, daily seasonal naive (48 steps), and weekly
seasonal naive (336 steps)** with expanding-window evaluation. Every run evaluates
both **4-hour (8-step)** and **day-ahead (48-step)** forecasts. SARIMA/ETS, tree models,
LSTM/GRU and prediction intervals are not implemented yet.

## 1. Set up

Run these commands in **PowerShell from the repository root**, using the existing
project environment. Activation is not required:

```powershell
$env:PYTHONPATH = 'src'
.\venv\python.exe -m eso.forecasting.pipeline --help
```

For a fresh environment, use Python 3.10 or later:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -e .
```

A standard Windows virtual environment uses `venv\Scripts\python.exe`. If you
created one this way, replace `venv\python.exe` below with that path. Install and
run with the same interpreter.

## 2. Prepare input

`--input` takes a **directory**, containing either:

- `observed_panel.csv` (preferred), or
- both `cleaned_panel.csv` and `value_provenance.csv`, with identical axes.

The first CSV column contains unique GSP IDs. Remaining headers are increasing,
timezone-aware half-hour interval starts. Rows hold numeric demand values; blanks
represent missing observations. Keep all half-hour columns, including missing
intervals. Negative signed net imports are valid. Values retain source Meter Volume
units; there is no conversion to MW.

The loader prefers observations. With the fallback files, it retains only values
labelled `observed`, `observed_import` or `observed_export`. Retrospective Phase 1
fills are excluded from history and scoring targets.

Use full-calendar panels, including weekends. Do not use median profiles,
weekday-only panels, or join separate seasons across a time gap. Defaults require
at least **16 elapsed days**: 14 training days, one validation day and one test day.
Longer validation and test periods are preferable for research. The default five-day
Phase 1 output is too short.

### Optional: build an observed panel from GP9 files

Skip this if you already have suitable input. This preparation step uses the existing
loader without fitting Phase 1 models. Adjust the dates to settlement dates covered
by your local `GP9_2023/*.csv` files. Publication filenames do not establish settlement
coverage; the June-July dates below are examples.

```powershell
@'
import json
from pathlib import Path
from eso.data import load_records, align_panel
from eso.forecasting import ForecastDataset

start, end = '2022-06-01', '2022-07-31'
paths = sorted(Path('GP9_2023').glob('*.csv'))
if not paths:
    raise FileNotFoundError('No GP9 CSV files found')
records, sources = load_records(paths, start, end, direction='signed_net')
panel, provenance, audit = align_panel(records, start, end, direction='signed_net')
ForecastDataset(panel)
audit.update(start=start, end=end, sources=sources)
output = Path('output/forecast_input')
output.mkdir(parents=True, exist_ok=False)
panel.to_csv(output / 'observed_panel.csv')
provenance.to_csv(output / 'value_provenance.csv')
(output / 'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
print('GSPs:', len(panel), 'half-hours:', panel.shape[1])
print('UTC range:', panel.columns[0], 'to', panel.columns[-1])
print('Observed fraction:', panel.notna().to_numpy().mean())
'@ | .\venv\python.exe -
```

Missing source periods remain blank. Inspect coverage before interpreting scores.
Use a new directory if `output/forecast_input` already exists.

## 3. Choose and check the final test boundary

Set your input directory and the first interval of the final test period. These
example values match the optional preparation step; change them for another panel:

```powershell
$forecastInput = 'output/forecast_input'
$forecastTestStart = '2022-07-15T00:00:00Z'

@'
import sys
from eso.forecasting import ForecastDataset, WalkForward

data = ForecastDataset.from_phase1(sys.argv[1])
plan = WalkForward(min_train=672, test_start=sys.argv[2], step=48)
print('GSPs:', len(data.panel), 'half-hours:', data.panel.shape[1])
print('UTC range:', data.panel.columns[0], 'to', data.panel.columns[-1])
print('Observed fraction:', data.panel.notna().to_numpy().mean())
for split in ('validation', 'test'):
    folds = plan.folds(data.panel.columns, split)
    print(split, 'origins:', len(folds), 'first:', data.panel.columns[folds[0]])
'@ | .\venv\python.exe - $forecastInput $forecastTestStart
```

This checks fold availability without scoring the test. The boundary must include
a timezone (`Z` means UTC), match an exact panel timestamp, leave at least 720
half-hours before it with defaults, and leave at least 48 half-hours from it onward.
Keep this boundary fixed during model comparison.

## 4. Run validation

```powershell
.\venv\python.exe -m eso.forecasting.pipeline --input $forecastInput --output output/forecast_validation_01 --test-start $forecastTestStart --min-train 672 --step 48 --split validation
```

The first origin follows 672 training half-hours. Training expands by 48 half-hours
at each subsequent origin. Only windows with all 48 target steps before the test
boundary enter validation. An origin is the first predicted interval; history ends
one half-hour earlier. Use validation results for model development.

## 5. Run the final test after freezing choices

```powershell
.\venv\python.exe -m eso.forecasting.pipeline --input $forecastInput --output output/forecast_test_01 --test-start $forecastTestStart --min-train 672 --step 48 --split test
```

The first test origin is exactly `--test-start`. Earlier test observations enter
later training histories once they are in the past. Do not revise model choices
using test scores. Each split runs independently; test execution does not load
validation artifacts. Every run needs a **new output directory**.

## Debug in VS Code

Open this repository folder in VS Code with the Python Debugger extension enabled.
In **Run and Debug**, select **Debug ESO Forecasting - Validation**, then press
**F5**. Enter a new output directory when prompted; increment the run suffix for
later runs (for example, `output/forecast_debug_validation_02`).

The configuration uses `venv/python.exe`, sets `PYTHONPATH` to `src`, and launches
`eso.forecasting.pipeline` with the prepared `output/forecast_input` panel, the
`2022-07-15T00:00:00Z` test boundary, 672 training half-hours and a 48-step origin
interval. It evaluates validation only. Edit `.vscode/launch.json` if your input,
interpreter or split settings differ.

Set breakpoints in `src/eso/forecasting/pipeline.py` at data loading or in
`src/eso/forecasting/backtesting.py` inside the origin loop to inspect model history
and predictions. **Debug ESO Phase 1** remains the screening-pipeline configuration.

## CLI options

| Option | Default | Meaning |
| --- | --- | --- |
| `--input` | Required | Directory containing the input panel files. |
| `--output` | Required | New directory for run artifacts. |
| `--test-start` | Required | Timezone-aware first test interval, present in the panel. |
| `--min-train` | `672` | Initial training length in half-hours (14 elapsed days). |
| `--step` | `48` | Distance between origins in half-hours (24 elapsed hours). |
| `--split` | `validation` | Evaluate `validation` or `test`. |

Both horizons and all three baselines always run; there is no CLI model/horizon
selection option. Use `--step 8` for an origin every four hours. This creates
overlapping day-ahead forecasts and more rows. Each origin/GSP produces 168 rows:
three models times (8 + 48) leads. Results accumulate in memory before export.

## Read the results

| File | Contents |
| --- | --- |
| `predictions.csv` | Model, split, origin, target timestamp, horizon, lead, GSP, actual, prediction and training MASE scale. |
| `metrics_per_gsp.csv` | MAE, RMSE, MASE and coverage for each model/split/horizon/GSP. |
| `metrics_pooled.csv` | The same metrics pooled across GSP forecast cases. |
| `run.json` | Fold settings, input path, units and availability/revision assumptions. |

```powershell
Import-Csv output/forecast_validation_01/metrics_pooled.csv | Format-Table model,horizon,mae,rmse,mase,coverage,n_scored -AutoSize
```

Lower errors are better when comparing the same scored cases. Check coverage first:
missing seasonal donors produce missing predictions, not a fallback. `n_cases`
counts requested cases, `n_observed` counts available actuals, and `n_scored` counts
cases with actual and prediction. `coverage = n_scored / n_observed`. MASE uses
training-only lag-48 differences; zero or unavailable scales produce undefined MASE,
reflected in `n_mase`.

Horizon 8 scores leads 1 through 8; horizon 48 scores leads 1 through 48. These are
not last-lead-only scores. Do not combine horizon groups: the first eight leads
overlap. Pooled metrics average GSP forecast cases, not errors of summed system
demand. Repeated targets at different origins are separate forecast cases.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `No module named eso` | Run from the repository root and set `$env:PYTHONPATH = 'src'`, or install with `-m pip install -e .` using the same interpreter. |
| Missing input CSV | Pass its containing directory; provide observations or both cleaned values and provenance. |
| `Require contiguous half-hours` | Build a full-calendar panel; keep missing values blank instead of deleting timestamps. |
| Timezone or test timestamp error | Use an exact panel interval with `Z` or an explicit offset. |
| `Need at least one full validation and test forecast window` | Supply more history or adjust the boundary/training length. Defaults need at least 768 total half-hours. |
| Weekly predictions are blank | Ensure at least 336 training half-hours and observed weekly donor values. |
| `FileExistsError` | Select a new output directory. |
| Empty scores or low coverage | Inspect missing actuals and donors; absent predictions do not indicate good accuracy. |

Run the forecasting regression checks:

```powershell
.\venv\python.exe -m unittest discover -s tests -p test_forecasting.py -v
```

This is a retrospective benchmark using latest revised observations and assuming
zero publication delay. Operational evaluation requires per-origin revision and
availability cutoffs. Seasonal lags use elapsed UTC steps, which can differ from
local-clock slots around daylight saving changes. See the
[forecasting contract](docs/forecasting.md) for assumptions and next steps.
