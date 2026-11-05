"""Frozen Phase 2D: development EDA and M2 + one oracle-temperature ablation."""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtesting import WalkForward, backtest
from .data import ForecastDataset
from .evaluation import common_cases
from .ml import GlobalTreeForecaster
from .oracle_weather import OracleTemperatureForecaster
from .pipeline import reproducibility, sha256
from .study import OriginPlan
from .study_analysis import origin_frames, sufficient_stats, finish_stats
from .temperature_eda import calendar, run_eda

KEYS = ['split', 'origin', 'timestamp', 'horizon', 'lead', 'gsp']
ORACLE = 'M2_temperature_oracle'


def match_cases(reference, candidate):
    """Fail closed on population, target, origin or training-scale changes."""
    if reference.duplicated(KEYS).any() or candidate.duplicated(KEYS).any():
        raise ValueError('Duplicate forecast cases')
    paired = reference.merge(candidate, on=KEYS, suffixes=('_m2', '_oracle'),
                             how='outer', indicator=True, validate='one_to_one')
    if not paired['_merge'].eq('both').all():
        raise ValueError('M2 and oracle forecasts must have exactly identical cases')
    for column in ('actual', 'mase_scale'):
        np.testing.assert_allclose(paired[column + '_m2'], paired[column + '_oracle'],
                                   rtol=1e-12, atol=1e-12, equal_nan=True)
    return paired.drop(columns='_merge')


def add_slices(frame, selection, coverage):
    result = frame.copy()
    cal = calendar(pd.DatetimeIndex(result.timestamp))
    for key in ('season', 'daytype', 'hour_band'):
        result[key] = cal[key].to_numpy()
    result['volume_stratum'] = result.gsp.map(selection.stratum)
    fraction = result.gsp.map(coverage.grid_area_coverage_fraction)
    result['weather_spatial_coverage'] = fraction
    result['weather_coverage_band'] = np.select(
        [fraction.isna(), fraction.eq(0), fraction.lt(.5), fraction.lt(.9)],
        ['no_boundary', 'no_land_overlap', 'below_50pct', '50_to_90pct'], default='at_least_90pct')
    return result


def report_metrics(parts, output, selection, coverage):
    tables = {}
    for dimension, records in parts.items():
        keys = ['model', 'split', 'horizon'] + ([] if dimension == 'pooled' else [dimension])
        tables[dimension] = finish_stats(records, keys)
        tables[dimension].to_csv(output / f'metrics_{dimension}.csv', index=False)
    benefits = []
    for dimension in tables:
        source = tables[dimension]
        keys = ['split', 'horizon'] + ([] if dimension == 'pooled' else [dimension])
        base = source.loc[source.model.eq('M2')].drop(columns='model')
        oracle = source.loc[source.model.eq(ORACLE)].drop(columns='model')
        joined = base.merge(oracle, on=keys, suffixes=('_m2', '_oracle'), validate='one_to_one')
        if not joined.n_scored_m2.equals(joined.n_scored_oracle):
            raise ValueError('Comparison must have identical scored cases')
        joined['delta_mae'] = joined.mae_m2 - joined.mae_oracle
        joined['relative_improvement_pct'] = 100 * joined.delta_mae / joined.mae_m2.replace(0, np.nan)
        joined.to_csv(output / f'benefits_{dimension}.csv', index=False)
        if dimension == 'gsp':
            joined = add_gsp_metadata(joined, selection, coverage)
            joined.to_csv(output / 'gsp_benefits.csv', index=False)
            for horizon, group in joined.groupby('horizon'):
                valid = group.n_scored_m2.gt(0)
                g = group.loc[valid]
                benefits.append(dict(horizon=int(horizon), n_gsps=len(group), n_compared=len(g),
                    wins=int(g.delta_mae.gt(1e-12).sum()), losses=int(g.delta_mae.lt(-1e-12).sum()),
                    ties=int(g.delta_mae.abs().le(1e-12).sum()),
                    macro_mae_m2=float(g.mae_m2.mean()), macro_mae_oracle=float(g.mae_oracle.mean()),
                    median_relative_improvement_pct=float(g.relative_improvement_pct.median()),
                    q25_relative_improvement_pct=float(g.relative_improvement_pct.quantile(.25)),
                    q75_relative_improvement_pct=float(g.relative_improvement_pct.quantile(.75))))
            plot_benefits(joined, output)
    pd.DataFrame(benefits).to_csv(output / 'gsp_benefit_summary.csv', index=False)
    return tables['pooled'], benefits


def add_gsp_metadata(frame, selection, coverage):
    frame = frame.copy()
    frame['volume_stratum'] = frame.gsp.map(selection.stratum)
    frame['weather_spatial_coverage'] = frame.gsp.map(coverage.grid_area_coverage_fraction)
    return frame


def plot_benefits(benefits, output):
    horizons = sorted(benefits.horizon.unique())
    fig, axes = plt.subplots(2, len(horizons), figsize=(11, 7), constrained_layout=True, squeeze=False)
    for column, horizon in enumerate(horizons):
        frame = benefits.loc[benefits.horizon.eq(horizon)]
        axes[0, column].hist(frame.relative_improvement_pct.dropna(), bins=35, color='#287a9b')
        axes[0, column].axvline(0, color='black', linewidth=1)
        axes[0, column].set(title=f'{horizon // 2}-hour horizon', xlabel='Per-GSP MAE improvement (%)', ylabel='GSP count')
        axes[1, column].scatter(frame.weather_spatial_coverage, frame.relative_improvement_pct, s=10, alpha=.55)
        axes[1, column].axhline(0, color='black', linewidth=1)
        axes[1, column].set(xlabel='Valid land grid / polygon area', ylabel='MAE improvement (%)')
    fig.suptitle('M2 versus M2 + oracle temperature: 2022 validation\nPositive improvement favours temperature; identical cases')
    fig.savefig(output / 'gsp_temperature_benefits.png', dpi=150)
    plt.close(fig)


def run(baseline, demand_path, weather_path, coverage_path, output):
    baseline, demand_path, weather_path, coverage_path, output = map(Path, (baseline, demand_path, weather_path, coverage_path, output))
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    metadata = json.loads((baseline / 'run.json').read_text())
    if metadata['split'] != 'validation' or json.loads((baseline / 'run_status.json').read_text())['status'] != 'complete':
        raise ValueError('Reference must be a completed validation run')
    current = reproducibility(demand_path)
    saved = metadata['reproducibility']
    if current['input_sha256'] != saved['input_sha256']:
        raise ValueError('Demand input hash differs from the frozen baseline')
    if current['packages'] != saved['packages']:
        raise ValueError('Package versions differ from the frozen baseline; rerun M2 first')
    changed_sources = [name for name, digest in saved['source_sha256'].items()
                       if name in current['source_sha256'] and current['source_sha256'][name] != digest]
    fitting_modules = {'ml.py', 'features.py', 'data.py', 'backtesting.py', 'evaluation.py', 'config.py'}
    for name in changed_sources:
        if Path(name).name in fitting_modules:
            raise ValueError(f'Baseline fitting implementation changed: {name}; rerun M2 first')
    config = metadata['resolved_config']
    spec = config['lightgbm']['feature_spec']
    if spec['rolling_means'] or spec['rolling_stds'] or not spec['target_lags']:
        raise ValueError('Reference must be frozen M2 seasonal features, not M3')
    full = ForecastDataset.from_phase1(demand_path)
    ids = metadata['selected_series']
    plan = WalkForward(**metadata['plan'])
    positions = list(plan.folds(full.panel.columns, 'validation'))
    if len(positions) != metadata['n_origins']:
        raise ValueError('Reference origin count differs from the reconstructed schedule')
    boundary = pd.Timestamp(plan.test_start)
    data = ForecastDataset(full.panel.loc[ids, full.panel.columns < boundary])
    del full
    # Even the weather lookup table excludes 2023 during this experiment.
    weather = pd.read_parquet(weather_path)
    weather = weather.loc[weather.index < boundary]
    selection = pd.read_csv(baseline / 'series_selection.csv', dtype={'gsp': str}).set_index('gsp')
    coverage = pd.read_csv(coverage_path, dtype={'gsp_id': str}).set_index('gsp_id')
    output.mkdir(parents=True)
    protocol = dict(study='Phase 2D controlled oracle-temperature feasibility',
        baseline_directory=str(baseline), demand_directory=str(demand_path),
        baseline_predictions_sha256=sha256(baseline / 'predictions.csv'),
        baseline_run_sha256=sha256(baseline / 'run.json'), weather_sha256=sha256(weather_path),
        coverage_sha256=sha256(coverage_path), reproducibility=current,
        changed_nonfitting_sources=changed_sources,
        selected_series=ids, n_gsps=len(ids), n_origins=len(positions),
        origins=[data.panel.columns[p].isoformat() for p in positions],
        plan=metadata['plan'], lightgbm=config['lightgbm'],
        only_added_feature='oracle_temperature_c at each target timestamp',
        fitting='same unweighted L2 LightGBM, rows, categories, schedules, targets, seed and settings as M2',
        weather_policy='realized reanalysis, deliberately perfect future weather; native NaN; no imputation or weather-based exclusions',
        comparison='cached M2 and naive forecasts verified against same inputs; M2 saved-model replay at every origin',
        eda_period='2021-2022 only; fixed 2C bins; GSP scaling from 2021 only; descriptive associations',
        final_test_evaluated=False)
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    selection.to_csv(output / 'series_selection.csv')
    started = time.monotonic()
    print('Running development-only temperature EDA', flush=True)
    run_eda(data.panel, weather, selection, coverage, output / 'eda')
    predictions_dir = output / 'predictions'
    predictions_dir.mkdir()
    parts = defaultdict(list)
    audits, importance = [], []
    baseline_origins = iter(origin_frames(baseline / 'predictions.csv'))
    try:
        for i, position in enumerate(positions):
            origin = data.panel.columns[position]
            print(f'Oracle temperature origin {i+1}/{len(positions)}: {origin}', flush=True)
            (output / 'study_status.json').write_text(json.dumps(dict(status='running', completed_origins=i,
                total_origins=len(positions), origin=origin.isoformat(), elapsed_seconds=round(time.monotonic()-started, 1))))
            cached = next(baseline_origins)
            if not cached.origin.eq(origin).all():
                raise ValueError('Baseline origin order does not match frozen plan')
            reference = cached.loc[cached.model.eq('lightgbm')].copy()
            reference['model'] = 'M2'
            saved_dir = baseline / 'models' / 'lightgbm' / f'origin_{position:08d}'
            baseline_model = GlobalTreeForecaster.load(saved_dir)
            future = data.panel.columns[position:position + max(plan.horizons)]
            replay = baseline_model.predict(future)
            expected = reference.loc[reference.horizon.eq(max(plan.horizons))].pivot(index='gsp', columns='timestamp', values='prediction')
            np.testing.assert_allclose(replay.to_numpy(), expected.reindex(index=replay.index, columns=replay.columns).to_numpy(),
                                       rtol=1e-11, atol=1e-10, equal_nan=True)
            oracle_model = OracleTemperatureForecaster(oracle_temperature=weather,
                horizon=max(plan.horizons), **config['lightgbm'])
            diagnostics = []
            forecasts = backtest(data, OriginPlan(plan, position), {ORACLE: lambda: oracle_model},
                diagnostics=diagnostics, artifact_dir=output / 'models')
            if forecasts.timestamp.max() >= boundary:
                raise ValueError('Reserved final test was accessed')
            if oracle_model.training_audit_ != baseline_model.training_audit_ or oracle_model.training_counts_ != baseline_model.training_counts_:
                raise ValueError('Training rows, targets or origins differ from M2')
            if oracle_model.feature_names_ != baseline_model.feature_names_ + ['oracle_temperature_c']:
                raise ValueError('Temperature must be the only feature change')
            if oracle_model.options != baseline_model.options:
                raise ValueError('Learner/training settings changed')
            paired = match_cases(reference, forecasts)
            paired = add_slices(paired, selection, coverage)
            paired.to_parquet(predictions_dir / f'origin_{position:08d}.parquet', index=False)
            # Canonicalize only CSV round-trip precision AFTER checking actuals
            # and scales. Matching never drops a GSP or a missing-weather case.
            canonical = paired[KEYS + ['actual_m2', 'mase_scale_m2', 'prediction_oracle']].rename(
                columns={'actual_m2': 'actual', 'mase_scale_m2': 'mase_scale', 'prediction_oracle': 'prediction'})
            canonical['model'] = ORACLE
            cached.loc[cached.model.eq('lightgbm'), 'model'] = 'M2'
            scored = add_slices(common_cases(pd.concat([cached, canonical], ignore_index=True)), selection, coverage)
            for dimension in ('pooled', 'gsp', 'season', 'daytype', 'volume_stratum', 'lead', 'weather_coverage_band'):
                keys = ['model', 'split', 'horizon'] + ([] if dimension == 'pooled' else [dimension])
                parts[dimension].append(sufficient_stats(scored, keys))
            n_weather_gsps = int(weather.reindex(index=future, columns=ids).notna().any().sum())
            audits.append(dict(origin=origin.isoformat(), baseline_replay_verified=True,
                training_audit_identical=True, training_counts_identical=True, only_temperature_added=True,
                n_gsps=len(ids), gsps_with_weather=n_weather_gsps, n_forecast_cases=len(paired),
                **oracle_model.training_audit_))
            pd.DataFrame(audits).to_csv(output / 'origin_audit.csv', index=False)
            pd.DataFrame(diagnostics).to_csv(output / 'fit_diagnostics.csv', mode='a', header=i == 0, index=False)
            imp = pd.DataFrame({'feature': oracle_model.feature_names_,
                'gain': oracle_model.booster_.feature_importance(importance_type='gain'),
                'split': oracle_model.booster_.feature_importance(importance_type='split')})
            imp['gain_fraction'] = imp.gain / imp.gain.sum()
            imp['origin'] = origin
            importance.append(imp)
            del baseline_model, oracle_model, forecasts, paired, scored, cached
            gc.collect()
        if next(baseline_origins, None) is not None:
            raise ValueError('Unexpected extra baseline origins')
        metrics, benefits = report_metrics(parts, output, selection, coverage)
        imp = pd.concat(importance, ignore_index=True)
        imp.to_csv(output / 'importance_by_origin.csv', index=False)
        imp.groupby('feature').agg(mean_gain_fraction=('gain_fraction', 'mean'),
            mean_splits=('split', 'mean')).sort_values('mean_gain_fraction', ascending=False).to_csv(output / 'importance_aggregated.csv')
        write_report(output, protocol, metrics, benefits)
        (output / 'study_status.json').write_text(json.dumps(dict(status='complete', completed_origins=len(positions),
            total_origins=len(positions), elapsed_seconds=round(time.monotonic()-started, 1), final_test_evaluated=False), indent=2))
    except Exception as exc:
        (output / 'study_status.json').write_text(json.dumps(dict(status='failed', error=str(exc),
            completed_origins=len(audits), final_test_evaluated=False), indent=2))
        raise


def write_report(output, protocol, metrics, benefits):
    feasibility = pd.read_csv(output / 'eda/gsp_feasibility.csv')
    pooled = pd.read_csv(output / 'benefits_pooled.csv')
    text = ['# Phase 2D: temperature feasibility and oracle ablation', '',
        f'2022 validation: {protocol["n_gsps"]} fixed GSPs, {protocol["n_origins"]} fixed origins. '
        'January-March 2023 was not evaluated. One realized target-time temperature feature was added to M2. '
        'There was no model tuning, imputation, population change, or spatial-coverage predictor.', '',
        f'{int(feasibility.n_weather.gt(0).sum())} GSPs have weather; '
        f'{int(feasibility.n_weather.eq(0).sum())} retain missing weather. '
        f'Median within-GSP raw temperature/demand correlation: {feasibility.pearson.median():.3f}; '
        f'after descriptive season/hour/daytype demeaning: {feasibility.calendar_adjusted_pearson.median():.3f}.', '',
        '| Horizon | M2 MAE | Oracle MAE | Improvement | M2 RMSE | Oracle RMSE | M2 MASE | Oracle MASE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in pooled.itertuples():
        text.append(f'| {r.horizon//2} hours | {r.mae_m2:.4f} | {r.mae_oracle:.4f} | {r.relative_improvement_pct:.2f}% | '
                    f'{r.rmse_m2:.4f} | {r.rmse_oracle:.4f} | {r.mase_m2:.4f} | {r.mase_oracle:.4f} |')
    text.extend(['', 'Positive improvement means lower oracle MAE. Errors use source Meter Volume units. '
                 'MASE uses each origin\'s training-only daily-naive scale. Constant/zero-scale series have undefined MASE, '
                 'but remain in MAE/RMSE. Scoring uses exactly common observed cases across M2, oracle and all three frozen naive references.', ''])
    for b in benefits:
        text.append(f'- {b["horizon"]//2}-hour: {b["wins"]} GSP wins, {b["losses"]} losses, {b["ties"]} ties; '
                    f'median GSP improvement {b["median_relative_improvement_pct"]:.2f}%; '
                    f'macro MAE {b["macro_mae_m2"]:.4f} to {b["macro_mae_oracle"]:.4f}.')
    text.extend(['', '## Interpretation and limits', '',
        'The EDA bins show associations, not causation or isolated temperature effects. '
        'Season, demand mix, solar output and time of day can confound the curves. '
        'Equal-GSP standardized plots use only 2021 means/stds; constant series remain in the experiment but have no z-score. '
        'Sparse plotted bins require at least 200 paired cases and 10 GSPs; all counts remain in the CSV exports.', '',
        'This is a perfect-weather feasibility experiment. Realized future reanalysis is not an operational forecast. '
        'Any positive result supports subsequent evaluation with archived weather forecasts issued before each demand origin. '
        'The result is descriptive validation evidence, without a claim of statistical significance. '
        'Spatial coverage and shared-region aliases limit representativeness; no automatic exclusions were made. '
        'Feature importance is descriptive and can distribute gain across correlated predictors.', '',
        '## Evidence', '',
        '- [Development curves](eda/temperature_demand_slices.png)',
        '- [Feasibility diagnostics](eda/feasibility_diagnostics.png)',
        '- [Per-GSP benefit and coverage](gsp_temperature_benefits.png)',
        '- `protocol.json`: frozen settings, hashes, population, dates and oracle declaration.',
        '- `origin_audit.csv`: all baseline replays and identical training audits.',
        '- `metrics_*.csv`, `benefits_*.csv`: pooled, GSP, season, daytype, volume, lead and weather coverage.',
        '- `predictions/`: paired M2/oracle cases, including all missing-weather GSPs.',
        '- `models/`: native boosters, prediction weather and reloadable schemas.',
        '- `eda/`: per-GSP correlations, paired counts, two-degree bins, local hour bins and coverage.', ''])
    (output / 'report.md').write_text('\n'.join(text), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='output/phase2c1_long_history_2022_validation/M2_seasonal')
    parser.add_argument('--demand', default='output/phase2c1_observed_2021_2023')
    parser.add_argument('--weather', default='Data/gsp_temperature/temperature_halfhourly_c.parquet')
    parser.add_argument('--coverage', default='Data/gsp_temperature/spatial_coverage.csv')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.baseline, args.demand, args.weather, args.coverage, args.output)


if __name__ == '__main__':
    main()
