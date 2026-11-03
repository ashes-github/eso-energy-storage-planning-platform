"""Bounded-memory, matched-case long-history metrics and GSP benefit analysis."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import common_cases
from .pipeline import sha256


def sufficient_stats(frame, groups):
    valid = frame.actual.notna() & frame.prediction.notna()
    error = (frame.prediction - frame.actual).where(valid)
    scaled = error.abs() / frame.mase_scale
    work = frame[list(groups)].copy()
    work['n_cases'] = 1
    work['n_observed'] = frame.actual.notna().astype(int)
    work['n_scored'] = valid.astype(int)
    work['n_mase'] = scaled.notna().astype(int)
    work['sum_abs'] = error.abs().fillna(0)
    work['sum_squared'] = (error ** 2).fillna(0)
    work['sum_scaled'] = scaled.fillna(0)
    return work.groupby(list(groups), observed=True, dropna=False, sort=True).sum().reset_index()


def finish_stats(parts, groups):
    totals = pd.concat(parts, ignore_index=True).groupby(list(groups), observed=True,
                                                        dropna=False, sort=True).sum().reset_index()
    scored = totals.n_scored.replace(0, np.nan)
    totals['coverage'] = totals.n_scored / totals.n_observed.replace(0, np.nan)
    totals['mae'] = totals.sum_abs / scored
    totals['rmse'] = np.sqrt(totals.sum_squared / scored)
    totals['mase'] = totals.sum_scaled / totals.n_mase.replace(0, np.nan)
    return totals.drop(columns=['sum_abs', 'sum_squared', 'sum_scaled'])


def origin_frames(path, chunksize=100_000):
    """Carry partial origins across CSV chunks so case intersections stay valid."""
    carry = None
    for chunk in pd.read_csv(path, dtype={'gsp': str}, parse_dates=['origin', 'timestamp'], chunksize=chunksize):
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        last = chunk.origin.iloc[-1]
        carry = chunk.loc[chunk.origin == last].copy()
        complete = chunk.loc[chunk.origin != last]
        for _, frame in complete.groupby('origin', sort=False):
            yield frame.reset_index(drop=True)
    if carry is not None:
        yield carry.reset_index(drop=True)


def benefit_table(metrics, dimensions=()):
    keys = ['split', 'horizon', 'gsp', *dimensions]
    selected = metrics.set_index([*keys, 'model'])[['mae', 'n_scored']].unstack('model')
    result = selected.reset_index()
    out = result[keys].copy()
    out.columns = keys
    out['daily_mae'] = result[('mae', 'seasonal_48')]
    out['lightgbm_mae'] = result[('mae', 'lightgbm')]
    if not result[('n_scored', 'seasonal_48')].equals(result[('n_scored', 'lightgbm')]):
        raise AssertionError('Head-to-head errors must use identical scored cases')
    out['n_paired_cases'] = result[('n_scored', 'lightgbm')]
    out['delta_mae'] = out.daily_mae - out.lightgbm_mae
    out['relative_improvement_pct'] = 100 * out.delta_mae / out.daily_mae.replace(0, np.nan)
    out['zero_baseline_mae'] = out.daily_mae.eq(0)
    out['comparison'] = np.select([out.delta_mae.gt(1e-12), out.delta_mae.lt(-1e-12),
                                    out.delta_mae.notna()], ['win', 'loss', 'tie'], default='unscored')
    return out


def benefit_summary(table, groups):
    rows = []
    for key, group in table.groupby(list(groups), observed=True, dropna=False):
        valid = group.comparison != 'unscored'
        compared = group.loc[valid]
        row = dict(zip(groups, key if isinstance(key, tuple) else (key,)))
        row.update(n_gsps=len(group), n_compared=int(valid.sum()),
                   win_fraction=float(compared.comparison.eq('win').mean()),
                   loss_fraction=float(compared.comparison.eq('loss').mean()),
                   tie_fraction=float(compared.comparison.eq('tie').mean()),
                   median_relative_improvement_pct=compared.relative_improvement_pct.median(),
                   q25_relative_improvement_pct=compared.relative_improvement_pct.quantile(.25),
                   q75_relative_improvement_pct=compared.relative_improvement_pct.quantile(.75),
                   n_relative_defined=int(compared.relative_improvement_pct.notna().sum()),
                   n_zero_baseline=int(compared.zero_baseline_mae.sum()),
                   macro_daily_mae=compared.daily_mae.mean(), macro_lightgbm_mae=compared.lightgbm_mae.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def analyse_run(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'run.json').read_text())
    if json.loads((directory / 'run_status.json').read_text())['status'] != 'complete' or metadata['split'] != 'validation':
        raise ValueError('Analyse completed validation runs only')
    analysis_hash = sha256(__file__)
    prediction_hash = sha256(directory / 'predictions.csv')
    status_path = directory / 'analysis_status.json'
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if (previous.get('analysis_source_sha256') == analysis_hash and
                previous.get('predictions_sha256') == prediction_hash):
            print(f'Analysis unchanged: {directory.name}', flush=True)
            return
    selection = pd.read_csv(directory / 'series_selection.csv', dtype={'gsp': str}).set_index('gsp')
    base = ('model', 'split', 'horizon')
    definitions = {
        'metrics_pooled': (base, 'raw'),
        'metrics_per_gsp': ((*base, 'gsp'), 'raw'),
        'metrics_by_lead': ((*base, 'lead'), 'raw'),
        'metrics_common_cases': (base, 'common'),
        'metrics_common_cases_by_lead': ((*base, 'lead'), 'common'),
        'metrics_pairwise_daily_per_gsp': ((*base, 'gsp'), 'paired'),
        'metrics_pairwise_volume': ((*base, 'volume_stratum'), 'paired'),
        'metrics_pairwise_daytype': ((*base, 'daytype'), 'paired'),
        'metrics_pairwise_season': ((*base, 'season'), 'paired'),
        'metrics_pairwise_lead': ((*base, 'lead'), 'paired'),
        'metrics_pairwise_per_gsp_daytype': ((*base, 'gsp', 'daytype'), 'paired'),
        'metrics_pairwise_per_gsp_season': ((*base, 'gsp', 'season'), 'paired'),
    }
    pieces = defaultdict(list)
    seen = set()
    for frame in origin_frames(directory / 'predictions.csv'):
        if set(frame.model) != set(metadata['resolved_config']['models']):
            raise ValueError('Incomplete model population at forecast origin')
        seen.update(frame.origin.unique())
        if frame.timestamp.max() >= pd.Timestamp(metadata['plan']['test_start']):
            raise ValueError('Analysis must not include reserved test targets')
        local = frame.timestamp.dt.tz_convert('Europe/London')
        frame['daytype'] = np.where(local.dt.dayofweek >= 5, 'weekend', 'weekday')
        frame['season'] = np.select([local.dt.month.isin([12, 1, 2]), local.dt.month.isin([3, 4, 5]),
                                      local.dt.month.isin([6, 7, 8])], ['winter', 'spring', 'summer'], default='autumn')
        frame['volume_stratum'] = frame.gsp.map(selection.stratum)
        frames = dict(raw=frame, common=common_cases(frame),
                      paired=common_cases(frame.loc[frame.model.isin(['seasonal_48', 'lightgbm'])]))
        for name, (groups, kind) in definitions.items():
            pieces[name].append(sufficient_stats(frames[kind], groups))
    if len(seen) != metadata['n_origins']:
        raise ValueError('Prediction origins do not match completed run metadata')
    outputs = {}
    for name, (groups, _) in definitions.items():
        outputs[name] = finish_stats(pieces[name], groups)
        outputs[name].to_csv(directory / f'{name}.csv', index=False)
    for dimension in (None, 'daytype', 'season'):
        source = 'metrics_pairwise_daily_per_gsp' if dimension is None else f'metrics_pairwise_per_gsp_{dimension}'
        table = benefit_table(outputs[source], () if dimension is None else (dimension,))
        table['volume_stratum'] = table.gsp.map(selection.stratum)
        suffix = '' if dimension is None else f'_{dimension}'
        table.to_csv(directory / f'gsp_benefits{suffix}.csv', index=False)
        groups = ('split', 'horizon') if dimension is None else ('split', 'horizon', dimension)
        benefit_summary(table, groups).to_csv(directory / f'gsp_benefit_summary{suffix}.csv', index=False)
        if dimension is None:
            benefit_summary(table, ('split', 'horizon', 'volume_stratum')).to_csv(
                directory / 'gsp_benefit_summary_volume.csv', index=False)
    importance = []
    for path in sorted((directory / 'models' / 'lightgbm').glob('origin_*/feature_importance.csv')):
        frame = pd.read_csv(path)
        total = frame.gain.sum()
        frame['gain_fraction'] = frame.gain / total if total > 0 else 0.0
        frame['origin_directory'] = path.parent.name
        importance.append(frame)
    if importance:
        all_importance = pd.concat(importance, ignore_index=True)
        all_importance.to_csv(directory / 'importance_by_origin.csv', index=False)
        all_importance.groupby('feature').agg(mean_gain_fraction=('gain_fraction', 'mean'),
            std_gain_fraction=('gain_fraction', 'std'), mean_splits=('split', 'mean'),
            n_origins=('origin_directory', 'nunique')).sort_values('mean_gain_fraction', ascending=False).to_csv(
                directory / 'importance_aggregated.csv')
    status_path.write_text(json.dumps(dict(status='complete',
        analysis_source_sha256=analysis_hash, predictions_sha256=prediction_hash,
        pairwise_reference='seasonal_48', units='source Meter Volume', final_test_evaluated=False)))
    run_status = json.loads((directory / 'run_status.json').read_text())
    run_status['metrics'] = 'complete'
    (directory / 'run_status.json').write_text(json.dumps(run_status))
    print(f'Analysed {directory.name}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True)
    args = parser.parse_args()
    study = Path(args.study)
    protocol = json.loads((study / 'protocol.json').read_text())
    pooled, benefits = [], []
    for experiment in protocol['experiments']:
        directory = study / experiment['name']
        analyse_run(directory)
        pooled.append(pd.read_csv(directory / 'metrics_common_cases.csv').assign(experiment=experiment['name']))
        benefits.append(pd.read_csv(directory / 'gsp_benefit_summary.csv').assign(experiment=experiment['name']))
    pd.concat(pooled, ignore_index=True).to_csv(study / 'ablation_metrics.csv', index=False)
    pd.concat(benefits, ignore_index=True).to_csv(study / 'ablation_gsp_benefits.csv', index=False)


if __name__ == '__main__':
    main()
