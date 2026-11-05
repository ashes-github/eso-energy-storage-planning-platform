"""Supplementary 2021-2022 diagnostics; does not refit the oracle ablation."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from .data import ForecastDataset
from .features import build_features
from .oracle_weather import OracleTemperatureForecaster
from .pipeline import reproducibility, sha256
from .temperature_eda import calendar, correlation

BOUNDARY = pd.Timestamp('2023-01-01', tz='UTC')


def stable_seed(label, seed=7):
    return (int.from_bytes(hashlib.sha256(str(label).encode()).digest()[:4], 'little') + seed) % (2**32)


def associations(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    result = dict(n_paired=len(x), n_unique_temperature=len(np.unique(x)), n_unique_demand=len(np.unique(y)),
                  pearson=np.nan, spearman=np.nan)
    if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
        result.update(pearson=correlation(x, y), spearman=correlation(rankdata(x), rankdata(y)))
    return result


def mi_diagnostic(x, y, seed, references=3):
    """kNN MI in nats; randomized references are descriptive, not p-values."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    result = dict(n_sample=len(x), mi_nats=np.nan, randomized_reference_mean=np.nan,
                  randomized_reference_std=np.nan, mi_minus_reference=np.nan)
    if len(x) < 20 or np.std(x) == 0 or np.std(y) == 0:
        return result
    def estimate(values):
        return float(mutual_info_regression(values.reshape(-1, 1), y,
            discrete_features=False, n_neighbors=5, random_state=seed)[0])
    value = estimate(x)
    rng = np.random.default_rng(seed)
    controls = np.array([estimate(rng.permutation(x)) for _ in range(references)])
    result.update(mi_nats=value, randomized_reference_mean=float(controls.mean()),
                  randomized_reference_std=float(controls.std(ddof=1)),
                  mi_minus_reference=float(value-controls.mean()))
    return result


def correlation_summary(rows):
    output = []
    for (scope, value), group in rows.groupby(['scope', 'slice'], dropna=False, sort=True):
        for metric in ('pearson', 'spearman'):
            valid = group[metric].dropna()
            output.append(dict(scope=scope, slice=value, metric=metric, n_gsps=len(group),
                n_defined=len(valid), median=valid.median(), q25=valid.quantile(.25), q75=valid.quantile(.75),
                fraction_negative=float(valid.lt(0).mean()), fraction_abs_above_0_3=float(valid.abs().gt(.3).mean())))
    return pd.DataFrame(output)


def association_study(demand, temperature, output, samples_per_gsp=2000, seed=7):
    if demand.columns.max() >= BOUNDARY or temperature.index.max() >= BOUNDARY:
        raise ValueError('Reserved final-test data must be removed before diagnostics')
    if demand.columns.min() < pd.Timestamp('2021-01-01', tz='UTC'):
        raise ValueError('This protocol covers 2021-2022 development only')
    cal = calendar(demand.columns)
    rows, samples, medians = [], [], []
    for i, gsp in enumerate(demand.index):
        if i % 60 == 0:
            print(f'Association diagnostics GSP {i+1}/{len(demand)}', flush=True)
        y = demand.loc[gsp].to_numpy()
        t = temperature.reindex(index=demand.columns, columns=[gsp]).iloc[:, 0].to_numpy()
        frame = cal.copy()
        frame['temperature_c'], frame['demand'] = t, y
        for scope in ('overall', 'season', 'hour', 'daytype', 'year'):
            groups = [('all', frame)] if scope == 'overall' else frame.groupby(scope, sort=True)
            for value, block in groups:
                rows.append(dict(gsp=gsp, scope=scope, slice=str(value),
                    **associations(block.temperature_c, block.demand)))
        paired = frame.loc[np.isfinite(t) & np.isfinite(y)].copy()
        means = paired.groupby(['season', 'hour', 'daytype'])[['temperature_c', 'demand']].transform('mean')
        residual = paired[['temperature_c', 'demand']] - means
        rows.append(dict(gsp=gsp, scope='calendar_adjusted', slice='all',
            **associations(residual.temperature_c, residual.demand)))
        initial = y[demand.columns.year == 2021]
        std = np.nanstd(initial)
        paired['demand_z'] = (paired.demand-np.nanmean(initial))/std if std > 0 else np.nan
        paired['temperature_bin_c'] = np.floor(paired.temperature_c / 2)*2+1
        median = paired.groupby('temperature_bin_c').agg(n=('demand', 'size'),
            median_demand=('demand', 'median'), median_demand_z=('demand_z', 'median')).reset_index()
        median['gsp'] = gsp
        medians.append(median)
        hourly = paired.loc[paired.index.minute == 0]
        rng = np.random.default_rng(stable_seed(gsp, seed))
        chosen = np.sort(rng.choice(len(hourly), size=min(samples_per_gsp, len(hourly)), replace=False))
        sample = hourly.iloc[chosen].copy()
        sample['gsp'] = gsp
        sample.index.name = 'timestamp_utc'
        samples.append(sample.reset_index())
    rows = pd.DataFrame(rows)
    rows.to_csv(output / 'correlations_per_gsp_and_slice.csv', index=False)
    summary = correlation_summary(rows)
    summary.to_csv(output / 'correlation_distribution_summary.csv', index=False)
    pd.concat(medians).to_csv(output / 'binned_gsp_medians.csv', index=False)
    sample = pd.concat(samples, ignore_index=True)
    sample.to_parquet(output / 'hourly_association_sample.parquet', index=False)
    pooled = []
    for scope in ('overall', 'season', 'hour', 'daytype', 'year'):
        groups = [('all', sample)] if scope == 'overall' else sample.groupby(scope, sort=True)
        for value, block in groups:
            for target in ('demand', 'demand_z'):
                pooled.append(dict(scope=scope, slice=str(value), target=target,
                    sample_policy='up to 2000 exact-hour paired observations per GSP; fixed seed',
                    **associations(block.temperature_c, block[target])))
    pd.DataFrame(pooled).to_csv(output / 'pooled_sample_correlations.csv', index=False)
    mi_rows = []
    for gsp in demand.index:
        block = sample.loc[sample.gsp.eq(gsp)]
        for year in ('all', 2021, 2022):
            selected = block if year == 'all' else block.loc[block.year.eq(year)]
            mi_rows.append(dict(gsp=gsp, year=str(year), **mi_diagnostic(
                selected.temperature_c, selected.demand, stable_seed(f'{gsp}:{year}', seed))))
    mi = pd.DataFrame(mi_rows)
    mi.to_csv(output / 'mutual_information_per_gsp.csv', index=False)
    pooled_mi = []
    for target in ('demand', 'demand_z'):
        finite = sample.loc[sample[target].notna()]
        chosen = finite.sample(n=min(30000, len(finite)), random_state=seed)
        pooled_mi.append(dict(target=target, **mi_diagnostic(chosen.temperature_c, chosen[target], seed)))
    pd.DataFrame(pooled_mi).to_csv(output / 'mutual_information_pooled_sample.csv', index=False)
    make_association_figures(rows, mi, output)
    spline_diagnostics(sample, output)
    # Paired year summaries provide descriptive stability, not a further test set.
    stability = rows.loc[rows.scope.eq('year')].pivot(index='gsp', columns='slice', values=['pearson', 'spearman'])
    stability.columns = [f'{metric}_{year}' for metric, year in stability.columns]
    for metric in ('pearson', 'spearman'):
        a, b = stability[f'{metric}_2021'], stability[f'{metric}_2022']
        stability[f'{metric}_both_defined'] = a.notna() & b.notna()
        stability[f'{metric}_same_sign'] = (np.sign(a) == np.sign(b)).where(a.notna() & b.notna())
    stability.to_csv(output / 'year_to_year_stability.csv')
    return rows, mi, sample


def fit_spline(frame):
    finite = frame.loc[np.isfinite(frame.temperature_c) & np.isfinite(frame.demand_z)].copy()
    if len(finite) < 20 or finite.temperature_c.nunique() < 6:
        raise ValueError('Insufficient temperature support for a spline')
    # Equal total weight per GSP, normalized to mean one for fixed Ridge alpha.
    weights = 1 / finite.groupby('gsp').gsp.transform('size')
    weights = weights / weights.mean()
    model = make_pipeline(SplineTransformer(n_knots=6, degree=3, include_bias=False), Ridge(alpha=1.0))
    model.fit(finite[['temperature_c']], finite.demand_z, ridge__sample_weight=weights)
    grid = np.linspace(*finite.temperature_c.quantile([.01, .99]), 150)
    prediction = model.predict(pd.DataFrame({'temperature_c': grid}))
    return pd.DataFrame({'temperature_c': grid, 'fitted_demand_z': prediction,
                         'n_sample': len(finite), 'n_gsps': finite.gsp.nunique()})


def spline_diagnostics(sample, output):
    curves = []
    for scope in ('overall', 'season', 'year'):
        groups = [('all', sample)] if scope == 'overall' else sample.groupby(scope, sort=True)
        for value, frame in groups:
            curve = fit_spline(frame)
            curve['scope'], curve['slice'] = scope, str(value)
            curves.append(curve)
    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(output / 'spline_curves.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, scope in zip(axes, ['overall', 'season', 'year']):
        for value, curve in curves.loc[curves.scope.eq(scope)].groupby('slice'):
            ax.plot(curve.temperature_c, curve.fitted_demand_z, label=value)
        ax.set(title=f'Descriptive spline: {scope}', xlabel='Temperature (C)', ylabel='Demand z-score')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('Fixed cubic spline + Ridge; equal GSP weight; 1st-99th percentile support\nHourly development sample; 2021 demand scaling; no forecast model refit')
    fig.savefig(output / 'smooth_temperature_response.png', dpi=150)
    plt.close(fig)


def make_association_figures(rows, mi, output):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    overall = rows.loc[rows.scope.eq('overall')]
    for metric in ('pearson', 'spearman'):
        axes[0].hist(overall[metric].dropna(), bins=25, alpha=.6, label=metric)
    axes[0].set(title='Per-GSP association', xlabel='Correlation', ylabel='GSP count')
    axes[0].legend(fontsize=8)
    axes[1].hist(mi.loc[mi.year.eq('all'), 'mi_nats'].dropna(), bins=25)
    axes[1].set(title='Per-GSP mutual information', xlabel='Estimated MI (nats)', ylabel='GSP count')
    hour = rows.loc[rows.scope.eq('hour')].copy()
    hour['hour'] = hour['slice'].astype(int)
    for metric in ('pearson', 'spearman'):
        group = hour.groupby('hour')[metric]
        median, q25, q75 = group.median(), group.quantile(.25), group.quantile(.75)
        axes[2].plot(median.index, median, label=metric)
        axes[2].fill_between(median.index, q25, q75, alpha=.15)
    axes[2].set(title='Local-hour association', xlabel='Europe/London hour', ylabel='Median and GSP interquartile range')
    axes[2].legend(fontsize=8)
    fig.suptitle('2021-2022 associations; undefined constant/missing cases retained in audit')
    fig.savefig(output / 'association_diagnostics.png', dpi=150)
    plt.close(fig)


def season_derangement(origins, repeat, seed=7):
    """Same-season whole-origin donor; no fixed points, no cross-GSP shuffles."""
    origins = pd.DatetimeIndex(origins)
    seasons = calendar(origins).season.to_numpy()
    rng = np.random.default_rng(seed + repeat)
    donors = {}
    for season in sorted(set(seasons)):
        members = origins[seasons == season]
        if len(members) < 2:
            raise ValueError('Permutation needs at least two origins per season')
        order = rng.permutation(len(members))
        shifted = np.roll(order, 1)
        for source, donor in zip(order, shifted):
            donors[members[source]] = members[donor]
    return donors


def permutation_study(study, weather, output, repeats=5, seed=7):
    protocol = json.loads((study / 'protocol.json').read_text())
    origins = pd.to_datetime(protocol['origins'], utc=True)
    maps = [season_derangement(origins, repeat, seed) for repeat in range(repeats)]
    donor_records, records, replays = [], [], []
    ids = protocol['selected_series']
    files = sorted((study / 'predictions').glob('*.parquet'))
    if len(files) != len(origins):
        raise ValueError('Expected one saved prediction file for every origin')
    for number, path in enumerate(files):
        saved = pd.read_parquet(path)
        origin = saved.origin.iloc[0]
        if origin != origins[number] or saved.timestamp.max() >= BOUNDARY or set(saved.gsp) != set(ids):
            raise ValueError('Permutation cases differ from the frozen development protocol')
        print(f'Permutation replay {number+1}/{len(origins)}: {origin}', flush=True)
        model_dir = study / 'models' / 'M2_temperature_oracle' / path.stem
        model = OracleTemperatureForecaster.load(model_dir)
        longest = max(protocol['plan']['horizons'])
        future = pd.date_range(origin, periods=longest, freq='30min')
        features = build_features(model.history, future, spec=model.feature_spec)
        matrix = model._matrix(features)
        reference = saved.loc[saved.horizon.eq(longest)]
        expected = reference.pivot(index='gsp', columns='timestamp', values='prediction_oracle').reindex(index=model.history.index, columns=future)
        actual = reference.pivot(index='gsp', columns='timestamp', values='actual_m2').reindex(index=model.history.index, columns=future).to_numpy()
        original = model.booster_.predict(matrix, num_threads=model.options['n_jobs']).reshape(len(ids), longest)
        np.testing.assert_allclose(original, expected.to_numpy(), rtol=1e-11, atol=1e-10)
        original_nan = matrix.oracle_temperature_c.isna().to_numpy()
        for repeat, mapping in enumerate(maps):
            donor = mapping[origin]
            donor_times = pd.date_range(donor, periods=longest, freq='30min')
            replacement = weather.reindex(index=donor_times, columns=model.history.index).to_numpy().T.reshape(-1).astype('float32')
            if not np.array_equal(np.isnan(replacement), original_nan):
                raise ValueError('Permutation must preserve original missingness')
            perturbed = matrix.copy()
            perturbed['oracle_temperature_c'] = replacement
            predicted = model.booster_.predict(perturbed, num_threads=model.options['n_jobs']).reshape(len(ids), longest)
            if not np.isfinite(predicted).all():
                raise ValueError('Nonfinite permuted predictions')
            no_weather = original_nan.reshape(len(ids), longest).all(axis=1)
            np.testing.assert_allclose(predicted[no_weather], original[no_weather], rtol=0, atol=0)
            donor_records.append(dict(origin=origin, donor_origin=donor, repeat=repeat,
                season=calendar(pd.DatetimeIndex([origin])).season.iloc[0]))
            for horizon in protocol['plan']['horizons']:
                y, base, altered = actual[:, :horizon], original[:, :horizon], predicted[:, :horizon]
                valid = np.isfinite(y) & np.isfinite(base) & np.isfinite(altered)
                base_error, changed_error = np.where(valid, base-y, 0), np.where(valid, altered-y, 0)
                records.append(pd.DataFrame(dict(gsp=model.history.index, origin=origin, repeat=repeat, horizon=horizon,
                    n=valid.sum(axis=1), original_abs=np.abs(base_error).sum(axis=1),
                    permuted_abs=np.abs(changed_error).sum(axis=1), original_squared=np.square(base_error).sum(axis=1),
                    permuted_squared=np.square(changed_error).sum(axis=1))))
        replays.append(dict(origin=origin, saved_prediction_replay=True, missingness_preserved=True,
                            no_weather_gsps_unchanged=True, n_gsps=len(ids)))
    detail = pd.concat(records, ignore_index=True)
    detail.to_parquet(output / 'permutation_per_gsp_origin_repeat.parquet', index=False)
    pd.DataFrame(donor_records).to_csv(output / 'permutation_donor_map.csv', index=False)
    pd.DataFrame(replays).to_csv(output / 'permutation_replay_audit.csv', index=False)
    tables = {}
    for dimension in ('pooled', 'gsp'):
        keys = ['horizon', 'repeat'] + ([] if dimension == 'pooled' else ['gsp'])
        grouped = detail.groupby(keys)[['n', 'original_abs', 'permuted_abs', 'original_squared', 'permuted_squared']].sum()
        grouped['original_mae'] = grouped.original_abs/grouped.n
        grouped['permuted_mae'] = grouped.permuted_abs/grouped.n
        grouped['delta_mae'] = grouped.permuted_mae-grouped.original_mae
        grouped['original_rmse'] = np.sqrt(grouped.original_squared/grouped.n)
        grouped['permuted_rmse'] = np.sqrt(grouped.permuted_squared/grouped.n)
        grouped['delta_rmse'] = grouped.permuted_rmse-grouped.original_rmse
        grouped.reset_index().to_csv(output / f'permutation_{dimension}_by_repeat.csv', index=False)
        keys = ['horizon'] + ([] if dimension == 'pooled' else ['gsp'])
        summary = grouped.reset_index().groupby(keys).agg(
            original_mae=('original_mae', 'mean'), mean_permuted_mae=('permuted_mae', 'mean'),
            mean_delta_mae=('delta_mae', 'mean'), std_delta_mae=('delta_mae', 'std'),
            min_delta_mae=('delta_mae', 'min'), max_delta_mae=('delta_mae', 'max'),
            mean_delta_rmse=('delta_rmse', 'mean'), n_repeats=('repeat', 'size')).reset_index()
        summary.to_csv(output / f'permutation_{dimension}_summary.csv', index=False)
        tables[dimension] = summary
    plot_permutation(tables, output)
    return tables['pooled']


def plot_permutation(tables, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    pooled = tables['pooled']
    axes[0].bar(pooled.horizon.astype(str), pooled.mean_delta_mae, color='#287a9b')
    axes[0].errorbar(pooled.horizon.astype(str), pooled.mean_delta_mae, yerr=pooled.std_delta_mae, fmt='none', color='black', capsize=5)
    axes[0].set(xlabel='Horizon (half-hour steps)', ylabel='Permuted MAE minus original MAE', title='Same-season profile permutation')
    for horizon, group in tables['gsp'].groupby('horizon'):
        axes[1].hist(group.mean_delta_mae, bins=30, alpha=.6, label=f'{horizon//2} hours')
    axes[1].set(xlabel='Per-GSP mean MAE increase', ylabel='GSP count', title='Model reliance varies by GSP')
    axes[1].legend()
    fig.suptitle('Five fixed profile swaps; positive values indicate model reliance\nError bars show repeat SD, not a confidence interval; no refitting')
    fig.savefig(output / 'permutation_importance.png', dpi=150)
    plt.close(fig)


def run(study, demand_path, weather_path, output):
    study, demand_path, weather_path, output = map(Path, (study, demand_path, weather_path, output))
    if output.exists():
        raise FileExistsError(f'Use a fresh supplemental output directory: {output}')
    status = json.loads((study / 'study_status.json').read_text())
    protocol = json.loads((study / 'protocol.json').read_text())
    if status['status'] != 'complete' or status['final_test_evaluated']:
        raise ValueError('Require the completed validation-only oracle study')
    current = reproducibility(demand_path)
    if current['input_sha256'] != protocol['reproducibility']['input_sha256'] or sha256(weather_path) != protocol['weather_sha256']:
        raise ValueError('Demand/weather inputs changed since the oracle ablation')
    if current['packages'] != protocol['reproducibility']['packages']:
        raise ValueError('Package versions differ from the saved models')
    # New supplementary module does not alter the archived fitting pipeline.
    for name in ('ml.py', 'features.py', 'oracle_weather.py'):
        matches = [(key, value) for key, value in protocol['reproducibility']['source_sha256'].items() if Path(key).name == name]
        for key, value in matches:
            if sha256(key) != value:
                raise ValueError(f'Saved model feature implementation changed: {name}')
    original_hashes = {str(path.relative_to(study)): sha256(path)
        for path in [study/'protocol.json', study/'metrics_pooled.csv', *sorted((study/'predictions').glob('*.parquet'))]}
    full = ForecastDataset.from_phase1(demand_path)
    demand = full.panel.loc[protocol['selected_series'], full.panel.columns < BOUNDARY].copy()
    del full
    weather = pd.read_parquet(weather_path)
    weather = weather.loc[weather.index < BOUNDARY]
    output.mkdir(parents=True)
    settings = dict(study='Supplementary temperature suitability diagnostics', seed=7,
        period='2021-2022 development only', correlation_sample='full half-hour data per GSP/slice',
        hourly_sample_per_gsp=2000, pooled_mi_sample_cap=30000,
        mi='kNN continuous estimator, 5 neighbours, nats; 3 randomized references; no p-value',
        spline='6-knot cubic B-spline + Ridge alpha=1; equal GSP weights; no parameter selection',
        permutation='5 same-season derangements of whole 48-step profiles; same donor origin for all GSPs; all other features fixed',
        demand_scaling='GSP mean/std from 2021 only', old_study=str(study),
        source_sha256=sha256(__file__), reproducibility=current, original_artifact_hashes=original_hashes,
        final_test_evaluated=False, models_refitted=False)
    (output/'protocol.json').write_text(json.dumps(settings, indent=2))
    started = time.monotonic()
    try:
        (output/'status.json').write_text(json.dumps(dict(status='associations_running')))
        rows, mi, sample = association_study(demand, weather, output)
        (output/'status.json').write_text(json.dumps(dict(status='permutation_running')))
        permutation = permutation_study(study, weather, output)
        for relative, digest in original_hashes.items():
            if sha256(study/relative) != digest:
                raise AssertionError('Original ablation artifacts were modified')
        write_report(output, rows, mi, permutation, len(demand))
        (output/'status.json').write_text(json.dumps(dict(status='complete', n_gsps=len(demand),
            n_origins=len(protocol['origins']), hourly_sample_rows=len(sample),
            original_artifacts_unchanged=True, models_refitted=False, final_test_evaluated=False,
            elapsed_seconds=round(time.monotonic()-started, 1)), indent=2))
    except Exception as exc:
        (output/'status.json').write_text(json.dumps(dict(status='failed', error=str(exc)), indent=2))
        raise


def write_report(output, rows, mi, permutation, n_gsps):
    overall = rows.loc[rows.scope.eq('overall')]
    mi_all = mi.loc[mi.year.eq('all')]
    text = ['# Supplementary temperature suitability diagnostics', '',
        f'2021-2022 development only; all {n_gsps} frozen GSPs retained. '
        'The completed M2-versus-temperature ablation was not refitted or modified. 2023 remains reserved.', '',
        '| Per-GSP association | Median | 25th percentile | 75th percentile | Negative | Absolute value > 0.3 |',
        '|---|---:|---:|---:|---:|---:|']
    for metric in ('pearson', 'spearman'):
        values = overall[metric].dropna()
        text.append(f'| {metric} ({len(values)} defined GSPs) | {values.median():.3f} | {values.quantile(.25):.3f} | '
            f'{values.quantile(.75):.3f} | {values.lt(0).mean():.1%} | {values.abs().gt(.3).mean():.1%} |')
    text.extend(['', 'Undefined constant/missing series remain in the audits and forecasting population. '
        'Stratified Pearson/Spearman tables cover GSP, season, all 24 local hours, weekday/weekend and development year; '
        'calendar-adjusted correlations are also exported. Pooled correlations are explicitly sample-based, '
        'with raw and 2021-standardized demand shown separately.', '',
        f'Median per-GSP estimated mutual information is {mi_all.mi_nats.median():.4f} nats '
        f'(median randomized reference {mi_all.randomized_reference_mean.median():.4f}). '
        'MI uses a fixed sample of up to 2000 exact-hour observations per GSP, five nearest neighbours and a fixed seed. '
        'All/2021/2022 estimates and sample sizes are saved. Randomized references are estimator diagnostics, '
        'not significance tests; neither temporal autocorrelation nor season/calendar confounding is removed by MI.', '',
        '![Association distributions](association_diagnostics.png)', '',
        '![Smooth temperature response](smooth_temperature_response.png)', '',
        'The cubic spline is a descriptive temperature-only response with fixed regularization, not a new forecasting model. '
        'Curves use equal GSP weights and stop at the 1st/99th percentile of observed temperature support. '
        'Season/year curves are separately fitted descriptive slices. Differences in GSP mix, hour and season can confound them. '
        'Per-GSP two-degree-bin medians are also exported. No curve was used to tune the forecast experiment.', '',
        '## Validation permutation importance', '',
        '| Horizon | Original oracle MAE | Mean permuted MAE | Mean MAE increase | Repeat SD |',
        '|---|---:|---:|---:|---:|'])
    for r in permutation.itertuples():
        text.append(f'| {r.horizon//2} hours | {r.original_mae:.4f} | {r.mean_permuted_mae:.4f} | {r.mean_delta_mae:.4f} | {r.std_delta_mae:.4f} |')
    text.extend(['', '![Permutation importance](permutation_importance.png)', '',
        'Positive importance means the fitted oracle model suffers when the temperature profile is replaced. '
        'Five fixed, same-season, whole-origin derangements preserve UTC diurnal order, GSP identity, '
        'cross-GSP weather coherence and missingness. Demand, calendar, lag features and model weights remain fixed. '
        'Every saved original prediction is replayed before permutation. Repeats measure sensitivity to donor choice; '
        'their SD is not a confidence interval. This is model reliance under a coarse season-restricted intervention, '
        'not formal conditional mutual information, causality, or the measured benefit of adding temperature.', '',
        '## Limits and retained scope', '',
        'Within-GSP Pearson/Spearman are unchanged by positive affine demand standardization; standardization is useful '
        'for pooled comparisons and response curves. Annual sign agreement measures only development-period stability. '
        'All correlations/MI/splines are descriptive, without independent-observation p-values. '
        'The original walk-forward ablation remains the decisive incremental prediction result. '
        'HDD/CDD, lagged/rolling temperature, additional weather attributes and operational archived forecasts remain deferred.', '',
        '## Methods and artifacts', '',
        '- [scikit-learn MI estimator](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html)',
        '- [Spline basis](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html)',
        '- [Permutation importance interpretation](https://scikit-learn.org/stable/modules/permutation_importance.html)',
        '- `protocol.json`, `status.json`: sampling, settings, hashes and unchanged-ablation checks.',
        '- `correlations_per_gsp_and_slice.csv`, `correlation_distribution_summary.csv`.',
        '- `year_to_year_stability.csv`, `pooled_sample_correlations.csv`.',
        '- `hourly_association_sample.parquet`, `mutual_information_*.csv`.',
        '- `spline_curves.csv`, `binned_gsp_medians.csv`.',
        '- `permutation_*`: donor maps, saved-model replay audits, per-origin/GSP/repeat errors and summaries.', ''])
    (output/'report.md').write_text('\n'.join(text), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='output/phase2d_temperature_oracle_2022_validation')
    parser.add_argument('--demand', default='output/phase2c1_observed_2021_2023')
    parser.add_argument('--weather', default='Data/gsp_temperature/temperature_halfhourly_c.parquet')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.study, args.demand, args.weather, args.output)


if __name__ == '__main__':
    main()
