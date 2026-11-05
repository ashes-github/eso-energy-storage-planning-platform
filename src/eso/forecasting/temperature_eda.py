"""Development-only temperature feasibility summaries and static figures."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calendar(index):
    local = index.tz_convert('Europe/London')
    return pd.DataFrame({
        'year': local.year,
        'season': np.select([local.month.isin([12, 1, 2]), local.month.isin([3, 4, 5]),
                             local.month.isin([6, 7, 8])], ['winter', 'spring', 'summer'], default='autumn'),
        'hour': local.hour,
        'hour_band': np.select([local.hour < 6, local.hour < 12, local.hour < 18],
                               ['00-05', '06-11', '12-17'], default='18-23'),
        'daytype': np.where(local.dayofweek >= 5, 'weekend', 'weekday'),
    }, index=index)


def correlation(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return np.nan
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def run_eda(demand, temperature, selection, coverage, output):
    output = Path(output)
    output.mkdir(exist_ok=False)
    if demand.columns.max() >= pd.Timestamp('2023-01-01', tz='UTC'):
        raise ValueError('EDA must not access reserved 2023 targets')
    temp = temperature.reindex(index=demand.columns, columns=demand.index)
    cal = calendar(demand.columns)
    correlations, bins, hourly_bins = [], [], []
    for gsp in demand.index:
        y = demand.loc[gsp].to_numpy()
        t = temp[gsp].to_numpy()
        initial = y[demand.columns.year == 2021]
        mean, std = np.nanmean(initial), np.nanstd(initial)
        z = (y - mean) / std if std > 0 else np.full(len(y), np.nan)
        frame = cal.copy()
        frame['temperature_c'], frame['demand'], frame['demand_z'] = t, y, z
        paired = frame.loc[np.isfinite(y) & np.isfinite(t)].copy()
        # Descriptive residualization within local season/hour/daytype, using
        # development data only. This is an EDA adjustment, not a fitted feature.
        cells = paired.groupby(['season', 'hour', 'daytype'])[['demand', 'temperature_c']].transform('mean')
        residual = paired[['demand', 'temperature_c']] - cells
        correlations.append(dict(gsp=gsp, n_demand=int(np.isfinite(y).sum()),
            n_weather=int(np.isfinite(t).sum()), n_paired=len(paired),
            weather_fraction=float(np.isfinite(t).mean()), initial_mean=mean, initial_std=std,
            pearson=correlation(t, y),
            calendar_adjusted_pearson=correlation(residual.temperature_c.to_numpy(), residual.demand.to_numpy()),
            volume_stratum=selection.loc[gsp, 'stratum'],
            weather_spatial_coverage=coverage.grid_area_coverage_fraction.get(gsp, np.nan)))
        if paired.empty:
            continue
        # Fixed two-Celsius-degree bins declared before looking at results.
        paired['temperature_bin_c'] = np.floor(paired.temperature_c / 2) * 2 + 1
        group = paired.groupby(['year', 'season', 'hour_band', 'daytype', 'temperature_bin_c'], observed=True)
        result = group.agg(n=('demand', 'size'), demand_sum=('demand', 'sum'),
                           z_sum=('demand_z', 'sum'), n_z=('demand_z', 'count'),
                           temperature_sum=('temperature_c', 'sum')).reset_index()
        result['gsp'], result['volume_stratum'] = gsp, selection.loc[gsp, 'stratum']
        bins.append(result)
        by_hour = paired.groupby(['hour', 'temperature_bin_c'], observed=True).agg(
            n=('demand', 'size'), demand_sum=('demand', 'sum'),
            z_sum=('demand_z', 'sum'), n_z=('demand_z', 'count')).reset_index()
        by_hour['gsp'] = gsp
        hourly_bins.append(by_hour)
    audit = pd.DataFrame(correlations)
    audit.to_csv(output / 'gsp_feasibility.csv', index=False)
    detail = pd.concat(bins, ignore_index=True)
    detail.to_parquet(output / 'binned_gsp_development.parquet', index=False)
    pd.concat(hourly_bins, ignore_index=True).to_parquet(output / 'binned_gsp_hour_of_day.parquet', index=False)
    tables = {}
    for dimension in (None, 'year', 'season', 'hour_band', 'daytype', 'volume_stratum', 'gsp'):
        keys = ['temperature_bin_c'] if dimension is None else [dimension, 'temperature_bin_c']
        totals = detail.groupby(keys, observed=True)[['n', 'demand_sum', 'z_sum', 'n_z', 'temperature_sum']].sum()
        totals['mean_demand'] = totals.demand_sum / totals.n
        totals['mean_demand_z'] = totals.z_sum / totals.n_z.replace(0, np.nan)
        # Give each GSP equal weight within the slice/bin, separately from the
        # case-weighted curve; both are descriptive and can be composition-biased.
        per_gsp_keys = list(dict.fromkeys([*keys, 'gsp']))
        per_gsp = detail.groupby(per_gsp_keys, observed=True)[['z_sum', 'n_z']].sum()
        per_gsp['mean_z'] = per_gsp.z_sum / per_gsp.n_z.replace(0, np.nan)
        macro = per_gsp.reset_index().groupby(keys, observed=True).mean_z.agg(['mean', 'count'])
        totals['equal_gsp_mean_demand_z'], totals['n_gsps_with_z'] = macro['mean'], macro['count']
        tables[dimension] = totals.reset_index()
        tables[dimension].to_csv(output / f'binned_{dimension or "overall"}.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, dimension in zip(axes.flat, ['season', 'hour_band', 'daytype', 'volume_stratum']):
        for name, frame in tables[dimension].groupby(dimension):
            frame = frame.loc[(frame.n >= 200) & (frame.n_gsps_with_z >= 10)]
            ax.plot(frame.temperature_bin_c, frame.equal_gsp_mean_demand_z, marker='.', label=name)
        ax.set(title=f'By {dimension.replace("_", " ")}', xlabel='Temperature bin centre (C)',
               ylabel='Mean demand z-score (equal GSP weight)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('2021-2022 development: binned temperature-demand association\n2021 GSP-specific mean/std; descriptive, not causal')
    fig.savefig(output / 'temperature_demand_slices.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    overall = tables[None]
    axes[0].bar(overall.temperature_bin_c, overall.n, width=1.8)
    axes[0].set(xlabel='Temperature bin centre (C)', ylabel='Paired half-hour cases', title='Temperature support')
    axes[1].hist(audit.pearson.dropna(), bins=25, alpha=.7, label='Raw')
    axes[1].hist(audit.calendar_adjusted_pearson.dropna(), bins=25, alpha=.6, label='Calendar adjusted')
    axes[1].set(xlabel='Within-GSP Pearson correlation', ylabel='GSP count', title='Linear association')
    axes[1].legend(fontsize=8)
    axes[2].hist(audit.weather_spatial_coverage.dropna(), bins=20)
    axes[2].set(xlabel='Valid grid / polygon area', ylabel='GSP count', title='Spatial representativeness')
    fig.savefig(output / 'feasibility_diagnostics.png', dpi=150)
    plt.close(fig)
    return audit
