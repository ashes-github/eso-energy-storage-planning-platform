"""Render static study figures and audit matched ablations without refitting."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def report(study, output):
    study, output = Path(study), Path(output)
    if json.loads((study / 'study_status.json').read_text())['status'] != 'complete':
        raise ValueError('Study forecasts must be complete')
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((study / 'protocol.json').read_text())
    combined = pd.read_csv(study / 'ablation_metrics.csv')
    full = study / 'M3_full'
    metrics = pd.read_csv(full / 'metrics_common_cases.csv')
    reference = None
    for experiment in protocol['experiments']:
        frame = combined.loc[(combined.experiment == experiment['name']) & (combined.model != 'lightgbm')]
        frame = frame.sort_values(['model', 'horizon']).reset_index(drop=True)
        numeric = frame[['horizon', 'n_cases', 'n_observed', 'n_scored', 'mae', 'rmse', 'mase']]
        if reference is None:
            reference = numeric
        else:
            np.testing.assert_allclose(numeric, reference, rtol=1e-12, atol=1e-12, equal_nan=True)
        if not combined.loc[combined.experiment == experiment['name'], 'coverage'].eq(1).all():
            raise ValueError('These report comparisons require fully shared observed cases')

    colors = {'persistence': '#7b7b7b', 'seasonal_48': '#e68a00',
              'seasonal_336': '#29934a', 'lightgbm': '#2166ac'}
    names = {'persistence': 'Persistence', 'seasonal_48': 'Daily naive',
             'seasonal_336': 'Weekly naive', 'lightgbm': 'LightGBM'}
    order = list(names)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for col, horizon in enumerate((8, 48)):
        scores = metrics.loc[metrics.horizon == horizon].set_index('model').loc[order]
        for row, metric in enumerate(('mae', 'rmse')):
            ax = axes[row, col]
            ax.bar(range(4), scores[metric], color=[colors[k] for k in order])
            ax.set(xticks=range(4), xticklabels=[names[k] for k in order],
                   title=f'{horizon // 2}-hour {metric.upper()}', ylabel='Source Meter Volume units')
            ax.tick_params(axis='x', labelrotation=15)
            ax.grid(axis='y', alpha=.2)
    fig.suptitle('2022 validation: 363 GSPs, 29 forecast origins; full shared coverage')
    fig.savefig(output / 'pooled_errors.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, dimension in zip(axes, ('volume', 'daytype')):
        frame = pd.read_csv(full / f'gsp_benefit_summary_{dimension}.csv')
        key = 'volume_stratum' if dimension == 'volume' else 'daytype'
        categories = ['low', 'medium', 'high'] if dimension == 'volume' else ['weekday', 'weekend']
        x = np.arange(len(categories))
        for j, horizon in enumerate((8, 48)):
            scores = frame.loc[frame.horizon == horizon].set_index(key).loc[categories]
            ax.bar(x + (j - .5) * .35, scores.win_fraction * 100, width=.35,
                   label=f'{horizon // 2}-hour', color=['#699bc8', '#173f6b'][j])
        ax.set(xticks=x, xticklabels=categories, ylim=(0, 100),
               ylabel='GSPs with lower MAE than daily naive (%)', title=f'Benefits by {dimension}')
        ax.legend()
        ax.grid(axis='y', alpha=.2)
    fig.savefig(output / 'gsp_benefits.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    experiments = ['M0_calendar', 'M1_origin', 'M2_seasonal', 'M3_full']
    for ax, horizon in zip(axes, (8, 48)):
        scores = combined.loc[(combined.model == 'lightgbm') & (combined.horizon == horizon)].set_index('experiment').loc[experiments]
        ax.bar(range(4), scores.mae, color=['#b4c9de', '#8eafce', '#5786b2', '#1f4e79'])
        baseline = metrics.loc[metrics.horizon == horizon].set_index('model')
        ax.axhline(baseline.loc['seasonal_48', 'mae'], color='#e68a00', linestyle='--', label='Daily naive')
        ax.axhline(baseline.loc['persistence', 'mae'], color='#555555', linestyle=':', label='Persistence')
        ax.set(xticks=range(4), xticklabels=['Calendar', '+ Origin lags', '+ Seasonal lags', '+ Rolling'],
               title=f'{horizon // 2}-hour feature ablation', ylabel='MAE (source Meter Volume units)')
        ax.tick_params(axis='x', labelrotation=15)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.2)
    fig.savefig(output / 'feature_ablation.png', dpi=170)
    plt.close(fig)

    importance = pd.read_csv(full / 'importance_aggregated.csv').head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.barh(importance.feature, importance.mean_gain_fraction * 100, color='#2166ac',
            xerr=importance.std_gain_fraction.fillna(0) * 100)
    ax.set(xlabel='Mean within-origin gain share (%); error bars: standard deviation',
           title='Full model: feature gain across 29 origins')
    ax.grid(axis='x', alpha=.2)
    fig.savefig(output / 'feature_importance.png', dpi=170)
    plt.close(fig)

    leads = pd.read_csv(full / 'metrics_by_lead.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, horizon in zip(axes, (8, 48)):
        for model in order:
            values = leads.loc[(leads.horizon == horizon) & (leads.model == model)].sort_values('lead')
            ax.plot(values.lead, values.mae, color=colors[model], label=names[model])
        ax.set(title=f'{horizon // 2}-hour lead errors', xlabel='Lead (half-hours)',
               ylabel='MAE (source Meter Volume units)')
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.savefig(output / 'lead_errors.png', dpi=170)
    plt.close(fig)

    seasons = pd.read_csv(full / 'metrics_pairwise_season.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    categories = ['winter', 'spring', 'summer', 'autumn']
    x = np.arange(4)
    for ax, horizon in zip(axes, (8, 48)):
        for j, model in enumerate(('seasonal_48', 'lightgbm')):
            scores = seasons.loc[(seasons.model == model) & (seasons.horizon == horizon)].set_index('season').loc[categories]
            ax.bar(x + (j - .5) * .35, scores.mae, width=.35, color=colors[model], label=names[model])
        ax.set(xticks=x, xticklabels=categories, title=f'{horizon // 2}-hour seasonal errors',
               ylabel='MAE (source Meter Volume units)')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.2)
    fig.savefig(output / 'season_errors.png', dpi=170)
    plt.close(fig)

    per_gsp = pd.read_csv(full / 'metrics_per_gsp.csv', dtype={'gsp': str})
    lgb = per_gsp.loc[per_gsp.model == 'lightgbm'].copy()
    lgb['scaled_error_sum'] = lgb.mase * lgb.n_mase
    lgb['pooled_mase_contribution_pct'] = 100 * lgb.scaled_error_sum / lgb.groupby('horizon').scaled_error_sum.transform('sum')
    lgb.sort_values(['horizon', 'scaled_error_sum'], ascending=[True, False]).to_csv(
        output / 'mase_contribution_audit.csv', index=False)
    per_gsp.groupby(['model', 'horizon']).agg(median_gsp_mase=('mase', 'median'),
        n_gsps_mase_defined=('mase', 'count')).to_csv(output / 'median_gsp_mase.csv')
    wide = per_gsp.pivot(index=['horizon', 'gsp'], columns='model', values='mae')
    wide['delta_vs_persistence'] = wide.persistence - wide.lightgbm
    wide.to_csv(output / 'gsp_mae_all_models.csv')
    (output / 'report_status.json').write_text(json.dumps(dict(status='complete',
        baseline_metrics_identical_across_ablations=True, final_test_evaluated=False), indent=2))
    print(f'Report figures and audits: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report(args.study, args.output)


if __name__ == '__main__':
    main()
