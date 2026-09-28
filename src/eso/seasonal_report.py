"""Summarize a completed seasonal experiment without retraining models."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .evaluation import distance_stability


def report(output):
    output = Path(output)
    status = json.loads((output/'COMPLETE.json').read_text())
    daily = pd.read_csv(output/'daily_features.csv')
    aggregate = pd.read_csv(output/'seasonal_recurrence.csv')
    full = aggregate[aggregate.phase == 'all']
    rows = []
    for (season, day_type), group in full.groupby(['season', 'day_type']):
        subset = aggregate[(aggregate.season == season) & (aggregate.day_type == day_type)]
        phases = subset.pivot(index='GSP Id', columns='phase', values='observed_pattern_frequency_6h')
        pairs = phases[['development', 'holdout']].dropna()
        days = daily[(daily.season == season) & (daily.day_type == day_type)]
        rows.append({'season': season, 'day_type': day_type, 'calendar_days': days.date.nunique(),
                     'gsps': len(group), 'sufficient_evidence_gsps': int(group.evidence_status.eq('sufficient').sum()),
                     'valid_gsp_days': int(group.valid_days.sum()), 'expected_gsp_days': int(group.expected_days.sum()),
                     'median_recurrence_6h': group.observed_pattern_frequency_6h.median(),
                     'median_recurrence_8h': group.observed_pattern_frequency_8h.median(),
                     'paired_gsps': len(pairs),
                     'early_late_spearman_6h': spearmanr(pairs.development, pairs.holdout).statistic,
                     'median_absolute_early_late_change_6h': (pairs.holdout-pairs.development).abs().median()})
    summary = pd.DataFrame(rows)
    summary.to_csv(output/'regime_summary.csv', index=False)
    reference = daily[(daily.season == 'summer_2022') & daily.date.between('2022-06-20', '2022-06-24')]
    valid = reference[reference.valid_day]
    comparison = reference.groupby('GSP Id').size().to_frame('reference_expected_days')
    comparison['reference_valid_days'] = valid.groupby('GSP Id').size().reindex(comparison.index, fill_value=0)
    for h in (6, 8):
        comparison[f'reference_frequency_{h}h'] = valid.groupby('GSP Id')[f'qualifies_{h}h'].mean()
    summer = full[(full.season == 'summer_2022') & (full.day_type == 'weekday')].set_index('GSP Id')
    comparison = comparison.join(summer[['observed_pattern_frequency_6h', 'observed_pattern_frequency_8h']])
    comparison['absolute_difference_6h'] = (comparison.reference_frequency_6h-comparison.observed_pattern_frequency_6h).abs()
    # The five reference days are inside the full-season estimate. Also compare
    # with disjoint summer weekdays so agreement is not inflated by overlap.
    other = daily[(daily.season == 'summer_2022') & (daily.day_type == 'weekday') &
                  ~daily.date.between('2022-06-20', '2022-06-24') & daily.valid_day]
    comparison['other_summer_weekdays_frequency_6h'] = other.groupby('GSP Id').qualifies_6h.mean()
    comparison['absolute_difference_other_days_6h'] = (comparison.reference_frequency_6h-comparison.other_summer_weekdays_frequency_6h).abs()
    comparison.to_csv(output/'reference_week_comparison.csv')
    diagnostics = []
    if status['models_complete']:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        metrics = pd.read_csv(output/'model_metrics.csv')
        model_summary = metrics.groupby(['season', 'day_type', 'representation', 'model']).agg(
            mean_later_median_rmse=('later_median_rmse', 'mean'),
            sd_later_median_rmse=('later_median_rmse', 'std'), runs=('seed', 'size')).reset_index()
        model_summary.to_csv(output/'model_summary.csv', index=False)
        for regime_dir in sorted((output/'models').iterdir()):
            split = json.loads((regime_dir/'split.json').read_text())
            # Match exact compound regime names without splitting season tokens.
            season, day_type = next((r.season, r.day_type) for r in summary.itertuples()
                                    if f'{r.season}_{r.day_type}' == regime_dir.name)
            recurrence = full[(full.season == season) & (full.day_type == day_type)].set_index('GSP Id')
            colours = recurrence.loc[split['train_gsps'], 'observed_pattern_frequency_6h'].to_numpy()
            for repdir in sorted(p for p in regime_dir.iterdir() if p.is_dir()):
                for model in ['autoencoder', 'gplvm']:
                    artifacts = sorted(repdir.glob(f'{model}_*.npz'))
                    arrays = [np.load(p) for p in artifacts]
                    for i, (path, arr) in enumerate(zip(artifacts, arrays)):
                        losses = arr['losses']
                        tail = min(50, len(losses)//2)
                        diagnostics.append({'regime': regime_dir.name, 'representation': repdir.name,
                                            'model_run': path.stem, 'steps': len(losses),
                                            'last_loss': losses[-1],
                                            'last_block_mean': losses[-tail:].mean(),
                                            'previous_block_mean': losses[-2*tail:-tail].mean()})
                        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
                        ax.plot(np.arange(1, len(losses)+1), losses)
                        ax.set(xlabel='Training step', ylabel='Training objective', title=path.stem)
                        fig.savefig(path.with_name(path.stem+'_loss.png'), dpi=120); plt.close(fig)
                        if i == 0:
                            fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
                            im = ax.scatter(arr['latent'][:, 0], arr['latent'][:, 1], c=colours, vmin=0, vmax=1, s=14)
                            fig.colorbar(im, ax=ax, label='All-season observed pattern frequency (6h)')
                            ax.set(title=f'{regime_dir.name}\n{repdir.name} {path.stem}', xlabel='Latent 1', ylabel='Latent 2')
                            fig.savefig(path.with_name(path.stem+'_latent.png'), dpi=140); plt.close(fig)
                    for i in range(len(arrays)):
                        for j in range(i+1, len(arrays)):
                            diagnostics.append({'regime': regime_dir.name, 'representation': repdir.name,
                                                'model_run': f'{artifacts[i].stem} vs {artifacts[j].stem}',
                                                'latent_distance_spearman': distance_stability(arrays[i]['latent'], arrays[j]['latent'])})
        pd.DataFrame(diagnostics).to_csv(output/'training_diagnostics.csv', index=False)
    text = ['# Seasonal Phase 1 results', '',
            'These are observed demand-pattern frequencies, not battery-feasibility probabilities.', '',
            summary.to_string(index=False), '',
            'June 20–24 versus the OTHER summer weekdays, using the same daily rule:',
            f"Median absolute frequency difference: {comparison.absolute_difference_other_days_6h.median():.4f}",
            f"GSPs differing by at least 0.20: {int(comparison.absolute_difference_other_days_6h.ge(.2).sum())}", '',
            'See seasonal_experiments.md in docs for definitions, uncertainty and evaluation limits.']
    (output/'RESULTS.md').write_text('\n'.join(text), encoding='utf-8')
    print(summary.to_string(index=False))
    print('\n'.join(text[-4:-1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    report(parser.parse_args().run)
