"""Artificial masking experiment on known readings, not independent calibration."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import spectral_features, temporal_features, screen
from .imputation import impute_panel


def validate(run, sample_size=30, seed=17):
    run = Path(run)
    config = json.loads((run / 'config.json').read_text())
    panel = pd.read_csv(run / 'observed_panel.csv', index_col=0)
    panel.columns = pd.to_datetime(panel.columns, utc=True)
    panel = panel.loc[panel.notna().all(axis=1) & (panel.std(axis=1) > 1e-12)]
    if panel.empty:
        raise ValueError('No fully observed varying profiles for artificial masking')
    rng = np.random.default_rng(seed)
    panel = panel.iloc[rng.choice(len(panel), min(sample_size, len(panel)), replace=False)]
    records = []
    screen_cfg = dict(config['screening'], allow_signed_exports=config['direction'] == 'signed_net')

    def features(y):
        f = spectral_features(y, config['features']['nperseg'])
        f.update(temporal_features(y, panel.columns, **config['features']['temporal']))
        return screen(pd.DataFrame([f]), screen_cfg).iloc[0]

    for gsp, truth in panel.iterrows():
        original = truth.to_numpy()
        reference = features(original)
        for pattern in ['random_5pct', 'random_20pct', 'block_2h', 'block_4h', 'block_6h']:
            mask = np.zeros(len(truth), dtype=bool)
            if pattern.startswith('random'):
                count = max(1, int(len(truth) * (0.05 if pattern == 'random_5pct' else 0.2)))
                mask[rng.choice(len(truth), count, replace=False)] = True
            else:
                length = {'block_2h': 4, 'block_4h': 8, 'block_6h': 12}[pattern]
                begin = int(rng.integers(1, len(truth)-length-1))
                mask[begin:begin+length] = True
            incomplete = truth.copy()
            incomplete.iloc[mask] = np.nan
            for method in ['hybrid', 'unbounded_linear_reference']:
                if method == 'hybrid':
                    filled, _, _ = impute_panel(incomplete.to_frame().T, config=config.get('missing'))
                    prediction = filled.iloc[0].to_numpy(dtype=float)
                else:
                    # Deliberately permissive comparator; not used by the pipeline.
                    prediction = incomplete.interpolate(limit_direction='both').to_numpy()
                recovered = mask & np.isfinite(prediction)
                error = prediction[recovered] - original[recovered]
                row = {'GSP Id': gsp, 'pattern': pattern, 'method': method,
                       'masked_periods': int(mask.sum()), 'filled_periods': int(recovered.sum()),
                       'rmse': float(np.sqrt(np.mean(error**2))) if len(error) else np.nan,
                       'normalized_rmse': float(np.sqrt(np.mean(error**2))/np.std(original)) if len(error) else np.nan}
                if np.isfinite(prediction).all():
                    f = features(prediction)
                    row.update(candidate_changed=bool(f.candidate != reference.candidate),
                               low_duration_error_hours=float(f.longest_observed_low_hours-reference.longest_observed_low_hours),
                               band_fraction_error=float(f.fraction_6_8h-reference.fraction_6_8h))
                records.append(row)
    details = pd.DataFrame(records)
    details.to_csv(run / 'imputation_validation.csv', index=False)
    summary = details.groupby(['method', 'pattern']).agg(
        profiles=('GSP Id', 'size'), masked_periods=('masked_periods', 'sum'),
        filled_periods=('filled_periods', 'sum'), mean_normalized_rmse=('normalized_rmse', 'mean'),
        assessed_labels=('candidate_changed', 'count'), label_flip_fraction=('candidate_changed', 'mean'))
    summary['filled_fraction'] = summary.filled_periods / summary.masked_periods
    summary.to_csv(run / 'imputation_validation_summary.csv')
    (run / 'imputation_validation_config.json').write_text(json.dumps(
        {'seed': seed, 'sample_size': len(panel), 'missing': config.get('missing'),
         'interpretation': 'Artificial masking on observed profiles; no ground-truth battery labels'}, indent=2))
    print(summary.to_string())
    return details


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--sample-size', type=int, default=30)
    args = parser.parse_args()
    validate(args.run, args.sample_size)
