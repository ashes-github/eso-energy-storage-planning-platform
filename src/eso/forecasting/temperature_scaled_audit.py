"""Post-hoc MASE attribution without exclusions from the primary experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import sha256


def audit(directory):
    directory = Path(directory)
    frame = pd.read_csv(directory / 'gsp_benefits.csv')
    pooled = pd.read_csv(directory / 'benefits_pooled.csv').set_index('horizon')
    scales = []
    for path in sorted((directory / 'predictions').glob('*.parquet')):
        rows = pd.read_parquet(path, columns=['origin', 'gsp', 'mase_scale_m2'])
        scales.append(rows.drop_duplicates(['origin', 'gsp']))
    scales = pd.concat(scales).groupby('gsp').mase_scale_m2.agg(['min', 'median', 'max'])
    scales.columns = ['minimum_training_scale', 'median_training_scale', 'maximum_training_scale']
    frame = frame.merge(scales, left_on='gsp', right_index=True, validate='many_to_one')
    output = []
    for horizon, block in frame.groupby('horizon'):
        block = block.copy()
        count = block.n_mase_m2.sum()
        if not block.n_mase_m2.equals(block.n_mase_oracle):
            raise AssertionError('MASE attribution requires shared scaled cases')
        block['mase_contribution_m2'] = block.mase_m2 * block.n_mase_m2 / count
        block['mase_contribution_oracle'] = block.mase_oracle * block.n_mase_oracle / count
        block['delta_pooled_mase_contribution'] = block.mase_contribution_oracle - block.mase_contribution_m2
        np.testing.assert_allclose(block.mase_contribution_m2.sum(), pooled.loc[horizon, 'mase_m2'])
        np.testing.assert_allclose(block.mase_contribution_oracle.sum(), pooled.loc[horizon, 'mase_oracle'])
        output.append(block)
    result = pd.concat(output).sort_values(['horizon', 'delta_pooled_mase_contribution'], ascending=[True, False])
    result.to_csv(directory / 'mase_gsp_contributions.csv', index=False)
    notes = ['# Post-hoc scaled-error diagnostic', '',
        'The full-population MASE remains the primary reported result. This diagnostic attributes '
        'its change to GSP contributions; no model was refitted and no GSP was removed from primary metrics.', '',
        '| Horizon | Largest increase contributor | Median training scale | Contribution to MASE increase | Total MASE increase |',
        '|---|---|---:|---:|---:|']
    sensitivity = []
    for horizon, block in result.groupby('horizon'):
        top = block.iloc[0]
        delta = pooled.loc[horizon, 'mase_oracle'] - pooled.loc[horizon, 'mase_m2']
        notes.append(f'| {horizon//2} hours | {top.gsp} | {top.median_training_scale:.8f} | '
                     f'{top.delta_pooled_mase_contribution:.4f} | {delta:.4f} |')
        remaining = block.loc[block.gsp.ne(top.gsp)]
        n = remaining.n_mase_m2.sum()
        sensitivity.append(dict(horizon=int(horizon), posthoc_excluded_gsp=top.gsp,
            reason='largest signed contribution to MASE increase; post-hoc sensitivity only',
            n_mase=int(n), mase_m2=float((remaining.mase_m2 * remaining.n_mase_m2).sum()/n),
            mase_oracle=float((remaining.mase_oracle * remaining.n_mase_oracle).sum()/n)))
    pd.DataFrame(sensitivity).to_csv(directory / 'mase_posthoc_sensitivity.csv', index=False)
    notes.extend(['', 'Tiny training-only seasonal-difference scales amplify modest absolute errors. '
        'The largest contributor must still be retained in the primary comparison. '
        'Missing temperature does not imply identical predictions: adding temperature changes the '
        'shared global model, including predictions for GSPs with missing weather.', '',
        '`mase_gsp_contributions.csv` gives every GSP\'s absolute/scaled errors, scale range and '
        'additive contribution to pooled MASE. `mase_posthoc_sensitivity.csv` removes only the '
        'largest signed contributor per horizon as an explicitly post-hoc diagnostic. '
        'This is not a replacement score, a changed evaluation population, or a basis for tuning.', ''])
    (directory / 'scaled_error_diagnostic.md').write_text('\n'.join(notes), encoding='utf-8')
    (directory / 'scaled_audit_provenance.json').write_text(json.dumps(dict(
        source_sha256=sha256(__file__), benefits_sha256=sha256(directory / 'gsp_benefits.csv'),
        pooled_sha256=sha256(directory / 'benefits_pooled.csv'),
        primary_population_changed=False, final_test_evaluated=False), indent=2))
    report = directory / 'report.md'
    contents = report.read_text(encoding='utf-8')
    link = '[Scaled-error attribution](scaled_error_diagnostic.md)'
    if link not in contents:
        report.write_text(contents + '\n## Scaled-error diagnostic\n\n' + link +
            ' explains the MASE deterioration and retains the full primary population.\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    audit(parser.parse_args().directory)
