"""Explore heuristic thresholds; this is not calibration against ground truth."""
import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from .features import temporal_features, screen


def analyse(run):
    run = Path(run)
    config = json.loads((run / 'config.json').read_text())
    panel = pd.read_csv(run / 'cleaned_panel.csv', index_col=0)
    panel.columns = pd.to_datetime(panel.columns, utc=True)
    baseline = pd.read_csv(run / 'screening.csv', index_col=0)
    if 'model_eligible' in baseline:
        baseline = baseline.loc[baseline.model_eligible]
        panel = panel.loc[baseline.index]
    rows = []
    for low, hours in itertools.product([0.25, 0.35, 0.45], [6, 8]):
        temporal = dict(config['features']['temporal'], low_fraction=low, min_hours=hours)
        features = baseline.copy()
        for gsp, values in panel.iterrows():
            for key, value in temporal_features(values.values, panel.columns, **temporal).items():
                features.loc[gsp, key] = value
        for threshold in [0.15, 0.25, 0.35]:
            result = screen(features, dict(config['screening'], score_threshold=threshold,
                                           allow_signed_exports=config['direction'] == 'signed_net'))
            rows.append({'low_fraction': low, 'min_low_hours': hours,
                         'score_threshold': threshold, 'candidate_count': int(result.candidate.sum()),
                         'changed_vs_default': int((result.candidate != baseline.candidate).sum()),
                         'total_gsps': len(result)})
    result = pd.DataFrame(rows)
    result.to_csv(run / 'screening_sensitivity.csv', index=False)
    print(result.to_string(index=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    analyse(parser.parse_args().run)
