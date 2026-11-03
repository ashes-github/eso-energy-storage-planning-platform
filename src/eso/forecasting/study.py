"""Run a frozen long-history protocol, streaming one forecast origin at a time."""
import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .backtesting import WalkForward, backtest
from .config import assess_training_history, load_config, model_factories, select_series
from .data import ForecastDataset
from .pipeline import reproducibility, sha256


class OriginPlan:
    def __init__(self, plan, position):
        self.plan, self.position = plan, position

    def __getattr__(self, name):
        return getattr(self.plan, name)

    def folds(self, timestamps, split='validation'):
        return (self.position,)


def run(protocol_path, input_directory, output):
    protocol_path, output = Path(protocol_path), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    frozen_configs = {e['name']: load_config(e['config']) for e in protocol['experiments']}
    config_hashes = {e['name']: sha256(e['config']) for e in protocol['experiments']}
    selection_config = frozen_configs[protocol['experiments'][0]['name']]
    for name, config in frozen_configs.items():
        for key in ('min_train_days', 'step', 'horizons', 'mase_period', 'models',
                    'selection_min_observed_fraction', 'representative_count', 'series_ids', 'save_models'):
            if config[key] != selection_config[key]:
                raise ValueError(f'Controlled experiments differ in {key}')
        settings = {k: v for k, v in config['lightgbm'].items() if k != 'feature_spec'}
        reference_settings = {k: v for k, v in selection_config['lightgbm'].items() if k != 'feature_spec'}
        if settings != reference_settings:
            raise ValueError('Controlled experiments differ in learner/training settings')
    (output / 'frozen_configs.json').write_text(json.dumps(frozen_configs, indent=2), encoding='utf-8')
    source = ForecastDataset.from_phase1(input_directory)
    min_train = selection_config['min_train_days'] * 48
    selected, selection = select_series(source.panel, min_train, selection_config)
    data = ForecastDataset(selected)
    del source, selected
    gc.collect()
    selection.to_csv(output / 'series_selection.csv')
    reference = dict(min_train=min_train, step=selection_config['step'],
                     horizons=tuple(selection_config['horizons']), scale_period=selection_config['mase_period'],
                     test_start=protocol['test_start'])
    metadata = reproducibility(Path(input_directory))
    started = time.monotonic()
    for experiment in protocol['experiments']:
        name = experiment['name']
        config = frozen_configs[name]
        if config['min_train_days'] * 48 != min_train or config['step'] != reference['step'] or tuple(config['horizons']) != reference['horizons']:
            raise ValueError('Controlled experiments must share folds and horizons')
        for key in ('selection_min_observed_fraction', 'representative_count', 'series_ids'):
            if config[key] != selection_config[key]:
                raise ValueError('Controlled experiments must share initial-history eligibility')
        directory = output / name
        directory.mkdir(exist_ok=False)
        plan = WalkForward(**reference)
        positions = list(plan.folds(data.panel.columns, 'validation'))
        factories = model_factories(config)
        run_metadata = dict(plan=asdict(plan), split='validation', resolved_config=config,
                            selected_series=list(data.panel.index), n_origins=len(positions),
                            training_history_assessment=assess_training_history(min_train, config),
                            reproducibility=metadata, config_sha256=config_hashes[name],
                            protocol_sha256=sha256(protocol_path),
                            units='source settlement-period Meter Volume; no conversion',
                            revision_policy='retrospective latest revisions; not operational as-of',
                            availability='zero publication delay assumed', imputation='none',
                            input=str(Path(input_directory).resolve()))
        (directory / 'run.json').write_text(json.dumps(run_metadata, indent=2), encoding='utf-8')
        (directory / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        selection.to_csv(directory / 'series_selection.csv')
        (directory / 'run_status.json').write_text(json.dumps({'status': 'running'}))
        try:
            for i, position in enumerate(positions):
                origin = data.panel.columns[position]
                print(f'{name}: origin {i+1}/{len(positions)} {origin}', flush=True)
                status = dict(status='running', experiment=name, completed_origins=i,
                              total_origins=len(positions), origin=origin.isoformat(),
                              elapsed_seconds=round(time.monotonic() - started, 1))
                (output / 'study_status.json').write_text(json.dumps(status, indent=2))
                diagnostics = []
                predictions = backtest(data, OriginPlan(plan, position), factories, 'validation',
                    diagnostics=diagnostics, artifact_dir=directory / 'models' if config['save_models'] else None)
                if predictions.timestamp.max() >= pd.Timestamp(protocol['test_start']):
                    raise AssertionError('Validation target crosses reserved final test boundary')
                predictions.to_csv(directory / 'predictions.csv', index=False, mode='a', header=i == 0)
                if diagnostics:
                    pd.DataFrame(diagnostics).to_csv(directory / 'fit_diagnostics.csv', index=False,
                                                       mode='a', header=i == 0)
                del predictions
                gc.collect()
            (directory / 'run_status.json').write_text(json.dumps({'status': 'complete', 'metrics': 'pending analysis'}))
        except Exception as exc:
            (directory / 'run_status.json').write_text(json.dumps({'status': 'failed', 'error': str(exc)}))
            (output / 'study_status.json').write_text(json.dumps({'status': 'failed', 'experiment': name, 'error': str(exc)}))
            raise
    (output / 'study_status.json').write_text(json.dumps(dict(status='complete',
        experiments=[e['name'] for e in protocol['experiments']],
        elapsed_seconds=round(time.monotonic() - started, 1), final_test_evaluated=False), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.protocol, args.input, args.output)


if __name__ == '__main__':
    main()
