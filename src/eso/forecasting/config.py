"""Strict Phase 2 settings and deterministic history-only GSP selection."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import SeasonalNaive
from .statistical import StatisticalForecaster


# The default verifies pipeline mechanics. It is not a modelling recommendation.
SMOKE_TRAIN_DAYS = 14
STUDY_MIN_TRAIN_DAYS = 180

DEFAULT = dict(horizons=[8, 48], min_train_days=SMOKE_TRAIN_DAYS, step=48, mase_period=48,
               models=['persistence', 'seasonal_48', 'seasonal_336'],
               series_ids=None, representative_count=None, save_models=False,
               sarima=dict(order=[1, 0, 0], seasonal_order=[1, 0, 0, 48],
                           trend=None, maxiter=200, min_observations=336, train_window=None),
               ets=dict(trend=None, damped_trend=False, seasonal_periods=48,
                        maxiter=1000, min_observations=336, train_window=None))


def assess_training_history(min_train, config):
    """Describe history length, without claiming data coverage or study readiness.

    180 elapsed days is an approximate six-month study floor. Prefer 6–12 months,
    or 1–2+ years where verified settlement coverage permits. A short configured
    fitting window remains a smoke benchmark even with a long input panel.
    """
    model_steps = {}
    for name in config['models']:
        window = config[name]['train_window'] if name in {'sarima', 'ets'} else None
        model_steps[name] = min(min_train, window) if window is not None else min_train
    short_models = [name for name, steps in model_steps.items()
                    if steps < STUDY_MIN_TRAIN_DAYS * 48]
    return dict(
        classification='smoke_benchmark' if short_models else 'long_history_benchmark',
        initial_history_half_hours=min_train,
        initial_history_elapsed_days=min_train / 48,
        initial_fitting_half_hours_by_model=model_steps,
        short_history_models=short_models,
        study_floor_elapsed_days=STUDY_MIN_TRAIN_DAYS,
        recommended_history='6–12 months; preferably 1–2+ years where coverage permits',
        scope='Length assessment only; not a check of observed coverage, seasonal diversity, '
              'features, model quality or operational availability',
    )


def load_config(path=None):
    config = copy.deepcopy(DEFAULT)
    if path is not None:
        supplied = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(supplied, dict):
            raise ValueError('Configuration must be a JSON object')
        unknown = supplied.keys() - config.keys()
        if unknown:
            raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
        for key, value in supplied.items():
            if key in {'sarima', 'ets'}:
                if not isinstance(value, dict) or value.keys() - config[key].keys():
                    raise ValueError(f'Invalid {key} settings')
                config[key].update(value)
            else:
                config[key] = value
    days = config['min_train_days']
    if isinstance(days, bool) or not isinstance(days, int) or days < 1:
        raise ValueError('min_train_days must be a positive integer')
    for key in ('step', 'mase_period'):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    horizons = config['horizons']
    if not isinstance(horizons, list) or not horizons or any(
            not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in horizons):
        raise ValueError('horizons must be positive integers')
    if len(set(horizons)) != len(horizons):
        raise ValueError('horizons must be unique')
    if not isinstance(config['save_models'], bool):
        raise ValueError('save_models must be boolean')
    count = config['representative_count']
    if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count < 3):
        raise ValueError('representative_count must be >= 3 for low/medium/high strata')
    ids = config['series_ids']
    if ids is not None and (not isinstance(ids, list) or not ids or
                            any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids)):
        raise ValueError('series_ids must be a nonempty list of unique string IDs')
    if ids is not None and count is not None:
        raise ValueError('Use series_ids or representative_count, not both')
    model_factories(config)  # Reject model options before loading/fitting data.
    return config


def model_factories(config):
    names = config['models']
    available = {'persistence': 1, 'seasonal_48': 48, 'seasonal_336': 336}
    if not isinstance(names, list) or not names or any(not isinstance(x, str) for x in names):
        raise ValueError('models must be a nonempty list of names')
    if len(set(names)) != len(names) or set(names) - (set(available) | {'sarima', 'ets'}):
        raise ValueError('Require unique supported model names')
    factories = {}
    for name in names:
        if name in available:
            factories[name] = lambda period=available[name]: SeasonalNaive(period)
        else:
            options = copy.deepcopy(config[name])
            StatisticalForecaster(name, **options)
            factories[name] = lambda kind=name, options=options: StatisticalForecaster(kind, **options)
    return factories


def select_series(panel, min_train, config):
    """Stratify by mean absolute quantity in initial history, never by holdout.

    Missing histories are marked unranked. Stable GSP ordering breaks ties.
    This is demand-magnitude stratification, not Phase 1 pattern stratification.
    """
    initial = panel.iloc[:, :min_train]
    selection = pd.DataFrame({'mean_absolute_volume': initial.abs().mean(axis=1),
                              'observed_fraction': initial.notna().mean(axis=1)})
    selection['stratum'] = 'unranked'
    ranked = selection.dropna(subset=['mean_absolute_volume']).sort_index().sort_values(
        'mean_absolute_volume', kind='stable').index.to_numpy()
    buckets = np.array_split(ranked, 3)
    for label, bucket in zip(('low', 'medium', 'high'), buckets):
        selection.loc[bucket, 'stratum'] = label
    ids = config['series_ids']
    if ids is not None:
        missing = set(ids) - set(panel.index)
        if missing:
            raise ValueError(f'Unknown GSP IDs: {sorted(missing)}')
        chosen = ids
    elif config['representative_count'] is not None:
        count = config['representative_count']
        if len(ranked) < count:
            raise ValueError('Not enough GSPs with observed initial history for requested sample')
        chosen = []
        for i, bucket in enumerate(buckets):
            size = count // 3 + (i < count % 3)
            positions = np.linspace(0, len(bucket) - 1, size, dtype=int)
            chosen.extend(bucket[positions].tolist())
    else:
        chosen = panel.index.tolist()
    selection['selected'] = selection.index.isin(chosen)
    selection.index.name = 'gsp'
    return panel.loc[chosen], selection
