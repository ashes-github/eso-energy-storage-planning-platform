"""One fold-fitted global LightGBM regressor shared by GSPs and direct leads."""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ForecastDataset
from .features import FeatureSpec, build_features, build_training_examples


def _positive(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


class GlobalTreeForecaster:
    """Pool observed targets across series and leads, using native missingness.

    There is no target encoding, scaling, feature filling, early-stopping holdout
    or validation-label access. GSP categories are fitted from observed training
    examples only. GSPs without any training targets receive NaN forecasts.
    """

    def __init__(self, *, horizon=48, min_history=336, training_step=48,
                 n_estimators=200, learning_rate=0.05, num_leaves=31,
                 min_child_samples=20, reg_lambda=1.0, random_state=7, n_jobs=4,
                 min_training_rows=100, max_training_rows=5_000_000,
                 train_window=None, feature_spec=None):
        for name, value in dict(horizon=horizon, min_history=min_history,
                                training_step=training_step, n_estimators=n_estimators,
                                num_leaves=num_leaves, min_child_samples=min_child_samples,
                                n_jobs=n_jobs, min_training_rows=min_training_rows).items():
            _positive(value, name)
        if num_leaves < 2:
            raise ValueError('num_leaves must be >= 2')
        for name, value in (('learning_rate', learning_rate), ('reg_lambda', reg_lambda)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f'{name} must be finite numeric')
        if not 0 < learning_rate <= 1 or reg_lambda < 0:
            raise ValueError('Require 0 < learning_rate <= 1 and reg_lambda >= 0')
        if not isinstance(random_state, int) or isinstance(random_state, bool) or not 0 <= random_state < 2**31:
            raise ValueError('random_state must be a nonnegative 32-bit integer')
        if max_training_rows is not None:
            _positive(max_training_rows, 'max_training_rows')
            if max_training_rows < min_training_rows:
                raise ValueError('max_training_rows must be >= min_training_rows')
        if train_window is not None:
            _positive(train_window, 'train_window')
            if train_window < min_history + horizon:
                raise ValueError('train_window must contain min_history plus horizon')
        if feature_spec is None:
            feature_spec = FeatureSpec()
        elif isinstance(feature_spec, dict):
            try:
                feature_spec = FeatureSpec(**feature_spec)
            except TypeError as exc:
                raise ValueError('Invalid feature_spec keys') from exc
        if not isinstance(feature_spec, FeatureSpec):
            raise ValueError('feature_spec must be a FeatureSpec or a settings dict')
        self.feature_spec = feature_spec
        self.options = dict(horizon=horizon, min_history=min_history, training_step=training_step,
                            n_estimators=n_estimators, learning_rate=learning_rate,
                            num_leaves=num_leaves, min_child_samples=min_child_samples,
                            reg_lambda=reg_lambda, random_state=random_state, n_jobs=n_jobs,
                            min_training_rows=min_training_rows, max_training_rows=max_training_rows,
                            train_window=train_window, feature_spec=asdict(feature_spec))
        self.horizon = horizon

    def _matrix(self, features):
        frame = features.astype('float32').copy()
        if np.isinf(frame.to_numpy()).any():
            raise ValueError('Nonfinite ML predictor after float32 conversion')
        identities = features.index.get_level_values('gsp')
        known = identities.isin(self.gsp_categories_)
        values = identities.to_numpy(dtype=object).copy()
        values[~known] = None
        frame['gsp_id'] = pd.Categorical(values, categories=self.gsp_categories_)
        return frame

    def _diagnostics(self, reason=''):
        rows = []
        for gsp in self.history.index:
            count = int(self.training_counts_.get(gsp, 0))
            local_reason = reason or ('no_training_targets' if count == 0 else '')
            rows.append(dict(gsp=gsp, status='failed' if local_reason else 'ok', reason=local_reason,
                             fit_start=pd.Timestamp(self.training_audit_['history_start']),
                             fit_end=pd.Timestamp(self.training_audit_['history_end']),
                             n_train_examples=count, n_global_train_examples=sum(self.training_counts_.values()),
                             n_estimators_fitted=self.booster_.current_iteration() if self.booster_ else 0))
        self.diagnostics = rows

    def fit(self, history):
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError('Install the ML extra: pip install -e ".[ml]"') from exc
        self.history = ForecastDataset(history).panel
        window = self.options['train_window']
        if window is not None:
            self.history = self.history.iloc[:, -window:]
        self.booster_ = None
        self.gsp_categories_ = []
        self.feature_names_ = []
        self.training_counts_ = {}
        self.training_audit_ = dict(history_start=self.history.columns[0].isoformat(),
                                   history_end=self.history.columns[-1].isoformat(),
                                   outer_origin=(self.history.columns[-1] + pd.Timedelta('30min')).isoformat())
        if self.history.shape[1] < self.options['min_history'] + self.horizon:
            self._diagnostics('insufficient_history')
            return self
        n_origins = len(range(self.options['min_history'], self.history.shape[1] - self.horizon + 1,
                              self.options['training_step']))
        estimated_rows = n_origins * len(self.history) * self.horizon
        limit = self.options['max_training_rows']
        if limit is not None and estimated_rows > limit:
            raise ValueError(f'ML training would allocate {estimated_rows} examples, above max_training_rows={limit}; '
                             'increase training_step, choose an explicit train_window/subset, or raise the budget')
        features, targets = build_training_examples(
            self.history, horizon=self.horizon, min_history=self.options['min_history'],
            step=self.options['training_step'], spec=self.feature_spec)
        valid = targets.notna()
        observed = targets.loc[valid]
        self.training_counts_ = {gsp: int(count) for gsp, count in observed.groupby(level='gsp').size().items()}
        self.training_audit_.update(n_inner_origins=n_origins, n_examples=estimated_rows,
                                    n_observed_targets=int(valid.sum()), n_missing_targets=int((~valid).sum()),
                                    training_target_start=features.index.get_level_values('timestamp').min().isoformat(),
                                    training_target_end=features.index.get_level_values('timestamp').max().isoformat(),
                                    last_inner_origin=features.index.get_level_values('origin').max().isoformat())
        if valid.sum() < self.options['min_training_rows']:
            self._diagnostics('insufficient_observed_targets')
            return self
        # Identity vocabularies are fitted only on rows with observed targets.
        self.gsp_categories_ = [gsp for gsp in self.history.index if gsp in self.training_counts_]
        matrix = self._matrix(features.loc[valid])
        self.feature_names_ = matrix.columns.tolist()
        options = self.options
        learner = LGBMRegressor(
            objective='regression', n_estimators=options['n_estimators'],
            learning_rate=options['learning_rate'], num_leaves=options['num_leaves'],
            min_child_samples=options['min_child_samples'], reg_lambda=options['reg_lambda'],
            random_state=options['random_state'], n_jobs=options['n_jobs'],
            device_type='cpu', deterministic=True, force_col_wise=True,
            use_missing=True, zero_as_missing=False, verbosity=-1)
        learner.fit(matrix, observed.to_numpy(), categorical_feature=['gsp_id'])
        self.booster_ = learner.booster_
        self._diagnostics()
        return self

    def predict(self, timestamps):
        if not hasattr(self, 'history'):
            raise ValueError('Call fit before predict')
        if len(timestamps) > self.horizon:
            raise ValueError('Prediction horizon exceeds the trained direct-lead range')
        features = build_features(self.history, timestamps, spec=self.feature_spec)
        future = timestamps.tz_convert('UTC').as_unit('ns')
        output = pd.DataFrame(np.nan, index=self.history.index, columns=future)
        if self.booster_ is None:
            return output
        matrix = self._matrix(features)
        if matrix.columns.tolist() != self.feature_names_:
            raise ValueError('Prediction feature schema differs from training')
        predictions = np.asarray(self.booster_.predict(matrix), dtype=float).reshape(len(self.history), -1)
        if not np.isfinite(predictions).all():
            raise ValueError('Global tree produced nonfinite predictions')
        output.iloc[:, :] = predictions
        # Same-GSP temporal forecasts are the objective, not unseen-GSP transfer.
        output.loc[~output.index.isin(self.gsp_categories_), :] = np.nan
        return output

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        metadata = dict(schema_version=1, options=self.options, feature_names=self.feature_names_,
                        gsp_categories=list(self.gsp_categories_), gsp_ids=list(self.history.index),
                        gsp_index_name=self.history.index.name,
                        training_audit=self.training_audit_,
                        training_counts=[dict(gsp=gsp, count=count) for gsp, count in self.training_counts_.items()],
                        fitted=self.booster_ is not None,
                        feature_policy='origin/target lags; native NaN; no target encoding or imputation',
                        holiday_policy='unknown; no holiday calendar supplied in demand-only ML')
        def scalar(value):
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(f'Unsupported metadata value: {type(value).__name__}')
        (directory / 'schema.json').write_text(json.dumps(metadata, indent=2, default=scalar), encoding='utf-8')
        spec = self.feature_spec
        tail = max((1, *spec.lags, *spec.target_lags, *spec.rolling_means, *spec.rolling_stds))
        self.history.iloc[:, -tail:].to_csv(directory / 'observed_panel.csv')
        if self.booster_ is not None:
            self.booster_.save_model(str(directory / 'model.txt'))
            pd.DataFrame(dict(feature=self.feature_names_,
                              gain=self.booster_.feature_importance(importance_type='gain'),
                              split=self.booster_.feature_importance(importance_type='split'))).to_csv(
                                  directory / 'feature_importance.csv', index=False)

    @classmethod
    def load(cls, directory):
        from lightgbm import Booster
        directory = Path(directory)
        metadata = json.loads((directory / 'schema.json').read_text(encoding='utf-8'))
        if metadata['schema_version'] != 1:
            raise ValueError('Unsupported global-tree schema version')
        model = cls(**metadata['options'])
        model.history = ForecastDataset.from_phase1(directory).panel
        model.history.index = pd.Index(metadata['gsp_ids'], name=metadata['gsp_index_name'])
        model.gsp_categories_ = metadata['gsp_categories']
        model.feature_names_ = metadata['feature_names']
        model.training_audit_ = metadata['training_audit']
        model.training_counts_ = {row['gsp']: row['count'] for row in metadata['training_counts']}
        model.booster_ = Booster(model_file=str(directory / 'model.txt')) if metadata['fitted'] else None
        if model.booster_ is not None and model.booster_.feature_name() != model.feature_names_:
            raise ValueError('Saved booster and feature schema disagree')
        model._diagnostics()
        return model
