"""Per-series statistical benchmarks with explicit failure and gap policies."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def _positive(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


class StatisticalForecaster:
    """Fresh independent fits per GSP; failed fits emit NaN, never a fallback.

    SARIMA keeps gaps in the elapsed axis for state-space missing-data handling.
    ETS requires a complete fitting window. Neither interpolates observations.
    Scaling uses only the fitting window, and forecasts retain the source units.
    """

    def __init__(self, kind, *, order=(1, 0, 0), seasonal_order=(1, 0, 0, 48),
                 trend=None, damped_trend=False, seasonal_periods=48,
                 maxiter=200, min_observations=336, train_window=None):
        if kind not in {'sarima', 'ets'}:
            raise ValueError('kind must be sarima or ets')
        _positive(maxiter, 'maxiter')
        _positive(min_observations, 'min_observations')
        if train_window is not None:
            _positive(train_window, 'train_window')
            if train_window < min_observations:
                raise ValueError('train_window must be >= min_observations')
        if len(order) != 3 or len(seasonal_order) != 4:
            raise ValueError('Require order (p,d,q) and seasonal_order (P,D,Q,s)')
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 0
               for x in (*order, *seasonal_order)):
            raise ValueError('SARIMA orders must be nonnegative integers')
        if seasonal_order[3] == 1 or (any(seasonal_order[:3]) and seasonal_order[3] < 2):
            raise ValueError('Active SARIMA seasonality requires a period >= 2')
        _positive(seasonal_periods, 'seasonal_periods')
        if seasonal_periods < 2:
            raise ValueError('ETS seasonal_periods must be >= 2')
        if kind == 'ets' and trend not in {None, 'add'}:
            raise ValueError('ETS uses additive error/seasonality and optional additive trend')
        if kind == 'sarima' and trend not in {None, 'n', 'c', 't', 'ct'}:
            raise ValueError('Invalid SARIMA trend')
        if not isinstance(damped_trend, bool) or (damped_trend and trend != 'add'):
            raise ValueError('Damping requires an additive ETS trend')
        self.kind = kind
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal_periods = seasonal_periods
        self.maxiter = maxiter
        self.min_observations = min_observations
        self.train_window = train_window

    def fit(self, history):
        # Import lazily so baseline-only users do not need the statistical extra.
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        self.history = history.copy()
        self.results = {}
        self.diagnostics = []
        window = history if self.train_window is None else history.iloc[:, -self.train_window:]
        for gsp, series in window.iterrows():
            values = series.to_numpy(dtype=float)
            observed = np.isfinite(values)
            info = dict(gsp=gsp, status='failed', reason='', n_train=len(values),
                        n_observed=int(observed.sum()), fit_start=window.columns[0],
                        fit_end=window.columns[-1], converged=False, warnings='')
            self.diagnostics.append(info)
            minimum = max(self.min_observations, 2 * self.seasonal_periods if self.kind == 'ets'
                          else 2 * self.seasonal_order[3])
            if observed.sum() < minimum:
                info['reason'] = 'insufficient_observations'
                continue
            if self.kind == 'ets' and not observed.all():
                info['reason'] = 'missing_training_values'
                continue
            # Additive ETS accepts signed/zero quantities. A scale of 1 retains
            # zero series without dividing by zero; no centering/donor fill.
            factor = float(np.max(np.abs(values[observed]))) or 1.0
            info['scale_factor'] = factor
            caught = []
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    if self.kind == 'sarima':
                        model = SARIMAX(values / factor, order=self.order,
                                        seasonal_order=self.seasonal_order, trend=self.trend,
                                        enforce_stationarity=True, enforce_invertibility=True)
                    else:
                        model = ETSModel(values / factor, error='add', trend=self.trend,
                                         damped_trend=self.damped_trend, seasonal='add',
                                         seasonal_periods=self.seasonal_periods,
                                         initialization_method='estimated')
                    result = model.fit(maxiter=self.maxiter, disp=False)
                info['converged'] = bool(result.mle_retvals.get('converged', False))
                if not info['converged']:
                    info['reason'] = 'nonconvergence'
                    continue
                if not np.isfinite(np.asarray(result.params)).all():
                    info['reason'] = 'nonfinite_parameters'
                    continue
                info.update(status='ok', aic=float(result.aic))
                self.results[gsp] = (result, factor)
            except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
                info['reason'] = f'{type(exc).__name__}: {exc}'
            finally:
                info['warnings'] = ' | '.join(str(w.message) for w in caught)
        return self

    def predict(self, timestamps):
        expected = pd.date_range(self.history.columns[-1] + pd.Timedelta('30min'),
                                 periods=len(timestamps), freq='30min')
        if not timestamps.equals(expected):
            raise ValueError('Statistical forecasts require contiguous half-hours immediately after history')
        output = pd.DataFrame(np.nan, index=self.history.index, columns=timestamps)
        for info in self.diagnostics:
            gsp = info['gsp']
            if gsp not in self.results:
                continue
            result, factor = self.results[gsp]
            try:
                values = np.asarray(result.forecast(len(timestamps)), dtype=float) * factor
                if values.shape != (len(timestamps),) or not np.isfinite(values).all():
                    raise ValueError('nonfinite or malformed forecast')
                output.loc[gsp] = values
            except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
                info.update(status='failed', reason=f'prediction: {type(exc).__name__}: {exc}')
        return output

    def save(self, directory):
        """Persist fitted statsmodels objects and scaling, for successful fits."""
        import json
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        manifest = []
        for i, info in enumerate(self.diagnostics):
            if info['status'] != 'ok':
                continue
            result, factor = self.results[info['gsp']]
            filename = f'series_{i:04d}.pickle'
            result.save(directory / filename)
            manifest.append(dict(gsp=str(info['gsp']), file=filename, scale_factor=factor))
        (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
