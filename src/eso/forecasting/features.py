"""Origin-available demand features and known-future local calendar features.

An origin is the FIRST forecast interval, matching backtesting.py. origin_lag_k
is Y[origin - k * 30 minutes], independent of lead. target_lag_k aligns the donor
to each target, and reads it only when strictly before the forecast origin.
"""
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import ForecastDataset


def _positive(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


@dataclass(frozen=True)
class FeatureSpec:
    """Explicit feature schema; full windows only, no fitted transformations."""

    lags: tuple = (1, 2, 48, 96, 336)
    rolling_means: tuple = (6, 12, 48, 336)
    rolling_stds: tuple = (48, 336)
    std_ddof: int = 1
    target_lags: tuple = (48, 96, 336)

    def __post_init__(self):
        for field in ('lags', 'rolling_means', 'rolling_stds', 'target_lags'):
            values = getattr(self, field)
            if not isinstance(values, (tuple, list)):
                raise ValueError(f'{field} must be a sequence of positive integers')
            for value in values:
                _positive(value, field)
            if len(set(values)) != len(values):
                raise ValueError(f'{field} must be unique')
            object.__setattr__(self, field, tuple(values))
        if not isinstance(self.std_ddof, int) or isinstance(self.std_ddof, bool) or self.std_ddof not in (0, 1):
            raise ValueError('std_ddof must be 0 (population) or 1 (sample)')
        if any(window <= self.std_ddof for window in self.rolling_stds):
            raise ValueError('Rolling standard-deviation windows must exceed std_ddof')


def _holidays(calendar):
    """Explicit date -> boolean flags, including known non-holidays.

    Missing dates mean unknown. The caller must supply the appropriate regional
    calendar/version known at the forecast origin, not retrospective additions.
    """
    if calendar is None:
        return {}
    if not isinstance(calendar, Mapping):
        raise ValueError('holiday_calendar must map local dates to boolean flags')
    result = {}
    for key, value in calendar.items():
        date = pd.Timestamp(key)
        if pd.isna(date) or date.tzinfo is not None or date != date.normalize():
            raise ValueError('Holiday keys must be timezone-free local calendar dates')
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError('Holiday flags must be explicit booleans')
        if date.date() in result:
            raise ValueError('Duplicate normalized holiday date')
        result[date.date()] = float(value)
    return result


def build_features(history, future, *, spec=FeatureSpec(), holiday_calendar=None):
    """Return numeric features indexed by (gsp, origin, timestamp).

    history has the ForecastDataset observed-panel contract. future contains
    consecutive half-hours immediately after history. Origin-relative features
    repeat across leads; target-relative lags and calendar features vary by lead.
    A target-relative donor at/after origin is unavailable and remains NaN.
    Missing lag donors, incomplete rolling windows and unknown holidays remain
    NaN. Rolling standard deviations use spec.std_ddof (sample by default).
    GSP identity remains an index field for a future fold-fitted encoder.
    """
    if not isinstance(spec, FeatureSpec):
        raise ValueError('spec must be a FeatureSpec')
    panel = ForecastDataset(history).panel
    if not isinstance(future, pd.DatetimeIndex) or future.tz is None or len(future) == 0:
        raise ValueError('future must be nonempty timezone-aware timestamps')
    future = future.tz_convert('UTC').as_unit('ns')
    origin = panel.columns[-1] + pd.Timedelta('30min')
    expected = pd.date_range(origin, periods=len(future), freq='30min')
    if not future.equals(expected):
        raise ValueError('future must be contiguous half-hours immediately after history')
    holiday_lookup = _holidays(holiday_calendar)
    h = len(future)
    index = pd.MultiIndex.from_product([panel.index, [origin], future],
                                      names=['gsp', 'origin', 'timestamp'])
    result = pd.DataFrame(index=index)
    result['lead'] = np.tile(np.arange(1, h + 1), len(panel))
    for lag in spec.lags:
        values = panel.iloc[:, -lag].to_numpy() if lag <= panel.shape[1] else np.full(len(panel), np.nan)
        result[f'origin_lag_{lag}'] = np.repeat(values, h)
    origin_position = panel.shape[1]
    target_positions = origin_position + np.arange(h)
    for lag in spec.target_lags:
        donor_positions = target_positions - lag
        available = (donor_positions >= 0) & (donor_positions < origin_position)
        donors = donor_positions[available]
        # Only guarded donor positions may be used for positional indexing.
        # In particular lead 49's daily donor is at origin and cannot be read.
        assert (donors < origin_position).all(), 'Target lag reaches the forecast window'
        values = np.full((len(panel), h), np.nan)
        values[:, available] = panel.iloc[:, donors].to_numpy()
        result[f'target_lag_{lag}'] = values.reshape(-1)
    for windows, statistic in ((spec.rolling_means, 'mean'), (spec.rolling_stds, 'std')):
        for window in windows:
            values = np.full(len(panel), np.nan)
            if panel.shape[1] >= window:
                block = panel.iloc[:, -window:]
                complete = block.notna().all(axis=1)
                estimate = (block.mean(axis=1) if statistic == 'mean'
                            else block.std(axis=1, ddof=spec.std_ddof))
                values = estimate.where(complete).to_numpy()
            result[f'{statistic}_{window}'] = np.repeat(values, h)

    local = future.tz_convert('Europe/London')
    half_hour = local.hour.to_numpy() * 2 + local.minute.to_numpy() // 30
    # Elapsed half-hours since local midnight distinguish repeated autumn slots.
    midnight = local.normalize().tz_convert('UTC')
    settlement_period = np.asarray((future - midnight) / pd.Timedelta('30min'), dtype=int) + 1
    dow = local.dayofweek.to_numpy()
    calendar = dict(
        settlement_period=settlement_period,
        half_hour_of_day=half_hour,
        hour=local.hour.to_numpy(),
        day_of_week=dow,
        weekend=(dow >= 5).astype(int),
        month=local.month.to_numpy(),
        holiday=np.array([holiday_lookup.get(date, np.nan) for date in local.date]),
        time_of_day_sin=np.sin(2 * np.pi * half_hour / 48),
        time_of_day_cos=np.cos(2 * np.pi * half_hour / 48),
        day_of_week_sin=np.sin(2 * np.pi * dow / 7),
        day_of_week_cos=np.cos(2 * np.pi * dow / 7),
    )
    for name, values in calendar.items():
        result[name] = np.tile(values, len(panel))
    return result


def build_training_examples(history, *, horizon=48, min_history=336, step=48,
                            spec=FeatureSpec(), holiday_calendar=None):
    """Build direct multi-lead examples entirely within an outer fit history.

    Each inner origin uses only its prefix for X. All its target intervals must
    already exist in the supplied outer history; incomplete final horizons are
    purged. Missing targets remain NaN for explicit exclusion by future learners.
    This returns (features, targets), with identical indexes. Nothing is encoded,
    scaled or imputed here. No validation or reserved test labels are accepted
    separately: pass only the history received by model.fit().
    """
    for name, value in (('horizon', horizon), ('min_history', min_history), ('step', step)):
        _positive(value, name)
    panel = ForecastDataset(history).panel
    if panel.shape[1] < min_history + horizon:
        raise ValueError('Need min_history plus at least one complete target horizon')
    frames, targets = [], []
    for position in range(min_history, panel.shape[1] - horizon + 1, step):
        future = panel.columns[position:position + horizon]
        frame = build_features(panel.iloc[:, :position], future, spec=spec,
                               holiday_calendar=holiday_calendar)
        frames.append(frame)
        targets.append(pd.Series(panel.iloc[:, position:position + horizon].to_numpy().reshape(-1),
                                 index=frame.index, name='target'))
    return pd.concat(frames), pd.concat(targets)
