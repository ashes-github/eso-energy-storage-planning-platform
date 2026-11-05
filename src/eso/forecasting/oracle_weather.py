"""Explicit perfect-weather experiment; never an operational forecast interface."""
from pathlib import Path

import numpy as np
import pandas as pd

from .ml import GlobalTreeForecaster


def lookup_temperature(temperature, index):
    """Vectorized lookup by case GSP/target UTC time; unknowns remain NaN."""
    timestamps = index.get_level_values('timestamp')
    gsps = index.get_level_values('gsp')
    rows = temperature.index.get_indexer(timestamps)
    columns = temperature.columns.get_indexer(gsps)
    valid = (rows >= 0) & (columns >= 0)
    result = np.full(len(index), np.nan, dtype='float32')
    result[valid] = temperature.to_numpy()[rows[valid], columns[valid]]
    return result


class OracleTemperatureForecaster(GlobalTreeForecaster):
    """Identical M2 fitting/targets plus one realized target-time temperature.

    The caller must supply only the declared development-period weather.
    Future target weather is intentionally oracle information. Demand labels
    retain all of GlobalTreeForecaster's chronological restrictions.
    """

    def __init__(self, *, oracle_temperature, **options):
        super().__init__(**options)
        t = oracle_temperature
        if not isinstance(t.index, pd.DatetimeIndex) or t.index.tz is None:
            raise ValueError('Temperature requires timezone-aware timestamps')
        if t.index.has_duplicates or t.columns.has_duplicates or not t.index.is_monotonic_increasing:
            raise ValueError('Temperature axes must be unique and timestamps increasing')
        if np.isinf(t.to_numpy()).any():
            raise ValueError('Infinite temperature is invalid')
        self.oracle_temperature = t.copy()
        self.oracle_temperature.index = t.index.tz_convert('UTC').as_unit('ns')

    def _matrix(self, features):
        matrix = super()._matrix(features)
        matrix['oracle_temperature_c'] = lookup_temperature(self.oracle_temperature, features.index)
        return matrix

    def save(self, directory):
        super().save(directory)
        future = pd.date_range(self.history.columns[-1] + pd.Timedelta('30min'),
                               periods=self.horizon, freq='30min')
        # A prediction artifact needs only its own forecast-period weather.
        self.oracle_temperature.reindex(index=future, columns=self.history.index).to_parquet(
            Path(directory) / 'oracle_prediction_temperature.parquet')
        (Path(directory) / 'ORACLE_WEATHER.txt').write_text(
            'Perfect realized target-time temperature. Reload with OracleTemperatureForecaster.load.\n', encoding='utf-8')

    @classmethod
    def load(cls, directory):
        baseline = GlobalTreeForecaster.load(directory)
        model = cls(oracle_temperature=pd.read_parquet(Path(directory) / 'oracle_prediction_temperature.parquet'),
                    **baseline.options)
        model.__dict__.update(baseline.__dict__)
        return model
