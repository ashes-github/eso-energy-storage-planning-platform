"""Forecasts repeat the last available seasonal block, without future reads."""
import numpy as np
import pandas as pd


class SeasonalNaive:
    """period=1: persistence; 48: daily; 336: weekly (elapsed UTC steps)."""

    def __init__(self, period=1):
        if not isinstance(period, int) or isinstance(period, bool) or period < 1:
            raise ValueError('period must be a positive integer')
        self.period = period

    def fit(self, history):
        self.history = history.copy()
        return self

    def predict(self, timestamps):
        h = self.history
        values = np.full((len(h), len(timestamps)), np.nan)
        if h.shape[1] >= self.period:
            values = h.iloc[:, -self.period:].to_numpy()[:, np.arange(len(timestamps)) % self.period]
        return pd.DataFrame(values, index=h.index, columns=timestamps)
