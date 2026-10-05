"""Observed half-hourly values, in original meter-volume units."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ForecastDataset:
    """Rows are GSP IDs; columns are unique contiguous UTC interval starts.

    NaN means unavailable. Negative net imports and constant series are valid.
    Revision availability is not reconstructed from Phase 1 CSV exports.
    """

    panel: pd.DataFrame

    def __post_init__(self):
        p = self.panel.copy()
        if p.empty or p.index.has_duplicates or p.index.isna().any():
            raise ValueError('Require nonempty panel with unique nonmissing GSP IDs')
        if not isinstance(p.columns, pd.DatetimeIndex) or p.columns.tz is None:
            raise ValueError('Require timezone-aware timestamp columns')
        p.columns = p.columns.tz_convert('UTC').as_unit('ns')
        if p.columns.has_duplicates or not p.columns.is_monotonic_increasing:
            raise ValueError('Timestamps must be unique and increasing')
        if (p.columns.asi8 % pd.Timedelta('30min').value != 0).any():
            raise ValueError('Timestamps must fall on half-hour boundaries')
        if len(p.columns) > 1 and not (np.diff(p.columns.asi8) == pd.Timedelta('30min').value).all():
            raise ValueError('Require contiguous half-hours; represent gaps with NaN')
        p = p.astype(float)
        if np.isinf(p.to_numpy()).any():
            raise ValueError('Infinite demand is invalid')
        self.panel = p

    @classmethod
    def from_phase1(cls, directory):
        """Prefer original observations; never trust retrospective fills."""
        directory = Path(directory)
        observed = directory / 'observed_panel.csv'
        def read(path):
            frame = pd.read_csv(path, index_col=0, dtype={0: str})
            frame.columns = pd.to_datetime(frame.columns)
            return frame
        if observed.exists():
            return cls(read(observed))
        panel = read(directory / 'cleaned_panel.csv')
        provenance = read(directory / 'value_provenance.csv')
        if not panel.index.equals(provenance.index) or not panel.columns.equals(provenance.columns):
            raise ValueError('Provenance must match the cleaned panel exactly')
        return cls(panel.where(provenance.isin(['observed', 'observed_import', 'observed_export'])))
