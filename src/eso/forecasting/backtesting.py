"""Expanding-window evaluation with an explicit final holdout boundary."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evaluation import mase_scale


@dataclass(frozen=True)
class WalkForward:
    min_train: int
    test_start: str
    step: int = 48
    horizons: tuple = (8, 48)
    scale_period: int = 48

    def folds(self, timestamps, split='validation'):
        if split not in {'validation', 'test'}:
            raise ValueError('split must be validation or test')
        for value in (self.min_train, self.step, self.scale_period, *self.horizons):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError('Window sizes and horizons must be positive integers')
        if not self.horizons or len(set(self.horizons)) != len(self.horizons):
            raise ValueError('Require nonempty unique horizons')
        boundary = pd.Timestamp(self.test_start)
        if boundary.tzinfo is None:
            raise ValueError('test_start must include a timezone')
        if boundary not in timestamps:
            raise ValueError('test_start must be an exact timestamp in the panel')
        cut = timestamps.get_loc(boundary)
        longest = max(self.horizons)
        if cut < self.min_train + longest or len(timestamps) - cut < longest:
            raise ValueError('Need at least one full validation and test forecast window')
        # Every validation label must precede the final holdout.
        start, stop = (self.min_train, cut) if split == 'validation' else (cut, len(timestamps))
        return range(start, stop - longest + 1, self.step)


def backtest(dataset, plan, models, split='validation', *, diagnostics=None, artifact_dir=None,
             progress=None):
    """Factories create fresh models at each origin; fit sees only past data.

    origin is the FIRST forecast interval start; history ends one half-hour
    earlier. Zero publication delay is assumed. During test walk-forward,
    earlier test observations enter later fits with hyperparameters frozen.
    """
    panel = dataset.panel
    origins = plan.folds(panel.columns, split)
    if not models:
        raise ValueError('Require at least one model factory')
    blocks = []
    for position in origins:
        history = panel.iloc[:, :position].copy()
        future = panel.columns[position:position + max(plan.horizons)]
        scale = mase_scale(history, plan.scale_period)
        for name, factory in models.items():
            if progress is not None:
                progress(future[0], name)
            model = factory()
            model.fit(history.copy())
            prediction = model.predict(future.copy())
            if not isinstance(prediction, pd.DataFrame) or not prediction.index.equals(panel.index) or not prediction.columns.equals(future):
                raise ValueError(f'{name}: predictions must preserve GSP and timestamp axes')
            prediction = prediction.astype(float)
            if np.isinf(prediction.to_numpy()).any():
                raise ValueError(f'{name}: infinite predictions')
            if diagnostics is not None:
                for info in getattr(model, 'diagnostics', []):
                    diagnostics.append(dict(info, model=name, split=split, origin=future[0]))
            if artifact_dir is not None and hasattr(model, 'save'):
                from pathlib import Path
                model.save(Path(artifact_dir) / name / f'origin_{position:08d}')
            for horizon in plan.horizons:
                # Preserve GSP-major/lead-minor case order without one Python
                # dictionary per forecast case. The final table is still in RAM.
                blocks.append(pd.DataFrame(dict(
                    model=name, split=split, origin=future[0],
                    timestamp=pd.DatetimeIndex(np.tile(future[:horizon].to_numpy(), len(panel))),
                    horizon=horizon, lead=np.tile(np.arange(1, horizon + 1), len(panel)),
                    gsp=np.repeat(panel.index.to_numpy(), horizon),
                    actual=panel.iloc[:, position:position + horizon].to_numpy().reshape(-1),
                    prediction=prediction.iloc[:, :horizon].to_numpy().reshape(-1),
                    mase_scale=np.repeat(scale.to_numpy(), horizon))))
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
