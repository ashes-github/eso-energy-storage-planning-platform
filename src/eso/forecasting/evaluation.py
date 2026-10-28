"""Observed-target metrics and explicit forecast coverage."""
import numpy as np
import pandas as pd


def mase_scale(history, period=48):
    """Seasonal absolute differences using training observations only."""
    differences = history.diff(period, axis=1).abs()
    return differences.mean(axis=1).replace(0, np.nan)


def summarize(predictions, groups=('model', 'split', 'horizon', 'gsp')):
    """Micro averages over forecast cases; horizons include leads 1 through H.

    Repeated targets at different origins are distinct forecasting cases.
    Missing forecasts do not enter errors, but do reduce coverage.
    """
    rows = []
    for key, frame in predictions.groupby(list(groups), sort=True, dropna=False):
        actual = frame.actual.notna()
        valid = actual & frame.prediction.notna()
        error = frame.loc[valid, 'prediction'] - frame.loc[valid, 'actual']
        scaled = error.abs() / frame.loc[valid, 'mase_scale']
        row = dict(zip(groups, key if isinstance(key, tuple) else (key,)))
        row.update(n_cases=len(frame), n_observed=int(actual.sum()), n_scored=int(valid.sum()),
                   coverage=float(valid.sum() / actual.sum()) if actual.any() else np.nan,
                   mae=error.abs().mean(), rmse=np.sqrt((error ** 2).mean()),
                   mase=scaled.mean(), n_mase=int(scaled.notna().sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def common_cases(predictions):
    """Mask errors to the same observed cases across ALL configured models.

    Preserve the original rows and observed denominators: coverage now means
    the fraction of observed cases shared by the entire model comparison.
    Horizon is part of case identity, so the 8/48 outputs remain separate.
    Missing model rows fail closed rather than shrinking the comparison set.
    """
    keys = ['split', 'origin', 'timestamp', 'horizon', 'lead', 'gsp']
    if predictions.duplicated(['model', *keys]).any():
        raise ValueError('Duplicate model/case rows')
    if predictions.empty:
        return predictions.copy()
    grouped = predictions.groupby(keys, dropna=False, sort=False)
    if (grouped.actual.nunique(dropna=False) > 1).any():
        raise ValueError('Models must share identical actuals for each case')
    if (grouped.mase_scale.nunique(dropna=False) > 1).any():
        raise ValueError('Models must share identical MASE scales for each case')
    valid = predictions.actual.notna() & predictions.prediction.notna()
    working = predictions.assign(_valid=valid)
    case_groups = working.groupby(keys, dropna=False, sort=False)
    common = (case_groups['_valid'].transform('sum') == predictions.model.nunique())
    common &= case_groups['model'].transform('nunique') == predictions.model.nunique()
    result = predictions.copy()
    result['prediction'] = result.prediction.where(common)
    return result
