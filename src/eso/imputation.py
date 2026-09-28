"""Within-GSP retrospective gap handling; observations are never overwritten."""
import numpy as np
import pandas as pd

DEFAULTS = {'method': 'hybrid', 'max_missing_fraction': 0.2,
            'max_gap_periods': 8, 'linear_max_gap_periods': 4,
            'min_seasonal_donors': 2}


def settings(config=None):
    result = dict(DEFAULTS, **(config or {}))
    if result['method'] not in {'hybrid', 'none'}:
        raise ValueError('missing.method must be hybrid or none')
    if not 0 <= result['max_missing_fraction'] < 1:
        raise ValueError('max_missing_fraction must be in [0,1)')
    for key in ['max_gap_periods', 'linear_max_gap_periods', 'min_seasonal_donors']:
        if not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    return result


def gaps(mask):
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def impute_panel(panel, provenance=None, config=None):
    """Bounded interpolation, then same-clock/day-type medians from ORIGINAL data.

    No cross-GSP borrowing or extrapolation. Sparse/long-gap rows stay in outputs
    with insufficient_evidence; they are not assigned a negative suitability label.
    These limits are conservative research defaults, not calibrated confidence.
    """
    cfg = settings(config)
    # CSV string dtypes can pivot to nullable/object columns. Normalize missing
    # sentinels once, and allow fractional interpolation into integer inputs.
    panel = pd.DataFrame(panel.to_numpy(dtype=float, na_value=np.nan),
                         index=panel.index, columns=panel.columns)
    filled = panel.copy()
    if provenance is None:
        provenance = pd.DataFrame(np.where(panel.isna(), 'missing', 'observed'),
                                  index=panel.index, columns=panel.columns)
    else:
        provenance = provenance.copy()
    local = panel.columns.tz_convert('Europe/London')
    slots = local.hour * 2 + local.minute // 30
    day_type = local.dayofweek >= 5
    dates = local.date
    quality = []
    for gsp, row in panel.iterrows():
        original = row.to_numpy(dtype=float)
        if np.isinf(original).any():
            raise ValueError('Infinite values cannot be imputed')
        missing = np.isnan(original)
        runs = gaps(missing)
        longest = max((end - start for start, end in runs), default=0)
        values = original.copy()
        methods = provenance.loc[gsp].to_numpy(copy=True)
        allowed = (missing.mean() <= cfg['max_missing_fraction']
                   and longest <= cfg['max_gap_periods'])
        if missing.any() and allowed and cfg['method'] == 'hybrid':
            for start, end in runs:
                if (end - start <= cfg['linear_max_gap_periods'] and start > 0
                        and end < len(values)):
                    values[start:end] = np.linspace(original[start-1], original[end], end-start+2)[1:-1]
                    methods[start:end] = 'linear'
            for pos in np.flatnonzero(np.isnan(values)):
                donor = (~missing & (slots == slots[pos]) & (day_type == day_type[pos])
                         & (dates != dates[pos]))
                # Count distinct donor DAYS, not duplicated clock slots on long days.
                donor_days = pd.Series(original[donor], index=dates[donor]).groupby(level=0).median()
                if len(donor_days) >= cfg['min_seasonal_donors']:
                    values[pos] = donor_days.median()
                    methods[pos] = 'seasonal_median'
        filled.loc[gsp] = values
        provenance.loc[gsp] = methods
        residual = int(np.isnan(values).sum())
        ready = residual == 0
        quality.append({'GSP Id': gsp, 'observed_periods': int((~missing).sum()),
                        'missing_periods_before': int(missing.sum()),
                        'missing_fraction_before': float(missing.mean()),
                        'longest_missing_run_periods': longest,
                        'imputed_periods': int((missing & np.isfinite(values)).sum()),
                        'remaining_missing_periods': residual,
                        'constant_profile': bool(ready and np.std(values) <= 1e-12),
                        'model_eligible': ready,
                        'data_status': ('observed' if not missing.any() else
                                        'imputed' if ready else 'insufficient_evidence')})
    return filled, provenance, pd.DataFrame(quality).set_index('GSP Id')
