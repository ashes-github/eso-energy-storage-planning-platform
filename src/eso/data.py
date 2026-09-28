"""Resolve revisions and flow direction before diagnosing missing observations."""
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ['GSP Id', 'Settlement Date', 'Settlement Period']
REVISION = ['Date of Aggregation', 'Flow Run Date', 'CDCA Run Number']


def load_records(paths, start, end, direction='I', chunksize=250_000):
    """Filter by settlement date, never by the publication filename.

    Revision policy is latest aggregation date, then flow date, then numeric
    CDCA run number. This is a declared research policy, not a run-type ranking.
    """
    if direction not in {'I', 'E', 'signed_net'}:
        raise ValueError('direction must be I, E or signed_net')
    lo, hi = pd.Timestamp(start).strftime('%Y%m%d'), pd.Timestamp(end).strftime('%Y%m%d')
    parts, sources = [], []
    columns = KEY + REVISION + ['Meter Volume', 'Import/Export Indicator', 'Estimate Indicator']
    for path in paths:
        path = Path(path)
        sources.append({'path': str(path.resolve()), 'bytes': path.stat().st_size,
                        'mtime_ns': path.stat().st_mtime_ns})
        for chunk in pd.read_csv(path, dtype='string', usecols=lambda c: c in columns,
                                 chunksize=chunksize):
            required = set(columns) - {'Estimate Indicator'}
            if not required.issubset(chunk.columns):
                raise ValueError(f'{path}: missing columns {required - set(chunk.columns)}')
            mask = chunk['Settlement Date'].between(lo, hi)
            if direction != 'signed_net':
                mask &= chunk['Import/Export Indicator'].eq(direction)
            parts.append(chunk.loc[mask].copy())
    records = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if records.empty:
        raise ValueError(f'No {direction} records for settlement dates {start} through {end}')
    return records, sources


def align_panel(records, start, end, direction='I'):
    """Keep all observed GSP identities and unresolved gaps.

    Europe/London midnight boundaries produce 46/48/50 half-hours. Timestamp
    conversion preserves elapsed time across clock changes for later forecasting.
    """
    r = records.copy()
    before = len(r)
    for col in ['GSP Id', 'Settlement Date']:
        if r[col].isna().any():
            raise ValueError(f'Missing {col}')
    for col in ['Settlement Period', 'Meter Volume'] + REVISION:
        r[col] = pd.to_numeric(r[col], errors='raise')
        numeric = r[col].to_numpy(dtype=float, na_value=np.nan)
        invalid = np.isinf(numeric).any() if col == 'Meter Volume' else not np.isfinite(numeric).all()
        if invalid:
            raise ValueError(f'Non-finite {col}')
    if (r['Settlement Period'] % 1 != 0).any():
        raise ValueError('Settlement periods must be integers')
    r['Settlement Period'] = r['Settlement Period'].astype(int)
    for col in REVISION[:2]:
        pd.to_datetime(r[col].astype('int64').astype(str), format='%Y%m%d', errors='raise')
    if direction == 'signed_net':
        flags = r['Import/Export Indicator']
        if not flags.isin(['I', 'E']).all():
            raise ValueError('Unknown or missing Import/Export Indicator')
        if (r['Meter Volume'] < 0).any():
            raise ValueError('GP9 Meter Volume must be unsigned before direction conversion')
        # Analysis convention: import demand positive, export negative. Convert
        # before conflict checking so conflicting directions cannot be hidden.
        r['Meter Volume'] *= np.where(flags.eq('E'), -1, 1)
    r = r.sort_values(KEY + REVISION, kind='stable')
    latest = r.groupby(KEY, sort=False).tail(1)[KEY + REVISION]
    ties = r.merge(latest, on=KEY + REVISION, how='inner')
    conflicts = ties.groupby(KEY)['Meter Volume'].nunique(dropna=False)
    if (conflicts > 1).any():
        raise ValueError('Conflicting volumes at identical latest revision; resolve source data')
    if direction == 'signed_net' and (ties.groupby(KEY)['Import/Export Indicator'].nunique() > 1).any():
        raise ValueError('Conflicting directions at identical latest revision')
    r = r.drop_duplicates(KEY, keep='last')
    days = pd.date_range(start, end, freq='D')
    if len(days) == 0:
        raise ValueError('Empty date window')
    expected, timestamps = [], []
    for day in days:
        begin = day.tz_localize('Europe/London').tz_convert('UTC')
        finish = (day + pd.Timedelta(days=1)).tz_localize('Europe/London').tz_convert('UTC')
        utc = pd.date_range(begin, finish, freq='30min', inclusive='left')
        expected.extend((day.strftime('%Y%m%d'), p + 1) for p in range(len(utc)))
        timestamps.extend(utc)
    columns = pd.MultiIndex.from_tuples(expected, names=KEY[1:])
    r['Settlement Date'] = r['Settlement Date'].astype(str)
    invalid_period = ~pd.MultiIndex.from_frame(r[KEY[1:]]).isin(columns)
    if invalid_period.any():
        raise ValueError('Records outside requested calendar or invalid settlement periods')
    panel = r.pivot(index='GSP Id', columns=KEY[1:], values='Meter Volume').reindex(columns=columns)
    panel = panel.sort_index()
    source = r.assign(source='observed')
    if direction == 'signed_net':
        source['source'] = np.where(r['Import/Export Indicator'].eq('I'), 'observed_import', 'observed_export')
    provenance = source.pivot(index='GSP Id', columns=KEY[1:], values='source')
    provenance = provenance.reindex(index=panel.index, columns=columns).fillna('missing')
    panel.columns = pd.DatetimeIndex(timestamps, name='timestamp_utc')
    provenance.columns = panel.columns
    provenance = provenance.mask(panel.isna(), 'missing')
    audit = {'input_rows': before, 'resolved_rows': len(r),
             'superseded_or_duplicate_rows': before - len(r), 'total_gsps': len(panel),
             'periods_per_gsp': len(columns), 'revision_order': REVISION,
             'flow_convention': 'I positive, E negative' if direction == 'signed_net' else direction,
             'observed_import_periods': int((provenance == 'observed_import').sum().sum()),
             'observed_export_periods': int((provenance == 'observed_export').sum().sum()),
             'genuinely_missing_periods': int(panel.isna().sum().sum()),
             'estimated_rows_after_resolution': int(r.get('Estimate Indicator', pd.Series(dtype=str)).eq('T').sum())}
    return panel, provenance, audit


def clean_panel(records, start, end):
    """Legacy complete-case helper retained for historical comparisons/tests."""
    panel, _, audit = align_panel(records, start, end)
    missing = panel.isna().sum(axis=1)
    flat = panel.std(axis=1).fillna(0).le(1e-12)
    excluded = pd.DataFrame({'missing_periods': missing, 'constant_profile': flat})
    excluded = excluded.loc[(missing > 0) | flat]
    panel = panel.loc[(missing == 0) & ~flat]
    if panel.empty:
        raise ValueError('No complete nonconstant GSPs remain')
    audit.update(retained_gsps=len(panel), excluded_gsps=len(excluded),
                 missing_policy='exclude entire GSP (legacy)')
    return panel, excluded, audit
