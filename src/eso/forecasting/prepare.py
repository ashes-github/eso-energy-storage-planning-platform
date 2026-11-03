"""Inventory historical GP9 settlement coverage and prepare observed panels."""
import argparse
import glob
import hashlib
import json
from pathlib import Path

import pandas as pd

from eso.data import KEY, REVISION, align_panel
from .data import ForecastDataset


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inventory(paths, output, chunksize=250_000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sources, coverage = [], []
    for path in paths:
        path = Path(path)
        counts = pd.Series(dtype='int64')
        for chunk in pd.read_csv(path, usecols=['Settlement Date'], dtype='string', chunksize=chunksize):
            if chunk['Settlement Date'].isna().any():
                raise ValueError(f'{path}: missing settlement dates')
            counts = counts.add(chunk['Settlement Date'].value_counts(), fill_value=0)
        dates = pd.to_datetime(counts.index, format='%Y%m%d', errors='raise')
        frame = pd.DataFrame(dict(source=str(path.resolve()), settlement_date=dates,
                                  raw_rows=counts.to_numpy(dtype='int64')))
        coverage.append(frame)
        source = dict(path=str(path.resolve()), bytes=path.stat().st_size,
                      sha256=sha256(path), rows=int(counts.sum()),
                      settlement_start=dates.min().date().isoformat(),
                      settlement_end=dates.max().date().isoformat())
        sources.append(source)
        print(f'{path.name}: {source["settlement_start"]} .. {source["settlement_end"]}; {source["rows"]:,} rows', flush=True)
    if not sources:
        raise ValueError('No GP9 sources')
    combined = pd.concat(coverage, ignore_index=True)
    combined.to_csv(output / 'source_date_coverage.csv', index=False)
    combined.groupby('settlement_date', sort=True).raw_rows.sum().to_csv(output / 'settlement_coverage.csv')
    (output / 'inventory.json').write_text(json.dumps(dict(sources=sources,
        policy='Raw settlement-date coverage only; revisions and GSP completeness not yet resolved'), indent=2), encoding='utf-8')
    return sources


def build_observed_panel(inventory_directory, start, end, output, chunksize=100_000):
    """Stage daily raw records; resolve revisions with the existing data policy.

    A day stays bounded in memory. All sources for that date are combined before
    align_panel, so revision order/conflicting latest ties match Phase 1 exactly.
    No imputation or GSP filtering occurs. Staging files remain as a local audit.
    """
    start = pd.Timestamp(start).strftime('%Y-%m-%d')
    end = pd.Timestamp(end).strftime('%Y-%m-%d')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    staging = output / 'daily_source_records'
    staging.mkdir()
    lo = pd.Timestamp(start).strftime('%Y%m%d')
    hi = pd.Timestamp(end).strftime('%Y%m%d')
    if lo > hi:
        raise ValueError('start must precede end')
    metadata = json.loads((Path(inventory_directory) / 'inventory.json').read_text(encoding='utf-8'))
    sources = []
    columns = KEY + REVISION + ['Meter Volume', 'Import/Export Indicator', 'Estimate Indicator']
    (output / 'prepare_status.json').write_text(json.dumps({'status': 'staging'}))
    for source in metadata['sources']:
        if source['settlement_end'] < start or source['settlement_start'] > end:
            continue
        path = Path(source['path'])
        if sha256(path) != source['sha256']:
            raise ValueError(f'Source changed since inventory: {path}')
        before = path.stat()
        selected = 0
        for chunk in pd.read_csv(path, dtype='string', usecols=lambda c: c in columns, chunksize=chunksize):
            required = set(columns) - {'Estimate Indicator'}
            if not required.issubset(chunk.columns):
                raise ValueError(f'Missing GP9 columns in {path}')
            chunk = chunk.loc[chunk['Settlement Date'].between(lo, hi)].reindex(columns=columns)
            selected += len(chunk)
            for date, records in chunk.groupby('Settlement Date', sort=False):
                if len(date) != 8 or not date.isdigit():
                    raise ValueError('Invalid settlement date')
                destination = staging / f'{date}.csv'
                records.to_csv(destination, index=False, mode='a', header=not destination.exists())
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'Source changed during read: {path}')
        sources.append(dict(source, selected_rows=selected))
        print(f'Staged {path.name}: {selected:,} selected rows', flush=True)
    frames, daily = [], []
    for i, day in enumerate(pd.date_range(start, end, freq='D')):
        path = staging / f'{day:%Y%m%d}.csv'
        if not path.exists():
            raise ValueError(f'No source observations for settlement day {day:%Y-%m-%d}')
        records = pd.read_csv(path, dtype='string')
        date = day.strftime('%Y-%m-%d')
        panel, provenance, audit = align_panel(records, date, date, direction='signed_net')
        frames.append(panel)
        daily.append(dict(settlement_date=date, **audit))
        del records, provenance
        if i % 30 == 0:
            print(f'Resolved settlement day {date}: {panel.shape[0]} GSPs', flush=True)
    panel = pd.concat(frames, axis=1).sort_index()
    del frames
    panel = ForecastDataset(panel).panel
    panel.to_csv(output / 'observed_panel.csv')
    pd.DataFrame(daily).to_csv(output / 'daily_data_audit.csv', index=False)
    monthly = panel.notna().T.groupby(panel.columns.tz_convert('Europe/London').strftime('%Y-%m')).sum()
    monthly.to_csv(output / 'observed_counts_by_month.csv')
    audit = dict(start=start, end=end, n_gsps=len(panel), n_half_hours=panel.shape[1],
                 utc_start=panel.columns[0].isoformat(), utc_end=panel.columns[-1].isoformat(),
                 observed_fraction=float(panel.notna().to_numpy().mean()),
                 sources=sources, revision_order=REVISION, direction='signed_net',
                 units='source settlement-period Meter Volume; no conversion',
                 imputation='none', availability='retrospective latest revisions; zero publication delay assumed',
                 input_rows=sum(d['input_rows'] for d in daily),
                 resolved_rows=sum(d['resolved_rows'] for d in daily))
    audit['observed_panel_sha256'] = sha256(output / 'observed_panel.csv')
    (output / 'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    (output / 'prepare_status.json').write_text(json.dumps({'status': 'complete'}))
    print(f'Prepared {panel.shape}: observed fraction {audit["observed_fraction"]:.6f}', flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', default='Data/GP9_20*/*.csv')
    parser.add_argument('--output', required=True)
    parser.add_argument('--inventory', help='Build using a previously completed inventory directory')
    parser.add_argument('--start')
    parser.add_argument('--end')
    args = parser.parse_args()
    if args.inventory:
        if not args.start or not args.end:
            parser.error('--inventory requires --start and --end')
        build_observed_panel(args.inventory, args.start, args.end, args.output)
    else:
        inventory(sorted(glob.glob(args.sources)), args.output)


if __name__ == '__main__':
    main()
