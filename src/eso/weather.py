"""Area-weighted GSP temperatures and explicitly interpolated half-hours."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from shapely.geometry import box
import xarray as xr


@contextmanager
def temperature_file(path):
    """Support NetCDF and the single-NetCDF ZIP supplied by CDS."""
    with tempfile.TemporaryDirectory() as directory:
        target = path
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                members = [n for n in archive.namelist() if n.endswith('.nc')]
                if len(members) != 1:
                    raise ValueError(f'{path}: expected exactly one NetCDF member')
                target = Path(directory) / 'temperature.nc'
                with archive.open(members[0]) as source, target.open('wb') as dest:
                    import shutil
                    shutil.copyfileobj(source, dest)
        with xr.open_dataset(target, engine='netcdf4') as dataset:
            yield dataset


def load_regions(path):
    geo = gpd.read_file(path)
    if geo.crs is None or 'GSPs' not in geo or geo.GSPs.isna().any():
        raise ValueError('Boundaries require a CRS and nonmissing GSPs identifiers')
    if geo.geometry.is_empty.any() or (~geo.is_valid).any():
        raise ValueError('Boundary geometries must be nonempty and valid')
    # Preserve original composite labels AND map their explicitly listed aliases.
    rows = []
    for row in geo.itertuples():
        label = str(row.GSPs).strip()
        for identifier in dict.fromkeys([label, *label.split('|')]):
            rows.append({'gsp_id': identifier.strip(), 'source_region': label,
                         'geometry': row.geometry})
    expanded = gpd.GeoDataFrame(rows, crs=geo.crs)
    sources = expanded.groupby('gsp_id').source_region.agg(lambda x: ';'.join(sorted(set(x))))
    regions = expanded[['gsp_id', 'geometry']].dissolve('gsp_id').sort_index()
    regions['source_regions'] = sources
    return regions.to_crs(3035)  # European equal-area CRS; areas in square metres.


def spatial_weights(regions, latitude, longitude, valid_land):
    """Intersect regular-grid footprints with polygons, excluding sea cells.

    Piecewise-constant grid footprints are an approximation to a continuous
    weather field. No nearest-cell substitution is made for uncovered regions.
    """
    lat = np.asarray(latitude)
    lon = np.asarray(longitude)
    if len(lat) < 2 or len(lon) < 2:
        raise ValueError('Require at least two coordinates on each grid axis')
    if not np.allclose(np.diff(lat), np.diff(lat)[0]) or not np.allclose(np.diff(lon), np.diff(lon)[0]):
        raise ValueError('Require a regular latitude/longitude grid')
    dy, dx = abs(np.diff(lat)[0]) / 2, abs(np.diff(lon)[0]) / 2
    flat = np.flatnonzero(valid_land.ravel())
    iy, ix = np.unravel_index(flat, valid_land.shape)
    cells = gpd.GeoSeries([box(lon[x]-dx, lat[y]-dy, lon[x]+dx, lat[y]+dy)
                          for y, x in zip(iy, ix)], crs=4326).to_crs(regions.crs)
    records = []
    rr, cc, aa = [], [], []
    for r, (gsp, region) in enumerate(regions.iterrows()):
        candidates = cells.sindex.query(region.geometry, predicate='intersects')
        for c in candidates:
            area = region.geometry.intersection(cells.iloc[c]).area
            if area > 0:
                rr.append(r)
                cc.append(int(flat[c]))
                aa.append(area)
                records.append({'gsp_id': gsp, 'cell_index': int(flat[c]),
                                'latitude': float(lat[iy[c]]), 'longitude': float(lon[ix[c]]),
                                'overlap_area_m2': area})
    matrix = csr_matrix((aa, (rr, cc)), shape=(len(regions), valid_land.size))
    coverage = regions[['source_regions']].copy()
    coverage['region_area_m2'] = regions.area
    coverage['valid_grid_overlap_m2'] = np.asarray(matrix.sum(axis=1)).ravel()
    coverage['grid_area_coverage_fraction'] = coverage.valid_grid_overlap_m2 / coverage.region_area_m2
    coverage['grid_cells'] = np.diff(matrix.indptr)
    coverage['spatial_method'] = np.where(coverage.grid_cells > 0, 'area_weighted_land_grid', 'no_land_grid_overlap')
    return matrix, coverage, pd.DataFrame(records)


def aggregate_temperature(values, weights):
    """Re-normalize weights over finite cells at each timestamp."""
    flat = np.asarray(values).reshape(len(values), -1)
    finite = np.isfinite(flat)
    numerator = (weights @ np.where(finite, flat, 0).T).T
    denominator = (weights @ finite.astype(float).T).T
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def half_hourly(hourly):
    """Retain exact hours; midpoint requires BOTH adjacent observations.

    Includes final :30 as missing. Never extrapolates or bridges missing hours.
    """
    if hourly.index.has_duplicates or not hourly.index.is_monotonic_increasing:
        raise ValueError('Hourly timestamps must be unique and increasing')
    if hourly.index.tz is None or (hourly.index.minute != 0).any() or (hourly.index.second != 0).any():
        raise ValueError('Require timezone-aware exact hourly timestamps')
    hourly = hourly.reindex(pd.date_range(hourly.index[0], hourly.index[-1], freq='h'))
    index = pd.date_range(hourly.index[0], hourly.index[-1] + pd.Timedelta('30min'), freq='30min')
    values = np.full((len(index), hourly.shape[1]), np.nan, dtype=np.float32)
    values[::2] = hourly.to_numpy()
    values[1::2] = ((hourly + hourly.shift(-1)) / 2).to_numpy()
    return pd.DataFrame(values, index=index.rename('timestamp_utc'), columns=hourly.columns)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('Data/era5_land_temperature'))
    parser.add_argument('--boundaries', type=Path, default=Path('Data/gsp_regions_20220314/GSP_regions_20220314.shp'))
    parser.add_argument('--output-dir', type=Path, default=Path('Data/gsp_temperature'))
    parser.add_argument('--demand-panel', type=Path, default=Path('output/phase2c1_observed_2021_2023/observed_panel.csv'))
    args = parser.parse_args(argv)
    files = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in {'.nc', '.zip'})
    if not files:
        raise ValueError('No monthly NetCDF/ZIP files found')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    regions = load_regions(args.boundaries)
    frames, inventory = [], []
    weights = None
    for path in files:
        print(f'Aggregating {path.name}', flush=True)
        with temperature_file(path) as dataset:
            data = dataset['t2m']
            time_dim = 'valid_time' if 'valid_time' in data.dims else 'time'
            data = data.transpose(time_dim, 'latitude', 'longitude')
            if data.attrs.get('units') != 'K':
                raise ValueError(f'{path}: expected Kelvin temperature')
            lat, lon = data.latitude.values, data.longitude.values
            values = data.values
            land = np.isfinite(values).any(axis=0)
            if weights is None:
                reference_lat, reference_lon, reference_land = lat.copy(), lon.copy(), land.copy()
                weights, coverage, detail = spatial_weights(regions, lat, lon, land)
                coverage.to_csv(args.output_dir / 'spatial_coverage.csv', index_label='gsp_id')
                detail.to_parquet(args.output_dir / 'grid_weights.parquet', index=False)
                print(f'{len(regions)} GSP identifiers; {int((coverage.grid_cells == 0).sum())} without land-grid overlap', flush=True)
            elif not (np.array_equal(lat, reference_lat) and np.array_equal(lon, reference_lon) and np.array_equal(land, reference_land)):
                raise ValueError(f'{path}: grid or land mask changed; refusing inconsistent weights')
            index = pd.DatetimeIndex(pd.to_datetime(data[time_dim].values, utc=True))
            frame = pd.DataFrame(aggregate_temperature(values - 273.15, weights).astype('float32'),
                                 index=index, columns=regions.index)
            frames.append(frame)
            inventory.append({'file': path.name, 'hours': len(frame), 'start': str(index.min()), 'end': str(index.max())})
    hourly = pd.concat(frames).sort_index()
    if hourly.index.has_duplicates:
        raise ValueError('Overlapping input timestamps; remove duplicate monthly inputs')
    complete_hours = pd.date_range(hourly.index.min(), hourly.index.max(), freq='h')
    missing_hours = complete_hours.difference(hourly.index)
    hourly = hourly.reindex(complete_hours)
    hourly.index.name = 'timestamp_utc'
    hourly.to_parquet(args.output_dir / 'temperature_hourly_c.parquet')
    half = half_hourly(hourly)
    half.to_parquet(args.output_dir / 'temperature_halfhourly_c.parquet')
    monthly = args.output_dir / 'halfhourly_by_month'
    monthly.mkdir(exist_ok=True)
    for month, block in half.groupby(half.index.strftime('%Y-%m')):
        long = block.rename_axis(columns='gsp_id').stack(future_stack=True).rename('temperature_c').reset_index()
        long['temporal_method'] = np.where(long.timestamp_utc.dt.minute == 0, 'hourly_reanalysis', 'linear_midpoint')
        long.loc[long.temperature_c.isna(), 'temporal_method'] = 'missing'
        long.to_parquet(monthly / f'temperature_{month}.parquet', index=False)
    unmatched = []
    if args.demand_panel.exists():
        demand = pd.read_csv(args.demand_panel, usecols=[0], dtype=str).iloc[:, 0]
        matching = pd.DataFrame({'gsp_id': demand})
        matching['boundary_match'] = matching.gsp_id.isin(regions.index)
        matching['has_temperature'] = matching.gsp_id.isin(coverage.index[coverage.grid_cells > 0])
        matching.to_csv(args.output_dir / 'demand_gsp_matching.csv', index=False)
        unmatched = matching.loc[~matching.has_temperature, 'gsp_id'].tolist()
    summary = {'units': 'degrees Celsius', 'timezone': 'UTC', 'timestamp_meaning': 'instantaneous value at interval start, not interval mean',
               'spatial_method': 'grid-cell footprint intersections in EPSG:3035, renormalized over valid land cells',
               'temporal_method': 'hourly retained; :30 = mean of adjacent hours; no extrapolation or missing-hour fills',
               'forecast_policy': 'historical reanalysis; future values are oracle covariates, not origin-available weather forecasts',
               'boundary_file': str(args.boundaries), 'gsp_count': len(regions), 'hourly_timestamps': len(hourly),
               'halfhourly_timestamps': len(half), 'start': str(half.index.min()), 'end': str(half.index.max()),
               'missing_input_hours': [str(t) for t in missing_hours], 'missing_halfhourly_values': int(half.isna().sum().sum()),
               'unmatched_demand_gsps': unmatched, 'input_files': inventory}
    (args.output_dir / 'metadata.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'input_files'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
