# GSP temperature covariates

Run from the project root with the project environment:

```powershell
.\venv\python.exe src\prepare_gsp_temperature.py
```

Dependencies are available as the `weather` optional dependency in
`pyproject.toml` (`xarray`, `netCDF4`, `pyarrow`), alongside the project's
geopandas, pandas, numpy and scipy dependencies.

## Inputs and spatial method

Monthly files in `Data/era5_land_temperature` may be NetCDF or ZIP archives
containing a single NetCDF. The variable must be `t2m` in Kelvin. Files are
processed one at a time, converted to Celsius, and aggregated before combining
months. Duplicate timestamps or inconsistent grids/land masks cause an error.

Use `Data/gsp_regions_20220314/GSP_regions_20220314.shp`. This has matching
`.dbf`, `.shx`, and `.prj` sidecars; the separately named `GSP region.shp`
does not have the same complete set of sidecars under that name.

The spatial average weights each 0.1 degree grid-cell footprint by its
intersection area with the region in the European equal-area CRS EPSG:3035.
This approximates a continuous temperature field as constant over each cell;
it is a geographical average, not population- or demand-weighted temperature.
Sea/missing cells are excluded and weights renormalized over finite values.
`grid_area_coverage_fraction` records the fraction of polygon area covered by
valid grid-cell footprints, so coastal regions can have partial coverage.
Regions without any land-grid overlap stay missing; no nearest-region guess
is made.

Repeated `GSPs` labels are dissolved into a single geometry. Shared labels such
as `HARK_1|HUTT_1|RRIG` are retained and also explicitly assigned to each listed
identifier. These aliases receive the same shared-region temperature, not
separately localized estimates. Arbitrary name similarities are not matched.
The source labels and coverage are exported for audit.

## Time interpretation

Hourly temperature is an instantaneous field, not an hourly accumulation.
See [ECMWF's time and parameter definitions](https://confluence.ecmwf.int/pages/viewpage.action?pageId=111150967).
The output is a point-in-time covariate attached to a UTC interval start:

```text
00:00 = original hourly temperature at 00:00
00:30 = (temperature at 00:00 + temperature at 01:00) / 2
01:00 = original hourly temperature at 01:00
```

These are interpolated estimates, not new observed half-hourly measurements
or means over settlement intervals. Interpolation spans monthly boundaries
after concatenation. Both adjacent hourly values must exist; missing hours
are not bridged. The final 2023-03-31 23:30 UTC is missing because
2023-04-01 00:00 UTC was not supplied. Downloading that additional hour would
allow its interpolation. UTC prevents daylight-saving duplicate/gap ambiguity;
the forecasting demand panels already use UTC interval starts.

## Outputs

All outputs are in `Data/gsp_temperature`:

- `temperature_halfhourly_c.parquet`: timestamp rows, GSP columns, Celsius.
- `temperature_hourly_c.parquet`: original hourly regional averages.
- `halfhourly_by_month/temperature_YYYY-MM.parquet`: join-ready rows with
  `timestamp_utc`, `gsp_id`, `temperature_c`, and `temporal_method`.
- `spatial_coverage.csv`: source labels, polygon/grid overlap, cell counts.
- `grid_weights.parquet`: contributing cells and intersection areas.
- `demand_gsp_matching.csv`: matching against the supplied demand panel.
- `metadata.json`: methods, input inventory, missingness, unmatched demand IDs.

Parquet preserves timezone and numeric types while avoiding a large text CSV.

```python
import pandas as pd

temperature = pd.read_parquet(
    'Data/gsp_temperature/temperature_halfhourly_c.parquet'
)
print(temperature.loc['2022-01-01':'2022-01-02'].iloc[:, :3])

# Same orientation as ForecastDataset.panel: GSP rows, UTC timestamp columns.
temperature_panel = temperature.T
# demand_panel is your existing ForecastDataset.panel DataFrame:
# aligned = temperature_panel.reindex(
#     index=demand_panel.index, columns=demand_panel.columns
# )
# Unknown GSPs and unavailable times remain NaN.
```

For a long demand table, merge on GSP identifier and UTC interval start with
`validate='many_to_one'`. Inspect `demand_gsp_matching.csv` before modelling;
some demand identifiers have no boundary match and cannot safely be assigned
a region using these files alone.

The supplied files produce 19,680 hourly / 39,360 half-hourly timestamps and
388 identifiers (including original shared labels and their explicit aliases).
Of 367 demand IDs in `output/phase2c1_observed_2021_2023/observed_panel.csv`,
357 have temperature. `HUER` has no land-grid overlap. Nine have no boundary
match: `AXMI_H`, `GLAG_P`, `HARG_P`, `ROTI3`, `SAFO_1`, `STOR_P`, `TONG1`,
`TOWH`, `WESA_H`. All other values are present except the final half-hour.
The smallest nonzero grid coverage is about 4.2%; consult the coverage report
before treating a partial-coverage region as representative of its full area.

## Forecast availability

ERA5-Land is historical reanalysis. Future-period temperatures are not weather
forecasts available at an earlier forecasting origin. Using them for the
forecast horizon is an oracle/perfect-weather experiment and must be labelled
as such. Operational evaluation needs weather forecasts issued by the origin.
For historical weather features, account for source publication availability;
an interpolated `:30` value also uses the following hour. This export does not
automatically inject weather into the existing forecasting model or assume
that observations were available in real time.
