import unittest
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from shapely.geometry import box

from eso.weather import aggregate_temperature, half_hourly, spatial_weights, load_regions


class WeatherTests(unittest.TestCase):
    def test_shared_labels_and_repeated_geometry_are_dissolved(self):
        geo = gpd.GeoDataFrame({'GSPs': ['A|B', 'A|B']},
                               geometry=[box(0, 50, 1, 51), box(0, 50, 1, 51)], crs=4326)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'regions.gpkg'
            geo.to_file(path, driver='GPKG')
            regions = load_regions(path)
        self.assertEqual(set(regions.index), {'A', 'B', 'A|B'})
        expected = geo.to_crs(3035).geometry.iloc[0].area
        np.testing.assert_allclose(regions.area, expected)

    def test_midpoints_cross_month_and_do_not_extrapolate_or_bridge_gaps(self):
        times = pd.to_datetime(['2021-01-31 23:00Z', '2021-02-01 00:00Z', '2021-02-01 02:00Z'])
        hourly = pd.DataFrame({'A': [2., 6., 10.]}, index=times)
        half = half_hourly(hourly)
        self.assertEqual(half.loc['2021-01-31 23:30Z', 'A'], 4.)
        self.assertEqual(half.loc['2021-02-01 00:00Z', 'A'], 6.)
        self.assertTrue(half.loc['2021-02-01 00:30Z':'2021-02-01 01:30Z', 'A'].isna().all())
        self.assertTrue(np.isnan(half.iloc[-1, 0]))

    def test_weighted_mean_missing_cells_and_no_coverage(self):
        weights = csr_matrix([[1., 3.], [0., 0.]])
        values = np.array([[[2., 6.]], [[2., np.nan]]])
        got = aggregate_temperature(values, weights)
        np.testing.assert_allclose(got[:, 0], [5., 2.])
        self.assertTrue(np.isnan(got[:, 1]).all())

    def test_spatial_partial_cell_overlap_and_sea_exclusion(self):
        regions = gpd.GeoDataFrame({'source_regions': ['A']}, geometry=[box(-.5, 49.5, 1., 50.5)], crs=4326).to_crs(3035)
        weights, coverage, _ = spatial_weights(regions, np.array([50., 51.]), np.array([0., 1.]), np.array([[True, True], [False, False]]))
        self.assertEqual(coverage.grid_cells.iloc[0], 2)
        ratio = weights.toarray()[0, 0] / weights.toarray()[0, 1]
        self.assertAlmostEqual(ratio, 2., delta=.03)


if __name__ == '__main__':
    unittest.main()
