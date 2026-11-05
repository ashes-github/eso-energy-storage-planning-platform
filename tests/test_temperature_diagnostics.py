import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from eso.forecasting.temperature_diagnostics import (
    associations, association_study, correlation_summary, fit_spline,
    mi_diagnostic, season_derangement,
)
from eso.forecasting.temperature_eda import calendar


class TemperatureDiagnosticTests(unittest.TestCase):
    def test_rank_ties_missing_and_constant_series(self):
        got = associations([1, 2, 2, 3, np.nan], [5, 4, 4, 1, 0])
        self.assertEqual(got['n_paired'], 4)
        self.assertAlmostEqual(got['spearman'], -1)
        self.assertTrue(np.isnan(associations([1, 1, 1], [2, 3, 4])['spearman']))
        self.assertTrue(np.isnan(associations([], [])['pearson']))
        summary = correlation_summary(pd.DataFrame([
            dict(scope='overall', slice='all', pearson=-.5, spearman=-.5),
            dict(scope='overall', slice='all', pearson=np.nan, spearman=np.nan),
        ]))
        self.assertTrue(summary.n_gsps.eq(2).all())
        self.assertTrue(summary.n_defined.eq(1).all())

    def test_permutation_preserves_season_and_has_no_fixed_points(self):
        origins = pd.date_range('2022-01-01', periods=29, freq='13D', tz='UTC')
        mapping = season_derangement(origins, 0)
        self.assertEqual(mapping, season_derangement(origins, 0))
        self.assertEqual(set(mapping), set(mapping.values()))
        self.assertTrue(all(source != donor for source, donor in mapping.items()))
        seasons = calendar(origins).season
        self.assertTrue(all(seasons.loc[source] == seasons.loc[donor] for source, donor in mapping.items()))
        with self.assertRaises(ValueError):
            season_derangement(origins[:1], 0)

    def test_nonlinear_dependence_and_deterministic_mi(self):
        x = np.linspace(-3, 3, 600)
        y = np.square(x)
        self.assertAlmostEqual(associations(x, y)['pearson'], 0, places=8)
        first = mi_diagnostic(x, y, seed=7)
        self.assertEqual(first, mi_diagnostic(x, y, seed=7))
        self.assertGreater(first['mi_minus_reference'], .5)
        self.assertTrue(np.isnan(mi_diagnostic(x, np.ones(600), seed=7)['mi_nats']))

    def test_spline_recovers_curved_response_within_support(self):
        x = np.linspace(-3, 3, 600)
        result = fit_spline(pd.DataFrame(dict(gsp='A', temperature_c=x, demand_z=x*x)))
        mid = result.fitted_demand_z.iloc[len(result)//2]
        self.assertGreater(result.fitted_demand_z.iloc[0], mid + 3)
        self.assertGreater(result.fitted_demand_z.iloc[-1], mid + 3)
        self.assertGreaterEqual(result.temperature_c.min(), np.quantile(x, .01))
        self.assertLessEqual(result.temperature_c.max(), np.quantile(x, .99))

    def test_2023_is_rejected_before_analysis(self):
        times = pd.date_range('2023-01-01', periods=3, freq='30min', tz='UTC')
        demand = pd.DataFrame([[1., 2., 3.]], index=['A'], columns=times)
        weather = demand.T
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                association_study(demand, weather, Path(directory))


if __name__ == '__main__':
    unittest.main()
