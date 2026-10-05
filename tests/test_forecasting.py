import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from eso.forecasting import ForecastDataset, SeasonalNaive, WalkForward, backtest
from eso.forecasting.evaluation import summarize


def panel(n=600):
    times = pd.date_range('2022-10-25', periods=n, freq='30min', tz='UTC')
    return pd.DataFrame([np.arange(n, dtype=float), -np.arange(n, dtype=float)],
                        index=['001', 'B'], columns=times)


class ForecastTests(unittest.TestCase):
    def test_contract_and_dst(self):
        p = panel()
        self.assertEqual(ForecastDataset(p).panel.shape, (2, 600))
        for invalid in [p.iloc[:, ::2], p.set_axis(p.columns.tz_localize(None), axis=1),
                        p.set_axis(['A', 'A']), p.iloc[:, ::-1]]:
            with self.assertRaises(ValueError):
                ForecastDataset(invalid)
        p.iloc[0, 0] = np.inf
        with self.assertRaises(ValueError):
            ForecastDataset(p)

    def test_reject_retrospective_fills_and_preserve_ids(self):
        p = panel(100)
        provenance = pd.DataFrame('observed', index=p.index, columns=p.columns)
        provenance.iloc[0, 3] = 'linear'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p.to_csv(root / 'cleaned_panel.csv')
            provenance.to_csv(root / 'value_provenance.csv')
            loaded = ForecastDataset.from_phase1(root).panel
            self.assertEqual(loaded.index[0], '001')
            self.assertTrue(np.isnan(loaded.iloc[0, 3]))
            p.iloc[0, 5] = np.nan
            p.to_csv(root / 'observed_panel.csv')
            self.assertTrue(np.isnan(ForecastDataset.from_phase1(root).panel.iloc[0, 5]))

    def test_seasonal_multistep(self):
        p = panel(12)
        forecast = SeasonalNaive(3).fit(p).predict(pd.RangeIndex(8))
        np.testing.assert_array_equal(forecast.iloc[0], [9, 10, 11, 9, 10, 11, 9, 10])
        self.assertTrue(SeasonalNaive(48).fit(p).predict(pd.RangeIndex(8)).isna().all().all())

    def test_holdout_and_future_perturbation(self):
        p = panel()
        plan = WalkForward(340, p.columns[500].isoformat(), step=48)
        factories = {'daily': lambda: SeasonalNaive(48)}
        before = backtest(ForecastDataset(p), plan, factories)
        self.assertLess(before.timestamp.max(), p.columns[500])
        changed = p.copy()
        changed.iloc[:, 340:] += 10000
        after = backtest(ForecastDataset(changed), plan, factories)
        first = before.origin.min()
        np.testing.assert_array_equal(before.loc[before.origin == first, 'prediction'],
                                      after.loc[after.origin == first, 'prediction'])
        np.testing.assert_array_equal(before.loc[before.origin == first, 'mase_scale'],
                                      after.loc[after.origin == first, 'mase_scale'])
        test = backtest(ForecastDataset(p), plan, factories, 'test')
        self.assertEqual(test.origin.min(), p.columns[500])
        self.assertGreaterEqual(test.timestamp.min(), p.columns[500])
        scores = summarize(before)
        np.testing.assert_allclose(scores.mae, 48)
        np.testing.assert_allclose(scores.mase, 1)

    def test_model_receives_only_history(self):
        p = panel()
        seen = []
        class Spy(SeasonalNaive):
            def fit(self, history):
                seen.append(history.columns[-1])
                history.iloc[:, :] = 0
                return super().fit(history)
        plan = WalkForward(340, p.columns[500].isoformat())
        original = p.copy()
        backtest(ForecastDataset(p), plan, {'spy': Spy})
        pd.testing.assert_frame_equal(p, original)
        self.assertTrue(all(t < p.columns[500] for t in seen))

    def test_missing_coverage_and_zero_scale(self):
        p = panel()
        p.iloc[0, :] = 0
        p.iloc[1, 339] = np.nan
        p.iloc[0, 340] = np.nan
        plan = WalkForward(340, p.columns[500].isoformat())
        result = backtest(ForecastDataset(p), plan, {'p': SeasonalNaive})
        first = result[result.origin == result.origin.min()]
        scores = summarize(first).set_index(['horizon', 'gsp'])
        self.assertEqual(scores.loc[(8, '001'), 'n_scored'], 7)
        self.assertEqual(scores.loc[(8, 'B'), 'coverage'], 0)
        self.assertTrue(np.isnan(scores.loc[(8, '001'), 'mase']))

    def test_invalid_split_and_short_panel(self):
        p = panel()
        for plan in [WalkForward(550, p.columns[500].isoformat()),
                     WalkForward(340, '2022-10-28'),
                     WalkForward(340, p.columns[500].isoformat(), step=0)]:
            with self.assertRaises(ValueError):
                plan.folds(p.columns)


if __name__ == '__main__':
    unittest.main()
