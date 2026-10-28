"""Statistical benchmark regressions: chronology, gaps, failures and comparisons."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from eso.forecasting import ForecastDataset, StatisticalForecaster, WalkForward, backtest
from eso.forecasting.config import load_config, select_series
from eso.forecasting.evaluation import common_cases, summarize
from eso.forecasting.pipeline import main


def seasonal_panel(n=180):
    rng = np.random.default_rng(42)
    values = 20 + np.tile([0, 2, -1, 4], (n + 3) // 4)[:n] + rng.normal(0, .2, n)
    return pd.DataFrame([values, -values], index=['001', 'B'],
                        columns=pd.date_range('2022-10-28', periods=n, freq='30min', tz='UTC'))


class ComparisonTests(unittest.TestCase):
    def test_common_case_skipping_hard_targets_and_lead_metrics(self):
        rows = []
        for model, predictions in [('A', [1., 10.]), ('B', [1., np.nan])]:
            for lead, prediction in enumerate(predictions, 1):
                rows.append(dict(model=model, split='validation', origin='t0', timestamp=f't{lead}',
                                 gsp='001', horizon=2, lead=lead, actual=0.,
                                 prediction=prediction, mase_scale=2.))
        frame = pd.DataFrame(rows)
        raw = summarize(frame, ('model',)).set_index('model')
        self.assertEqual(raw.loc['A', 'mae'], 5.5)
        matched = summarize(common_cases(frame), ('model',)).set_index('model')
        np.testing.assert_array_equal(matched.mae, [1., 1.])
        np.testing.assert_array_equal(matched.coverage, [.5, .5])
        leads = summarize(frame, ('model', 'lead')).set_index(['model', 'lead'])
        self.assertEqual(leads.loc[('A', 2), 'mae'], 10.)
        self.assertEqual(leads.loc[('B', 2), 'n_scored'], 0)
        with self.assertRaises(ValueError):
            common_cases(pd.concat([frame, frame.iloc[:1]]))
        changed = frame.copy()
        changed.loc[2, 'actual'] = 100
        with self.assertRaises(ValueError):
            common_cases(changed)
        # A missing row from B must not make A's difficult case comparable.
        missing = common_cases(frame.iloc[:3])
        self.assertTrue(np.isnan(missing.loc[1, 'prediction']))

    def test_history_only_representative_selection(self):
        p = seasonal_panel()
        p = pd.concat([p, p.rename(index={'001': 'C', 'B': 'D'}) * 10])
        config = load_config()
        config['representative_count'] = 3
        chosen, audit = select_series(p, 80, config)
        changed = p.copy()
        changed.iloc[:, 80:] *= -10000
        after, after_audit = select_series(changed, 80, config)
        self.assertEqual(chosen.index.tolist(), after.index.tolist())
        pd.testing.assert_frame_equal(audit, after_audit)
        self.assertEqual(set(audit.loc[audit.selected, 'stratum']), {'low', 'medium', 'high'})

    def test_config_and_output_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = root / 'config.json'
            cfg.write_text(json.dumps({'models': ['unknown']}), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_config(cfg)
            cfg.write_text(json.dumps({'sarima': {'unknown': 3}}), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_config(cfg)
            with self.assertRaises(FileExistsError):
                main(['--input', 'missing', '--output', directory, '--test-start', '2022-01-01Z'])


@unittest.skipUnless(importlib.util.find_spec('statsmodels'), 'Install statistical extra')
class StatisticalTests(unittest.TestCase):
    def factory(self, kind):
        return StatisticalForecaster(kind, order=(1, 0, 0), seasonal_order=(0, 1, 0, 4),
                                     seasonal_periods=4, min_observations=40, maxiter=300)

    def test_signed_units_axes_and_saved_models(self):
        p = seasonal_panel()
        future = pd.date_range(p.columns[-1] + pd.Timedelta('30min'), periods=8, freq='30min')
        for kind in ('sarima', 'ets'):
            with self.subTest(kind=kind):
                model = self.factory(kind).fit(p)
                forecast = model.predict(future)
                self.assertTrue(forecast.notna().all().all(), model.diagnostics)
                self.assertTrue((forecast.loc['001'] > 0).all())
                self.assertTrue((forecast.loc['B'] < 0).all())
                self.assertTrue(forecast.index.equals(p.index))
                self.assertTrue(forecast.columns.equals(future))
                scaled = self.factory(kind).fit(p * 1000).predict(future)
                np.testing.assert_allclose(scaled / 1000, forecast, rtol=1e-5, atol=1e-5)
                with tempfile.TemporaryDirectory() as directory:
                    destination = Path(directory) / 'models'
                    model.save(destination)
                    from statsmodels.iolib.smpickle import load_pickle
                    manifest = json.loads((destination / 'manifest.json').read_text())
                    saved = load_pickle(destination / manifest[0]['file'])
                    restored = np.asarray(saved.forecast(8)) * manifest[0]['scale_factor']
                    np.testing.assert_allclose(restored, forecast.iloc[0])

    def test_missing_policies_and_trailing_window(self):
        p = seasonal_panel()
        p.iloc[0, 5] = np.nan
        future = pd.date_range(p.columns[-1] + pd.Timedelta('30min'), periods=8, freq='30min')
        ets = self.factory('ets').fit(p)
        result = ets.predict(future)
        self.assertTrue(result.loc['001'].isna().all())
        self.assertEqual(ets.diagnostics[0]['reason'], 'missing_training_values')
        sarima = self.factory('sarima').fit(p)
        self.assertTrue(sarima.predict(future).loc['001'].notna().all(), sarima.diagnostics)
        trailing = StatisticalForecaster('ets', seasonal_periods=4, min_observations=40,
                                         train_window=80, maxiter=300).fit(p)
        self.assertTrue(trailing.predict(future).notna().all().all())
        self.assertEqual(trailing.diagnostics[0]['n_train'], 80)
        short = self.factory('ets').fit(p.iloc[:, :20])
        self.assertEqual(short.diagnostics[0]['reason'], 'insufficient_observations')

    def test_future_perturbation_in_statistical_backtest(self):
        p = seasonal_panel()
        plan = WalkForward(80, p.columns[140].isoformat(), step=100, horizons=(8, 12), scale_period=4)
        factories = {kind: lambda kind=kind: self.factory(kind) for kind in ('sarima', 'ets')}
        before = backtest(ForecastDataset(p), plan, factories)
        self.assertTrue(before.prediction.notna().all())
        changed = p.copy()
        changed.iloc[:, 80:] += 10000
        after = backtest(ForecastDataset(changed), plan, factories)
        np.testing.assert_allclose(before.prediction, after.prediction, equal_nan=True)
        np.testing.assert_allclose(before.mase_scale, after.mase_scale, equal_nan=True)
        self.assertLess(before.timestamp.max(), p.columns[140])

    def test_nonconvergence_is_missing_not_fallback(self):
        p = seasonal_panel()
        future = pd.date_range(p.columns[-1] + pd.Timedelta('30min'), periods=8, freq='30min')
        with patch('statsmodels.tsa.exponential_smoothing.ets.ETSModel.fit') as fit:
            fit.return_value.mle_retvals = {'converged': False}
            model = self.factory('ets').fit(p)
        self.assertTrue(model.predict(future).isna().all().all())
        self.assertTrue(all(d['reason'] == 'nonconvergence' for d in model.diagnostics))

    def test_pipeline_artifacts(self):
        p = seasonal_panel(800)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'input', root / 'run'
            source.mkdir()
            p.to_csv(source / 'observed_panel.csv')
            config = root / 'config.json'
            config.write_text(json.dumps(dict(models=['persistence', 'sarima', 'ets'],
                save_models=True, sarima=dict(seasonal_order=[0, 1, 0, 4]),
                ets=dict(seasonal_periods=4))), encoding='utf-8')
            main(['--input', str(source), '--output', str(output), '--config', str(config),
                  '--test-start', p.columns[744].isoformat()])
            self.assertEqual(json.loads((output / 'run_status.json').read_text())['status'], 'complete')
            self.assertEqual(len(pd.read_csv(output / 'fit_diagnostics.csv')), 4)
            self.assertTrue((output / 'metrics_common_cases_by_lead.csv').exists())
            metadata = json.loads((output / 'run.json').read_text())
            self.assertIn('observed_panel.csv', metadata['reproducibility']['input_sha256'])
            self.assertEqual(metadata['selected_series'], ['001', 'B'])
            self.assertTrue(list((output / 'models').rglob('manifest.json')))


if __name__ == '__main__':
    unittest.main()
