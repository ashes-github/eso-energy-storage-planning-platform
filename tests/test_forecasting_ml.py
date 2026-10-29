"""Global-tree chronology, pooled identity, native missingness and artifacts."""
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from eso.forecasting import (ForecastDataset, GlobalTreeForecaster, SeasonalNaive,
                             WalkForward, backtest, build_training_examples)
from eso.forecasting.config import load_config, model_factories, select_series, assess_training_history
from eso.forecasting.pipeline import main


def panel(n=240):
    times = pd.date_range('2022-10-24', periods=n, freq='30min', tz='UTC')
    wave = 20 + 5 * np.sin(np.arange(n) * 2 * np.pi / 48)
    return pd.DataFrame([wave, -wave, np.zeros(n)], index=pd.Index(['001', 'B', 'ZERO'], name='gsp'),
                        columns=times)


def future_after(p, n=8):
    return pd.date_range(p.columns[-1] + pd.Timedelta('30min'), periods=n, freq='30min')


class MLConfigTests(unittest.TestCase):
    def test_presets_and_strict_options(self):
        root = Path(__file__).resolve().parents[1]
        study = load_config(root / 'configs/phase2_ml.json')
        smoke = load_config(root / 'configs/phase2_ml_smoke.json')
        self.assertIsNone(study['representative_count'])
        self.assertEqual(smoke['representative_count'], 20)
        self.assertEqual(study['models'], smoke['models'])
        self.assertIn('lightgbm', model_factories(study))
        self.assertEqual(assess_training_history(8640, study)['classification'], 'long_history_benchmark')
        study['lightgbm']['train_window'] = 672
        self.assertIn('lightgbm', assess_training_history(8640, study)['short_history_models'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            for value in [{'selection_min_observed_fraction': 1.1},
                          {'models': ['lightgbm'], 'lightgbm': {'typo': 1}},
                          {'models': ['lightgbm'], 'lightgbm': {'learning_rate': 0}},
                          {'models': ['lightgbm'], 'lightgbm': {'feature_spec': {'typo': 1}}}]:
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_eligibility_is_frozen_from_initial_coverage(self):
        p = panel()
        p.iloc[0, :30] = np.nan
        config = load_config()
        config['selection_min_observed_fraction'] = .95
        selected, audit = select_series(p, 100, config)
        self.assertEqual(selected.index.tolist(), ['B', 'ZERO'])
        self.assertFalse(audit.loc['001', 'eligible'])
        changed = p.copy()
        changed.iloc[:, 100:] = np.nan
        after, after_audit = select_series(changed, 100, config)
        self.assertEqual(selected.index.tolist(), after.index.tolist())
        pd.testing.assert_frame_equal(audit, after_audit)
        self.assertTrue(audit.loc['ZERO', 'selected'])
        config['series_ids'] = ['001']
        with self.assertRaises(ValueError):
            select_series(p, 100, config)

    def test_vectorized_backtest_case_order(self):
        p = panel()
        plan = WalkForward(100, p.columns[160].isoformat(), horizons=(8, 48))
        result = backtest(ForecastDataset(p), plan, {'persistence': SeasonalNaive})
        expected = [(h, gsp, lead, p.columns[100 + lead - 1])
                    for h in (8, 48) for gsp in p.index for lead in range(1, h + 1)]
        actual = list(result[['horizon', 'gsp', 'lead', 'timestamp']].itertuples(index=False, name=None))
        self.assertEqual(actual, expected)
        for row in result.itertuples():
            self.assertEqual(row.actual, p.at[row.gsp, row.timestamp])
            self.assertEqual(row.prediction, p.loc[row.gsp].iloc[99])


@unittest.skipUnless(importlib.util.find_spec('lightgbm'), 'Install ML extra')
class GlobalTreeTests(unittest.TestCase):
    def make_model(self, **options):
        defaults = dict(horizon=8, min_history=48, training_step=8, n_estimators=25,
                        num_leaves=7, min_child_samples=2, min_training_rows=1,
                        n_jobs=1, random_state=7)
        defaults.update(options)
        return GlobalTreeForecaster(**defaults)

    def test_one_pooled_fit_native_missing_values_and_gsp_identity(self):
        from lightgbm import LGBMRegressor
        p = panel()
        p.iloc[0, 52] = np.nan
        p.iloc[0, 55] = np.nan  # Missing origin_lag_1 at the inner origin 56.
        captured = []
        original_fit = LGBMRegressor.fit
        def spy(learner, X, y, **kwargs):
            captured.append((X.copy(), np.asarray(y).copy(), kwargs))
            return original_fit(learner, X, y, **kwargs)
        with patch.object(LGBMRegressor, 'fit', spy):
            model = self.make_model().fit(p)
        self.assertEqual(len(captured), 1)
        X, y, kwargs = captured[0]
        self.assertTrue(np.isfinite(y).all())
        self.assertTrue(X.target_lag_336.isna().all())
        self.assertTrue(X.holiday.isna().all())
        self.assertTrue(X.origin_lag_1.isna().any())
        self.assertEqual(kwargs['categorical_feature'], ['gsp_id'])
        self.assertEqual(X.gsp_id.cat.categories.tolist(), ['001', 'B', 'ZERO'])
        expected_X, expected_y = build_training_examples(p, horizon=8, min_history=48, step=8)
        self.assertEqual(len(y), expected_y.notna().sum())
        self.assertEqual(model.training_audit_['n_missing_targets'], expected_y.isna().sum())
        self.assertEqual(model.training_audit_['n_inner_origins'], expected_X.index.get_level_values('origin').nunique())
        forecast = model.predict(future_after(p))
        self.assertTrue(forecast.index.equals(p.index))
        self.assertTrue(forecast.columns.equals(future_after(p)))
        self.assertTrue(np.isfinite(forecast.to_numpy()).all())
        self.assertTrue((forecast.loc['B'] < 0).all())
        self.assertLess(pd.Timestamp(model.training_audit_['training_target_end']), future_after(p)[0])

    def test_future_perturbation_and_frozen_first_origin_categories(self):
        p = panel()
        plan = WalkForward(100, p.columns[160].isoformat(), step=100, horizons=(4, 8))
        seen = []
        def factory():
            model = self.make_model()
            seen.append(model)
            return model
        before = backtest(ForecastDataset(p), plan, {'lightgbm': factory})
        changed = p.copy()
        changed.iloc[:, 100:] += 1_000_000
        after = backtest(ForecastDataset(changed), plan, {'lightgbm': factory})
        self.assertTrue(before.prediction.notna().all())
        np.testing.assert_array_equal(before.prediction, after.prediction)
        np.testing.assert_array_equal(before.mase_scale, after.mase_scale)
        self.assertEqual(seen[0].gsp_categories_, seen[1].gsp_categories_)
        self.assertEqual(seen[0].training_audit_, seen[1].training_audit_)
        self.assertLess(before.timestamp.max(), p.columns[160])

    def test_zero_targets_valid_untrained_series_missing_and_no_fallback(self):
        p = panel()
        p.loc['UNOBSERVED'] = np.nan
        model = self.make_model().fit(p)
        result = model.predict(future_after(p))
        self.assertTrue(result.loc['UNOBSERVED'].isna().all())
        self.assertNotIn('UNOBSERVED', model.gsp_categories_)
        self.assertIn('ZERO', model.gsp_categories_)
        self.assertGreater(model.training_counts_['ZERO'], 0)
        self.assertEqual(model.diagnostics[-1]['reason'], 'no_training_targets')
        missing = self.make_model().fit(p * np.nan)
        self.assertTrue(missing.predict(future_after(p)).isna().all().all())
        short = self.make_model().fit(p.iloc[:, :50])
        self.assertEqual(short.diagnostics[0]['reason'], 'insufficient_history')

    def test_budget_guard_precedes_feature_allocation_and_window_is_explicit(self):
        p = panel()
        with patch('eso.forecasting.ml.build_training_examples') as builder:
            with self.assertRaisesRegex(ValueError, 'max_training_rows'):
                self.make_model(max_training_rows=2).fit(p)
            builder.assert_not_called()
        fitted = self.make_model(train_window=100).fit(p)
        self.assertEqual(fitted.history.shape[1], 100)
        self.assertEqual(fitted.training_audit_['history_start'], p.columns[-100].isoformat())
        with self.assertRaises(ValueError):
            fitted.predict(future_after(p, 9))

    def test_serialization_and_repeated_fits_are_reproducible(self):
        p = panel(400)
        model = self.make_model().fit(p)
        expected = model.predict(future_after(p))
        repeated = self.make_model().fit(p).predict(future_after(p))
        np.testing.assert_array_equal(expected, repeated)
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / 'model'
            model.save(saved)
            self.assertEqual(ForecastDataset.from_phase1(saved).panel.shape[1], 336)
            restored = GlobalTreeForecaster.load(saved)
            pd.testing.assert_frame_equal(expected, restored.predict(future_after(p)))
            self.assertTrue((saved / 'feature_importance.csv').exists())
            self.assertEqual(restored.training_counts_, model.training_counts_)
            self.assertEqual(restored.feature_names_, model.feature_names_)
            self.assertEqual(restored.diagnostics, model.diagnostics)

    def test_pipeline_outputs_model_schema_metrics_and_metadata(self):
        p = panel(800)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'input', root / 'run'
            source.mkdir()
            p.to_csv(source / 'observed_panel.csv')
            config = root / 'config.json'
            config.write_text(json.dumps(dict(models=['seasonal_48', 'lightgbm'], save_models=True,
                horizons=[8, 48], lightgbm=dict(n_estimators=10, n_jobs=1))), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                main(['--input', str(source), '--output', str(output), '--config', str(config),
                      '--test-start', p.columns[744].isoformat()])
            self.assertEqual(json.loads((output / 'run_status.json').read_text())['status'], 'complete')
            run = json.loads((output / 'run.json').read_text())
            self.assertIsNotNone(run['reproducibility']['packages']['lightgbm'])
            schema = next((output / 'models' / 'lightgbm').rglob('schema.json'))
            metadata = json.loads(schema.read_text())
            self.assertIn('gsp_id', metadata['feature_names'])
            self.assertIn('target_lag_48', metadata['feature_names'])
            metrics = pd.read_csv(output / 'metrics_common_cases_by_lead.csv')
            self.assertEqual(set(metrics.model), {'lightgbm', 'seasonal_48'})
            self.assertEqual(set(metrics.loc[metrics.horizon == 48, 'lead']), set(range(1, 49)))


if __name__ == '__main__':
    unittest.main()
