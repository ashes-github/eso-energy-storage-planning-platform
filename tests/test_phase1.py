"""Scientific regressions using independent synthetic signals and data cases."""
import unittest
import numpy as np
import pandas as pd
import torch

from eso.data import clean_panel
from eso.features import spectral_features, temporal_features, screen
from eso.models import pca_model, autoencoder_model, gplvm_model
from eso.evaluation import latent_metrics, distance_stability


def records(days=('20220101', '20220102'), gsps=('A', 'B', 'C')):
    rows = []
    for gsp in gsps:
        for day in days:
            date = pd.Timestamp(day)
            begin = date.tz_localize('Europe/London').tz_convert('UTC')
            end = (date + pd.Timedelta(days=1)).tz_localize('Europe/London').tz_convert('UTC')
            count = int((end - begin).total_seconds() / 1800)
            for p in range(1, count + 1):
                rows.append({'GSP Id': gsp, 'Settlement Date': day, 'Settlement Period': p,
                             'Meter Volume': float(p), 'Date of Aggregation': 20230101,
                             'Flow Run Date': 20230102, 'CDCA Run Number': 2})
    return pd.DataFrame(rows)


class DataTests(unittest.TestCase):
    def test_missing_middle_gsp_and_zero_exclusions(self):
        r = records()
        complete, excluded, _ = clean_panel(r, '2022-01-01', '2022-01-02')
        self.assertEqual(complete.shape, (3, 96))
        self.assertTrue(excluded.empty)
        r = r.drop(r[(r['GSP Id'] == 'B') & (r['Settlement Period'] == 2)].index)
        panel, excluded, _ = clean_panel(r, '2022-01-01', '2022-01-02')
        self.assertEqual(panel.index.tolist(), ['A', 'C'])
        np.testing.assert_equal(panel.iloc[1].values, complete.loc['C'].values)

    def test_revision_order_numeric_and_conflict(self):
        r = records()
        newer = r.iloc[[0]].copy()
        newer['CDCA Run Number'], newer['Meter Volume'] = 10, 999
        panel, _, _ = clean_panel(pd.concat([newer, r]), '2022-01-01', '2022-01-02')
        self.assertEqual(panel.loc['A'].iloc[0], 999)
        conflict = newer.copy()
        conflict['Meter Volume'] = 1000
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            clean_panel(pd.concat([r, newer, conflict]), '2022-01-01', '2022-01-02')

    def test_dst_is_continuous_not_padded(self):
        for days, count in [(('20220326', '20220327'), 94), (('20221029', '20221030'), 98)]:
            panel, _, _ = clean_panel(records(days), days[0], days[-1])
            self.assertEqual(panel.shape[1], count)
            self.assertTrue(((panel.columns[1:] - panel.columns[:-1]) == pd.Timedelta(minutes=30)).all())

    def test_invalid_period_and_constant_profile(self):
        r = records()
        r.loc[0, 'Settlement Period'] = 49
        with self.assertRaisesRegex(ValueError, 'invalid settlement'):
            clean_panel(r, '2022-01-01', '2022-01-02')
        r = records()
        r.loc[r['GSP Id'] == 'B', 'Meter Volume'] = 0
        panel, excluded, _ = clean_panel(r, '2022-01-01', '2022-01-02')
        self.assertNotIn('B', panel.index)
        self.assertTrue(excluded.loc['B', 'constant_profile'])


class FeatureTests(unittest.TestCase):
    def test_physical_band_and_scale(self):
        hours = np.arange(960) / 2
        short = spectral_features(np.sin(2 * np.pi * hours / 7), 480)
        daily = spectral_features(np.sin(2 * np.pi * hours / 24), 480)
        scaled = spectral_features(3 * np.sin(2 * np.pi * hours / 7) + 100, 480)
        self.assertGreater(short['fraction_6_8h'], 0.9)
        self.assertLess(daily['fraction_6_8h'], 0.01)
        self.assertAlmostEqual(scaled['power_6_8h'], 9 * short['power_6_8h'])
        self.assertAlmostEqual(scaled['fraction_6_8h'], short['fraction_6_8h'])

    def test_cross_midnight_and_spectral_signature_not_enough(self):
        t = pd.date_range('2022-01-01', periods=96, freq='30min', tz='UTC')
        contiguous = np.ones(96) * 10
        contiguous[40:54] = 1  # Seven hours, spanning midnight.
        broken = np.ones(96) * 10
        broken[np.arange(20, 76, 2)] = 1
        a, b = temporal_features(contiguous, t), temporal_features(broken, t)
        self.assertEqual(a['longest_observed_low_hours'], 7)
        self.assertEqual(a['qualifying_start_days'], 1)
        self.assertEqual(b['qualifying_start_days'], 0)
        f = pd.DataFrame([dict(b, fraction_20_28h=1.0)])
        result = screen(f, dict(temporal_weight=0.8, min_day_fraction=0.2, score_threshold=0.1))
        self.assertFalse(result.candidate.iloc[0])


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_heldout_values_do_not_enter_pca_inference(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(12, 10))
        train, test = x[:10], x[10:]
        observed = np.arange(7)
        a = pca_model(train, test, observed, 2, 1)[2]
        changed = test.copy()
        changed[:, 7:] = 1e8
        b = pca_model(train, changed, observed, 2, 1)[2]
        np.testing.assert_allclose(a, b)

    def test_gplvm_and_ae_holdout_and_reproducibility(self):
        x = np.random.default_rng(8).normal(size=(12, 10))
        observed = np.arange(7)
        changed = x[10:].copy()
        changed[:, 7:] = 1e8
        for kind in ['ae', 'independent', 'label_informed']:
            def fit(test):
                if kind == 'ae':
                    return autoencoder_model(x[:10], test, observed, 2, 1, 3, 3)
                return gplvm_model(x[:10], test, observed, np.arange(10) % 2,
                                   2, 1, kind, 0.5, 4, 'random', 3, 3)
            a, b = fit(x[10:]), fit(changed)
            self.assertTrue(np.isfinite(a[2]).all())
            np.testing.assert_allclose(a[0], b[0])
            np.testing.assert_allclose(a[2], b[2])

    def test_metrics_degenerate_and_rotation(self):
        z = np.random.default_rng(1).normal(size=(10, 2))
        self.assertIsNone(latent_metrics(z, np.zeros(10))['silhouette'])
        rotated = z @ np.array([[0., -1.], [1., 0.]]) + 7
        self.assertAlmostEqual(distance_stability(z, rotated), 1.0)


if __name__ == '__main__':
    unittest.main()
