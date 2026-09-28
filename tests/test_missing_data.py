import unittest
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from eso.data import align_panel, load_records
from eso.features import screen_with_quality, temporal_features, screen
from eso.imputation import impute_panel
from eso.pipeline import run, ROOT
from test_phase1 import records


FEATURE_CONFIG = {'nperseg': 96, 'temporal': {'low_fraction': 0.35, 'min_hours': 6,
                  'following_hours': 8, 'min_peak_fraction': 0.75}}
SCREEN_CONFIG = {'temporal_weight': 0.8, 'min_day_fraction': 0.2,
                 'score_threshold': 0.25, 'allow_signed_exports': True}


def daily_panel(days=5):
    times = pd.date_range('2022-06-20', periods=days*48, freq='30min', tz='Europe/London').tz_convert('UTC')
    y = 20 + 10*np.cos(2*np.pi*np.arange(days*48)/48)
    return pd.DataFrame([y], index=pd.Index(['A'], name='GSP Id'), columns=times)


class DirectionTests(unittest.TestCase):
    def test_exports_are_observations_not_gaps(self):
        r = records()
        r['Import/Export Indicator'] = 'I'
        r.loc[(r['GSP Id'] == 'B') & (r['Settlement Period'] < 20), 'Import/Export Indicator'] = 'E'
        r.loc[r['GSP Id'] == 'C', 'Import/Export Indicator'] = 'E'
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'publication_2023.csv'
            r.to_csv(path, index=False)
            loaded, _ = load_records([path], '2022-01-01', '2022-01-02', 'signed_net')
        p, provenance, audit = align_panel(loaded, '2022-01-01', '2022-01-02', 'signed_net')
        self.assertEqual(p.shape, (3, 96))
        self.assertEqual(audit['genuinely_missing_periods'], 0)
        self.assertEqual(p.loc['B'].iloc[0], -1)
        self.assertEqual(provenance.loc['B'].iloc[0], 'observed_export')
        self.assertTrue((p.loc['C'] < 0).all())

    def test_direction_revision_and_conflict(self):
        r = records()
        r['Import/Export Indicator'] = 'I'
        updated = r.iloc[[0]].copy()
        updated['Import/Export Indicator'] = 'E'
        updated['CDCA Run Number'] = 10
        p, _, _ = align_panel(pd.concat([r, updated]), '2022-01-01', '2022-01-02', 'signed_net')
        self.assertEqual(p.iloc[0, 0], -1)
        updated['CDCA Run Number'] = 2
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            align_panel(pd.concat([r, updated]), '2022-01-01', '2022-01-02', 'signed_net')

    def test_missing_value_is_a_gap_and_unsigned_guard(self):
        r = records()
        r['Import/Export Indicator'] = 'I'
        r.loc[0, 'Meter Volume'] = np.nan
        p, source, audit = align_panel(r, '2022-01-01', '2022-01-02', 'signed_net')
        self.assertEqual(audit['genuinely_missing_periods'], 1)
        self.assertEqual(source.iloc[0, 0], 'missing')
        r.loc[0, 'Meter Volume'] = -1
        with self.assertRaisesRegex(ValueError, 'unsigned'):
            align_panel(r, '2022-01-01', '2022-01-02', 'signed_net')


class ImputationTests(unittest.TestCase):
    def test_short_gap_preserves_every_observation(self):
        p = daily_panel()
        truth = p.copy()
        p.iloc[0, 20:22] = np.nan
        filled, source, q = impute_panel(p)
        np.testing.assert_allclose(filled.iloc[0, 20:22], np.linspace(truth.iloc[0, 19], truth.iloc[0, 22], 4)[1:-1])
        observed = p.notna().to_numpy()
        np.testing.assert_array_equal(filled.to_numpy()[observed], p.to_numpy()[observed])
        self.assertEqual(q.loc['A', 'imputed_periods'], 2)
        self.assertEqual(source.iloc[0, 20], 'linear')

    def test_seasonal_gap_uses_original_same_clock_days(self):
        p = daily_panel()
        truth = p.copy()
        p.iloc[0, 60:66] = np.nan
        filled, source, q = impute_panel(p)
        np.testing.assert_allclose(filled, truth, atol=1e-12)
        self.assertEqual(source.iloc[0, 60], 'seasonal_median')
        self.assertEqual(q.loc['A', 'data_status'], 'imputed')

    def test_sparse_and_long_gap_stay_unknown(self):
        p = daily_panel()
        p.loc['sparse'] = np.nan
        p.loc['sparse', p.columns[0]] = 1
        p.iloc[0, 40:60] = np.nan
        filled, _, q = impute_panel(p)
        result = screen_with_quality(filled, q, FEATURE_CONFIG, SCREEN_CONFIG)
        self.assertTrue(result.candidate.isna().all())
        self.assertTrue(result.screening_status.eq('insufficient_evidence').all())
        self.assertEqual(q.loc['sparse', 'imputed_periods'], 0)

    def test_no_recursive_donors_or_weekday_weekend_mix(self):
        p = daily_panel(days=3)
        p.iloc[0, 10:16] = np.nan
        p.iloc[0, 58:64] = np.nan
        filled, _, q = impute_panel(p)
        self.assertEqual(q.loc['A', 'remaining_missing_periods'], 12)
        p = daily_panel(days=7)
        # First Saturday: weekday donors must not count; only one Sunday donor.
        p.iloc[0, 5*48+10:5*48+16] = np.nan
        _, _, q = impute_panel(p)
        self.assertEqual(q.loc['A', 'remaining_missing_periods'], 6)

    def test_constant_retained_and_imputed_labels_marked(self):
        p = daily_panel()
        p.loc['constant'] = 0
        p.iloc[0, 60:66] = np.nan
        filled, _, q = impute_panel(p)
        result = screen_with_quality(filled, q, FEATURE_CONFIG, SCREEN_CONFIG)
        self.assertTrue(q.loc['constant', 'model_eligible'])
        self.assertFalse(result.loc['constant', 'candidate'])
        self.assertTrue(result.loc['A', 'screening_status'].endswith('_imputed'))

    def test_export_trough_can_qualify_but_export_only_cannot(self):
        p = daily_panel(days=2)
        y = np.full(96, 10.0)
        y[40:54] = -2
        a = temporal_features(y, p.columns)
        self.assertEqual(a['qualifying_start_days'], 1)
        f = pd.DataFrame([dict(a, fraction_20_28h=1.)])
        self.assertTrue(screen(f, SCREEN_CONFIG).candidate.iloc[0])
        self.assertEqual(temporal_features(y-20, p.columns)['qualifying_start_days'], 0)


class PipelineMissingTests(unittest.TestCase):
    def test_all_gsps_reported_and_holdout_never_uses_imputed_truth(self):
        cfg = json.loads((ROOT / 'configs/phase1.json').read_text())
        rows = []
        for g in range(14):
            for d in range(5):
                for period in range(1, 49):
                    if g == 12 and d == 2 and period in [20, 21]:
                        continue
                    if g == 13 and (d != 0 or period != 1):
                        continue
                    volume = 15 + g + 10*np.cos(2*np.pi*(period+g)/48)
                    rows.append({'GSP Id': f'G{g}', 'Settlement Date': f'202206{20+d}',
                                 'Settlement Period': period, 'Meter Volume': volume,
                                 'Import/Export Indicator': 'I', 'Date of Aggregation': 20230101,
                                 'Flow Run Date': 20230102, 'CDCA Run Number': 1})
        (ROOT / 'tmp').mkdir(exist_ok=True)
        with TemporaryDirectory(dir=ROOT / 'tmp') as temp:
            path = Path(temp)
            pd.DataFrame(rows).to_csv(path / 'fixture.csv', index=False)
            cfg.update(input_glob=(path / 'fixture.csv').relative_to(ROOT).as_posix(),
                       shapefile=None, seeds=[1], prior_modes=['independent'], prior_scales=[1.0],
                       inducing_strategies=['random'], inducing_count=4, steps=2, ae_steps=2,
                       infer_steps=2)
            with contextlib.redirect_stdout(io.StringIO()):
                run(cfg, path / 'result')
            result = pd.read_csv(path / 'result/screening.csv', index_col=0)
            split = json.loads((path / 'result/split.json').read_text())
            self.assertEqual(len(result), 14)
            self.assertTrue(pd.isna(result.loc['G13', 'candidate']))
            self.assertEqual(result.loc['G13', 'screening_status'], 'insufficient_evidence')
            self.assertEqual(result.loc['G12', 'data_status'], 'imputed')
            self.assertIn('G12', split['train_gsps'])
            self.assertNotIn('G12', split['test_gsps'])
            self.assertNotIn('G13', split['train_gsps'] + split['test_gsps'])


if __name__ == '__main__':
    unittest.main()
