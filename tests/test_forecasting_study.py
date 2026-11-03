"""Bounded historical preparation, streaming case integrity and study metrics."""
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from eso.data import align_panel, load_records
from eso.forecasting.evaluation import summarize
from eso.forecasting.prepare import inventory, build_observed_panel
from eso.forecasting.study import run
from eso.forecasting.study_analysis import sufficient_stats, finish_stats, origin_frames, benefit_table, analyse_run


class StudyTests(unittest.TestCase):
    def test_partitioned_preparation_matches_existing_revision_policy(self):
        rows = [dict(**{'GSP Id': gsp, 'Settlement Date': date, 'Settlement Period': str(period),
                        'Date of Aggregation': '20210105', 'Flow Run Date': '20210106',
                        'CDCA Run Number': '2', 'Meter Volume': str(period),
                        'Import/Export Indicator': flag, 'Estimate Indicator': 'F'})
                for date in ('20210101', '20210102') for gsp, flag in [('001', 'I'), ('B', 'E')]
                for period in range(1, 49)]
        older = dict(rows[0], **{'Flow Run Date': '20210104', 'Meter Volume': '999'})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / 'a.csv', root / 'b.csv'
            pd.DataFrame([older, *rows[:96]]).to_csv(a, index=False)
            pd.DataFrame(rows[96:]).iloc[:, ::-1].to_csv(b, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                inventory([a, b], root / 'inventory', chunksize=30)
                build_observed_panel(root / 'inventory', '2021-01-01', '2021-01-02', root / 'panel', chunksize=25)
            records, _ = load_records([a, b], '2021-01-01', '2021-01-02', direction='signed_net')
            expected, _, _ = align_panel(records, '2021-01-01', '2021-01-02', direction='signed_net')
            from eso.forecasting import ForecastDataset
            actual = ForecastDataset.from_phase1(root / 'panel').panel
            expected = ForecastDataset(expected).panel
            actual.columns.name = expected.columns.name
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_index_type=False)
            self.assertEqual(actual.loc['001'].iloc[0], 1)
            self.assertEqual(actual.loc['B'].iloc[0], -1)
            with self.assertRaises(FileExistsError):
                build_observed_panel(root / 'inventory', '2021-01-01', '2021-01-02', root / 'panel')

    def test_stats_combine_squared_errors_not_root_means(self):
        frame = pd.DataFrame(dict(model=['a'] * 4, split=['validation'] * 4, horizon=[8] * 4,
                                  actual=[0., 0., np.nan, 0.], prediction=[1., 9., 2., np.nan],
                                  mase_scale=[2., 2., np.nan, 2.]))
        keys = ('model', 'split', 'horizon')
        result = finish_stats([sufficient_stats(frame.iloc[:1], keys), sufficient_stats(frame.iloc[1:], keys)], keys)
        expected = summarize(frame, keys)
        for column in ('mae', 'rmse', 'mase', 'coverage', 'n_scored', 'n_observed', 'n_mase'):
            np.testing.assert_allclose(result[column], expected[column], equal_nan=True)
        self.assertAlmostEqual(result.rmse.iloc[0], np.sqrt(41))

    def test_origin_chunk_carry_and_zero_baseline_benefits(self):
        frame = pd.DataFrame(dict(origin=pd.to_datetime(['2022-01-01T00:00Z'] * 7 + ['2022-01-02T00:00Z'] * 5),
                                  timestamp=pd.to_datetime(['2022-01-01T00:00Z'] * 12),
                                  gsp=['001'] * 12, prediction=np.arange(12)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.csv'
            frame.to_csv(path, index=False)
            blocks = list(origin_frames(path, chunksize=3))
            self.assertEqual([len(f) for f in blocks], [7, 5])
        metrics = pd.DataFrame(dict(model=['seasonal_48', 'lightgbm'], split=['validation'] * 2,
                                    horizon=[8] * 2, gsp=['001'] * 2, mae=[0., 1.], n_scored=[5, 5]))
        table = benefit_table(metrics)
        self.assertTrue(table.zero_baseline_mae.iloc[0])
        self.assertTrue(np.isnan(table.relative_improvement_pct.iloc[0]))
        self.assertEqual(table.comparison.iloc[0], 'loss')
        metrics.loc[1, 'n_scored'] = 4
        with self.assertRaises(AssertionError):
            benefit_table(metrics)

    @unittest.skipUnless(importlib.util.find_spec('lightgbm'), 'Install ML extra')
    def test_streaming_driver_and_analysis_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input'
            source.mkdir()
            times = pd.date_range('2022-01-01', periods=850, freq='30min', tz='UTC')
            pd.DataFrame([np.arange(850, dtype=float), -np.arange(850, dtype=float)],
                          index=['001', 'B'], columns=times).to_csv(source / 'observed_panel.csv')
            cfg = root / 'config.json'
            cfg.write_text(json.dumps(dict(models=['persistence', 'seasonal_48', 'seasonal_336', 'lightgbm'],
                min_train_days=14, horizons=[8, 48], step=48, save_models=True,
                lightgbm=dict(n_estimators=3, n_jobs=1))))
            protocol = root / 'protocol.json'
            protocol.write_text(json.dumps(dict(test_start=times[800].isoformat(),
                experiments=[dict(name='M3_full', config=str(cfg))])))
            with contextlib.redirect_stdout(io.StringIO()):
                run(protocol, source, root / 'study')
                analyse_run(root / 'study' / 'M3_full')
            output = root / 'study' / 'M3_full'
            raw = pd.read_csv(output / 'predictions.csv')
            expected = summarize(raw, ('model', 'split', 'horizon')).sort_values(['model', 'horizon'])
            actual = pd.read_csv(output / 'metrics_pooled.csv').sort_values(['model', 'horizon'])
            np.testing.assert_allclose(actual.rmse, expected.rmse)
            self.assertTrue((output / 'gsp_benefits.csv').exists())
            self.assertTrue((output / 'importance_aggregated.csv').exists())
            timestamp = (output / 'metrics_pooled.csv').stat().st_mtime_ns
            with contextlib.redirect_stdout(io.StringIO()):
                analyse_run(output)
            self.assertEqual(timestamp, (output / 'metrics_pooled.csv').stat().st_mtime_ns)
            self.assertEqual(json.loads((output / 'run_status.json').read_text())['metrics'], 'complete')
            self.assertTrue((pd.to_datetime(raw.timestamp) < times[800]).all())


if __name__ == '__main__':
    unittest.main()
