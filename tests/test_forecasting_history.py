"""Short software checks must remain distinguishable from longer-history runs."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from eso.forecasting.config import assess_training_history, load_config
from eso.forecasting.pipeline import main


class HistoryAssessmentTests(unittest.TestCase):
    def test_default_is_smoke(self):
        config = load_config()
        self.assertEqual(config['min_train_days'], 14)
        assessment = assess_training_history(672, config)
        self.assertEqual(assessment['classification'], 'smoke_benchmark')
        self.assertEqual(assessment['initial_history_elapsed_days'], 14)
        self.assertEqual(assessment['short_history_models'], config['models'])

    def test_long_history_and_short_fitting_window(self):
        config = load_config()
        for days in (180, 365, 730):
            self.assertEqual(assess_training_history(days * 48, config)['classification'],
                             'long_history_benchmark')
        config['models'] = ['seasonal_48', 'sarima', 'ets']
        config['sarima']['train_window'] = 672
        result = assess_training_history(365 * 48, config)
        self.assertEqual(result['classification'], 'smoke_benchmark')
        self.assertEqual(result['short_history_models'], ['sarima'])
        self.assertEqual(result['initial_fitting_half_hours_by_model']['sarima'], 672)
        self.assertEqual(result['initial_fitting_half_hours_by_model']['ets'], 365 * 48)
        self.assertEqual(assess_training_history(180 * 48 - 1, load_config())['classification'],
                         'smoke_benchmark')

    def test_cli_labels_actual_override_in_console_and_metadata(self):
        for initial in (672, 8640):
            with self.subTest(initial=initial), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, output = root / 'input', root / 'run'
                source.mkdir()
                timestamps = pd.date_range('2021-01-01', periods=initial + 96,
                                           freq='30min', tz='UTC')
                pd.DataFrame([np.arange(len(timestamps), dtype=float)], index=['001'],
                             columns=timestamps).to_csv(source / 'observed_panel.csv')
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    main(['--input', str(source), '--output', str(output),
                          '--min-train', str(initial),
                          '--test-start', timestamps[initial + 48].isoformat()])
                assessment = json.loads((output / 'run.json').read_text())['training_history_assessment']
                expected = 'smoke_benchmark' if initial == 672 else 'long_history_benchmark'
                self.assertEqual(assessment['classification'], expected)
                self.assertEqual(assessment['initial_history_half_hours'], initial)
                self.assertIn(expected, stream.getvalue())
                self.assertEqual('Use these results to verify the pipeline' in stream.getvalue(),
                                 initial == 672)


if __name__ == '__main__':
    unittest.main()
