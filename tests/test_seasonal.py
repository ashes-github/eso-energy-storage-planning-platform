import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eso.seasonal import (analyse_days, calendar_phase, daily_features,
                          representative_inputs, summarize, utc_midnight)


CONFIG = json.loads((Path(__file__).resolve().parents[1]/'configs/phase1_seasonal.json').read_text())


def context(day='2022-06-10', low_hours=7):
    day = pd.Timestamp(day)
    times = pd.date_range(utc_midnight(day-pd.Timedelta(days=1)),
                          utc_midnight(day+pd.Timedelta(days=2)), freq='30min', inclusive='left')
    local = times.tz_convert('Europe/London')
    hour = local.hour+local.minute/60
    # Start at 22:00, ending the next morning: exercises midnight attribution.
    low = ((hour-22) % 24) < low_hours
    return np.where(low, 10., 100.), times


class SeasonalTests(unittest.TestCase):
    def test_cross_midnight_run_and_duration_sensitivity(self):
        y, times = context()
        f, p = daily_features(y, times, '2022-06-10', CONFIG)
        self.assertTrue(f['valid_day'])
        self.assertTrue(f['qualifies_6h'])
        self.assertFalse(f['qualifies_8h'])
        self.assertEqual(f['longest_low_hours'], 7.)
        self.assertEqual(p.shape, (48,))

    def test_missing_context_is_unknown_even_if_focal_day_complete(self):
        y, times = context()
        y[-1] = np.nan
        f, _ = daily_features(y, times, '2022-06-10', CONFIG)
        self.assertFalse(f['valid_day'])
        self.assertEqual(f['observed_periods'], 48)
        self.assertTrue(np.isnan(f['qualifies_6h']))

    def test_constant_and_negative_only_are_not_positive_peak_candidates(self):
        y, times = context(low_hours=8)
        for values in [np.zeros(len(y)), y-200]:
            f, _ = daily_features(values, times, '2022-06-10', CONFIG)
            self.assertTrue(f['valid_day'])
            self.assertFalse(f['qualifies_6h'])

    def test_dst_uses_elapsed_time_and_local_profile_slots(self):
        y, times = context('2022-03-27')
        f, p = daily_features(y, times, '2022-03-27', CONFIG)
        self.assertTrue(f['valid_day'])
        self.assertEqual(f['expected_periods'], 46)
        self.assertEqual(np.isnan(p).sum(), 2)

    def test_split_purges_context_overlap(self):
        self.assertEqual(calendar_phase('2022-07-29', '2022-07-31'), 'development')
        self.assertEqual(calendar_phase('2022-07-30', '2022-07-31'), 'boundary_purged')
        self.assertEqual(calendar_phase('2022-07-31', '2022-07-31'), 'boundary_purged')
        self.assertEqual(calendar_phase('2022-08-01', '2022-07-31'), 'holdout')

    def test_valid_denominator_and_holdout_isolation(self):
        season = {'name': 'summer', 'start': '2022-06-01', 'end': '2022-06-30'}
        times = pd.date_range(utc_midnight('2022-05-31'), utc_midnight('2022-07-02'),
                              freq='30min', inclusive='left')
        hour = times.tz_convert('Europe/London').hour
        y = np.where(((hour-22) % 24) < 7, 10., 100.)
        panel = pd.DataFrame([y, np.full(len(y), np.nan)], index=['A', 'B'], columns=times)
        daily, profiles, split = analyse_days(panel, season, CONFIG)
        summary = summarize(daily, CONFIG)
        a = summary[(summary['GSP Id']=='A') & (summary.phase=='all')]
        b = summary[summary['GSP Id']=='B']
        self.assertTrue((a.observed_pattern_frequency_6h == 1).all())
        self.assertTrue(b.observed_pattern_frequency_6h.isna().all())
        self.assertTrue(b.evidence_status.eq('insufficient_evidence').all())
        original = representative_inputs(daily, profiles, ('summer', 'weekday'), 'development')
        changed = panel.copy()
        changed.loc[:, changed.columns >= utc_midnight(split)] *= 100
        d2, p2, _ = analyse_days(changed, season, CONFIG)
        pd.testing.assert_frame_equal(original, representative_inputs(d2, p2, ('summer', 'weekday'), 'development'))
        pd.testing.assert_frame_equal(summary, summarize(daily, CONFIG))

    def test_missing_days_do_not_become_negative_outcomes(self):
        rows = []
        for i, day in enumerate(pd.date_range('2022-06-01', periods=35)):
            valid = i % 3 != 0
            rows.append({'GSP Id': 'A', 'season': 'summer', 'day_type': 'weekday',
                         'phase': 'development', 'week': str(day.to_period('W').start_time.date()),
                         'valid_day': valid, 'qualifies_6h': True if valid else np.nan,
                         'qualifies_8h': False if valid else np.nan})
        summary = summarize(pd.DataFrame(rows), CONFIG)
        full = summary[summary.phase == 'all'].iloc[0]
        self.assertEqual(full.valid_days, 23)
        self.assertEqual(full.expected_days, 35)
        self.assertEqual(full.observed_pattern_frequency_6h, 1.)
        self.assertEqual(full.observed_pattern_frequency_8h, 0.)
        self.assertEqual(full.evidence_status, 'insufficient_evidence')
        self.assertEqual(full.ci_low_6h, 1.)


if __name__ == '__main__':
    unittest.main()
