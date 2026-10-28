"""Forecast feature regressions: origin boundaries, missingness, labels and DST."""
import unittest

import numpy as np
import pandas as pd

from eso.forecasting import (FeatureSpec, ForecastDataset, WalkForward, backtest,
                             build_features, build_training_examples)


def panel(n=400):
    times = pd.date_range('2022-06-01', periods=n, freq='30min', tz='UTC')
    return pd.DataFrame([np.arange(n, dtype=float), -np.arange(n, dtype=float)],
                        index=['001', 'B'], columns=times)


def future_after(history, n=48):
    return pd.date_range(history.columns[-1] + pd.Timedelta('30min'), periods=n, freq='30min')


class FeatureTests(unittest.TestCase):
    def test_origin_relative_lags_and_complete_rolling_statistics(self):
        p = panel()
        original = p.copy()
        features = build_features(p, future_after(p))
        for gsp, sign in [('001', 1), ('B', -1)]:
            group = features.xs(gsp, level='gsp')
            np.testing.assert_array_equal(group.lead, np.arange(1, 49))
            for lag in (1, 2, 48, 96, 336):
                np.testing.assert_allclose(group[f'origin_lag_{lag}'], sign * (400 - lag))
            for window in (6, 12, 48, 336):
                np.testing.assert_allclose(group[f'mean_{window}'],
                                           sign * np.arange(400 - window, 400).mean())
            for window in (48, 336):
                np.testing.assert_allclose(group[f'std_{window}'],
                                           np.arange(400 - window, 400).std(ddof=1))
        self.assertEqual(features.index.names, ['gsp', 'origin', 'timestamp'])
        self.assertEqual(set(features.index.get_level_values('origin')), {future_after(p)[0]})
        pd.testing.assert_frame_equal(p, original)

    def test_missing_donors_and_windows_are_not_filled(self):
        p = panel()
        p.iloc[0, -2] = np.nan
        p.iloc[1] = 0
        result = build_features(p, future_after(p, 2))
        a, b = result.xs('001', level='gsp'), result.xs('B', level='gsp')
        self.assertTrue(a.origin_lag_2.isna().all())
        self.assertTrue(a.mean_6.isna().all())
        self.assertTrue(a.std_48.isna().all())
        self.assertTrue(a.origin_lag_1.notna().all())
        self.assertTrue((b.std_48 == 0).all())
        short = build_features(panel(4), future_after(panel(4), 2))
        self.assertTrue(short.origin_lag_48.isna().all())
        self.assertTrue(short.mean_6.isna().all())
        self.assertTrue(short.origin_lag_1.notna().all())

    def test_future_targets_cannot_change_first_origin_features(self):
        p = panel(100)
        kwargs = dict(horizon=8, min_history=48, step=8)
        features, targets = build_training_examples(p, **kwargs)
        changed = p.copy()
        changed.iloc[:, 48:] += 1_000_000
        other, other_targets = build_training_examples(changed, **kwargs)
        first = p.columns[48]
        pd.testing.assert_frame_equal(features.xs(first, level='origin'),
                                      other.xs(first, level='origin'))
        self.assertFalse(targets.xs(first, level='origin').equals(
            other_targets.xs(first, level='origin')))

    def test_training_labels_are_purged_and_axes_match(self):
        p = panel(101)
        p.iloc[0, 50] = np.nan
        features, targets = build_training_examples(p, horizon=8, min_history=48, step=7)
        self.assertTrue(features.index.equals(targets.index))
        self.assertEqual(targets.name, 'target')
        self.assertTrue(targets.isna().any())
        self.assertLess(features.index.get_level_values('timestamp').max(), future_after(p)[0])
        self.assertEqual(set(features.index.get_level_values('origin')),
                         set(p.columns[list(range(48, 101 - 8 + 1, 7))]))
        for (gsp, origin, timestamp), target in targets.items():
            position = p.columns.get_loc(origin)
            self.assertEqual(features.loc[(gsp, origin, timestamp), 'origin_lag_1'],
                             p.iloc[p.index.get_loc(gsp), position - 1])
            if pd.notna(target):
                self.assertEqual(target, p.at[gsp, timestamp])
        # Perturb observations outside an outer model.fit(history) boundary.
        changed = p.copy()
        changed.iloc[:, 80:] += 9999
        before = build_training_examples(p.iloc[:, :80], horizon=8, min_history=48)
        after = build_training_examples(changed.iloc[:, :80], horizon=8, min_history=48)
        pd.testing.assert_frame_equal(before[0], after[0])
        pd.testing.assert_series_equal(before[1], after[1])

    def test_calendar_holidays_and_cyclic_encodings(self):
        p = panel(48)  # Last observation June 1 at 23:30 UTC.
        result = build_features(p, future_after(p, 48), holiday_calendar={'2022-06-02': True,
                                                                       '2022-06-03': False})
        group = result.xs('001', level='gsp')
        first, last = group.iloc[0], group.iloc[-1]
        self.assertEqual(first.hour, 1)  # UTC midnight is BST 01:00.
        self.assertEqual(first.half_hour_of_day, 2)
        self.assertEqual(first.settlement_period, 3)
        self.assertEqual(first.day_of_week, 3)
        self.assertEqual(first.weekend, 0)
        self.assertEqual(first.month, 6)
        self.assertEqual(first.holiday, 1)
        self.assertEqual(last.holiday, 0)
        np.testing.assert_allclose(group.time_of_day_sin ** 2 + group.time_of_day_cos ** 2, 1)
        np.testing.assert_allclose(group.day_of_week_sin ** 2 + group.day_of_week_cos ** 2, 1)
        unknown = build_features(p, future_after(p, 2))
        self.assertTrue(unknown.holiday.isna().all())

    def test_dst_settlement_periods_and_repeated_local_slots(self):
        for day, count in [('2022-03-27', 46), ('2022-10-30', 50)]:
            start = pd.Timestamp(day).tz_localize('Europe/London').tz_convert('UTC')
            future = pd.date_range(start, periods=count, freq='30min')
            history = pd.DataFrame([[1.]], index=['001'],
                                   columns=pd.DatetimeIndex([start - pd.Timedelta('30min')]))
            result = build_features(history, future)
            np.testing.assert_array_equal(result.settlement_period, np.arange(1, count + 1))
            self.assertTrue((result.weekend == 1).all())
            if count == 50:
                self.assertEqual((result.half_hour_of_day == 2).sum(), 2)
            else:
                self.assertFalse((result.half_hour_of_day == 2).any())

    def test_fit_predict_backtest_integration_and_future_isolation(self):
        seen = []
        class FeatureModel:
            def fit(self, history):
                self.history = history
            def predict(self, timestamps):
                features = build_features(self.history, timestamps)
                seen.append(features)
                return pd.DataFrame(features.origin_lag_1.to_numpy().reshape(len(self.history), -1),
                                    index=self.history.index, columns=timestamps)
        p = panel(220)
        plan = WalkForward(100, p.columns[160].isoformat(), horizons=(8, 48))
        before = backtest(ForecastDataset(p), plan, {'features': FeatureModel})
        first_features = seen[0]
        seen.clear()
        changed = p.copy()
        changed.iloc[:, 100:] += 12345
        after = backtest(ForecastDataset(changed), plan, {'features': FeatureModel})
        pd.testing.assert_frame_equal(first_features, seen[0])
        np.testing.assert_array_equal(before.prediction, after.prediction)

    def test_spec_and_invalid_inputs(self):
        for kwargs in [dict(lags=(0,)), dict(lags=(1, 1)), dict(lags=(True,)),
                       dict(std_ddof=1.0), dict(rolling_stds=(1,)),
                       dict(target_lags=(0,)), dict(target_lags=(48, 48)),
                       dict(target_lags=(True,))]:
            with self.assertRaises(ValueError):
                FeatureSpec(**kwargs)
        p = panel()
        future = future_after(p)
        for invalid in [future.tz_localize(None), future[1:], future[::-1], future[:0]]:
            with self.assertRaises(ValueError):
                build_features(p, invalid)
        for calendar in [{'2022-06-01': 1}, {'2022-06-01T12:00': True}, []]:
            with self.assertRaises(ValueError):
                build_features(p, future, holiday_calendar=calendar)
        with self.assertRaises(ValueError):
            build_training_examples(panel(10), min_history=8, horizon=8)

    def test_target_aligned_seasonal_lags_match_naive_donors(self):
        from eso.forecasting import SeasonalNaive
        p = panel()
        future = future_after(p)
        features = build_features(p, future)
        for lag in (48, 96, 336):
            expected = SeasonalNaive(lag).fit(p).predict(future)
            for gsp, sign in [('001', 1), ('B', -1)]:
                group = features.xs(gsp, level='gsp')
                np.testing.assert_array_equal(group[f'target_lag_{lag}'],
                                              sign * np.arange(400 - lag, 448 - lag))
                np.testing.assert_array_equal(group[f'target_lag_{lag}'], expected.loc[gsp])
        # Origin daily lag is fixed; target daily lag follows all 48 donor slots.
        group = features.xs('001', level='gsp')
        self.assertEqual(group.origin_lag_48.nunique(), 1)
        self.assertEqual(group.target_lag_48.nunique(), 48)
        self.assertEqual(group.target_lag_48.iloc[-1], p.loc['001'].iloc[-1])

    def test_unavailable_target_lags_never_wrap_or_reuse_future_predictions(self):
        p = panel(24)
        p.iloc[0, 3] = np.nan
        features = build_features(p, future_after(p, 60),
                                  spec=FeatureSpec(target_lags=(1, 48, 96)))
        group = features.xs('001', level='gsp')
        self.assertTrue(group.target_lag_48.iloc[:24].isna().all())
        np.testing.assert_allclose(group.target_lag_48.iloc[24:48], p.loc['001'], equal_nan=True)
        self.assertTrue(group.target_lag_48.iloc[48:].isna().all())
        self.assertTrue(group.target_lag_96.isna().all())
        self.assertEqual(group.target_lag_1.iloc[0], p.loc['001'].iloc[-1])
        self.assertTrue(group.target_lag_1.iloc[1:].isna().all())

    def test_training_target_lags_are_strictly_before_each_inner_origin(self):
        p = panel(160)
        kwargs = dict(horizon=60, min_history=70, step=7)
        features, _ = build_training_examples(p, **kwargs)
        for (gsp, origin, timestamp), row in features.iterrows():
            origin_position = p.columns.get_loc(origin)
            target_position = p.columns.get_loc(timestamp)
            for lag in (48, 96, 336):
                donor = target_position - lag
                if 0 <= donor < origin_position:
                    self.assertEqual(row[f'target_lag_{lag}'], p.loc[gsp].iloc[donor])
                else:
                    self.assertTrue(np.isnan(row[f'target_lag_{lag}']))
        changed = p.copy()
        changed.iloc[:, 70:] += 10_000_000
        after, _ = build_training_examples(changed, **kwargs)
        pd.testing.assert_frame_equal(features.xs(p.columns[70], level='origin'),
                                      after.xs(p.columns[70], level='origin'))


if __name__ == '__main__':
    unittest.main()
