import tempfile
import unittest

import numpy as np
import pandas as pd

from eso.forecasting.features import FeatureSpec, build_features
from eso.forecasting.ml import GlobalTreeForecaster
from eso.forecasting.oracle_weather import OracleTemperatureForecaster, lookup_temperature
from eso.forecasting.temperature_study import match_cases


class OracleWeatherTests(unittest.TestCase):
    def setUp(self):
        self.times = pd.date_range('2022-01-01', periods=160, freq='30min', tz='UTC')
        self.history = pd.DataFrame([np.sin(np.arange(140) / 5) + 3, np.arange(140) / 50],
                                    index=['B', 'A'], columns=self.times[:140])
        self.temperature = pd.DataFrame({'A': np.arange(160), 'B': np.arange(160) + 100}, index=self.times)
        self.options = dict(horizon=8, min_history=48, training_step=16, n_estimators=5,
                            min_training_rows=10, n_jobs=1,
                            feature_spec=FeatureSpec(lags=(1,), target_lags=(48,), rolling_means=(), rolling_stds=()))

    def test_alignment_uses_names_not_axis_order_and_keeps_unknowns(self):
        features = build_features(self.history, self.times[140:148], spec=self.options['feature_spec'])
        t = self.temperature.drop(columns='A')
        got = lookup_temperature(t, features.index).reshape(2, 8)
        np.testing.assert_array_equal(got[0], np.arange(140, 148) + 100)
        self.assertTrue(np.isnan(got[1]).all())
        t = self.temperature.drop(index=self.times[140])
        self.assertTrue(np.isnan(lookup_temperature(t, features.index).reshape(2, 8)[:, 0]).all())

    def test_only_one_feature_added_same_targets_and_future_not_used_in_fit(self):
        baseline = GlobalTreeForecaster(**self.options).fit(self.history)
        oracle = OracleTemperatureForecaster(oracle_temperature=self.temperature, **self.options).fit(self.history)
        self.assertEqual(oracle.training_audit_, baseline.training_audit_)
        self.assertEqual(oracle.training_counts_, baseline.training_counts_)
        self.assertEqual(oracle.feature_names_, baseline.feature_names_ + ['oracle_temperature_c'])
        altered = self.temperature.copy()
        altered.loc[self.times[140]:] = -999
        other = OracleTemperatureForecaster(oracle_temperature=altered, **self.options).fit(self.history)
        self.assertEqual(oracle.booster_.model_to_string(), other.booster_.model_to_string())
        self.assertLess(pd.Timestamp(oracle.training_audit_['training_target_end']), self.times[140])

    def test_native_missing_weather_keeps_gsps_and_model_roundtrip(self):
        t = self.temperature.drop(columns='B')
        model = OracleTemperatureForecaster(oracle_temperature=t, **self.options).fit(self.history)
        forecast = model.predict(self.times[140:148])
        self.assertTrue(np.isfinite(forecast.to_numpy()).all())
        self.assertTrue(forecast.index.equals(self.history.index))
        with tempfile.TemporaryDirectory() as directory:
            path = directory + '/model'
            model.save(path)
            restored = OracleTemperatureForecaster.load(path)
            pd.testing.assert_frame_equal(restored.predict(self.times[140:148]), forecast)

    def test_comparison_refuses_population_and_target_changes(self):
        frame = pd.DataFrame(dict(split=['validation'], origin=[self.times[140]],
            timestamp=[self.times[140]], horizon=[8], lead=[1], gsp=['A'],
            actual=[2.], prediction=[3.], mase_scale=[1.]))
        self.assertEqual(len(match_cases(frame, frame)), 1)
        with self.assertRaises(ValueError):
            match_cases(frame, frame.assign(gsp='unknown'))
        with self.assertRaises(AssertionError):
            match_cases(frame, frame.assign(actual=999.))
        with self.assertRaises(ValueError):
            match_cases(frame, pd.concat([frame, frame]))


if __name__ == '__main__':
    unittest.main()
