"""Physical frequency units and explicit contiguous low-demand checks."""
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import welch


def band_power(frequency, density, low, high):
    """Trapezoidal PSD area with interpolated band endpoints (cycles/hour)."""
    interior = (frequency > low) & (frequency < high)
    x = np.r_[low, frequency[interior], high]
    return float(trapezoid(np.interp(x, frequency, density), x=x))


def spectral_features(values, nperseg=96):
    values = np.asarray(values, dtype=float)
    if min(nperseg, len(values)) < 96:
        raise ValueError('Use at least 48 hours / 96 samples for the spectral bands')
    segment = min(nperseg, len(values))
    frequency, density = welch(values, fs=2.0, window='hann', nperseg=segment,
                               noverlap=segment // 2, detrend='constant', scaling='density')
    total = float(trapezoid(density, x=frequency))
    short = band_power(frequency, density, 1/8, 1/6)
    daily = band_power(frequency, density, 1/28, 1/20)
    return {'power_6_8h': short, 'power_20_28h': daily, 'total_power': total,
            'fraction_6_8h': short / total if total > 0 else 0.0,
            'fraction_20_28h': daily / total if total > 0 else 0.0,
            'frequency_resolution_cph': 2.0 / segment}


def temporal_features(values, timestamps, low_fraction=0.35, min_hours=6,
                      following_hours=8, min_peak_fraction=0.75):
    """Low threshold = p10 + low_fraction*(p90-p10) over this window.

    A qualifying run is followed within following_hours by demand at least
    p10 + min_peak_fraction*(p90-p10). Boundary-touching runs are censored;
    runs crossing midnight are retained. Output is a demand-pattern proxy.
    """
    y = np.asarray(values, dtype=float)
    p10, p90 = np.quantile(y, [0.1, 0.9])
    amplitude = p90 - p10
    low = y <= p10 + low_fraction * amplitude
    changes = np.diff(np.r_[False, low, False].astype(int))
    starts, ends = np.where(changes == 1)[0], np.where(changes == -1)[0]
    durations, qualifies, ratios, observed = [], set(), [], 0
    for start, end in zip(starts, ends):
        if start == 0 or end + int(2 * following_hours) > len(y):
            continue
        observed += 1
        hours = (end - start) / 2
        peak = float(np.max(y[end:end + int(2 * following_hours)]))
        trough = float(np.mean(y[start:end]))
        durations.append(hours)
        if trough > 0:
            ratios.append(peak / trough)
        if amplitude > 0 and hours >= min_hours and peak > 0 and peak >= p10 + min_peak_fraction * amplitude:
            qualifies.add(timestamps[start].tz_convert('Europe/London').date())
    days = len(set(timestamps.tz_convert('Europe/London').date))
    return {'longest_observed_low_hours': max(durations, default=0.0),
            'qualifying_start_days': len(qualifies),
            'qualifying_day_fraction': len(qualifies) / days,
            'observed_low_runs': observed, 'censored_low_runs': len(starts) - observed,
            'max_peak_trough_ratio': max(ratios) if ratios else np.nan,
            'max_up_ramp_per_hour': float(max(0, np.max(np.diff(y)) * 2)),
            'p90_minus_p10': amplitude, 'negative_periods': int((y < 0).sum())}


def extract_features(panel, config):
    rows = []
    for gsp, row in panel.iterrows():
        result = spectral_features(row.to_numpy(), config['nperseg'])
        result.update(temporal_features(row.to_numpy(), panel.columns,
                                       **config['temporal']))
        rows.append({'GSP Id': gsp, **result})
    return pd.DataFrame(rows).set_index('GSP Id')


def screen(features, config):
    """Transparent uncalibrated score; spectra cannot override temporal failure."""
    result = features.copy()
    daily = result['fraction_20_28h'].clip(0, 1)
    temporal = result['qualifying_day_fraction'].clip(0, 1)
    weight = config['temporal_weight']
    result['screening_score'] = weight * temporal + (1 - weight) * daily
    eligible = (temporal > 0) & (temporal >= config['min_day_fraction'])
    if not config.get('allow_signed_exports', False):
        eligible &= result['negative_periods'] == 0
    result['candidate'] = eligible & (result['screening_score'] >= config['score_threshold'])
    result['screening_status'] = np.where(result['candidate'], 'candidate', 'not_flagged')
    if not config.get('allow_signed_exports', False):
        result.loc[result['negative_periods'] > 0, 'screening_status'] = 'requires_review_negative_volume'
    return result


def screen_with_quality(panel, quality, feature_config, screening_config):
    """Every GSP receives a result row; insufficient evidence is never False."""
    ready = quality.index[quality['model_eligible']]
    if len(ready):
        result = screen(extract_features(panel.loc[ready], feature_config), screening_config)
    else:
        result = pd.DataFrame(columns=['candidate', 'screening_score', 'screening_status'])
    result = result.reindex(panel.index).join(quality)
    result['candidate'] = result['candidate'].astype('boolean')
    unavailable = ~result['model_eligible']
    result.loc[unavailable, 'candidate'] = pd.NA
    result.loc[unavailable, 'screening_status'] = 'insufficient_evidence'
    imputed = result['data_status'].eq('imputed')
    result.loc[imputed, 'screening_status'] = result.loc[imputed, 'screening_status'] + '_imputed'
    return result
