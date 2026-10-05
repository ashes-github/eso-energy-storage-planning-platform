"""Observed-day recurrence and chronological representative-profile experiments.

Run independently of the historical five-day experiment. Recurrence describes a
demand pattern, not a calibrated probability of battery feasibility.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import periodogram
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data import align_panel, load_records
from .features import band_power
from .models import autoencoder_model, gplvm_model, pca_model

ROOT = Path(__file__).resolve().parents[2]
ZONE = "Europe/London"
BEHAVIOUR = [
    "longest_low_hours",
    "max_up_ramp_per_hour",
    "p90_minus_p10",
    "fraction_6_8h",
    "fraction_20_28h",
]


def utc_midnight(day):
    return pd.Timestamp(day).tz_localize(ZONE).tz_convert("UTC")


def calendar_phase(day, split):
    """Purge focal days whose three-calendar-day context crosses the split."""
    day, split = pd.Timestamp(day), pd.Timestamp(split)
    if day + pd.Timedelta(days=2) <= split:
        return "development"
    if day - pd.Timedelta(days=1) >= split:
        return "holdout"
    return "boundary_purged"


def daily_features(values, times, day, config):
    """Attribute bounded low runs to their local START date; no extrapolation."""
    y = np.asarray(values, dtype=float)
    day = pd.Timestamp(day)
    focal = (times >= utc_midnight(day)) & (
        times < utc_midnight(day + pd.Timedelta(days=1))
    )
    expected = int(
        (utc_midnight(day + pd.Timedelta(days=1)) - utc_midnight(day)).total_seconds()
        / 1800
    )
    observed = int(np.isfinite(y[focal]).sum())
    result = {
        "expected_periods": expected,
        "observed_periods": observed,
        "valid_day": False,
        "invalid_reason": "missing_context",
        **{f"qualifies_{h}h": np.nan for h in config["duration_hours"]},
    }
    profile = np.full(48, np.nan)
    expected_context = pd.date_range(
        utc_midnight(day - pd.Timedelta(days=1)),
        utc_midnight(day + pd.Timedelta(days=2)),
        freq="30min",
        inclusive="left",
    )
    if not times.equals(expected_context) or not np.isfinite(y).all():
        return result, profile
    local = times[focal].tz_convert(ZONE)
    slots = local.hour * 2 + local.minute // 30
    profile_series = pd.Series(y[focal]).groupby(slots).mean()
    profile[profile_series.index] = profile_series.to_numpy()
    p10, p90 = np.quantile(y[focal], [0.1, 0.9])
    amplitude = p90 - p10
    result.update(
        valid_day=True,
        invalid_reason="",
        p90_minus_p10=amplitude,
        max_up_ramp_per_hour=float(max(0, np.max(np.diff(y[focal])) * 2)),
        negative_periods=int((y[focal] < 0).sum()),
    )
    # A single-day PSD is descriptive only: resolution is about 1/24 cycles/h.
    f, density = periodogram(y[focal], fs=2, window="hann", detrend="constant")
    total = float(trapezoid(density, x=f))
    result.update(
        fraction_6_8h=(
            band_power(f, density, 1 / 8, 1 / 6) / total if total > 0 else 0.0
        ),
        fraction_20_28h=(
            band_power(f, density, 1 / 28, 1 / 20) / total if total > 0 else 0.0
        ),
        frequency_resolution_cph=2 / expected,
    )
    low = y <= p10 + config["low_fraction"] * amplitude
    changes = np.diff(np.r_[False, low, False].astype(int))
    starts, ends = np.where(changes == 1)[0], np.where(changes == -1)[0]
    future = int(config["following_hours"] * 2)
    qualifying, durations, censored = [], [], 0
    for start, end in zip(starts, ends):
        if not focal[start]:
            continue
        if start == 0 or end + future > len(y):
            censored += 1
            continue
        hours = (end - start) / 2
        durations.append(hours)
        peak = float(y[end : end + future].max())
        if (
            amplitude > 0
            and peak > 0
            and peak >= p10 + config["min_peak_fraction"] * amplitude
        ):
            qualifying.append(hours)
    result.update(longest_low_hours=max(durations, default=0.0), censored_runs=censored)
    # Long runs of unknown start crossing the context boundary cannot establish
    # a valid negative. Constant focal profiles remain valid nonqualifying days.
    if amplitude > 0 and ((low[0] and low[focal].all()) or censored):
        result.update(valid_day=False, invalid_reason="censored_run")
        return result, profile
    for h in config["duration_hours"]:
        result[f"qualifies_{h}h"] = bool(any(d >= h for d in qualifying))
    return result, profile


def load_season(paths, season):
    """Bound peak memory by resolving revisions in disjoint 28-day chunks."""
    first = pd.Timestamp(season["start"]) - pd.Timedelta(days=1)
    last = pd.Timestamp(season["end"]) + pd.Timedelta(days=1)
    parts, audits, sources = [], [], []
    for begin in pd.date_range(first, last, freq="28D"):
        end = min(begin + pd.Timedelta(days=27), last)
        print(f"Loading {season['name']}: {begin.date()} .. {end.date()}", flush=True)
        records, sources = load_records(
            paths, str(begin.date()), str(end.date()), "signed_net"
        )
        panel, _, audit = align_panel(
            records, str(begin.date()), str(end.date()), "signed_net"
        )
        parts.append(panel.astype(float))
        audits.append({"start": str(begin.date()), "end": str(end.date()), **audit})
        del records
    return pd.concat(parts, axis=1).sort_index(), audits, sources


def analyse_days(panel, season, config):
    dates = pd.date_range(season["start"], season["end"])
    split = dates[int(len(dates) * config["development_fraction"])]
    rows, profiles = [], []
    for day in dates:
        mask = (panel.columns >= utc_midnight(day - pd.Timedelta(days=1))) & (
            panel.columns < utc_midnight(day + pd.Timedelta(days=2))
        )
        context = panel.loc[:, mask]
        for gsp, values in zip(panel.index, context.to_numpy()):
            features, profile = daily_features(values, context.columns, day, config)
            rows.append(
                {
                    "GSP Id": gsp,
                    "season": season["name"],
                    "date": str(day.date()),
                    "day_type": "weekend" if day.dayofweek >= 5 else "weekday",
                    "week": str((day - pd.Timedelta(days=day.dayofweek)).date()),
                    "phase": calendar_phase(day, split),
                    **features,
                }
            )
            profiles.append(profile)
    return pd.DataFrame(rows), np.array(profiles), str(split.date())


def summarize(daily, config, weekly=False):
    keys = ["season", "day_type", "GSP Id"] + (["week"] if weekly else ["phase"])
    data = (
        daily
        if weekly
        else pd.concat([daily, daily.assign(phase="all")], ignore_index=True)
    )
    rng = np.random.default_rng(config["split_seed"])
    rows = []
    for key, group in data.groupby(keys, sort=True):
        valid = group[group.valid_day]
        n, expected = len(valid), len(group)
        row = dict(zip(keys, key))
        row.update(
            expected_days=expected,
            valid_days=n,
            coverage=n / expected,
            evidence_status=(
                "sufficient"
                if n >= config["minimum_valid_days"]
                and n / expected >= config["minimum_coverage"]
                else "insufficient_evidence"
            ),
        )
        for h in config["duration_hours"]:
            column = f"qualifies_{h}h"
            count = int(valid[column].sum())
            row.update(
                {
                    f"qualifying_days_{h}h": count,
                    f"observed_pattern_frequency_{h}h": count / n if n else np.nan,
                }
            )
            if not weekly:
                blocks = (
                    valid.groupby("week")[column]
                    .agg(["sum", "count"])
                    .to_numpy(dtype=float)
                )
                lo = hi = np.nan
                if len(blocks) >= 4:
                    picks = rng.integers(
                        len(blocks), size=(config["bootstrap_repetitions"], len(blocks))
                    )
                    draws = blocks[picks].sum(axis=1)
                    lo, hi = np.quantile(draws[:, 0] / draws[:, 1], [0.025, 0.975])
                row.update({f"ci_low_{h}h": lo, f"ci_high_{h}h": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def representative_inputs(daily, profiles, regime, phase):
    """Never mix development and later holdout days in a representation."""
    mask = (
        (daily.season == regime[0])
        & (daily.day_type == regime[1])
        & (daily.phase == phase)
        & daily.valid_day
    )
    rows, ids = [], []
    for gsp, group in daily[mask].groupby("GSP Id", sort=True):
        p = profiles[group.index.to_numpy()]
        median = np.nanmedian(p, axis=0)
        iqr = np.nanquantile(p, 0.75, axis=0) - np.nanquantile(p, 0.25, axis=0)
        behaviour = group[BEHAVIOUR].median().to_numpy(dtype=float)
        rows.append(np.r_[median, iqr, behaviour])
        ids.append(gsp)
    return pd.DataFrame(rows, index=pd.Index(ids, name="GSP Id"), columns=range(101))


def benchmark(daily, profiles, aggregate, config, output):
    """Held-out GSPs: earlier representative inputs -> later median target.

    This is temporal transfer of a typical profile, not half-hour forecasting.
    """
    import torch

    torch.set_num_threads(config["torch_threads"])
    metrics, eligibility = [], []
    for regime in sorted(set(zip(daily.season, daily.day_type))):
        development = representative_inputs(daily, profiles, regime, "development")
        future = representative_inputs(daily, profiles, regime, "holdout")
        a = aggregate[
            (aggregate.season == regime[0]) & (aggregate.day_type == regime[1])
        ]
        good = a[a.evidence_status == "sufficient"]
        ids = sorted(
            set(good[good.phase == "development"]["GSP Id"])
            & set(good[good.phase == "holdout"]["GSP Id"])
            & set(development.index)
            & set(future.index)
        )
        ids = [
            g
            for g in ids
            if np.isfinite(development.loc[g]).all()
            and np.isfinite(future.loc[g, :47]).all()
        ]
        eligibility.append(
            {
                "season": regime[0],
                "day_type": regime[1],
                "eligible_gsps": len(ids),
                "total_gsps": a["GSP Id"].nunique(),
            }
        )
        if len(ids) < 10:
            continue
        train_ids, test_ids = train_test_split(
            ids, test_size=config["test_fraction"], random_state=config["split_seed"]
        )
        target = future.loc[test_ids, :47].to_numpy()
        regime_dir = output / "models" / ("_".join(regime))
        regime_dir.mkdir(parents=True, exist_ok=True)
        (regime_dir / "split.json").write_text(
            json.dumps({"train_gsps": train_ids, "test_gsps": test_ids}, indent=2)
        )

        def score(name, representation, seed, prediction, reconstruction_mse=np.nan):
            error = prediction - target
            metrics.append(
                {
                    "season": regime[0],
                    "day_type": regime[1],
                    "representation": representation,
                    "model": name,
                    "seed": seed,
                    "train_gsps": len(train_ids),
                    "test_gsps": len(test_ids),
                    "later_median_rmse": float(np.sqrt(np.mean(error**2))),
                    "later_median_mae": float(np.mean(np.abs(error))),
                    "development_reconstruction_mse_scaled": reconstruction_mse,
                }
            )

        baseline = development.loc[test_ids, :47].to_numpy()
        score("persistence", "median48", 0, baseline)
        np.savez_compressed(
            regime_dir / "targets.npz", later_median=target, earlier_median=baseline
        )
        for representation, width in [
            ("median48", 48),
            ("median48_iqr48_behaviour5", 101),
        ]:
            scaler = StandardScaler().fit(development.loc[train_ids].iloc[:, :width])
            train = scaler.transform(development.loc[train_ids].iloc[:, :width])
            test = scaler.transform(development.loc[test_ids].iloc[:, :width])
            observed = np.arange(width)
            repdir = regime_dir / representation
            repdir.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, repdir / "scaler.joblib")
            np.savez_compressed(repdir / "inputs.npz", train=train, test=test)
            for name in ["pca", "autoencoder", "gplvm"]:
                for seed in (
                    [config["seeds"][0]] if name == "pca" else config["seeds"]
                ):
                    print(
                        f"Fitting {regime} {representation} {name} seed={seed}",
                        flush=True,
                    )
                    if name == "pca":
                        result = pca_model(
                            train, test, observed, config["latent_dim"], seed
                        )
                    elif name == "autoencoder":
                        result = autoencoder_model(
                            train,
                            test,
                            observed,
                            config["latent_dim"],
                            seed,
                            config["ae_steps"],
                            config["infer_steps"],
                        )
                    else:
                        result = gplvm_model(
                            train,
                            test,
                            observed,
                            np.zeros(len(train)),
                            config["latent_dim"],
                            seed,
                            "independent",
                            1.0,
                            config["inducing_count"],
                            "kmeans",
                            config["steps"],
                            config["infer_steps"],
                        )
                    z, recon, prediction, artifact = result[:4]
                    if not all(np.isfinite(x).all() for x in [z, recon, prediction]):
                        raise ValueError(f"Non-finite {name} output")
                    uncertainty = {}
                    if name == "gplvm":
                        # Variational SDs belong to training GSPs in split order.
                        # Held-out decoder inference produces point estimates only.
                        sd = np.asarray(result[5])
                        if sd.shape != z.shape or not np.isfinite(sd).all() or (sd < 0).any():
                            raise ValueError("Invalid GPLVM latent posterior SD")
                        uncertainty["latent_posterior_sd"] = sd
                        pd.DataFrame(
                            sd,
                            index=pd.Index(train_ids, name="GSP Id"),
                            columns=[f"z{i+1}_sd" for i in range(sd.shape[1])],
                        ).to_csv(repdir / f"{name}_{seed}_latent_posterior_sd.csv")
                    raw = scaler.inverse_transform(prediction)[:, :48]
                    score(
                        name,
                        representation,
                        seed,
                        raw,
                        float(np.mean((recon - train) ** 2)),
                    )
                    joblib.dump(artifact, repdir / f"{name}_{seed}.joblib")
                    np.savez_compressed(
                        repdir / f"{name}_{seed}.npz",
                        latent=z,
                        prediction=raw,
                        losses=np.array(result[4] if len(result) > 4 else []),
                        **uncertainty,
                    )
            pd.DataFrame(metrics).to_csv(output / "model_metrics.csv", index=False)
    pd.DataFrame(eligibility).to_csv(output / "model_eligibility.csv", index=False)
    return pd.DataFrame(metrics)


def plots(daily, aggregate, weekly, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full = aggregate[aggregate.phase == "all"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, ((season, day_type), g) in zip(
        axes.flat, weekly.groupby(["season", "day_type"])
    ):
        matrix = g.pivot(
            index="GSP Id", columns="week", values="observed_pattern_frequency_6h"
        )
        matrix = matrix.loc[matrix.mean(axis=1).sort_values().index]
        im = ax.imshow(
            matrix.to_numpy(), aspect="auto", vmin=0, vmax=1, interpolation="nearest"
        )
        ax.set_title(f"{season} {day_type}")
        ax.set_ylabel("GSPs (ordered by recurrence)")
        ax.set_xticks(
            range(len(matrix.columns)), matrix.columns, rotation=90, fontsize=7
        )
    fig.colorbar(im, ax=axes, label="Observed qualifying-day fraction (6h)")
    fig.savefig(output / "weekly_recurrence.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    groups = list(full.groupby(["season", "day_type"]))
    ax.boxplot(
        [g.observed_pattern_frequency_6h.dropna() for _, g in groups],
        tick_labels=["\n".join(k) for k, _ in groups],
    )
    ax.set_ylabel("Observed qualifying-day fraction (6h)")
    ax.set_ylim(-0.03, 1.03)
    fig.savefig(output / "regime_recurrence.png", dpi=150)
    plt.close(fig)


def run(config_path, output, skip_models=False):
    config = json.loads(Path(config_path).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    paths = sorted(ROOT.glob(config["input_glob"]))
    daily_parts, profile_parts, panels, audits, splits = [], [], {}, {}, {}
    for season in config["seasons"]:
        panel, audit, sources = load_season(paths, season)
        panels[season["name"]] = panel
        audits[season["name"]] = audit
    # GSPs absent from a whole season still receive explicit unknown rows.
    ids = sorted(set().union(*(set(p.index) for p in panels.values())))
    for season in config["seasons"]:
        name = season["name"]
        panel = panels[name].reindex(ids)
        panel.to_csv(output / f"{name}_observed_panel.csv")
        days, profiles, split = analyse_days(panel, season, config)
        daily_parts.append(days)
        profile_parts.append(profiles)
        splits[name] = split
    daily = pd.concat(daily_parts, ignore_index=True)
    profiles = np.concatenate(profile_parts)
    daily.to_csv(output / "daily_features.csv", index=False)
    np.savez_compressed(output / "daily_profiles.npz", profiles=profiles)
    aggregate, weekly = summarize(daily, config), summarize(daily, config, weekly=True)
    aggregate.to_csv(output / "seasonal_recurrence.csv", index=False)
    weekly.to_csv(output / "weekly_recurrence.csv", index=False)
    for regime in sorted(set(zip(daily.season, daily.day_type))):
        representative_inputs(daily, profiles, regime, "development").to_csv(
            output / f"{'_'.join(regime)}_development_profiles.csv"
        )
    plots(daily, aggregate, weekly, output)
    metadata = {
        "sources": sources,
        "ingestion": audits,
        "chronological_split_dates": splits,
        "total_gsps": len(ids),
        "daily_rows": len(daily),
        "valid_daily_rows": int(daily.valid_day.sum()),
        "imputation": "none: recurrence denominator is valid source-observed days including source estimates",
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "pandas", "scipy", "scikit-learn", "torch", "pyro-ppl"]
        },
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "src/eso").glob("*.py")
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    if not skip_models:
        benchmark(daily, profiles, aggregate, config, output)
    (output / "COMPLETE.json").write_text(
        json.dumps({"screening_complete": True, "models_complete": not skip_models})
    )
    print(f"Completed: {output}", flush=True)
    return daily, aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/phase1_seasonal.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-models", action="store_true")
    args = parser.parse_args()
    run(args.config, args.output, args.skip_models)


if __name__ == "__main__":
    main()
