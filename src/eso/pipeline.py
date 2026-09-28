"""Run from the project root: python -m eso.pipeline --config configs/phase1.json."""

import argparse
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data import load_records, align_panel
from .imputation import impute_panel, settings as missing_settings
from .features import screen_with_quality
from .models import pca_model, autoencoder_model, gplvm_model
from .evaluation import latent_metrics, distance_stability
from .visualization import latent_plot, loss_plot, geospatial_plot

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def validate(config):
    missing_settings(config.get("missing"))
    if config["direction"] not in {"I", "E", "signed_net"}:
        raise ValueError("direction must be I, E or signed_net")
    if (
        config["direction"] != "signed_net"
        and missing_settings(config.get("missing"))["method"] != "none"
    ):
        raise ValueError(
            "Imputation requires signed_net so exports are not mistaken for missing imports; use missing.method=none for a legacy direction-only run"
        )
    if config["latent_dim"] < 2 or config["steps"] < 1 or config["infer_steps"] < 1:
        raise ValueError(
            "At least two latent dimensions and positive training/inference steps required"
        )
    if not 0 < config["test_fraction"] < 1 or not 0 < config["observed_fraction"] < 1:
        raise ValueError("Split fractions must be between zero and one")
    if not config["seeds"] or any(s <= 0 for s in config["prior_scales"]):
        raise ValueError("Seeds and positive prior scales required")
    if config["inducing_count"] < 1 or config["ae_steps"] < 1:
        raise ValueError("Positive inducing count and AE training steps required")
    if not config["prior_modes"] or not set(config["prior_modes"]) <= {
        "independent",
        "label_informed",
    }:
        raise ValueError("Specify independent and/or label_informed prior modes")
    if not config["inducing_strategies"] or not set(config["inducing_strategies"]) <= {
        "random",
        "kmeans",
    }:
        raise ValueError("Specify random and/or kmeans inducing initialization")
    temporal = config["features"]["temporal"]
    if not 0 <= temporal["low_fraction"] < temporal["min_peak_fraction"] <= 1:
        raise ValueError("Temporal fractions must satisfy 0 <= low < peak <= 1")
    if temporal["min_hours"] <= 0 or temporal["following_hours"] < 0.5:
        raise ValueError(
            "Positive low-run duration and at least half-hour following window required"
        )
    for key in ["temporal_weight", "min_day_fraction", "score_threshold"]:
        if not 0 <= config["screening"][key] <= 1:
            raise ValueError(f"Invalid screening {key}")


def run(config, output):
    validate(config)
    torch.set_num_threads(config.get("torch_threads", 2))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    paths = sorted(ROOT.glob(config["input_glob"]))
    print(f"Reading {len(paths)} source files", flush=True)
    records, sources = load_records(
        paths, config["start"], config["end"], config["direction"]
    )
    raw_panel, provenance, audit = align_panel(
        records, config["start"], config["end"], config["direction"]
    )
    raw_panel.to_csv(output / "observed_panel.csv")
    panel, provenance, quality = impute_panel(
        raw_panel, provenance, config.get("missing")
    )
    quality["export_periods"] = (provenance == "observed_export").sum(axis=1)
    quality["import_periods"] = (provenance == "observed_import").sum(axis=1)
    quality.to_csv(output / "data_quality.csv")
    provenance.to_csv(output / "value_provenance.csv")
    raw_panel.isna().to_csv(output / "missing_mask.csv")
    # Compatibility file: these rows remain in screening as insufficient evidence.
    quality.loc[~quality.model_eligible].to_csv(output / "excluded_gsps.csv")
    audit.update(
        retained_gsps=len(panel),
        model_eligible_gsps=int(quality.model_eligible.sum()),
        excluded_gsps=int((~quality.model_eligible).sum()),
        missing_policy=missing_settings(config.get("missing")),
        imputed_gsps=int(quality.data_status.eq("imputed").sum()),
        imputed_periods=int(quality.imputed_periods.sum()),
        insufficient_evidence_gsps=int(
            quality.data_status.eq("insufficient_evidence").sum()
        ),
        constant_profiles_retained=int(quality.constant_profile.sum()),
    )
    # This full panel is reusable by a forecasting pipeline; retain UTC and identities.
    panel.to_csv(output / "cleaned_panel.csv")
    features = screen_with_quality(
        panel,
        quality,
        config["features"],
        dict(
            config["screening"],
            allow_signed_exports=config["direction"] == "signed_net",
        ),
    )
    features.to_csv(output / "screening.csv")
    audit["sources"] = sources
    audit["settlement_window"] = [config["start"], config["end"]]
    audit["panel_sha256"] = hashlib.sha256(
        (output / "cleaned_panel.csv").read_bytes()
    ).hexdigest()
    if config.get("shapefile"):
        audit["unmatched_map_gsps"] = geospatial_plot(
            features, ROOT / config["shapefile"], output
        )
    write_json(output / "data_audit.json", audit)
    print(
        f"Clean panel: {panel.shape}; candidates: {int(features.candidate.sum())}",
        flush=True,
    )
    panel = panel.loc[quality.model_eligible]
    if config.get("max_gsps"):
        ids = np.random.default_rng(config["split_seed"]).choice(
            panel.index, min(config["max_gsps"], len(panel)), replace=False
        )
        panel = panel.loc[sorted(ids)]

    print(f"Training matrix: {panel.shape}", flush=True)
    # Never score against imputed 'ground truth', or use test future values to
    # impute its observed prefix. Holdout GSPs must be originally fully observed.
    complete_ids = np.flatnonzero(
        quality.loc[panel.index, "data_status"].eq("observed")
    )
    if len(complete_ids) < 10:
        write_json(
            output / "run_status.json",
            {
                "status": "complete",
                "model_runs": 0,
                "validation": "Screening only: fewer than ten fully observed GSPs for holdout",
            },
        )
        return output
    _, test_ids = train_test_split(
        complete_ids,
        test_size=config["test_fraction"],
        random_state=config["split_seed"],
    )
    train_ids = np.setdiff1d(np.arange(len(panel)), test_ids)
    train_ids, test_ids = np.sort(train_ids), np.sort(test_ids)
    if config["latent_dim"] >= min(len(train_ids), panel.shape[1]):
        raise ValueError(
            "latent_dim must be smaller than the training matrix dimensions"
        )
    # Train-only column standardization; test withheld values never fit preprocessing.
    scaler = StandardScaler().fit(panel.iloc[train_ids])
    train = scaler.transform(panel.iloc[train_ids])
    test = scaler.transform(panel.iloc[test_ids])
    observed_count = int(panel.shape[1] * config["observed_fraction"])
    observed = np.arange(observed_count)
    withheld = np.arange(observed_count, panel.shape[1])
    labels = features.loc[panel.index[train_ids], "candidate"].to_numpy(dtype=int)
    joblib.dump(scaler, output / "scaler.joblib")
    write_json(
        output / "split.json",
        {
            "train_gsps": panel.index[train_ids].tolist(),
            "test_gsps": panel.index[test_ids].tolist(),
            "observed_periods": observed.tolist(),
            "withheld_periods": withheld.tolist(),
            "timestamps_utc": [str(t) for t in panel.columns],
            "holdout_policy": "Originally fully observed GSPs only; imputed GSPs may enter training",
            "imputed_training_gsps": quality.loc[panel.index[train_ids]]
            .query("data_status == 'imputed'")
            .index.tolist(),
        },
    )
    versions = {
        p: importlib.metadata.version(p)
        for p in [
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
            "torch",
            "pyro-ppl",
            "matplotlib",
        ]
    }
    versions["python"] = platform.python_version()
    write_json(output / "versions.json", versions)
    write_json(
        output / "source_hashes.json",
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    )
    summaries, reference = [], {}

    def save(
        name, z, reconstruction, prediction, artifact, losses=None, sd=None, group=None
    ):
        folder = output / name
        folder.mkdir(exist_ok=True, parents=True)
        if not all(np.isfinite(a).all() for a in [z, reconstruction, prediction]):
            raise ValueError(f"{name}: non-finite model output")
        metrics = latent_metrics(z, labels)
        metrics["train_reconstruction_mse_standardized"] = float(
            np.mean((reconstruction - train) ** 2)
        )
        metrics["heldout_gsp_withheld_period_mse_standardized"] = float(
            np.mean((prediction[:, withheld] - test[:, withheld]) ** 2)
        )
        raw_prediction = scaler.inverse_transform(prediction)
        raw_target = panel.iloc[test_ids].to_numpy()
        metrics["heldout_gsp_withheld_period_rmse_meter_volume"] = float(
            np.sqrt(
                np.mean((raw_prediction[:, withheld] - raw_target[:, withheld]) ** 2)
            )
        )
        if group in reference:
            metrics["distance_rank_stability_vs_first_seed"] = distance_stability(
                reference[group], z
            )
        elif group:
            reference[group] = z.copy()
        latent = pd.DataFrame(
            z,
            index=panel.index[train_ids],
            columns=[f"z{i+1}" for i in range(z.shape[1])],
        )
        latent["candidate"] = labels
        latent.to_csv(folder / "latent.csv")
        np.savez_compressed(
            folder / "predictions.npz",
            train_reconstruction=reconstruction,
            test_prediction_standardized=prediction,
            test_prediction_meter_volume=raw_prediction,
        )
        if sd is not None:
            pd.DataFrame(
                sd,
                index=latent.index,
                columns=[f"z{i+1}_sd" for i in range(sd.shape[1])],
            ).to_csv(folder / "latent_posterior_sd.csv")
            metrics["mean_latent_posterior_sd"] = float(np.mean(sd))
        if isinstance(artifact, dict):
            torch.save(artifact, folder / "model.pt")
        else:
            joblib.dump(artifact, folder / "model.joblib")
        if losses is not None:
            pd.DataFrame({"loss": losses}).to_csv(folder / "loss.csv", index=False)
            loss_plot(losses, folder / "loss.png")
        latent_plot(z, labels, folder / "latent.png", name.replace("_", " "))
        write_json(folder / "metrics.json", metrics)
        summaries.append({"model": name, **metrics})
        pd.DataFrame(summaries).to_csv(output / "comparison.csv", index=False)
        print(
            f'{name}: withheld MSE {metrics["heldout_gsp_withheld_period_mse_standardized"]:.4f}',
            flush=True,
        )

    save(
        "pca",
        *pca_model(train, test, observed, config["latent_dim"], config["split_seed"]),
    )
    for seed in config["seeds"]:
        save(
            f"autoencoder_seed{seed}",
            *autoencoder_model(
                train,
                test,
                observed,
                config["latent_dim"],
                seed,
                config["ae_steps"],
                config["infer_steps"],
            ),
            group="autoencoder",
        )
        for mode, scale, strategy in itertools.product(
            config["prior_modes"], config["prior_scales"], config["inducing_strategies"]
        ):
            group = f"gplvm_{mode}_scale{scale}_{strategy}"
            save(
                f"{group}_seed{seed}",
                *gplvm_model(
                    train,
                    test,
                    observed,
                    labels,
                    config["latent_dim"],
                    seed,
                    mode,
                    scale,
                    config["inducing_count"],
                    strategy,
                    config["steps"],
                    config["infer_steps"],
                ),
                group=group,
            )
    write_json(
        output / "run_status.json",
        {
            "status": "complete",
            "model_runs": len(summaries),
            "validation": "Held-out GSP completion; labels are uncalibrated descriptive proxies",
            "quick": config.get("quick", False),
        },
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1.json")
    parser.add_argument("--output", default="output/phase1")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Short validation run; not a convergence study",
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.quick:
        config.update(
            seeds=[1],
            prior_scales=[1.0],
            inducing_strategies=["kmeans"],
            steps=40,
            ae_steps=80,
            infer_steps=60,
            max_gsps=80,
            quick=True,
        )
    run(config, args.output)


if __name__ == "__main__":
    main()
