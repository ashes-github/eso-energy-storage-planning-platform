"""Run observed-data Phase 2 baseline or statistical benchmarks."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

from .data import ForecastDataset
import pandas as pd

from .config import assess_training_history, load_config, model_factories, select_series
from .backtesting import WalkForward, backtest
from .evaluation import common_cases, summarize


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reproducibility(directory):
    root = Path(__file__).resolve().parents[3]
    packages = {}
    for name in ("numpy", "pandas", "scipy", "statsmodels"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    files = (
        ["observed_panel.csv"]
        if (directory / "observed_panel.csv").exists()
        else ["cleaned_panel.csv", "value_provenance.csv"]
    )
    return dict(
        python=platform.python_version(),
        packages=packages,
        git_commit=commit,
        git_status=dirty,
        source_sha256={
            str(p.relative_to(root)): sha256(p)
            for p in sorted((root / "src" / "eso" / "forecasting").glob("*.py"))
        },
        input_sha256={name: sha256(directory / name) for name in files},
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Observed-panel directory")
    parser.add_argument(
        "--output", required=True, help="New directory; existing paths are refused"
    )
    parser.add_argument("--config", help="Phase 2 JSON configuration")
    parser.add_argument(
        "--test-start", required=True, help="UTC timestamp, e.g. 2022-08-15T00:00:00Z"
    )
    parser.add_argument(
        "--min-train", type=int,
        help="Initial history in half-hours; default 672 is a smoke minimum. "
             "Study floor: 8640 (180 elapsed days); prefer 6–12 months or 1–2+ years."
    )
    parser.add_argument(
        "--step", type=int, help="Override origin spacing in half-hours"
    )
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output}")
    config = load_config(args.config)
    min_train = (
        args.min_train if args.min_train is not None else config["min_train_days"] * 48
    )
    step = args.step if args.step is not None else config["step"]
    plan = WalkForward(
        min_train,
        args.test_start,
        step,
        tuple(config["horizons"]),
        config["mase_period"],
    )
    data = ForecastDataset.from_phase1(args.input)
    origins = list(plan.folds(data.panel.columns, args.split))
    selected, selection = select_series(data.panel, min_train, config)
    data = ForecastDataset(selected)
    models = model_factories(config)
    history_assessment = assess_training_history(min_train, config)
    print(
        f"Training history: {min_train} half-hours "
        f"({min_train / 48:g} elapsed days); {history_assessment['classification']}.",
        flush=True,
    )
    if history_assessment['short_history_models']:
        print(
            "Short-history models: " + ', '.join(history_assessment['short_history_models'])
            + ". Use these results to verify the pipeline. For a modelling study, "
              "aim for 6–12 months, preferably 1–2+ years where coverage permits.",
            flush=True,
        )
    if {"sarima", "ets"} & set(models):
        try:
            import statsmodels  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                'Install the statistical extra: pip install -e ".[statistical]"'
            ) from exc
    config.update(
        min_train=min_train, step=step, test_start=args.test_start, split=args.split
    )
    metadata = dict(
        plan=asdict(plan),
        split=args.split,
        input=str(Path(args.input).resolve()),
        units="source settlement-period Meter Volume; no conversion to MW",
        revision_policy="retrospective latest revisions; not operational as-of",
        availability="zero publication delay assumed",
        imputation="none; original observations only",
        resolved_config=config,
        training_history_assessment=history_assessment,
        reproducibility=reproducibility(Path(args.input)),
        selected_series=list(data.panel.index),
        selection_history_end=data.panel.columns[min_train - 1].isoformat(),
        initial_training_range=[
            data.panel.columns[0].isoformat(),
            data.panel.columns[min_train - 1].isoformat(),
        ],
        evaluation_range=[
            data.panel.columns[origins[0]].isoformat(),
            data.panel.columns[origins[-1] + max(plan.horizons) - 1].isoformat(),
        ],
        n_origins=len(origins),
        missing_policy=dict(
            sarima="state-space gaps retained", ets="refuse incomplete fitting windows"
        ),
        convergence_policy="nonconverged fits emit NaN; no baseline fallback",
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    selection.to_csv(output / "series_selection.csv")
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "run_status.json").write_text(
        json.dumps({"status": "running"}), encoding="utf-8"
    )
    print(
        f"Evaluating {len(data.panel)} GSPs, {len(origins)} origins, models: {list(models)}",
        flush=True,
    )
    diagnostics = []
    try:
        predictions = backtest(
            data,
            plan,
            models,
            args.split,
            diagnostics=diagnostics,
            artifact_dir=output / "models" if config["save_models"] else None,
            progress=lambda origin, name: print(
                f"{origin}: fitting {name}", flush=True
            ),
        )
        predictions.to_csv(output / "predictions.csv", index=False)
        groups = ("model", "split", "horizon")
        summarize(predictions).to_csv(output / "metrics_per_gsp.csv", index=False)
        summarize(predictions, groups).to_csv(
            output / "metrics_pooled.csv", index=False
        )
        summarize(predictions, (*groups, "lead")).to_csv(
            output / "metrics_by_lead.csv", index=False
        )
        common = common_cases(predictions)
        summarize(common, groups).to_csv(
            output / "metrics_common_cases.csv", index=False
        )
        summarize(common, (*groups, "lead")).to_csv(
            output / "metrics_common_cases_by_lead.csv", index=False
        )
        if diagnostics:
            pd.DataFrame(diagnostics).to_csv(
                output / "fit_diagnostics.csv", index=False
            )
        (output / "run.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        (output / "run_status.json").write_text(
            json.dumps({"status": "complete"}), encoding="utf-8"
        )
    except Exception as exc:
        (output / "run_status.json").write_text(
            json.dumps({"status": "failed", "error": str(exc)}), encoding="utf-8"
        )
        raise
    print(f"Wrote {len(predictions)} forecast cases to {output}")


if __name__ == "__main__":
    main()
