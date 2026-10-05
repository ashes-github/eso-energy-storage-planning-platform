"""Run the Phase 2 baseline benchmark against a saved Phase 1 panel."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .data import ForecastDataset
from .baselines import SeasonalNaive
from .backtesting import WalkForward, backtest
from .evaluation import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Phase 1 output directory")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--test-start", required=True, help="UTC timestamp, e.g. 2022-08-15T00:00:00Z"
    )
    parser.add_argument("--min-train", type=int, default=48 * 14)
    parser.add_argument("--step", type=int, default=48)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    args = parser.parse_args()
    data = ForecastDataset.from_phase1(args.input)
    print(
        f"Loaded {data.panel.shape[0]} series with {data.panel.shape[1]} timestamps from {args.input}"
    )
    plan = WalkForward(args.min_train, args.test_start, args.step)
    models = {
        "persistence": lambda: SeasonalNaive(1),
        "seasonal_48": lambda: SeasonalNaive(48),
        "seasonal_336": lambda: SeasonalNaive(336),
    }
    predictions = backtest(data, plan, models, args.split)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    summarize(predictions).to_csv(output / "metrics_per_gsp.csv", index=False)
    summarize(predictions, ("model", "split", "horizon")).to_csv(
        output / "metrics_pooled.csv", index=False
    )
    metadata = dict(
        plan=asdict(plan),
        split=args.split,
        input=str(Path(args.input).resolve()),
        units="source Meter Volume; no conversion",
        revision_policy="retrospective latest revisions; not operational as-of",
        availability="zero publication delay assumed",
        imputation="none; original observations only",
    )
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(predictions)} forecast cases to {output}")


if __name__ == "__main__":
    main()
