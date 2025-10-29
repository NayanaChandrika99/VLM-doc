# Location: triage/calibration/evaluate_calibration.py
# Purpose: Evaluate calibrated logits and produce reliability metrics.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from triage.calibration.plotting import CalibrationPlots, generate_plots
from triage.calibration.reliability_plots import compute_reliability, coverage_vs_error
from triage.calibration.temp_scaling import TemperatureScaler, load_temperature


def load_logits(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    required = {"logits", "labels"}
    if not required.issubset(data.files):
        raise ValueError(f"Logits NPZ must contain {required}; found {data.files}")
    logits = data["logits"]
    labels = data["labels"]
    return {"logits": logits, "labels": labels}


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    if args.dry_run:
        rng = np.random.default_rng(args.random_seed)
        logits = rng.standard_normal((args.dry_run_size, args.num_classes)).astype(np.float32)
        labels = rng.integers(0, args.num_classes, size=args.dry_run_size)
        scaler = TemperatureScaler(temperature=1.2)
    else:
        payload = load_logits(Path(args.logits))
        logits = payload["logits"]
        labels = payload["labels"]
        if args.temperature:
            scaler = load_temperature(args.temperature)
        else:
            scaler = TemperatureScaler(temperature=1.0)

    scaled_logits = scaler.transform(logits)
    probabilities = np.exp(scaled_logits) / np.exp(scaled_logits).sum(axis=1, keepdims=True)

    reliability = compute_reliability(probabilities, labels, num_bins=args.num_bins)
    coverage = coverage_vs_error(probabilities, labels, args.thresholds)

    summary = {
        "temperature": scaler.temperature,
        "ece": reliability.ece,
        "reliability": {
            "bins": reliability.bins,
            "accuracies": reliability.accuracies,
            "confidences": reliability.confidences,
            "counts": reliability.counts,
        },
        "coverage_vs_error": coverage,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        metadata = {
            "temperature": scaler.temperature,
            "ece": reliability.ece,
        }
        plots: CalibrationPlots = generate_plots(
            reliability,
            coverage,
            plot_dir,
            prefix=args.prefix,
            metadata=metadata,
        )
        summary["plots"] = {
            "reliability_png": str(plots.reliability_png),
            "coverage_png": str(plots.coverage_png),
            "report_html": str(plots.report_html),
        }
        output_path.write_text(json.dumps(summary, indent=2))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate calibration performance.")
    parser.add_argument("--logits", default="triage/artifacts/rvl_val_logits.npz", help="NPZ containing logits + labels.")
    parser.add_argument("--temperature", default="triage/artifacts/calibration_temp.json", help="Temperature JSON path.")
    parser.add_argument("--output", default="metrics/calibration.json", help="Where to write calibration metrics.")
    parser.add_argument("--plot-dir", help="Directory to store reliability/coverage plots.")
    parser.add_argument("--prefix", default="calibration_reliability", help="Filename prefix for generated plots.")
    parser.add_argument("--num-bins", type=int, default=10, help="Number of bins for reliability diagram.")
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.90, 0.95, 0.97, 0.99], help="Confidence thresholds for coverage vs error.")
    parser.add_argument("--dry-run", action="store_true", help="Generate synthetic logits for development.")
    parser.add_argument("--dry-run-size", type=int, default=256, help="Synthetic sample count.")
    parser.add_argument("--num-classes", type=int, default=16, help="Number of classes for synthetic logits.")
    parser.add_argument("--random-seed", type=int, default=21, help="Random seed for dry-run generation.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = evaluate(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
