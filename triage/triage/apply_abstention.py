# Location: triage/triage/apply_abstention.py
# Purpose: CLI tool to apply calibration + abstention thresholds to classifier outputs.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from triage.calibration.temp_scaling import TemperatureScaler, load_temperature
from triage.io.abstention import apply_threshold


def load_logits(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    required = {"logits", "labels", "probabilities"}
    if not required.issubset(data.files):
        raise ValueError(f"Expected logits NPZ to contain {required}; found {data.files}")
    return {key: data[key] for key in data.files}


def apply_abstention(args: argparse.Namespace) -> dict[str, object]:
    if args.dry_run:
        rng = np.random.default_rng(args.random_seed)
        logits = rng.standard_normal((args.dry_run_size, args.num_classes)).astype(np.float32)
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        labels = rng.integers(0, args.num_classes, size=args.dry_run_size)
        uids = [f"dry-{i}" for i in range(args.dry_run_size)]
    else:
        payload = load_logits(Path(args.logits))
        logits = payload["logits"]
        probabilities = payload.get("probabilities")
        if probabilities is None:
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        labels = payload.get("labels")
        uids = payload.get("uids", [str(i) for i in range(len(labels))])

    scaler = TemperatureScaler()
    if args.temperature and Path(args.temperature).exists():
        scaler = load_temperature(args.temperature)
        scaled_logits = logits / scaler.temperature
        exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    prediction_labels = [str(p) for p in predictions.tolist()]

    threshold_results = {}
    for threshold in args.thresholds:
        res = apply_threshold(prediction_labels, confidences, threshold)
        threshold_results[f"{threshold:.2f}"] = {
            "coverage": res.coverage,
            "abstain_rate": res.abstain_rate,
        }

    output_payload = {
        "uids": list(uids),
        "predictions": prediction_labels,
        "confidences": confidences.tolist(),
        "thresholds": args.thresholds,
        "results": threshold_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2))
    return output_payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply abstention thresholds to classifier outputs.")
    parser.add_argument("--logits", default="triage/artifacts/rvl_val_logits.npz", help="NPZ with logits/probabilities/labels.")
    parser.add_argument("--temperature", default="triage/artifacts/calibration_temp.json", help="Temperature calibration JSON.")
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.95], help="Confidence thresholds to evaluate.")
    parser.add_argument("--output", default="metrics/abstention.json", help="Path to store abstention metrics.")
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic logits for development.")
    parser.add_argument("--dry-run-size", type=int, default=128, help="Synthetic sample count.")
    parser.add_argument("--num-classes", type=int, default=16, help="Number of classes for synthetic data.")
    parser.add_argument("--random-seed", type=int, default=31, help="Random seed for synthetic data.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = apply_abstention(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
