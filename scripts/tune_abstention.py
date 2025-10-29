"""Sweep abstention thresholds to recommend an operating point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ModuleNotFoundError:  # pragma: no cover
    HAS_MPL = False

from triage.calibration.temp_scaling import TemperatureScaler, load_temperature
from triage.io.structured import allowed_labels


def load_logits(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    if "logits" not in data or "labels" not in data:
        raise ValueError("Logits NPZ must contain 'logits' and 'labels' arrays.")
    payload: Dict[str, np.ndarray] = {"logits": data["logits"], "labels": data["labels"]}
    if "probabilities" in data:
        payload["probabilities"] = data["probabilities"]
    if "uids" in data:
        payload["uids"] = data["uids"]
    return payload


def compute_probabilities(logits: np.ndarray, temperature_path: Path | None) -> np.ndarray:
    scaler = TemperatureScaler()
    if temperature_path and temperature_path.exists():
        scaler = load_temperature(temperature_path)
    scaled = logits / scaler.temperature
    exp_logits = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def sweep_thresholds(probabilities: np.ndarray, labels: np.ndarray, thresholds: Sequence[float], min_coverage: float) -> Dict[str, object]:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    label_names = list(allowed_labels(include_unknown=False))

    results = []
    best = {"threshold": None, "score": -1.0}
    for threshold in thresholds:
        mask = confidences >= threshold
        coverage = float(mask.mean())
        if mask.sum() == 0:
            accuracy = 0.0
            macro_f1 = 0.0
        else:
            accuracy = float((predictions[mask] == labels[mask]).mean())
            macro_f1 = _macro_f1(labels[mask], predictions[mask], label_names)
        entry = {
            "threshold": threshold,
            "coverage": coverage,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        }
        results.append(entry)
        if coverage >= min_coverage and macro_f1 > best["score"]:
            best = {"threshold": threshold, "score": macro_f1}

    if best["threshold"] is None and results:
        best_entry = max(results, key=lambda item: item["macro_f1"])
        best = {"threshold": best_entry["threshold"], "score": best_entry["macro_f1"]}
    return {"results": results, "recommended_threshold": best["threshold"]}


def _macro_f1(true_labels: np.ndarray, predicted_labels: np.ndarray, label_names: Sequence[str]) -> float:
    num_classes = len(label_names)
    f1_sum = 0.0
    for class_index in range(num_classes):
        tp = int(((true_labels == class_index) & (predicted_labels == class_index)).sum())
        fp = int(((true_labels != class_index) & (predicted_labels == class_index)).sum())
        fn = int(((true_labels == class_index) & (predicted_labels != class_index)).sum())
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom else 0.0
        f1_sum += f1
    return f1_sum / num_classes if num_classes else 0.0


def plot_results(results: Sequence[Dict[str, float]], output_path: Path) -> None:
    if not HAS_MPL:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        return
    thresholds = [item["threshold"] for item in results]
    coverage = [item["coverage"] for item in results]
    accuracy = [item["accuracy"] for item in results]
    macro_f1 = [item["macro_f1"] for item in results]

    plt.figure(figsize=(6, 5))
    plt.plot(thresholds, coverage, marker="o", label="Coverage")
    plt.plot(thresholds, accuracy, marker="s", label="Accuracy")
    plt.plot(thresholds, macro_f1, marker="^", label="Macro F1")
    plt.xlabel("Confidence threshold")
    plt.ylabel("Metric value")
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.title("Abstention Threshold Sweep")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep abstention thresholds and recommend an operating point.")
    parser.add_argument("--logits", default="triage/artifacts/rvl_val_logits.npz", help="NPZ file with logits + labels.")
    parser.add_argument("--temperature", help="Optional temperature calibration JSON.")
    parser.add_argument("--thresholds", nargs="*", type=float, help="Explicit thresholds to evaluate.")
    parser.add_argument("--grid-start", type=float, default=0.5, help="Start of automatic threshold grid.")
    parser.add_argument("--grid-stop", type=float, default=0.99, help="End of automatic threshold grid.")
    parser.add_argument("--grid-step", type=float, default=0.05, help="Step size for automatic grid.")
    parser.add_argument("--min-coverage", type=float, default=0.6, help="Minimum coverage required for recommended threshold.")
    parser.add_argument("--output", default="metrics/abstention_sweep.json", help="Where to store JSON summary.")
    parser.add_argument("--plot", default="reports/abstention_curve.png", help="Where to store PNG plot.")
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic logits for development.")
    parser.add_argument("--dry-run-size", type=int, default=256, help="Synthetic sample count.")
    parser.add_argument("--num-classes", type=int, default=16, help="Number of classes for synthetic data.")
    parser.add_argument("--random-seed", type=int, default=7, help="Random seed for synthetic data.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dry_run:
        rng = np.random.default_rng(args.random_seed)
        logits = rng.standard_normal((args.dry_run_size, args.num_classes)).astype(np.float32)
        labels = rng.integers(0, args.num_classes, size=args.dry_run_size)
    else:
        payload = load_logits(Path(args.logits))
        logits = payload["logits"]
        labels = payload["labels"]

    if args.thresholds:
        thresholds = sorted(set(args.thresholds))
    else:
        thresholds = list(np.arange(args.grid_start, args.grid_stop + 1e-9, args.grid_step))

    probabilities = compute_probabilities(logits, Path(args.temperature) if args.temperature else None)
    summary = sweep_thresholds(probabilities, labels, thresholds, args.min_coverage)
    plot_results(summary["results"], Path(args.plot))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
