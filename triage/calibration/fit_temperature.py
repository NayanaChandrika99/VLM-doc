# Location: triage/calibration/fit_temperature.py
# Purpose: CLI to fit temperature scaling from stored logits.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from triage.calibration.temp_scaling import TemperatureScaler, save_temperature


def load_logits(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    required = {"logits", "labels"}
    if not required.issubset(data.files):
        raise ValueError(f"Logits NPZ must contain {required}; found {data.files}")
    return {key: data[key] for key in data.files}


def fit_temperature_cli(args: argparse.Namespace) -> Dict[str, object]:
    if args.dry_run:
        rng = np.random.default_rng(args.random_seed)
        num_samples = args.dry_run_size
        num_classes = args.num_classes
        logits = rng.standard_normal((num_samples, num_classes)).astype(np.float32)
        labels = rng.integers(0, num_classes, size=num_samples)
    else:
        path = Path(args.logits)
        data = load_logits(path)
        logits = data["logits"]
        labels = data.get("labels")
        if labels is None:
            raise ValueError("Logits NPZ must include `labels` array")

    scaler = TemperatureScaler()
    temperature = scaler.fit(logits, labels, max_iter=args.max_iter, lr=args.lr)

    metadata = {
        "source": str(args.logits) if not args.dry_run else "synthetic",
        "samples": int(logits.shape[0]),
        "classes": int(logits.shape[1]),
    }
    save_temperature(args.output, scaler, metadata)
    return {"temperature": temperature, "metadata": metadata, "output": str(args.output)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit temperature scaling from stored logits.")
    parser.add_argument("--logits", default="triage/artifacts/rvl_val_logits.npz", help="NPZ file containing logits + labels.")
    parser.add_argument("--output", default="triage/artifacts/calibration_temp.json", help="Where to store the temperature JSON.")
    parser.add_argument("--max-iter", type=int, default=500, help="Optimizer max iterations.")
    parser.add_argument("--lr", type=float, default=0.01, help="Optimizer learning rate.")
    parser.add_argument("--dry-run", action="store_true", help="Generate synthetic logits for development.")
    parser.add_argument("--dry-run-size", type=int, default=256, help="Number of synthetic samples when using dry-run.")
    parser.add_argument("--num-classes", type=int, default=16, help="Number of classes for synthetic logits.")
    parser.add_argument("--random-seed", type=int, default=13, help="Random seed for synthetic data.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = fit_temperature_cli(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
