"""Run staged regression checks for the triage project."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List


DEFAULT_COMMANDS: List[List[str]] = [
    ["pytest", "tests/test_data_adapters.py"],
    ["pytest", "tests/test_layout_features.py", "tests/test_layout_dataset.py", "tests/test_layout_head.py"],
    ["pytest", "tests/test_rvl_dataset.py", "tests/test_rvl_model.py", "tests/test_rvl_training.py", "tests/test_rvl_evaluation.py"],
    ["pytest", "tests/test_calibration.py", "tests/test_calibration_plotting.py", "tests/test_abstention.py", "tests/test_abstention_tuning.py"],
    ["pytest", "tests/test_serving_app.py", "tests/test_adapter_management.py", "tests/test_lora_scaffolding.py"],
]


def run_commands(dry_run: bool) -> None:
    for command in DEFAULT_COMMANDS:
        if dry_run:
            print("[DRY-RUN]", " ".join(command))
        else:
            print("Running:", " ".join(command))
            subprocess.run(command, check=True)


def build_run_card(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run_card = {
        "artifacts": {
            "classifier_checkpoint": str(Path("triage/artifacts/rvl_classifier.pt")),
            "validation_logits": str(Path("triage/artifacts/rvl_val_logits.npz")),
            "calibration": str(Path("triage/artifacts/calibration_temp.json")),
            "metrics": str(Path("metrics/rvl_classifier.json")),
        },
        "commands": [" ".join(cmd) for cmd in DEFAULT_COMMANDS],
    }
    output.write_text(json.dumps(run_card, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute regression tests and generate run card.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--output", default="reports/run_card.json", help="Path to run card JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_commands(args.dry_run)
    build_run_card(Path(args.output))


if __name__ == "__main__":
    main()
