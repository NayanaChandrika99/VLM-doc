from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from triage.calibration.fit_temperature import fit_temperature_cli, parse_args as parse_fit_args
from triage.calibration.evaluate_calibration import evaluate, parse_args as parse_eval_args


def test_fit_temperature_dry_run(tmp_path: Path) -> None:
    output = tmp_path / "temp.json"
    args = parse_fit_args([
        "--output",
        str(output),
        "--dry-run",
        "--dry-run-size",
        "64",
        "--num-classes",
        "8",
    ])
    result = fit_temperature_cli(args)
    assert output.exists()
    payload = json.loads(output.read_text())
    assert payload["state"]["temperature"] == result["temperature"]


def test_evaluate_calibration_dry_run(tmp_path: Path) -> None:
    output = tmp_path / "calibration.json"
    args = parse_eval_args([
        "--output",
        str(output),
        "--dry-run",
        "--dry-run-size",
        "32",
        "--num-classes",
        "4",
    ])
    summary = evaluate(args)
    assert output.exists()
    data = json.loads(output.read_text())
    assert data["ece"] >= 0.0
    assert "coverage_vs_error" in summary
