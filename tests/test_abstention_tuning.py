from __future__ import annotations

import json
from pathlib import Path

from scripts.tune_abstention import main


def test_tune_abstention_dry_run(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "abstention.json"
    plot = tmp_path / "curve.png"
    argv = [
        "--output",
        str(output),
        "--plot",
        str(plot),
        "--dry-run",
        "--dry-run-size",
        "64",
        "--grid-start",
        "0.5",
        "--grid-stop",
        "0.8",
        "--grid-step",
        "0.1",
    ]

    monkeypatch.setattr("sys.argv", ["tune_abstention.py", *argv])
    main(argv=None)
    assert output.exists()
    assert plot.exists()
    data = json.loads(output.read_text())
    assert data["recommended_threshold"] is not None
