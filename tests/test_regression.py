from __future__ import annotations

from pathlib import Path

from scripts.run_regression import build_run_card, run_commands


def test_run_regression_dry_run(tmp_path: Path) -> None:
    # Ensure dry-run does not raise
    run_commands(dry_run=True)
    output = tmp_path / "run_card.json"
    build_run_card(output)
    assert output.exists()
