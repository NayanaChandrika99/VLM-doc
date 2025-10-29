from __future__ import annotations

from pathlib import Path

from triage.triage.apply_abstention import apply_abstention, parse_args


def test_apply_abstention_dry_run(tmp_path: Path) -> None:
    output = tmp_path / "abstention.json"
    args = parse_args([
        "--output",
        str(output),
        "--dry-run",
        "--thresholds",
        "0.80",
        "0.95",
    ])
    result = apply_abstention(args)
    assert output.exists()
    assert "0.80" in result["results"]
