from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from triage.calibration.plotting import generate_plots
from triage.calibration.reliability_plots import ReliabilityResult
from triage.calibration.evaluate_calibration import evaluate, parse_args


def test_generate_plots(tmp_path: Path) -> None:
    reliability = ReliabilityResult(
        bins=[0.0, 0.5, 1.0],
        accuracies=[0.2, 0.8, 0.0],
        confidences=[0.1, 0.9, 0.0],
        counts=[10, 20, 0],
        ece=0.12,
    )
    coverage = {"0.50": 0.75, "0.50_coverage": 0.60}
    plots = generate_plots(reliability, coverage, tmp_path, metadata={"ece": 0.12})
    assert plots.reliability_png.exists()
    assert plots.coverage_png.exists()
    assert plots.report_html.exists()


def test_evaluate_calibration_with_plots(tmp_path: Path) -> None:
    output = tmp_path / "calibration.json"
    plot_dir = tmp_path / "reports"
    args = parse_args([
        "--output",
        str(output),
        "--dry-run",
        "--plot-dir",
        str(plot_dir),
        "--prefix",
        "test_calib",
    ])
    summary = evaluate(args)
    assert output.exists()
    assert "plots" in summary
    plots = summary["plots"]
    assert Path(plots["reliability_png"]).exists()
    assert Path(plots["report_html"]).exists()
