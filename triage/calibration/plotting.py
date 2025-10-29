# Location: triage/calibration/plotting.py
# Purpose: Generate reliability diagrams and coverage plots for calibration analysis.
# Why: Phase 5 requires visual artefacts so operators can inspect calibration quality at a glance.

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, Sequence

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ModuleNotFoundError:  # pragma: no cover
    HAS_MPL = False

from triage.calibration.reliability_plots import ReliabilityResult


@dataclass
class CalibrationPlots:
    reliability_png: Path
    coverage_png: Path
    report_html: Path


def generate_plots(
    reliability: ReliabilityResult,
    coverage: Dict[str, float],
    output_dir: Path,
    *,
    prefix: str = "calibration_reliability",
    metadata: Dict[str, float] | None = None,
) -> CalibrationPlots:
    output_dir.mkdir(parents=True, exist_ok=True)
    reliability_path = output_dir / f"{prefix}.png"
    coverage_path = output_dir / f"{prefix}_coverage.png"
    _plot_reliability(reliability, reliability_path)
    _plot_coverage(coverage, coverage_path)
    report_path = output_dir / f"{prefix}.html"
    _write_html_report(reliability_path, coverage_path, report_path, coverage, metadata)
    return CalibrationPlots(
        reliability_png=reliability_path,
        coverage_png=coverage_path,
        report_html=report_path,
    )


def _plot_reliability(result: ReliabilityResult, output_path: Path) -> None:
    if not HAS_MPL:
        output_path.write_text("Matplotlib unavailable; reliability plot not generated.")
        return
    plt.figure(figsize=(6, 5))
    plt.plot(result.confidences, result.accuracies, marker="o", label="Observed accuracy")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    plt.title("Reliability Diagram")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_coverage(coverage: Dict[str, float], output_path: Path) -> None:
    if not HAS_MPL:
        output_path.write_text("Matplotlib unavailable; coverage plot not generated.")
        return
    thresholds = sorted(float(t) for t in coverage.keys() if t.endswith("_coverage"))
    coverage_values = [coverage[f"{t:.2f}_coverage"] for t in thresholds]
    accuracy_values = [coverage.get(f"{t:.2f}", 0.0) for t in thresholds]
    plt.figure(figsize=(6, 5))
    plt.plot(thresholds, coverage_values, marker="o", label="Coverage")
    plt.plot(thresholds, accuracy_values, marker="s", label="Accuracy")
    plt.title("Coverage vs Accuracy")
    plt.xlabel("Threshold")
    plt.ylabel("Value")
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _write_html_report(
    reliability_png: Path,
    coverage_png: Path,
    report_path: Path,
    coverage: Dict[str, float],
    metadata: Dict[str, float] | None,
) -> None:
    reliability_b64 = _encode_png(reliability_png)
    coverage_b64 = _encode_png(coverage_png)
    rows = []
    for key in sorted(coverage.keys()):
        rows.append(f"<tr><td>{key}</td><td>{coverage[key]:.4f}</td></tr>")
    meta_rows = []
    if metadata:
        for key, value in metadata.items():
            meta_rows.append(f"<tr><td>{key}</td><td>{value:.4f}</td></tr>")
    html = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Calibration Report</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 2rem; }}
      img {{ max-width: 600px; display: block; margin-bottom: 2rem; }}
      table {{ border-collapse: collapse; margin-bottom: 2rem; }}
      th, td {{ border: 1px solid #ccc; padding: 0.5rem 0.75rem; text-align: left; }}
      th {{ background-color: #f5f5f5; }}
    </style>
  </head>
  <body>
    <h1>Calibration Reliability Report</h1>
    <section>
      <h2>Reliability Diagram</h2>
      <img src="data:image/png;base64,{reliability_b64}" alt="Reliability diagram" />
    </section>
    <section>
      <h2>Coverage vs Accuracy</h2>
      <img src="data:image/png;base64,{coverage_b64}" alt="Coverage vs accuracy" />
    </section>
    <section>
      <h2>Threshold Metrics</h2>
      <table>
        <thead><tr><th>Metric</th><th>Value</th></tr></thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    </section>
    <section>
      <h2>Metadata</h2>
      <table>
        <thead><tr><th>Key</th><th>Value</th></tr></thead>
        <tbody>
          {meta_rows}
        </tbody>
      </table>
    </section>
  </body>
</html>
    """.format(rows="\n".join(rows), meta_rows="\n".join(meta_rows), reliability_b64=reliability_b64, coverage_b64=coverage_b64)
    report_path.write_text(html)


def _encode_png(path: Path) -> str:
    buffer = BytesIO()
    with open(path, "rb") as handle:
        buffer.write(handle.read())
    return base64.b64encode(buffer.getvalue()).decode("ascii")
