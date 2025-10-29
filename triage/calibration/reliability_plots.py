# Location: triage/calibration/reliability_plots.py
# Purpose: Compute reliability diagrams, expected calibration error, and coverage metrics.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


@dataclass
class ReliabilityResult:
    bins: Sequence[float]
    accuracies: Sequence[float]
    confidences: Sequence[float]
    counts: Sequence[int]
    ece: float


def compute_reliability(probabilities: np.ndarray, labels: np.ndarray, *, num_bins: int = 10) -> ReliabilityResult:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    bin_confidences = np.zeros(num_bins, dtype=np.float32)
    bin_accuracies = np.zeros(num_bins, dtype=np.float32)
    bin_counts = np.zeros(num_bins, dtype=np.int32)

    for idx in range(num_bins):
        mask = bin_indices == idx
        count = mask.sum()
        if count > 0:
            bin_counts[idx] = int(count)
            bin_confidences[idx] = float(confidences[mask].mean())
            bin_accuracies[idx] = float(accuracies[mask].mean())

    total = len(labels)
    ece = 0.0
    for idx in range(num_bins):
        weight = bin_counts[idx] / max(total, 1)
        ece += weight * abs(bin_accuracies[idx] - bin_confidences[idx])

    return ReliabilityResult(
        bins=bin_edges.tolist(),
        accuracies=bin_accuracies.tolist(),
        confidences=bin_confidences.tolist(),
        counts=bin_counts.tolist(),
        ece=float(ece),
    )


def coverage_vs_error(probabilities: np.ndarray, labels: np.ndarray, thresholds: Sequence[float]) -> Dict[str, float]:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    results: Dict[str, float] = {}
    for threshold in thresholds:
        mask = confidences >= threshold
        coverage = float(mask.mean())
        if mask.any():
            accuracy = float((predictions[mask] == labels[mask]).mean())
        else:
            accuracy = 0.0
        results[f"{threshold:.2f}"] = accuracy
        results[f"{threshold:.2f}_coverage"] = coverage
    return results
