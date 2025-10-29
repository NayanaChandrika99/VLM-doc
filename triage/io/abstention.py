# Location: triage/io/abstention.py
# Purpose: Helper utilities for applying abstention policies to classifier outputs.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


@dataclass
class AbstentionResult:
    labels: np.ndarray
    mask: np.ndarray
    coverage: float
    abstain_rate: float


def apply_threshold(labels: Sequence[str], confidences: Sequence[float], threshold: float) -> AbstentionResult:
    confidences_arr = np.asarray(confidences, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=object)
    mask = confidences_arr >= threshold
    adjusted_labels = labels_arr.copy()
    adjusted_labels[~mask] = "unknown"
    coverage = float(mask.mean())
    abstain_rate = 1.0 - coverage
    return AbstentionResult(labels=adjusted_labels, mask=mask, coverage=coverage, abstain_rate=abstain_rate)
