# Location: triage/calibration/isotonic.py
# Purpose: Provide a lightweight isotonic calibration scaffold (optional in dry-run).

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np


@dataclass
class IsotonicCalibrator:
    """Minimal isotonic regression wrapper (uses numpy for dry-run)."""

    thresholds: np.ndarray | None = None
    values: np.ndarray | None = None

    def fit(self, confidences: np.ndarray, labels: np.ndarray) -> None:
        # Sort by confidence and compute cumulative positive rate (monotonic)
        order = np.argsort(confidences)
        sorted_conf = confidences[order]
        sorted_labels = labels[order]
        cumulative = np.cumsum(sorted_labels)
        total = np.arange(1, len(sorted_labels) + 1)
        isotonic = cumulative / total
        self.thresholds = sorted_conf
        self.values = isotonic

    def predict(self, confidences: np.ndarray) -> np.ndarray:
        if self.thresholds is None or self.values is None:
            raise ValueError("Calibrator not fitted.")
        return np.interp(confidences, self.thresholds, self.values, left=self.values[0], right=self.values[-1])

    def state_dict(self) -> Dict[str, list[float]]:
        return {
            "thresholds": self.thresholds.tolist() if self.thresholds is not None else [],
            "values": self.values.tolist() if self.values is not None else [],
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, list[float]]) -> "IsotonicCalibrator":
        thresholds = np.array(state.get("thresholds", []), dtype=np.float32)
        values = np.array(state.get("values", []), dtype=np.float32)
        return cls(thresholds=thresholds, values=values)
