# Location: triage/calibration/temp_scaling.py
# Purpose: Provide temperature scaling utilities for calibration.
# Why: Phase 5 requires fitting calibration parameters from stored logits without heavy model execution.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TemperatureScaler:
    """Simple temperature scaling wrapper."""

    temperature: float = 1.0

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        *,
        max_iter: int = 1000,
        lr: float = 0.01,
        tolerance: float = 1e-4,
    ) -> float:
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        temperature_param = torch.nn.Parameter(torch.ones(1) * self.temperature)
        optimizer = torch.optim.LBFGS([temperature_param], lr=lr, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = F.cross_entropy(logits_tensor / temperature_param, labels_tensor)
            loss.backward()
            return loss

        prev_loss = float("inf")
        for _ in range(max_iter):
            loss = optimizer.step(closure)
            current = float(loss.item())
            if abs(prev_loss - current) < tolerance:
                break
            prev_loss = current

        self.temperature = float(temperature_param.item())
        return self.temperature

    def transform(self, logits: np.ndarray) -> np.ndarray:
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        scaled = logits_tensor / self.temperature
        return np.asarray(scaled.tolist(), dtype=np.float32)

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        scaled = self.transform(logits)
        probs = torch.softmax(torch.tensor(scaled, dtype=torch.float32), dim=-1)
        return np.asarray(probs.tolist(), dtype=np.float32)

    def state_dict(self) -> Dict[str, float]:
        return {"temperature": self.temperature}

    @classmethod
    def from_state_dict(cls, state: Dict[str, float]) -> "TemperatureScaler":
        return cls(temperature=float(state.get("temperature", 1.0)))


def save_temperature(path: str | Path, scaler: TemperatureScaler, metadata: Dict[str, object] | None = None) -> None:
    payload: Dict[str, object] = {"state": scaler.state_dict()}
    if metadata:
        payload["metadata"] = metadata
    Path(path).write_text(json_dumps(payload))


def load_temperature(path: str | Path) -> TemperatureScaler:
    payload = json_loads(Path(path).read_text())
    return TemperatureScaler.from_state_dict(payload.get("state", {}))


# Local JSON helpers keep dependencies minimal.
def json_dumps(obj: object) -> str:
    import json

    return json.dumps(obj, indent=2, sort_keys=True)


def json_loads(text: str) -> Dict[str, object]:
    import json

    return json.loads(text)
