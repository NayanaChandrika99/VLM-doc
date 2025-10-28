# Location: triage/io/structured.py
# Purpose: Validate and assemble structured JSON responses for baseline inference.
# Why: Phase 2 requires guaranteed `{label, confidence, meta}` outputs without extra deps.
"""Lightweight helpers for producing and validating triage JSON predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, MutableMapping, Sequence


_ALLOWED_LABELS: Sequence[str] = (
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific_report",
    "scientific_publication",
    "specification",
    "file_folder",
    "news_article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
    "unknown",
)


@dataclass(frozen=True)
class PredictionMeta:
    """Metadata block returned with each prediction."""

    model_id: str
    calibration_id: str
    adapter_id: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "model_id": self.model_id,
            "calibration_id": self.calibration_id,
            "adapter_id": self.adapter_id,
        }


def allowed_labels(include_unknown: bool = True) -> Sequence[str]:
    """Expose the label taxonomy for prompt builders and validators."""

    if include_unknown:
        return _ALLOWED_LABELS
    return _ALLOWED_LABELS[:-1]


def build_prediction(
    label: str,
    confidence: float,
    meta: PredictionMeta | MutableMapping[str, str],
) -> Dict[str, object]:
    """
    Assemble a structured prediction dict and validate the payload.

    Args:
        label: Candidate RVL label (case-insensitive).
        confidence: Score in [0, 1].
        meta: PredictionMeta dataclass or dict with required fields.
    """

    payload = {
        "label": _normalise_label(label),
        "confidence": float(confidence),
        "meta": dict(meta.as_dict() if isinstance(meta, PredictionMeta) else meta),
    }
    validate_prediction(payload)
    return payload


def validate_prediction(payload: MutableMapping[str, object]) -> None:
    """Raise ValueError if payload is not schema compliant."""

    missing = {"label", "confidence", "meta"} - payload.keys()
    if missing:
        raise ValueError(f"Prediction missing required keys: {sorted(missing)}")

    label = payload["label"]
    if not isinstance(label, str):
        raise ValueError("`label` must be a string.")
    if label not in _ALLOWED_LABELS:
        raise ValueError(f"Unsupported label '{label}'.")

    confidence = payload["confidence"]
    if not isinstance(confidence, (int, float)):
        raise ValueError("`confidence` must be numeric.")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError(f"`confidence` out of range: {confidence}")

    meta = payload["meta"]
    if not isinstance(meta, MutableMapping):
        raise ValueError("`meta` must be a mapping.")

    required_meta = {"model_id", "calibration_id", "adapter_id"}
    missing_meta = required_meta - meta.keys()
    if missing_meta:
        raise ValueError(f"`meta` missing required keys: {sorted(missing_meta)}")

    for key in required_meta:
        value = meta[key]
        if not isinstance(value, str) or not value:
            raise ValueError(f"`meta.{key}` must be a non-empty string.")


def _normalise_label(raw: str) -> str:
    candidate = raw.strip().lower().replace(" ", "_")
    if candidate not in _ALLOWED_LABELS:
        raise ValueError(f"Unsupported label '{raw}'. Allowed: {', '.join(_ALLOWED_LABELS)}")
    return candidate
