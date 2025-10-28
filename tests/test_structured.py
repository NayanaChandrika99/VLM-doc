# Location: tests/test_structured.py
# Purpose: Validate structured JSON helpers for baseline inference.
# Why: Ensures schema checks catch obvious regressions.
"""Unit tests for triage.io.structured utilities."""

import pytest

from triage.io.structured import PredictionMeta, build_prediction, validate_prediction


def test_build_prediction_happy_path():
    meta = PredictionMeta(model_id="m", calibration_id="c", adapter_id="a")
    payload = build_prediction("invoice", 0.9, meta)
    assert payload["label"] == "invoice"
    assert payload["confidence"] == pytest.approx(0.9)
    validate_prediction(payload)  # should not raise


def test_build_prediction_rejects_unknown_label():
    meta = PredictionMeta(model_id="m", calibration_id="c", adapter_id="a")
    with pytest.raises(ValueError):
        build_prediction("nonexistent", 0.1, meta)
