# Location: tests/test_package_imports.py
# Purpose: Guard the triage package surface for ingestion utilities.
# Why: Prevent regressions where triage.data is missing __init__ and cannot be imported in pytest.
"""Smoke tests for triage package imports used across the ingestion stack."""


def test_triage_data_importable():
    """triage.data should expose adapter modules without manual PYTHONPATH tweaks."""

    import triage.data

    assert hasattr(triage.data, "doclaynet_adapter")
    assert hasattr(triage.data, "rvl_adapter")


def test_prompts_and_inference_importable():
    """Baseline prompt helpers and inference module should import cleanly."""

    import triage.prompts
    import triage_infer

    assert hasattr(triage.prompts, "build_baseline_prompt")
    assert hasattr(triage_infer, "BaselineInference")
