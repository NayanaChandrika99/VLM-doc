# Location: tests/test_package_imports.py
# Purpose: Guard the triage package surface for ingestion utilities.
# Why: Prevent regressions where triage.data is missing __init__ and cannot be imported in pytest.
"""Smoke tests for triage package imports used across the ingestion stack."""


def test_triage_data_importable():
    """triage.data should expose adapter modules without manual PYTHONPATH tweaks."""

    import triage.data

    assert hasattr(triage.data, "doclaynet_adapter")
    assert hasattr(triage.data, "rvl_adapter")
