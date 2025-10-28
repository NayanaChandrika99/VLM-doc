# Location: triage/prompts/__init__.py
# Purpose: Expose reusable prompt templates for the triage baseline.
# Why: Keeps prompt variants organised and importable by inference utilities.
"""Prompt templates used by the Phase 2 baseline inference stack."""

from .baseline import BASELINE_PROMPT_TEMPLATE, build_baseline_prompt

__all__ = ["BASELINE_PROMPT_TEMPLATE", "build_baseline_prompt"]
