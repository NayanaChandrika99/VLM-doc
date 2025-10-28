# Location: triage/prompts/baseline.py
# Purpose: Centralise baseline prompt wording for the baseline classifier.
# Why: Consistency across CLI, tests, and future experiments.
"""Prompt text helpers for the baseline document classifier."""

from __future__ import annotations

from typing import Iterable, Sequence

from triage.io.structured import allowed_labels


BASELINE_PROMPT_TEMPLATE = """You are a meticulous document specialist.
Analyse the supplied page and return ONLY a JSON object with the following schema:
{{
  "label": "<one of {label_count} categories>",
  "confidence": <float between 0.0 and 1.0>
}}

Allowed categories (case-sensitive values to use in the JSON):
{labels}

Guidelines:
- Pick the single best fitting category for the page content/layout.
- Express confidence as a decimal with at most two decimal places.
- Do not include explanations, markdown, or extra keys.
"""


def build_baseline_prompt(extra_guidance: Iterable[str] | None = None) -> str:
    """Render the baseline prompt with optional extra bullet guidance."""

    label_list = [label for label in allowed_labels(include_unknown=False)]
    lines: list[str] = [
        BASELINE_PROMPT_TEMPLATE.format(
            label_count=len(label_list),
            labels="\n".join(f"- {label}" for label in label_list),
        )
    ]
    extras: Sequence[str] = tuple(extra_guidance or ())
    if extras:
        lines.append("Additional instructions:")
        lines.extend(f"- {line}" for line in extras)
    return "\n".join(lines).strip()
