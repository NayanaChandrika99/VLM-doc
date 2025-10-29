from __future__ import annotations

import math

import numpy as np
import pytest

from triage.data.doclaynet_adapter import DOCLAYNET_LABELS, DocLayNetRegion
from triage.layout.features import (
    AREA_FEATURE_PREFIX,
    COUNT_FEATURE_PREFIX,
    LAYOUT_FEATURE_NAMES,
    compute_layout_feature_dict,
    compute_layout_features,
)


def _feature_index(name: str) -> int:
    try:
        return LAYOUT_FEATURE_NAMES.index(name)
    except ValueError:
        raise AssertionError(f"Feature {name} missing from descriptor list")


def test_no_regions_yields_whitespace_page() -> None:
    page_size = (1000, 500)
    vector = compute_layout_features([], page_size)
    assert vector.shape[0] == len(LAYOUT_FEATURE_NAMES)

    # Area ratios all zero
    assert np.allclose(vector[: len(DOCLAYNET_LABELS)], 0.0)

    coverage_idx = _feature_index("coverage_ratio")
    whitespace_idx = _feature_index("whitespace_ratio")
    region_count_idx = _feature_index("region_count_normalized")
    aspect_idx = _feature_index("page_aspect_ratio")
    entropy_idx = _feature_index("layout_entropy")

    assert vector[coverage_idx] == pytest.approx(0.0)
    assert vector[whitespace_idx] == pytest.approx(1.0)
    assert vector[region_count_idx] == pytest.approx(0.0)
    assert vector[aspect_idx] == pytest.approx(page_size[0] / page_size[1])
    assert vector[entropy_idx] == pytest.approx(0.0)


def test_compute_layout_features_simple_regions() -> None:
    page_size = (800, 600)
    regions = [
        DocLayNetRegion(bbox=(0, 0, 400, 600), label="text"),
        DocLayNetRegion(bbox=(400, 0, 800, 300), label="table"),
    ]

    vector = compute_layout_features(regions, page_size)
    feature_map = compute_layout_feature_dict(regions, page_size)

    text_area_idx = LAYOUT_FEATURE_NAMES.index(f"{AREA_FEATURE_PREFIX}text")
    table_area_idx = LAYOUT_FEATURE_NAMES.index(f"{AREA_FEATURE_PREFIX}table")
    text_count_idx = LAYOUT_FEATURE_NAMES.index(f"{COUNT_FEATURE_PREFIX}text")
    table_count_idx = LAYOUT_FEATURE_NAMES.index(f"{COUNT_FEATURE_PREFIX}table")

    expected_text_area = (400 * 600) / (800 * 600)
    expected_table_area = (400 * 300) / (800 * 600)

    assert vector[text_area_idx] == pytest.approx(expected_text_area)
    assert vector[table_area_idx] == pytest.approx(expected_table_area)
    assert vector[text_count_idx] == pytest.approx(0.5)
    assert vector[table_count_idx] == pytest.approx(0.5)

    coverage_idx = _feature_index("coverage_ratio")
    whitespace_idx = _feature_index("whitespace_ratio")
    count_idx = _feature_index("region_count_normalized")
    entropy_idx = _feature_index("layout_entropy")
    textual_idx = _feature_index("textual_area_ratio")
    visual_idx = _feature_index("visual_area_ratio")

    total_area = expected_text_area + expected_table_area
    assert vector[coverage_idx] == pytest.approx(total_area)
    assert vector[whitespace_idx] == pytest.approx(1.0 - total_area)
    assert vector[count_idx] == pytest.approx(min(2 / 32.0, 1.0))

    expected_entropy = -(
        (expected_text_area / total_area) * math.log2(expected_text_area / total_area)
        + (expected_table_area / total_area) * math.log2(expected_table_area / total_area)
    ) / math.log2(len(DOCLAYNET_LABELS))
    assert vector[entropy_idx] == pytest.approx(expected_entropy)

    assert vector[textual_idx] == pytest.approx(expected_text_area)
    assert vector[visual_idx] == pytest.approx(expected_table_area)

    # Dict conversion mirrors vector
    for name, value in feature_map.items():
        idx = _feature_index(name)
        assert vector[idx] == pytest.approx(value)
