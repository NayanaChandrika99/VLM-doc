"""Deterministic layout descriptor utilities for DocLayNet pages."""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from triage.data.doclaynet_adapter import DOCLAYNET_LABELS, DocLayNetRegion

AREA_FEATURE_PREFIX = "area_ratio__"
COUNT_FEATURE_PREFIX = "count_ratio__"
TEXTUAL_LABELS = {"text", "title", "section_header", "caption", "list_item"}
VISUAL_LABELS = {"picture", "table", "formula"}
REGION_NORMALISATION_FACTOR = 32.0


LAYOUT_FEATURE_NAMES: list[str] = [
    *(f"{AREA_FEATURE_PREFIX}{label}" for label in DOCLAYNET_LABELS),
    *(f"{COUNT_FEATURE_PREFIX}{label}" for label in DOCLAYNET_LABELS),
    "coverage_ratio",
    "whitespace_ratio",
    "region_count_normalized",
    "page_aspect_ratio",
    "layout_entropy",
    "textual_area_ratio",
    "visual_area_ratio",
]


def compute_layout_features(
    regions: Sequence[DocLayNetRegion | Mapping[str, object]] | Iterable[DocLayNetRegion | Mapping[str, object]],
    page_size: Sequence[float] | tuple[int, int],
) -> np.ndarray:
    """
    Compute a deterministic layout descriptor vector for a single DocLayNet page.

    Args:
        regions: Sequence of DocLayNetRegion objects (or dicts with ``bbox``/``label`` keys).
        page_size: (width, height) of the original page in pixels.

    Returns:
        numpy.ndarray ordered according to :const:`LAYOUT_FEATURE_NAMES`.
    """

    width, height = _coerce_page_size(page_size)
    page_area = max(width * height, 1.0)

    area_accumulator = np.zeros(len(DOCLAYNET_LABELS), dtype=np.float32)
    count_accumulator = np.zeros(len(DOCLAYNET_LABELS), dtype=np.float32)

    total_area_ratio = 0.0
    total_regions = 0.0

    for region in regions:
        label, bbox = _coerce_region(region)
        if label not in DOCLAYNET_LABELS:
            continue
        area_ratio = _bbox_area_ratio(bbox, width, height, page_area)
        if area_ratio <= 0:
            continue
        idx = DOCLAYNET_LABELS.index(label)
        area_accumulator[idx] += area_ratio
        count_accumulator[idx] += 1.0
        total_area_ratio += area_ratio
        total_regions += 1.0

    coverage_ratio = float(min(total_area_ratio, 1.0))
    whitespace_ratio = float(max(0.0, 1.0 - coverage_ratio))

    if total_regions > 0:
        count_accumulator /= total_regions

    textual_area_ratio = float(
        sum(area_accumulator[DOCLAYNET_LABELS.index(label)] for label in TEXTUAL_LABELS if label in DOCLAYNET_LABELS)
    )
    visual_area_ratio = float(
        sum(area_accumulator[DOCLAYNET_LABELS.index(label)] for label in VISUAL_LABELS if label in DOCLAYNET_LABELS)
    )

    layout_entropy = _normalized_entropy(area_accumulator)
    region_count_normalized = float(min(total_regions / REGION_NORMALISATION_FACTOR, 1.0))
    page_aspect_ratio = float(width / height) if height else 0.0

    feature_vector = np.concatenate(
        [
            area_accumulator,
            count_accumulator,
            np.array(
                [
                    coverage_ratio,
                    whitespace_ratio,
                    region_count_normalized,
                    page_aspect_ratio,
                    layout_entropy,
                    textual_area_ratio,
                    visual_area_ratio,
                ],
                dtype=np.float32,
            ),
        ]
    )
    return feature_vector.astype(np.float32, copy=False)


def compute_layout_feature_dict(
    regions: Sequence[DocLayNetRegion | Mapping[str, object]] | Iterable[DocLayNetRegion | Mapping[str, object]],
    page_size: Sequence[float] | tuple[int, int],
) -> dict[str, float]:
    """Return layout features as a name → value mapping."""

    values = compute_layout_features(regions, page_size)
    return {name: float(value) for name, value in zip(LAYOUT_FEATURE_NAMES, values, strict=True)}


def _coerce_page_size(page_size: Sequence[float] | tuple[int, int]) -> tuple[float, float]:
    try:
        width, height = float(page_size[0]), float(page_size[1])
    except (TypeError, IndexError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid page_size {page_size!r}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"Page dimensions must be positive, got {page_size!r}")
    return width, height


def _coerce_region(region: DocLayNetRegion | Mapping[str, object]) -> tuple[str, Sequence[float]]:
    if isinstance(region, DocLayNetRegion):
        return region.label, region.bbox
    if isinstance(region, Mapping):
        label = str(region.get("label"))
        bbox = region.get("bbox")
        if bbox is None:
            raise ValueError(f"Region missing bbox: {region!r}")
        return label, bbox
    raise TypeError(f"Unsupported region type: {type(region)!r}")


def _bbox_area_ratio(bbox: Sequence[float], page_width: float, page_height: float, page_area: float) -> float:
    try:
        x1, y1, x2, y2 = map(float, bbox[:4])
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid bbox: {bbox!r}") from exc

    x1 = max(0.0, min(x1, page_width))
    x2 = max(0.0, min(x2, page_width))
    y1 = max(0.0, min(y1, page_height))
    y2 = max(0.0, min(y2, page_height))

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    if page_area <= 0:
        return 0.0
    return float(area / page_area)


def _normalized_entropy(area_ratios: np.ndarray) -> float:
    coverage = float(area_ratios.sum())
    if coverage <= 0.0:
        return 0.0
    distribution = area_ratios / coverage
    distribution = np.clip(distribution, 1e-12, 1.0)
    entropy = -float(np.sum(distribution * np.log2(distribution)))
    max_entropy = math.log2(len(area_ratios)) if len(area_ratios) > 1 else 1.0
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0
