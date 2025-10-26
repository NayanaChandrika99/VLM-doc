"""
DocLayNet dataset adapter providing layout annotations for downstream use.

The adapter targets the `nevernever69/dit-doclaynet-segmentation` subset,
enforcing document-level splits to avoid template leakage and returning
tensorised pages alongside region metadata derived from bounding boxes and
segmentation polygons.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Iterable, Iterator, List, Literal, Optional, Sequence

from datasets import Dataset, IterableDataset, concatenate_datasets, load_dataset
from PIL import Image
import torch
from torch.utils.data import Dataset as TorchDataset

from .transforms import get_default_image_transform

DOCLAYNET_DATASET_ID = "nevernever69/dit-doclaynet-segmentation"
SPLIT_NAMES = ("train", "validation", "test")
DOC_KEYS = ("document_id", "document", "doc_id", "doc_name", "source")
DOCLAYNET_LABELS = [
    "background",
    "title",
    "paragraph",
    "figure",
    "table",
    "list",
    "header",
    "footer",
    "page_number",
    "footnote",
    "caption",
]


@dataclass(frozen=True)
class DocLayNetRegion:
    """Structured representation of a single layout region (bbox + optional masks)."""

    bbox: Sequence[float]
    label: str
    score: Optional[float] = None
    segmentation: Optional[Sequence[Sequence[float]]] = None
    area: Optional[float] = None


@dataclass(frozen=True)
class DocLayNetSample:
    """Container for page tensor, metadata, and region annotations."""

    image: torch.Tensor
    regions: Sequence[DocLayNetRegion]
    uid: str
    document_id: str
    page_index: int
    original_size: Sequence[int]


class DocLayNetDataset(TorchDataset[DocLayNetSample]):
    """PyTorch dataset wrapper producing :class:`DocLayNetSample` records."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        split: Literal["train", "validation", "test"],
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        self._dataset = dataset
        self.split = split
        self.transform = transform or get_default_image_transform()

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> DocLayNetSample:
        record = self._dataset[index]
        image = _ensure_pil_image(record["image"])
        original_size = image.size
        tensor = self.transform(image)
        document_id = _resolve_document_id(record)
        metadata = record.get("metadata") or {}
        page_index = int(
            record.get("page_index")
            or record.get("page")
            or record.get("page_id")
            or record.get("page_no")
            or metadata.get("page_no")
            or metadata.get("page_index")
            or 0
        )
        uid = f"{document_id}-{page_index}"
        regions = _extract_regions(record)
        return DocLayNetSample(
            image=tensor,
            regions=regions,
            uid=uid,
            document_id=document_id,
            page_index=page_index,
            original_size=original_size,
        )

    def as_iterable(self) -> Iterator[DocLayNetSample]:
        """Yield samples sequentially (handy for feature extraction pipelines)."""

        for index in range(len(self._dataset)):
            yield self[index]


def load_doclaynet(
    split: Literal["train", "validation", "test"],
    *,
    dataset_id: str = DOCLAYNET_DATASET_ID,
    cache_dir: Optional[str] = None,
    random_seed: int = 13,
    transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    base_split: str = "train+validation+test",
) -> DocLayNetDataset:
    """
    Load DocLayNet with deterministic document-level split assignment.

    Args:
        split: Desired logical split (train/validation/test).
        dataset_id: Override HF dataset ID if needed.
        cache_dir: Optional HF cache directory for offline reuse.
        random_seed: Seed controlling document shuffling before split assignment.
        transform: Optional PIL -> tensor callable.
        base_split: HF split string to read prior to repartitioning.

    Returns:
        :class:`DocLayNetDataset` representing the requested split.
    """

    if split not in SPLIT_NAMES:
        raise ValueError(f"Unsupported split '{split}'. Expected one of {SPLIT_NAMES}.")

    base_ds = _load_combined_dataset(dataset_id, base_split, cache_dir)
    assignments = _document_split_assignments(dataset_id, base_split, cache_dir, random_seed)
    indices = assignments[split]
    subset = base_ds.select(indices)
    return DocLayNetDataset(subset, split=split, transform=transform)


def iter_regions(sample: DocLayNetSample) -> Iterator[DocLayNetRegion]:
    """Expose region iterator helper for downstream feature extraction."""

    yield from sample.regions


def _load_combined_dataset(dataset_id: str, base_split: str, cache_dir: Optional[str]) -> Dataset:
    """
    Load the requested HF dataset split, falling back gracefully if unavailable.
    """

    try:
        dataset = load_dataset(dataset_id, split=base_split, cache_dir=cache_dir)
    except ValueError:
        splits = []
        for candidate in ("train", "validation", "test"):
            try:
                splits.append(load_dataset(dataset_id, split=candidate, cache_dir=cache_dir))
            except ValueError:
                continue
        if not splits:
            raise
        dataset = concatenate_datasets(splits)
    if isinstance(dataset, IterableDataset):
        raise TypeError("DocLayNet adapter requires a finite Dataset, not IterableDataset.")
    return dataset


@lru_cache(maxsize=8)
def _document_split_assignments(
    dataset_id: str,
    base_split: str,
    cache_dir: Optional[str],
    random_seed: int,
) -> Dict[str, List[int]]:
    """Compute document-level split lists keyed by split name."""

    dataset = _load_combined_dataset(dataset_id, base_split, cache_dir)
    doc_to_indices: Dict[str, List[int]] = {}
    for idx, record in enumerate(dataset):
        document_id = _resolve_document_id(record)
        doc_to_indices.setdefault(document_id, []).append(idx)

    documents = list(doc_to_indices.keys())
    rng = random.Random(random_seed)
    rng.shuffle(documents)

    total_docs = len(documents)
    if total_docs == 0:
        raise ValueError("DocLayNet dataset appears empty; verify dataset availability.")

    train_end = max(1, math.floor(total_docs * 0.8))
    val_end = max(train_end + 1, train_end + max(1, math.floor(total_docs * 0.1)))
    doc_groups = {
        "train": documents[:train_end],
        "validation": documents[train_end:val_end],
        "test": documents[val_end:],
    }

    while any(len(doc_groups[name]) == 0 for name in doc_groups):
        empty_name = next(name for name, docs in doc_groups.items() if not docs)
        donor_name = max(doc_groups.keys(), key=lambda key: len(doc_groups[key]))
        if not doc_groups[donor_name]:
            raise ValueError("Unable to allocate documents across splits; dataset too small.")
        doc_groups[empty_name].append(doc_groups[donor_name].pop())

    splits: Dict[str, List[int]] = {"train": [], "validation": [], "test": []}
    for split_name, doc_ids in doc_groups.items():
        for doc_id in doc_ids:
            splits[split_name].extend(doc_to_indices[doc_id])

    return splits


def _ensure_pil_image(value: Image.Image) -> Image.Image:
    if isinstance(value, Image.Image):
        if value.mode != "RGB":
            return value.convert("RGB")
        return value
    raise TypeError(f"Unsupported image type: {type(value)!r}")


def _resolve_document_id(record: dict) -> str:
    metadata = record.get("metadata") or {}
    for key in ("original_filename", "document_id", "doc_id", "source", "collection"):
        value = metadata.get(key)
        if value:
            return str(value)
    for key in DOC_KEYS:
        if key in record and record[key] is not None:
            return str(record[key])
    if "id" in record:
        return str(record["id"]).split("_")[0]
    raise KeyError("Unable to determine document identifier from DocLayNet record.")


def _extract_regions(record: dict) -> List[DocLayNetRegion]:
    bboxes = record.get("bboxes")
    categories = record.get("category_id")
    segmentations = record.get("segmentation")
    areas = record.get("area")
    regions: List[DocLayNetRegion] = []

    if bboxes is not None and categories is not None:
        limit = min(len(bboxes), len(categories))
        for idx in range(limit):
            bbox = _ensure_bbox(bboxes[idx])
            label = _resolve_label(categories[idx])
            segmentation = segmentations[idx] if segmentations is not None and idx < len(segmentations) else None
            area = areas[idx] if areas is not None and idx < len(areas) else None
            regions.append(DocLayNetRegion(bbox=bbox, label=label, segmentation=segmentation, area=area))
        return regions

    annotations = record.get("annotations") or record.get("segments") or record.get("layout")
    if annotations is None:
        return regions

    if isinstance(annotations, dict):
        boxes = annotations.get("bbox") or annotations.get("bboxes") or annotations.get("boxes")
        labels = annotations.get("label") or annotations.get("labels") or annotations.get("category")
        scores = annotations.get("score") or annotations.get("scores")
        if boxes is None or labels is None:
            return regions
        for idx, bbox in enumerate(boxes):
            label = labels[idx] if idx < len(labels) else "unknown"
            score = scores[idx] if scores is not None and idx < len(scores) else None
            regions.append(DocLayNetRegion(bbox=_ensure_bbox(bbox), label=str(label), score=score))
    elif isinstance(annotations, list):
        for entry in annotations:
            bbox = _ensure_bbox(entry.get("bbox") or entry.get("bboxes") or entry.get("box"))
            label = entry.get("label") or entry.get("category") or "unknown"
            score = entry.get("score")
            regions.append(DocLayNetRegion(bbox=bbox, label=str(label), score=score))
    return regions


def _ensure_bbox(value) -> Sequence[float]:
    if value is None:
        return (0.0, 0.0, 0.0, 0.0)
    if isinstance(value, (list, tuple)):
        return tuple(float(v) for v in value)
    raise TypeError(f"Unexpected bbox type {type(value)!r}")


def _resolve_label(category: int | str) -> str:
    try:
        index = int(category)
    except (TypeError, ValueError):
        return str(category)
    if 0 <= index < len(DOCLAYNET_LABELS):
        return DOCLAYNET_LABELS[index]
    return str(index)
