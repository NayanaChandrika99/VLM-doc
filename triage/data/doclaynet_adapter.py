"""
DocLayNet dataset adapter providing layout annotations for downstream use.

The adapter targets the `pierreguillou/DocLayNet-base` subset, enforcing
document-level splits to avoid template leakage and returning tensorised pages
alongside region metadata derived from block-level bounding boxes.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Literal, Optional, Sequence, Union, Tuple

from datasets import Dataset, IterableDataset, concatenate_datasets, load_dataset
from PIL import Image
import torch
from torch.utils.data import Dataset as TorchDataset
from huggingface_hub import hf_hub_download

from .transforms import get_default_image_transform

DOCLAYNET_DATASET_ID = "pierreguillou/DocLayNet-base"
SPLIT_NAMES = ("train", "validation", "test")
DOC_KEYS = ("document_id", "document", "doc_id", "doc_name", "source")
DOCLAYNET_LABELS = [
    "caption",
    "footnote",
    "formula",
    "list_item",
    "page_footer",
    "page_header",
    "picture",
    "section_header",
    "table",
    "text",
    "title",
]


@dataclass(frozen=True)
class DocLayNetRegion:
    """Structured representation of a single layout region (bbox + optional metadata)."""

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


def _load_combined_dataset(
    dataset_id: str,
    base_split: str,
    cache_dir: Optional[str],
) -> Union[Dataset, "_DocLayNetMemoryDataset"]:
    """
    Load the requested HF dataset split, falling back gracefully if unavailable.
    """

    try:
        dataset = load_dataset(
            dataset_id,
            split=base_split,
            cache_dir=cache_dir,
        )
    except (ValueError, RuntimeError):
        if dataset_id == DOCLAYNET_DATASET_ID:
            dataset = _load_doclaynet_base_from_archive(cache_dir, splits=tuple(_parse_splits(base_split)))
        else:
            splits = []
            for candidate in ("train", "validation", "test"):
                try:
                    splits.append(
                        load_dataset(
                            dataset_id,
                            split=candidate,
                            cache_dir=cache_dir,
                        )
                    )
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
        return value if value.mode == "RGB" else value.convert("RGB")
    if isinstance(value, (str, Path)):
        image = Image.open(value)
        return image if image.mode == "RGB" else image.convert("RGB")
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
    bboxes = record.get("bboxes_block") or record.get("bboxes")
    categories = record.get("categories") or record.get("category_id")
    texts = record.get("texts") or []
    regions: List[DocLayNetRegion] = []

    if bboxes is not None and categories is not None:
        limit = min(len(bboxes), len(categories))
        for idx in range(limit):
            bbox = _ensure_bbox(bboxes[idx])
            label = _resolve_label(categories[idx])
            area = _compute_area(bbox)
            regions.append(DocLayNetRegion(bbox=bbox, label=label, area=area))
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
    if isinstance(category, str):
        normalized = category.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in DOCLAYNET_LABELS:
            return normalized
        return normalized
    try:
        index = int(category)
    except (TypeError, ValueError):
        return str(category)
    if 0 <= index < len(DOCLAYNET_LABELS):
        return DOCLAYNET_LABELS[index]
    return str(index)


def _compute_area(bbox: Sequence[float]) -> Optional[float]:
    if len(bbox) < 4:
        return None
    _, _, width, height = bbox[:4]
    return float(width) * float(height)


def _parse_splits(base_split: str) -> List[str]:
    return [token.strip() for token in base_split.split("+") if token.strip()]


class _DocLayNetMemoryDataset:
    """Lightweight dataset emulating Hugging Face Dataset essentials."""

    def __init__(self, records: List[dict]) -> None:
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict:
        return self._records[index]

    def select(self, indices: List[int]) -> "_DocLayNetMemoryDataset":
        return _DocLayNetMemoryDataset([self._records[i] for i in indices])


@lru_cache(maxsize=2)
def _load_doclaynet_base_from_archive(
    cache_dir: Optional[str],
    *,
    splits: Tuple[str, ...],
) -> _DocLayNetMemoryDataset:
    """
    Load DocLayNet base dataset directly from the published zip archive.

    Returns:
        Memory-backed dataset with `__len__`, `__getitem__`, and `select`.
    """

    zip_path = hf_hub_download(
        repo_id=DOCLAYNET_DATASET_ID,
        filename="data/dataset_base.zip",
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    extract_root = Path(zip_path).with_suffix("")
    if not extract_root.exists():
        import zipfile

        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_root)

    base_dir = extract_root / "base_dataset"
    if not base_dir.exists():
        raise FileNotFoundError(f"DocLayNet base archive missing 'base_dataset' directory at {base_dir}")

    combined_records: List[dict] = []
    for split in splits:
        split_dir = base_dir / split
        if not split_dir.exists():
            continue
        annotations_dir = split_dir / "annotations"
        images_dir = split_dir / "images"
        for guid, annotation_path in enumerate(sorted(annotations_dir.glob("*.json"))):
            record = _parse_doclaynet_base_record(annotation_path, images_dir, guid)
            combined_records.append(record)

    return _DocLayNetMemoryDataset(combined_records)  # type: ignore[return-value]


def _parse_doclaynet_base_record(annotation_path: Path, images_dir: Path, guid: int) -> dict:
    import json

    with annotation_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    page_metadata = data.get("metadata", {})
    forms = data.get("form", [])
    texts: List[str] = []
    categories: List[str] = []
    bboxes_block: List[Sequence[float]] = []
    bboxes_line: List[Sequence[float]] = []

    for item in forms:
        texts.append(item.get("text", ""))
        categories.append(item.get("category", "unknown"))
        bboxes_block.append(_ensure_bbox(item.get("box")))
        bboxes_line.append(_ensure_bbox(item.get("box_line")))

    image_filename = annotation_path.name.replace(".json", ".png")
    image_path = images_dir / image_filename
    if not image_path.exists():
        raise FileNotFoundError(f"DocLayNet base image missing: {image_path}")

    return {
        "id": str(guid),
        "image": str(image_path),
        "texts": texts,
        "categories": [_normalize_category_name(cat) for cat in categories],
        "bboxes_block": bboxes_block,
        "bboxes_line": bboxes_line,
        "metadata": {
            "original_filename": page_metadata.get("original_filename", ""),
            "page_no": page_metadata.get("page_no", 0),
            "num_pages": page_metadata.get("num_pages", 0),
            "collection": page_metadata.get("collection", ""),
            "doc_category": page_metadata.get("doc_category", ""),
            "page_hash": page_metadata.get("page_hash", ""),
            "original_width": page_metadata.get("original_width", 0),
            "original_height": page_metadata.get("original_height", 0),
            "coco_width": page_metadata.get("coco_width", 0),
            "coco_height": page_metadata.get("coco_height", 0),
        },
    }


def _normalize_category_name(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or "unknown"
