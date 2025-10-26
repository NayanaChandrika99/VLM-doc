"""
RVL-CDIP dataset adapter yielding tensorised images with label metadata.

This module wraps the Hugging Face `vaclavpechtor/rvl_cdip-small-200` mirror,
enforcing the published train/validation splits (with test aliased to validation)
and returning samples compatible with
PyTorch training loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Literal, Optional, Sequence

from datasets import Dataset, IterableDataset, load_dataset
from PIL import Image
import torch
from torch.utils.data import Dataset as TorchDataset

from .transforms import get_default_image_transform

RVL_DATASET_ID = "vaclavpechtor/rvl_cdip-small-200"
RVL_SPLITS: Dict[str, str] = {
    "train": "train",
    "validation": "validation",
    "test": "validation",
}
RVL_SPLIT_SIZES = {
    "train": 2_560,
    "validation": 640,
    "test": 640,
}


@dataclass(frozen=True)
class RVLSample:
    """Container for a single RVL-CDIP example."""

    image: torch.Tensor
    label_id: int
    label_str: str
    uid: str


class RVLDataset(TorchDataset[RVLSample]):
    """PyTorch dataset that materialises RVL-CDIP samples with transforms."""

    def __init__(
        self,
        split: Literal["train", "validation", "test"],
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        dataset_id: str = RVL_DATASET_ID,
        cache_dir: Optional[str] = None,
        streaming: bool = False,
    ) -> None:
        if split not in RVL_SPLITS:
            raise ValueError(f"Unsupported split '{split}'. Expected one of {tuple(RVL_SPLITS)}.")

        self._hf_dataset = load_dataset(
            dataset_id,
            split=RVL_SPLITS[split],
            cache_dir=cache_dir,
            streaming=streaming,
        )
        self.split = split
        self.streaming = streaming
        self.transform = transform or get_default_image_transform()
        self.label_names = _extract_label_names(self._hf_dataset)

        if not streaming:
            expected = RVL_SPLIT_SIZES[split]
            actual = len(self._hf_dataset)
            if actual != expected:
                raise ValueError(
                    f"RVL split '{split}' expected {expected} examples but found {actual}. "
                    "Confirm dataset integrity."
                )

    def __len__(self) -> int:
        if self.streaming:
            raise TypeError("Length is undefined for streaming RVL datasets.")
        return len(self._hf_dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> RVLSample:
        if self.streaming:
            raise TypeError("Indexing a streaming RVL dataset is not supported.")

        record = self._hf_dataset[index]
        image = _ensure_pil_image(record["image"])
        tensor = self.transform(image)
        label_id = int(record["label"])
        label_str = self.label_names[label_id]
        uid = str(
            record.get("id")
            or record.get("image_id")
            or record.get("path")
            or f"{self.split}-{index}"
        )
        return RVLSample(image=tensor, label_id=label_id, label_str=label_str, uid=uid)

    def as_iterable(self) -> Iterable[RVLSample]:
        """
        Yield samples sequentially; useful for streaming evaluations.

        Returns:
            Generator of :class:`RVLSample`.
        """

        if isinstance(self._hf_dataset, IterableDataset):
            iterator: Iterable = self._hf_dataset
        else:
            iterator = (self._hf_dataset[i] for i in range(len(self._hf_dataset)))  # type: ignore[arg-type]

        for idx, record in enumerate(iterator):
            image = _ensure_pil_image(record["image"])
            tensor = self.transform(image)
            label_id = int(record["label"])
            label_str = self.label_names[label_id]
            uid = str(
                record.get("id")
                or record.get("image_id")
                or record.get("path")
                or f"{self.split}-{idx}"
            )
            yield RVLSample(image=tensor, label_id=label_id, label_str=label_str, uid=uid)


def load_rvl(
    split: Literal["train", "validation", "test"],
    *,
    transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    dataset_id: str = RVL_DATASET_ID,
    cache_dir: Optional[str] = None,
    streaming: bool = False,
) -> RVLDataset:
    """
    Convenience wrapper returning the canonical RVL-CDIP dataset object.

    Args:
        split: One of ``train``, ``validation``, or ``test``.
        transform: Optional callable converting PIL -> tensor; defaults to
            :func:`get_default_image_transform`.
        dataset_id: HF dataset identifier override.
        cache_dir: Optional path where the HF dataset is cached.
        streaming: Whether to request HF streaming mode.

    Returns:
        :class:`RVLDataset` ready for PyTorch DataLoaders.
    """

    return RVLDataset(
        split=split,
        transform=transform,
        dataset_id=dataset_id,
        cache_dir=cache_dir,
        streaming=streaming,
    )


def _extract_label_names(dataset: Dataset | IterableDataset) -> Sequence[str]:
    """Read label names from the Hugging Face dataset features."""

    features = getattr(dataset, "features", None)
    if features is None or "label" not in features:
        raise ValueError("RVL dataset is missing label metadata; verify dataset version.")
    return features["label"].names


def _ensure_pil_image(value: Image.Image) -> Image.Image:
    """Convert the incoming image-like object to RGB PIL format."""

    if isinstance(value, Image.Image):
        if value.mode != "RGB":
            return value.convert("RGB")
        return value
    raise TypeError(f"Unsupported image type: {type(value)!r}")
