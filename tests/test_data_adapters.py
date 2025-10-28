"""
Unit tests for RVL-CDIP data adapter utilities.

Tests rely on lightweight stubs instead of fetching the full dataset.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from PIL import Image

from triage.data import doclaynet_adapter, rvl_adapter


class _DummyDataset:
    """Mimics a Hugging Face dataset with label metadata and dict access."""

    def __init__(self, records):
        self._records = records
        self.features = {"label": SimpleNamespace(names=["letter", "memo"])}

    def __len__(self):
        return len(self._records)

    def __getitem__(self, index):
        return self._records[index]


def test_rvl_dataset_length_validation(monkeypatch):
    """Adapter should raise if split counts differ from expected values."""

    base_image = Image.new("RGB", (8, 8))
    dummy_records = [
        {"image": base_image, "label": 0, "id": "sample-0"},
    ]
    dataset = _DummyDataset(dummy_records)

    def _fake_load_dataset(*args, **kwargs):
        return dataset

    monkeypatch.setattr(rvl_adapter, "load_dataset", _fake_load_dataset)

    expected = rvl_adapter.RVL_SPLIT_SIZES["train"]
    with pytest.raises(ValueError, match=f"expected {expected}"):
        rvl_adapter.RVLDataset(split="train")


def test_rvl_dataset_returns_sample(monkeypatch):
    """Routed samples should include tensor image, ids, and label strings."""

    base_image = Image.new("RGB", (8, 8))
    dummy_records = [
        {"image": base_image, "label": 1, "id": "sample-1"},
    ]
    dataset = _DummyDataset(dummy_records)

    def _fake_load_dataset(*args, **kwargs):
        return dataset

    monkeypatch.setattr(rvl_adapter, "load_dataset", _fake_load_dataset)

    # Align expected count to the dummy dataset before instantiating.
    monkeypatch.setitem(rvl_adapter.RVL_SPLIT_SIZES, "validation", len(dataset))
    monkeypatch.setitem(rvl_adapter.RVL_SPLIT_SIZES, "test", len(dataset))

    result = rvl_adapter.RVLDataset(
        split="validation",
        transform=lambda image: torch.zeros(3, 4, 4),
    )[0]

    assert isinstance(result.image, torch.Tensor)
    assert result.image.shape == (3, 4, 4)
    assert result.label_id == 1
    assert result.label_str == "memo"
    assert result.uid == "sample-1"


class _DummyDocDataset:
    """Lightweight dataset emulating select() for DocLayNet adapter tests."""

    def __init__(self, records):
        self._records = list(records)

    def __len__(self):
        return len(self._records)

    def __getitem__(self, index):
        return self._records[index]

    def select(self, indices):
        return _DummyDocDataset([self._records[i] for i in indices])


def test_doclaynet_split_assignment(monkeypatch):
    """Ensure document-level splits keep pages from same doc together."""

    base_image = Image.new("RGB", (10, 10))
    dataset = _DummyDocDataset(
        [
            {
                "image": base_image,
                "bboxes_block": [(0, 0, 1, 1)],
                "categories": [1],
                "metadata": {"original_filename": "docA.pdf", "page_no": 0},
            },
            {
                "image": base_image,
                "bboxes_block": [(0, 0, 1, 1)],
                "categories": [1],
                "metadata": {"original_filename": "docA.pdf", "page_no": 1},
            },
            {
                "image": base_image,
                "bboxes_block": [(0, 0, 1, 1)],
                "categories": [3],
                "metadata": {"original_filename": "docB.pdf", "page_no": 0},
            },
            {
                "image": base_image,
                "bboxes_block": [(0, 0, 1, 1)],
                "categories": [4],
                "metadata": {"original_filename": "docC.pdf", "page_no": 0},
            },
        ]
    )

    monkeypatch.setattr(doclaynet_adapter, "load_dataset", lambda *args, **kwargs: dataset)

    assignments = doclaynet_adapter._document_split_assignments("dummy", "any", None, random_seed=7)
    # All pages must be assigned to splits
    assert sum(len(v) for v in assignments.values()) == len(dataset)
    # Pages from the same document must share the split
    docA_splits = [
        name
        for name, idxs in assignments.items()
        if any(dataset[i]["metadata"]["original_filename"].startswith("docA") for i in idxs)
    ]
    assert len(docA_splits) == 1


def test_doclaynet_dataset_returns_regions(monkeypatch):
    """Dataset should parse annotations into DocLayNetRegion structures."""

    base_image = Image.new("RGB", (10, 10))
    records = [
        {
            "image": base_image,
            "bboxes_block": [(0, 0, 1, 1), (1, 1, 2, 2)],
            "categories": [9, 3],  # text, list_item
            "texts": ["hello world", "bullet"],
            "metadata": {"original_filename": "docX.pdf", "page_no": 3},
        },
        {
            "image": base_image,
            "bboxes_block": [(0, 0, 1, 1)],
            "categories": [1],
            "metadata": {"original_filename": "docY.pdf", "page_no": 0},
        },
        {
            "image": base_image,
            "bboxes_block": [(0, 0, 1, 1)],
            "categories": [10],
            "metadata": {"original_filename": "docZ.pdf", "page_no": 0},
        },
    ]
    dataset = _DummyDocDataset(records)

    monkeypatch.setattr(doclaynet_adapter, "load_dataset", lambda *args, **kwargs: dataset)

    ds = doclaynet_adapter.load_doclaynet("train", transform=lambda image: torch.zeros(3, 4, 4))
    sample = ds[0]

    assert sample.uid == "docX.pdf-3"
    assert sample.original_size == base_image.size
    assert len(sample.regions) == 2
    assert sample.regions[0].label == "text"
    assert sample.regions[0].area == pytest.approx(1.0)
    assert isinstance(sample.image, torch.Tensor)


def test_rvl_test_split_aliases_validation():
    """The small RVL mirror exposes only train/validation; test should alias validation."""

    assert rvl_adapter.RVL_SPLITS["test"] == "validation"
    assert rvl_adapter.RVL_SPLIT_SIZES["test"] == rvl_adapter.RVL_SPLIT_SIZES["validation"]
