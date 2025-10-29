from __future__ import annotations

from pathlib import Path

import torch

from triage.triage.dataset import RVLClassifierDataset


def test_rvl_classifier_dataset_dry_run_without_layout() -> None:
    dataset = RVLClassifierDataset(
        "train",
        embedding_dim=32,
        use_layout=False,
        dry_run=True,
        dry_run_size=10,
    )

    assert len(dataset) == 10
    sample = dataset[0]
    assert isinstance(sample.embedding, torch.Tensor)
    assert sample.embedding.shape == (32,)
    assert sample.layout is None
    assert 0 <= sample.label_id < 16


def test_rvl_classifier_dataset_dry_run_with_layout() -> None:
    dataset = RVLClassifierDataset(
        "validation",
        embedding_dim=16,
        use_layout=True,
        dry_run=True,
        dry_run_size=5,
    )

    example = dataset[0]
    assert example.layout is not None
    assert example.layout.shape[0] > 0

