from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from triage.data.doclaynet_adapter import DocLayNetRegion, DocLayNetSample
from triage.layout.dataset import DocLayNetFeatureDataset, load_embedding_lookup
from triage.layout.features import compute_layout_features


class DummyDocLayNetDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def _build_sample(uid: str, bbox, label: str, page_size=(400, 400)) -> DocLayNetSample:
    region = DocLayNetRegion(bbox=bbox, label=label)
    image = torch.zeros(3, page_size[1], page_size[0])
    return DocLayNetSample(
        image=image,
        regions=[region],
        uid=uid,
        document_id="doc",
        page_index=0,
        original_size=page_size,
    )


def test_dataset_uses_lookup_embeddings(tmp_path) -> None:
    sample = _build_sample("doc-0", (0, 0, 200, 400), "text")
    base = DummyDocLayNetDataset([sample])

    embeddings = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    np.savez(tmp_path / "embeddings.npz", uids=np.array(["doc-0"]), embeddings=embeddings)

    lookup, dim = load_embedding_lookup(tmp_path / "embeddings.npz")
    dataset = DocLayNetFeatureDataset(base, embedding_dim=dim, embedding_lookup=lookup)
    example = dataset[0]

    assert example.embedding.shape == (3,)
    expected = torch.tensor(embeddings[0].tolist())
    assert torch.allclose(example.embedding, expected)

    expected_features = compute_layout_features(sample.regions, sample.original_size)
    expected_target = torch.tensor(expected_features.tolist())
    assert torch.allclose(example.target, expected_target)


def test_dataset_generates_deterministic_random_embeddings() -> None:
    sample = _build_sample("doc-1", (0, 0, 400, 200), "table")
    base = DummyDocLayNetDataset([sample])

    dataset = DocLayNetFeatureDataset(
        base,
        embedding_dim=4,
        embedding_lookup=None,
        allow_random=True,
        random_seed=42,
    )

    ex_a = dataset[0].embedding

    # Recreate dataset to ensure determinism
    dataset_b = DocLayNetFeatureDataset(
        base,
        embedding_dim=4,
        embedding_lookup=None,
        allow_random=True,
        random_seed=42,
    )
    ex_b = dataset_b[0].embedding

    assert torch.allclose(ex_a, ex_b)
