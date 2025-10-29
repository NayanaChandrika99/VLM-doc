"""Datasets wrapping DocLayNet samples with deterministic layout features."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from triage.data.doclaynet_adapter import DocLayNetDataset, DocLayNetSample, load_doclaynet
from triage.layout.features import LAYOUT_FEATURE_NAMES, compute_layout_features


@dataclass
class LayoutExample:
    """Container representing a single feature-learning example."""

    embedding: torch.Tensor
    target: torch.Tensor
    uid: str
    document_id: str
    page_index: int


class DocLayNetFeatureDataset(Dataset[LayoutExample]):
    """
    Dataset yielding (embedding, layout_descriptor) pairs for feature learning.

    Embeddings are sourced from a lookup table when available; otherwise,
    deterministic random vectors can be generated when ``allow_random=True``.
    """

    def __init__(
        self,
        base_dataset: DocLayNetDataset,
        *,
        embedding_dim: int,
        embedding_lookup: Optional[Mapping[str, Sequence[float]]] = None,
        allow_random: bool = False,
        random_seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._base = base_dataset
        self.embedding_dim = int(embedding_dim)
        self._embedding_lookup = embedding_lookup
        self.allow_random = allow_random
        self.random_seed = random_seed
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> LayoutExample:
        sample = self._base[index]
        features = compute_layout_features(sample.regions, sample.original_size)
        target_tensor = torch.tensor(features.tolist(), dtype=self.dtype)
        emb_tensor = self._resolve_embedding(sample.uid)
        return LayoutExample(
            embedding=emb_tensor,
            target=target_tensor,
            uid=sample.uid,
            document_id=sample.document_id,
            page_index=sample.page_index,
        )

    def _resolve_embedding(self, uid: str) -> torch.Tensor:
        if self._embedding_lookup is not None and uid in self._embedding_lookup:
            vector = np.asarray(self._embedding_lookup[uid], dtype=np.float32)
            if vector.shape[0] != self.embedding_dim:
                raise ValueError(
                    f"Embedding for uid '{uid}' has dim {vector.shape[0]}, expected {self.embedding_dim}."
                )
            return torch.tensor(vector.tolist(), dtype=self.dtype)

        if not self.allow_random:
            raise KeyError(f"Embedding for uid '{uid}' not found and random fallback disabled.")

        rng = _uid_rng(uid, seed=self.random_seed)
        vector = rng.standard_normal(self.embedding_dim).astype(np.float32)
        return torch.tensor(vector.tolist(), dtype=self.dtype)


def load_doclaynet_feature_dataset(
    split: str,
    *,
    embedding_store: Optional[str | Path] = None,
    embedding_dim: Optional[int] = None,
    allow_random: bool = False,
    random_seed: int = 0,
    cache_dir: Optional[str] = None,
) -> DocLayNetFeatureDataset:
    """
    Convenience loader wrapping :func:`load_doclaynet` with layout features.

    Args:
        split: Logical split (train/validation/test).
        embedding_store: Optional ``.npz`` file containing ``uids`` and ``embeddings`` arrays.
        embedding_dim: Required when ``embedding_store`` is omitted.
        allow_random: If True, generate deterministic random embeddings when missing.
        random_seed: Seed controlling random embedding generation.
        cache_dir: Optional Hugging Face cache directory.

    Returns:
        :class:`DocLayNetFeatureDataset`.
    """

    base = load_doclaynet(split, cache_dir=cache_dir)
    lookup: Optional[MutableMapping[str, Sequence[float]]] = None
    store_dim = None
    if embedding_store is not None:
        lookup, store_dim = load_embedding_lookup(embedding_store)
    dim = store_dim or embedding_dim
    if dim is None:
        raise ValueError("Either embedding_store or embedding_dim must be provided.")
    return DocLayNetFeatureDataset(
        base,
        embedding_dim=int(dim),
        embedding_lookup=lookup,
        allow_random=allow_random,
        random_seed=random_seed,
    )


def load_embedding_lookup(path: str | Path) -> tuple[Dict[str, np.ndarray], int]:
    """
    Load ``uid`` → embedding vectors from a NumPy ``.npz`` store.

    The file must contain arrays named ``uids`` and ``embeddings``. Embeddings are
    expected to be two-dimensional ``(N, D)``. Returns the lookup dictionary and the
    inferred embedding dimension ``D``.
    """

    store = np.load(Path(path), allow_pickle=False)
    if "uids" not in store or "embeddings" not in store:
        raise KeyError("Embedding store must contain 'uids' and 'embeddings' arrays.")
    uids = store["uids"]
    embeddings = store["embeddings"]
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings with shape (N, D); received {embeddings.shape}.")
    lookup: Dict[str, np.ndarray] = {}
    for uid, vector in zip(uids, embeddings, strict=True):
        lookup[str(uid)] = np.asarray(vector, dtype=np.float32)
    return lookup, embeddings.shape[1]


def _uid_rng(uid: str, seed: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{uid}-{seed}".encode("utf-8")).digest()
    seed_int = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(seed_int)
