# Location: triage/triage/dataset.py
# Purpose: Provide reusable dataset wrappers for RVL classifier training/evaluation.
# Why: Phase 4 requires a consistent way to pair embeddings (and optional layout signals) with RVL labels without running heavy models locally.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from triage.data import rvl_adapter
from triage.io.structured import allowed_labels
from triage.layout.predict_head import LayoutHead, LayoutHeadConfig


@dataclass
class ClassifierExample:
    """Single RVL training instance."""

    embedding: torch.Tensor
    label_id: int
    label_str: str
    uid: str
    layout: Optional[torch.Tensor] = None


class RVLClassifierDataset(Dataset[ClassifierExample]):
    """
    Dataset yielding embeddings, optional layout features, and RVL labels.

    Supports two operating modes:
      * ``dry_run=True`` – generates deterministic synthetic data with configurable size.
      * ``dry_run=False`` – loads real RVL samples and looks up embeddings/layout features
        from supplied stores. When ``use_layout`` is True and a layout head checkpoint is
        provided, layout features are predicted from embeddings on the fly.
    """

    def __init__(
        self,
        split: str,
        *,
        embedding_dim: int,
        embedding_store: Optional[str | Path] = None,
        layout_store: Optional[str | Path] = None,
        layout_head_checkpoint: Optional[str | Path] = None,
        use_layout: bool = False,
        dry_run: bool = False,
        dry_run_size: int = 128,
        random_seed: int = 13,
        dataset_id: str = rvl_adapter.RVL_DATASET_ID,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.split = split
        self.embedding_dim = int(embedding_dim)
        self.use_layout = use_layout
        self.dry_run = dry_run
        self.random_seed = random_seed
        self.label_names = tuple(allowed_labels(include_unknown=False))
        self.label_to_index = {label: idx for idx, label in enumerate(self.label_names)}

        self._embedding_lookup: Optional[MutableMapping[str, np.ndarray]] = None
        self._layout_lookup: Optional[MutableMapping[str, np.ndarray]] = None
        self._layout_head: Optional[LayoutHead] = None

        if dry_run:
            self._examples = _build_synthetic_examples(
                size=dry_run_size,
                embedding_dim=self.embedding_dim,
                layout_dim=len(self.label_names) if use_layout else 0,
                use_layout=use_layout,
                random_seed=random_seed,
                label_names=self.label_names,
            )
            self._rvl_dataset = None
            if use_layout and self._examples:
                first_layout = self._examples[0].layout
                self.layout_dim = len(first_layout) if first_layout is not None else 0
            else:
                self.layout_dim = 0
            return

        # Real dataset path
        self._rvl_dataset = rvl_adapter.load_rvl(
            split, dataset_id=dataset_id, cache_dir=cache_dir, streaming=False
        )
        self._examples = None

        if embedding_store is None:
            raise ValueError("embedding_store must be provided for non-dry-run datasets.")
        self._embedding_lookup, inferred_dim = load_embedding_store(embedding_store)
        if inferred_dim != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: store={inferred_dim} expected={self.embedding_dim}"
            )

        if layout_store is not None:
            lookup, layout_dim = load_embedding_store(layout_store)
            self._layout_lookup = lookup
            self._layout_dim = layout_dim
        else:
            self._layout_lookup = None
            self._layout_dim = None

        if use_layout and layout_head_checkpoint is not None:
            self._layout_head = _load_layout_head(layout_head_checkpoint, self.embedding_dim)
            self._layout_head.eval()
            self.layout_dim = self._layout_head(torch.zeros(1, self.embedding_dim)).shape[-1]
        elif self._layout_lookup is not None:
            first_vector = next(iter(self._layout_lookup.values()))
            self.layout_dim = len(first_vector)
        else:
            self.layout_dim = 0

    def __len__(self) -> int:
        if self.dry_run:
            return len(self._examples)  # type: ignore[arg-type]
        assert self._rvl_dataset is not None
        return len(self._rvl_dataset)

    def __getitem__(self, index: int) -> ClassifierExample:
        if self.dry_run:
            assert self._examples is not None
            return self._examples[index]

        assert self._rvl_dataset is not None
        sample = self._rvl_dataset[index]
        uid = sample.uid
        embedding = self._resolve_embedding(uid)

        layout_tensor: Optional[torch.Tensor] = None
        if self.use_layout:
            layout_tensor = self._resolve_layout(uid, embedding)

        label_str = sample.label_str
        label_id = self.label_to_index[label_str]
        return ClassifierExample(
            embedding=embedding,
            label_id=label_id,
            label_str=label_str,
            uid=uid,
            layout=layout_tensor,
        )

    def _resolve_embedding(self, uid: str) -> torch.Tensor:
        if self._embedding_lookup is None:
            raise KeyError("Embedding lookup not initialised.")
        if uid not in self._embedding_lookup:
            raise KeyError(f"Embedding for uid '{uid}' not found.")
        vector = self._embedding_lookup[uid]
        return torch.tensor(vector.tolist(), dtype=torch.float32)

    def _resolve_layout(self, uid: str, embedding: torch.Tensor) -> torch.Tensor:
        if self._layout_lookup is not None:
            vector = self._layout_lookup.get(uid)
            if vector is None:
                raise KeyError(f"Layout vector for uid '{uid}' not found in layout store.")
            return torch.tensor(vector.tolist(), dtype=torch.float32)

        if self._layout_head is None:
            raise ValueError(
                "Layout features requested but no layout_store or layout_head_checkpoint provided."
            )
        with torch.no_grad():
            layout = self._layout_head(embedding.float())
        return layout.squeeze(0) if layout.dim() == 2 and layout.size(0) == 1 else layout


def load_embedding_store(path: str | Path) -> tuple[Dict[str, np.ndarray], int]:
    """Load uid → vector lookup from an NPZ file containing `uids` and `embeddings`."""

    store = np.load(Path(path), allow_pickle=False)
    if "uids" not in store or "embeddings" not in store:
        raise KeyError("Embedding store must contain 'uids' and 'embeddings'.")
    uids = store["uids"]
    embeddings = store["embeddings"]
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings with shape (N, D); found {embeddings.shape}.")
    dim = embeddings.shape[1]
    lookup: Dict[str, np.ndarray] = {}
    for uid, vector in zip(uids, embeddings, strict=True):
        lookup[str(uid)] = np.asarray(vector, dtype=np.float32)
    return lookup, dim


def _build_synthetic_examples(
    *,
    size: int,
    embedding_dim: int,
    layout_dim: int,
    use_layout: bool,
    random_seed: int,
    label_names: Sequence[str],
) -> list[ClassifierExample]:
    rng = np.random.default_rng(random_seed)
    examples: list[ClassifierExample] = []
    for idx in range(size):
        uid = f"dry-{idx}"
        embedding = torch.tensor(
            rng.standard_normal(embedding_dim).astype(np.float32).tolist(), dtype=torch.float32
        )
        label_str = label_names[idx % len(label_names)]
        label_id = idx % len(label_names)
        layout_tensor = None
        if use_layout:
            layout_tensor = torch.tensor(
                rng.standard_normal(layout_dim).astype(np.float32).tolist(),
                dtype=torch.float32,
            )
        examples.append(
            ClassifierExample(
                embedding=embedding,
                label_id=label_id,
                label_str=label_str,
                uid=uid,
                layout=layout_tensor,
            )
        )
    return examples


def _load_layout_head(path: str | Path, embedding_dim: int) -> LayoutHead:
    checkpoint = torch.load(Path(path), map_location="cpu")
    metadata = checkpoint.get("metadata") or {}
    descriptor_dim = metadata.get("descriptor_dim")
    if descriptor_dim is None:
        raise KeyError("Layout head checkpoint missing 'descriptor_dim' metadata.")
    cfg = LayoutHeadConfig(
        embedding_dim=embedding_dim,
        descriptor_dim=int(descriptor_dim),
        hidden_dim=metadata.get("config", {}).get("hidden_dim", 512),
        num_hidden_layers=metadata.get("config", {}).get("num_hidden_layers", 2),
        dropout=metadata.get("config", {}).get("dropout", 0.1),
    )
    head = LayoutHead(cfg)
    state = checkpoint.get("state_dict")
    if state is None:
        raise KeyError("Layout head checkpoint missing 'state_dict'.")
    head.load_state_dict(state)
    return head
