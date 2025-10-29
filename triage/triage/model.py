# Location: triage/triage/model.py
# Purpose: Define trainable heads for the RVL classifier track.
# Why: Phase 4 requires lightweight, configurable models that operate on backbone embeddings and optional layout descriptors.

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


@dataclass
class ClassifierConfig:
    """Configuration for the RVL classifier MLP."""

    embedding_dim: int
    num_classes: int
    layout_dim: int = 0
    hidden_dims: Sequence[int] = (512, 256)
    dropout: float = 0.1
    activation: nn.Module = nn.GELU()


class RVLClassifier(nn.Module):
    """Simple feed-forward head for RVL classification."""

    def __init__(self, config: ClassifierConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = config.embedding_dim + max(0, config.layout_dim)

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(config.activation)
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, config.num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, embedding: torch.Tensor, layout: torch.Tensor | None = None) -> torch.Tensor:
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        if layout is not None:
            if layout.dim() == 1:
                layout = layout.unsqueeze(0)
            features = torch.cat([embedding, layout], dim=-1)
        else:
            features = embedding
        return self.mlp(features)


def build_classifier(config: ClassifierConfig) -> RVLClassifier:
    """Helper mirroring the config dataclass interface."""

    return RVLClassifier(config)


def logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert raw logits to probabilities via softmax."""

    return torch.softmax(logits, dim=-1)
