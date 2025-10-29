"""Predictor head mapping backbone embeddings to layout descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import torch
from torch import nn


@dataclass
class LayoutHeadConfig:
    """Configuration parameters for :class:`LayoutHead`."""

    embedding_dim: int
    descriptor_dim: int
    hidden_dim: int = 512
    num_hidden_layers: int = 2
    dropout: float = 0.1
    activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.GELU()


class LayoutHead(nn.Module):
    """Multi-layer perceptron that predicts layout descriptors from embeddings."""

    def __init__(self, config: LayoutHeadConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        in_dim = config.embedding_dim
        for _ in range(max(0, config.num_hidden_layers)):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(config.activation)
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = config.hidden_dim

        layers.append(nn.Linear(in_dim, config.descriptor_dim))
        self.net = nn.Sequential(*layers)

        self.embedding_dim = config.embedding_dim
        self.descriptor_dim = config.descriptor_dim

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        return self.net(embeddings)


def build_loss_fn(name: str = "smooth_l1", beta: float = 0.1) -> nn.Module:
    """Factory producing a regression loss for layout descriptor training."""

    key = name.lower()
    if key in {"smooth_l1", "huber"}:
        return nn.SmoothL1Loss(beta=beta)
    if key in {"l2", "mse"}:
        return nn.MSELoss()
    if key in {"l1", "mae"}:
        return nn.L1Loss()
    raise ValueError(f"Unsupported loss function '{name}'.")
