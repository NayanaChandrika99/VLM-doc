from __future__ import annotations

import torch

from triage.triage.model import ClassifierConfig, RVLClassifier, build_classifier, logits_to_probabilities


def test_rvl_classifier_forward_without_layout() -> None:
    config = ClassifierConfig(embedding_dim=8, num_classes=4, hidden_dims=(16,), dropout=0.0)
    model = build_classifier(config)

    inputs = torch.randn(3, 8)
    logits = model(inputs)
    assert logits.shape == (3, 4)


def test_rvl_classifier_forward_with_layout() -> None:
    config = ClassifierConfig(embedding_dim=6, layout_dim=2, num_classes=3, hidden_dims=(12,), dropout=0.0)
    model = RVLClassifier(config)

    embedding = torch.randn(2, 6)
    layout = torch.randn(2, 2)
    logits = model(embedding, layout)
    assert logits.shape == (2, 3)


def test_logits_to_probabilities() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    probs = logits_to_probabilities(logits)
    assert torch.allclose(probs.sum(dim=-1), torch.tensor([1.0]))
    assert probs.shape == (1, 3)

