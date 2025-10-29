from __future__ import annotations

import json
from pathlib import Path

import torch

from triage.layout.predict_head import LayoutHead, LayoutHeadConfig
from triage.layout.train_layout_head import TrainingConfig, train_layout_head


def test_layout_head_forward_shapes() -> None:
    config = LayoutHeadConfig(embedding_dim=16, descriptor_dim=8, hidden_dim=32, num_hidden_layers=1, dropout=0.0)
    model = LayoutHead(config)

    input_tensor = torch.randn(4, 16)
    output = model(input_tensor)
    assert output.shape == (4, 8)


def test_train_layout_head_dry_run(tmp_path: Path) -> None:
    config = TrainingConfig(
        embedding_dim=16,
        batch_size=8,
        epochs=2,
        dry_run=True,
        dry_run_size=32,
        output_dir=tmp_path / "artifacts",
        metrics_path=tmp_path / "metrics.json",
    )

    metrics = train_layout_head(config)
    assert (config.output_dir / "layout_head.pt").exists()
    assert config.metrics_path.exists()

    written = json.loads(config.metrics_path.read_text())
    assert written["dry_run"] is True
    assert written["num_train_samples"] > 0
    assert metrics["best_val_loss"] == written["best_val_loss"]
