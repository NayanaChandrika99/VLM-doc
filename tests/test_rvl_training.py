from __future__ import annotations

from pathlib import Path

from triage.triage.train_rvl_classifier import TrainingConfig, train_classifier


def test_train_classifier_dry_run(tmp_path: Path) -> None:
    config = TrainingConfig(
        embedding_dim=16,
        use_layout=True,
        dry_run=True,
        dry_run_size=32,
        epochs=1,
        batch_size=8,
        output_dir=tmp_path / "artifacts",
        embedding_store=None,
    )

    metrics = train_classifier(config)
    checkpoint = config.output_dir / config.checkpoint_name
    logits = config.output_dir / config.logits_name
    metrics_path = config.output_dir / config.metrics_name

    assert checkpoint.exists()
    assert logits.exists()
    assert metrics_path.exists()
    assert metrics["best_val_loss"] >= 0
