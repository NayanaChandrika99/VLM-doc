from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from triage.triage.eval_rvl_classifier import evaluate_classifier
from triage.triage.train_rvl_classifier import TrainingConfig, train_classifier


def test_evaluate_classifier_dry_run(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    config = TrainingConfig(
        embedding_dim=12,
        use_layout=False,
        dry_run=True,
        dry_run_size=32,
        epochs=1,
        batch_size=8,
        output_dir=output_dir,
        embedding_store=None,
    )
    train_classifier(config)

    metrics_path = tmp_path / "metrics.json"
    args = Namespace(
        split="validation",
        embedding_dim=12,
        embedding_store=None,
        layout_store=None,
        layout_head_checkpoint=None,
        use_layout=False,
        dry_run=True,
        dry_run_size=32,
        random_seed=42,
        dataset_id="vaclavpechtor/rvl_cdip-small-200",
        cache_dir=None,
        batch_size=16,
        checkpoint=str(output_dir / config.checkpoint_name),
        device="cpu",
        save_logits=str(tmp_path / "logits.npz"),
        metrics_path=str(metrics_path),
    )

    metrics = evaluate_classifier(args)
    assert metrics_path.exists()
    assert "metrics" in metrics
    assert metrics["metrics"]["macro_f1"] >= 0
