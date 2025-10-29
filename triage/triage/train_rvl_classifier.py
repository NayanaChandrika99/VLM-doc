# Location: triage/triage/train_rvl_classifier.py
# Purpose: Train the RVL classifier head (with optional layout fusion) while supporting lightweight dry-run workflows.
# Why: Phase 4 requires reproducible training scripts that produce checkpoints, logits, and metrics even on machines without GPU access.

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from triage.triage.dataset import ClassifierExample, RVLClassifierDataset
from triage.triage.model import ClassifierConfig, RVLClassifier, logits_to_probabilities


@dataclass
class TrainingConfig:
    train_split: str = "train"
    validation_split: str = "validation"
    embedding_store: Optional[str] = None
    embedding_dim: int = 1024
    layout_store: Optional[str] = None
    layout_head_checkpoint: Optional[str] = None
    use_layout: bool = False
    dry_run: bool = False
    dry_run_size: int = 128
    random_seed: int = 13
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dims: Sequence[int] = (512, 256)
    dropout: float = 0.1
    device: str = "cpu"
    dataset_id: str = "vaclavpechtor/rvl_cdip-small-200"
    cache_dir: Optional[str] = None
    output_dir: Path = Path("triage/artifacts")
    checkpoint_name: str = "rvl_classifier.pt"
    logits_name: str = "rvl_val_logits.npz"
    metrics_name: str = "rvl_training_metrics.json"

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data


def build_dataloaders(config: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    train_dataset = RVLClassifierDataset(
        config.train_split,
        embedding_dim=config.embedding_dim,
        embedding_store=config.embedding_store,
        layout_store=config.layout_store,
        layout_head_checkpoint=config.layout_head_checkpoint,
        use_layout=config.use_layout,
        dry_run=config.dry_run,
        dry_run_size=config.dry_run_size,
        random_seed=config.random_seed,
        dataset_id=config.dataset_id,
        cache_dir=config.cache_dir,
    )
    val_dataset = RVLClassifierDataset(
        config.validation_split,
        embedding_dim=config.embedding_dim,
        embedding_store=config.embedding_store,
        layout_store=config.layout_store,
        layout_head_checkpoint=config.layout_head_checkpoint,
        use_layout=config.use_layout,
        dry_run=config.dry_run,
        dry_run_size=max(32, config.dry_run_size // 2),
        random_seed=config.random_seed + 1,
        dataset_id=config.dataset_id,
        cache_dir=config.cache_dir,
    )

    collate_fn = _build_collate(config.use_layout)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def train_classifier(config: TrainingConfig) -> Dict[str, object]:
    torch.manual_seed(config.random_seed)

    train_loader, val_loader = build_dataloaders(config)

    base_dataset = train_loader.dataset.dataset if isinstance(train_loader.dataset, Subset) else train_loader.dataset  # type: ignore[attr-defined]
    num_classes = len(getattr(base_dataset, "label_names"))
    layout_dim = getattr(base_dataset, "layout_dim", 0)
    if layout_dim == 0 and config.use_layout:
        sample_example = base_dataset[0]
        if sample_example.layout is not None:
            layout_dim = sample_example.layout.shape[0]

    classifier_config = ClassifierConfig(
        embedding_dim=config.embedding_dim,
        layout_dim=layout_dim,
        num_classes=num_classes,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )
    model = RVLClassifier(classifier_config).to(config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(config.epochs):
        train_loss, train_acc = _run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device=config.device,
            train=True,
        )
        val_loss, val_acc = _run_epoch(
            model,
            val_loader,
            optimizer=None,
            criterion=criterion,
            device=config.device,
            train=False,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / config.checkpoint_name
    torch.save(
        {
            "state_dict": best_state or model.state_dict(),
            "config": classifier_config.__dict__,
            "training_config": config.to_dict(),
            "history": history,
        },
        checkpoint_path,
    )

    logits_path = output_dir / config.logits_name
    logits_dict = _collect_logits(model, val_loader, device=config.device, use_layout=config.use_layout)
    np.savez(logits_path, **logits_dict)

    metrics = {
        "best_val_loss": best_val_loss,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "logits": str(logits_path),
        "config": config.to_dict(),
    }
    metrics_path = output_dir / config.metrics_name
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def _run_epoch(
    model: RVLClassifier,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: nn.Module,
    *,
    device: str,
    train: bool,
) -> Tuple[float, float]:
    model.train(mode=train)
    total_loss = 0.0
    correct = 0
    total = 0

    for embeddings, labels, layouts, _ in loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        layouts = layouts.to(device) if layouts is not None else None

        if train:
            optimizer.zero_grad()  # type: ignore[union-attr]
        logits = model(embeddings, layouts)
        loss = criterion(logits, labels)

        if train:
            loss.backward()
            optimizer.step()  # type: ignore[union-attr]

        total_loss += float(loss.item()) * embeddings.size(0)
        preds = logits.argmax(dim=-1)
        correct += int((preds == labels).sum().item())
        total += embeddings.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def _collect_logits(model: RVLClassifier, loader: DataLoader, *, device: str, use_layout: bool) -> Dict[str, np.ndarray]:
    model.eval()
    logits_list: list[list[float]] = []
    labels_list: list[int] = []
    probs_list: list[list[float]] = []
    uids: list[str] = []

    with torch.no_grad():
        for embeddings, labels, layouts, batch_uids in loader:
            embeddings = embeddings.to(device)
            layouts = layouts.to(device) if (layouts is not None) else None
            logits = model(embeddings, layouts)
            probs = logits_to_probabilities(logits)

            logits_list.extend(logits.cpu().float().tolist())
            probs_list.extend(probs.cpu().float().tolist())
            labels_list.extend(labels.cpu().tolist())
            uids.extend(batch_uids)

    return {
        "uids": np.array(uids),
        "logits": np.array(logits_list, dtype=np.float32),
        "probabilities": np.array(probs_list, dtype=np.float32),
        "labels": np.array(labels_list, dtype=np.int64),
    }


def _build_collate(use_layout: bool):
    def collate(batch: Sequence[ClassifierExample]):
        embeddings = torch.stack([example.embedding for example in batch])
        labels = torch.tensor([example.label_id for example in batch], dtype=torch.long)
        layouts = None
        if use_layout:
            layout_tensors = []
            for example in batch:
                if example.layout is not None:
                    layout_tensors.append(example.layout)
                else:
                    if layout_tensors:
                        layout_tensors.append(torch.zeros_like(layout_tensors[0]))
                    else:
                        layout_tensors.append(torch.zeros(1, dtype=torch.float32))
            layouts = torch.stack(layout_tensors)
        uids = [example.uid for example in batch]
        return embeddings, labels, layouts, uids

    return collate


def parse_args(argv: Optional[Sequence[str]] = None) -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train the RVL classifier head.")
    parser.add_argument("--embedding-store", help="Path to NPZ file containing embeddings.")
    parser.add_argument("--embedding-dim", type=int, default=1024, help="Dimensionality of embeddings.")
    parser.add_argument("--layout-store", help="Optional NPZ file containing layout vectors.")
    parser.add_argument("--layout-head-checkpoint", help="Optional path to layout head checkpoint.")
    parser.add_argument("--use-layout", action="store_true", help="Concatenate layout features during training.")
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic data for a fast development loop.")
    parser.add_argument("--dry-run-size", type=int, default=128, help="Synthetic dataset size per split.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--output-dir", default="triage/artifacts", help="Directory to store artefacts.")
    parser.add_argument("--device", default="cpu", help="Computation device (cpu or cuda).")
    parser.add_argument("--random-seed", type=int, default=13, help="Random seed.")
    parser.add_argument("--dataset-id", default="vaclavpechtor/rvl_cdip-small-200", help="HF dataset identifier.")
    parser.add_argument("--cache-dir", help="Hugging Face cache directory.")
    args = parser.parse_args(argv)

    return TrainingConfig(
        embedding_store=args.embedding_store,
        embedding_dim=args.embedding_dim,
        layout_store=args.layout_store,
        layout_head_checkpoint=args.layout_head_checkpoint,
        use_layout=args.use_layout,
        dry_run=args.dry_run,
        dry_run_size=args.dry_run_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        output_dir=Path(args.output_dir),
        device=args.device,
        random_seed=args.random_seed,
        dataset_id=args.dataset_id,
        cache_dir=args.cache_dir,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    metrics = train_classifier(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
