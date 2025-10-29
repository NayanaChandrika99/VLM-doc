# Location: triage/layout/train_layout_head.py
# Purpose: Train a lightweight predictor that maps embeddings to layout descriptors.
# Why: Phase 3 requires deterministic layout features and a learned head ready for fusion.

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from triage.layout.dataset import DocLayNetFeatureDataset, LayoutExample, load_doclaynet_feature_dataset
from triage.layout.features import LAYOUT_FEATURE_NAMES
from triage.layout.predict_head import LayoutHead, LayoutHeadConfig, build_loss_fn


@dataclass
class TrainingConfig:
    split: str = "train"
    embedding_store: Optional[str] = None
    embedding_dim: int = 1024
    allow_random: bool = False
    random_seed: int = 13
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 1e-3
    dropout: float = 0.1
    hidden_dim: int = 512
    num_hidden_layers: int = 2
    loss: str = "smooth_l1"
    beta: float = 0.1
    val_fraction: float = 0.1
    device: str = "cpu"
    output_dir: Path = Path("layout/artifacts")
    metrics_path: Optional[Path] = None
    dry_run: bool = False
    dry_run_size: int = 64
    dataset_limit: Optional[int] = None
    cache_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        if self.metrics_path is not None:
            data["metrics_path"] = str(self.metrics_path)
        return data


class SyntheticLayoutDataset(Dataset[LayoutExample]):
    """Synthetic dataset used for dry-run training loops."""

    def __init__(self, *, size: int, embedding_dim: int, descriptor_dim: int, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        embeddings = torch.randn(size, embedding_dim, generator=generator)
        targets = torch.sigmoid(torch.randn(size, descriptor_dim, generator=generator))
        self.examples = [
            LayoutExample(
                embedding=embeddings[i],
                target=targets[i],
                uid=f"synth-{i}",
                document_id="synth",
                page_index=i,
            )
            for i in range(size)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> LayoutExample:
        return self.examples[index]


def build_dataset(config: TrainingConfig) -> DocLayNetFeatureDataset | SyntheticLayoutDataset:
    if config.dry_run:
        return SyntheticLayoutDataset(
            size=config.dry_run_size,
            embedding_dim=config.embedding_dim,
            descriptor_dim=len(LAYOUT_FEATURE_NAMES),
            seed=config.random_seed,
        )
    return load_doclaynet_feature_dataset(
        config.split,
        embedding_store=config.embedding_store,
        embedding_dim=config.embedding_dim,
        allow_random=config.allow_random,
        random_seed=config.random_seed,
        cache_dir=config.cache_dir,
    )


def collate_examples(batch: Sequence[LayoutExample]) -> Tuple[torch.Tensor, torch.Tensor]:
    embeddings = torch.stack([example.embedding for example in batch])
    targets = torch.stack([example.target for example in batch])
    return embeddings, targets


def prepare_loaders(
    dataset: Dataset[LayoutExample],
    *,
    batch_size: int,
    val_fraction: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least two samples.")

    generator = torch.Generator().manual_seed(seed)
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_examples),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_examples),
    )


def train_layout_head(config: TrainingConfig) -> Dict[str, object]:
    torch.manual_seed(config.random_seed)

    dataset = build_dataset(config)
    if config.dataset_limit is not None and not config.dry_run:
        indices = list(range(min(len(dataset), config.dataset_limit)))
        dataset = Subset(dataset, indices)

    train_loader, val_loader = prepare_loaders(
        dataset,
        batch_size=config.batch_size,
        val_fraction=config.val_fraction,
        seed=config.random_seed,
    )

    descriptor_dim = len(LAYOUT_FEATURE_NAMES)
    model_config = LayoutHeadConfig(
        embedding_dim=config.embedding_dim,
        descriptor_dim=descriptor_dim,
        hidden_dim=config.hidden_dim,
        num_hidden_layers=config.num_hidden_layers,
        dropout=config.dropout,
    )
    model = LayoutHead(model_config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = build_loss_fn(config.loss, beta=config.beta)

    best_state = None
    best_loss = float("inf")
    history: list[Dict[str, float]] = []

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        total = 0
        for embeddings, targets in train_loader:
            embeddings = embeddings.to(config.device)
            targets = targets.to(config.device)

            optimizer.zero_grad()
            predictions = model(embeddings)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item()) * embeddings.size(0)
            total += embeddings.size(0)

        avg_train_loss = train_loss / max(total, 1)

        model.eval()
        val_loss = 0.0
        val_total = 0
        with torch.no_grad():
            for embeddings, targets in val_loader:
                embeddings = embeddings.to(config.device)
                targets = targets.to(config.device)
                predictions = model(embeddings)
                loss = loss_fn(predictions, targets)
                val_loss += float(loss.item()) * embeddings.size(0)
                val_total += embeddings.size(0)
        avg_val_loss = val_loss / max(val_total, 1)

        history.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = model.state_dict()

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = output_dir / "layout_head.pt"
    metadata = {
        "feature_names": LAYOUT_FEATURE_NAMES,
        "embedding_dim": config.embedding_dim,
        "descriptor_dim": descriptor_dim,
        "config": config.to_dict(),
    }

    torch.save({"state_dict": best_state, "metadata": metadata, "history": history}, artifact_path)

    metrics = {
        "best_val_loss": best_loss,
        "history": history,
        "num_train_samples": len(train_loader.dataset),  # type: ignore[attr-defined]
        "num_val_samples": len(val_loader.dataset),  # type: ignore[attr-defined]
        "dry_run": config.dry_run,
    }

    metrics_path = config.metrics_path or (output_dir / "layout_head_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def _parse_args(argv: Optional[Sequence[str]] = None) -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train layout descriptor predictor head.")
    parser.add_argument("--split", default="train", help="DocLayNet split to use (train/validation/test).")
    parser.add_argument("--embedding-store", help="Path to NPZ containing precomputed embeddings.")
    parser.add_argument("--embedding-dim", type=int, default=1024, help="Embedding dimension if no store is provided.")
    parser.add_argument("--allow-random", action="store_true", help="Generate deterministic random embeddings on miss.")
    parser.add_argument("--random-seed", type=int, default=13, help="Random seed for dataset split and training.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout applied between hidden layers.")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden layer width.")
    parser.add_argument("--num-hidden-layers", type=int, default=2, help="Number of hidden layers.")
    parser.add_argument("--loss", default="smooth_l1", help="Loss function (smooth_l1|mse|mae).")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta parameter for SmoothL1Loss.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of data reserved for validation.")
    parser.add_argument("--output-dir", default="layout/artifacts", help="Directory to store weights/metrics.")
    parser.add_argument("--metrics-path", help="Override metrics output file.")
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic data instead of DocLayNet.")
    parser.add_argument("--dry-run-size", type=int, default=64, help="Synthetic dataset size for dry run.")
    parser.add_argument("--dataset-limit", type=int, help="Maximum number of samples from DocLayNet.")
    parser.add_argument("--cache-dir", help="Hugging Face cache directory.")
    args = parser.parse_args(argv)

    return TrainingConfig(
        split=args.split,
        embedding_store=args.embedding_store,
        embedding_dim=args.embedding_dim,
        allow_random=args.allow_random or args.dry_run,
        random_seed=args.random_seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        loss=args.loss,
        beta=args.beta,
        val_fraction=args.val_fraction,
        output_dir=Path(args.output_dir),
        metrics_path=Path(args.metrics_path) if args.metrics_path else None,
        dry_run=args.dry_run,
        dry_run_size=args.dry_run_size,
        dataset_limit=args.dataset_limit,
        cache_dir=args.cache_dir,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = _parse_args(argv)
    metrics = train_layout_head(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
