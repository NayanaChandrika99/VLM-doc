# Location: triage/adapters/train_lora_adapter.py
# Purpose: Provide scaffolding for LoRA adapter creation with a dry-run development mode.
# Why: Phase 4 requires a reproducible entry point for future per-tenant fine-tuning without forcing GPU workloads on local machines.

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch

from triage.adapters.registry import get_default, register_adapter, set_default


@dataclass
class AdapterConfig:
    adapter_id: str
    base_model: str = "richardyoung/olmOCR-2-7B-1025-MLX-4bit"
    output_dir: Path = Path("triage/adapters/artifacts")
    dry_run: bool = False
    metadata_only: bool = False
    random_seed: int = 13
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data


def train_adapter(config: AdapterConfig) -> Dict[str, object]:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_path = output_dir / f"{config.adapter_id}.pt"
    metadata_path = output_dir / f"{config.adapter_id}.json"

    if config.dry_run:
        torch.manual_seed(config.random_seed)
        dummy_weights = {"lora_A": torch.randn(4, 4), "lora_B": torch.randn(4, 4)}
        torch.save(dummy_weights, adapter_path)
    else:
        raise RuntimeError(
            "Real LoRA training is not implemented in this scaffold. "
            "Run with --dry-run locally, and execute actual PEFT training on a GPU machine."
        )

    metadata = {
        "adapter_id": config.adapter_id,
        "base_model": config.base_model,
        "path": str(adapter_path),
        "config": config.to_dict(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    register_adapter(
        config.adapter_id,
        adapter_path,
        metadata={
            "base_model": config.base_model,
            "notes": config.notes or "",
        },
    )
    if get_default() is None:
        set_default(config.adapter_id)

    return metadata


def parse_args(argv: Optional[Sequence[str]] = None) -> AdapterConfig:
    parser = argparse.ArgumentParser(description="Train or register a LoRA adapter (scaffold).")
    parser.add_argument("--adapter-id", required=True, help="Logical identifier for the adapter.")
    parser.add_argument("--base-model", default="richardyoung/olmOCR-2-7B-1025-MLX-4bit", help="Base model identifier.")
    parser.add_argument("--output-dir", default="triage/adapters/artifacts", help="Directory to store adapter artefacts.")
    parser.add_argument("--dry-run", action="store_true", help="Create synthetic adapter weights for development.")
    parser.add_argument("--random-seed", type=int, default=13, help="Random seed for synthetic artefacts.")
    parser.add_argument("--notes", help="Optional notes stored in adapter metadata.")
    args = parser.parse_args(argv)
    return AdapterConfig(
        adapter_id=args.adapter_id,
        base_model=args.base_model,
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
        random_seed=args.random_seed,
        notes=args.notes,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    metadata = train_adapter(config)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
