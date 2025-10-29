# Location: triage/triage/eval_rvl_classifier.py
# Purpose: Evaluate the trained RVL classifier, compute metrics, and export confusion matrices/logits.
# Why: Phase 4 needs reproducible evaluation tooling that works with dry-run data locally and real artefacts on heavier machines.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from triage.calibration.reliability_plots import compute_reliability, coverage_vs_error
from triage.calibration.temp_scaling import TemperatureScaler, load_temperature
from triage.io.abstention import apply_threshold
from triage.triage.dataset import ClassifierExample, RVLClassifierDataset
from triage.triage.model import ClassifierConfig, RVLClassifier, logits_to_probabilities


def evaluate_classifier(config: argparse.Namespace) -> Dict[str, object]:
    dataset = RVLClassifierDataset(
        config.split,
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

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=_collate_examples(config.use_layout),
    )

    checkpoint = torch.load(Path(config.checkpoint), map_location=config.device)
    cfg_dict = checkpoint.get("config")
    if cfg_dict is None:
        raise KeyError("Checkpoint missing 'config' metadata.")
    classifier_config = ClassifierConfig(**cfg_dict)
    model = RVLClassifier(classifier_config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(config.device)
    model.eval()

    all_logits: list[list[float]] = []
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_uids: list[str] = []
    all_confidences: list[float] = []

    with torch.no_grad():
        for embeddings, labels, layouts, batch_uids in loader:
            embeddings = embeddings.to(config.device)
            layouts = layouts.to(config.device) if layouts is not None else None
            logits = model(embeddings, layouts)
            all_logits.extend(logits.cpu().float().tolist())
            all_labels.extend(labels.tolist())
            preds = logits.argmax(dim=-1)
            probs = logits_to_probabilities(logits)
            confidences = probs.max(dim=-1).values
            all_preds.extend(preds.cpu().tolist())
            all_confidences.extend(confidences.cpu().tolist())
            all_uids.extend(batch_uids)

    logits_array = np.array(all_logits, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int64)

    scaler = TemperatureScaler()
    temperature_path = getattr(config, "temperature", None)
    if temperature_path and Path(temperature_path).exists():
        scaler = load_temperature(temperature_path)
    scaled_logits = logits_array / scaler.temperature
    exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    label_names = dataset.label_names
    metrics = _compute_metrics(labels_array, all_preds, label_names)

    num_bins = getattr(config, "num_bins", 10)
    reliability = compute_reliability(probabilities, labels_array, num_bins=num_bins)
    thresholds = getattr(config, "thresholds", [0.90, 0.95, 0.97, 0.99])
    coverage = coverage_vs_error(probabilities, labels_array, thresholds)
    metrics["calibration"] = {
        "temperature": scaler.temperature,
        "ece": reliability.ece,
        "reliability": {
            "bins": reliability.bins,
            "accuracies": reliability.accuracies,
            "confidences": reliability.confidences,
            "counts": reliability.counts,
        },
        "coverage_vs_error": coverage,
    }

    confidences = probabilities.max(axis=1)
    predicted_labels = [label_names[idx] for idx in all_preds]
    abstention_summary = {}
    for threshold in thresholds:
        result = apply_threshold(predicted_labels, confidences, threshold)
        abstention_summary[f"{threshold:.2f}"] = {
            "coverage": result.coverage,
            "abstain_rate": result.abstain_rate,
        }
    metrics["abstention"] = abstention_summary

    if config.save_logits:
        logits_path = Path(config.save_logits)
        logits_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            logits_path,
            uids=np.array(all_uids),
            logits=np.array(all_logits, dtype=np.float32),
            probabilities=probabilities,
            labels=labels_array,
            predictions=np.array(all_preds, dtype=np.int64),
            confidences=np.array(confidences, dtype=np.float32),
        )
        metrics["logits_path"] = str(logits_path)

    abstention_output = getattr(config, "abstention_output", None)
    if abstention_output:
        abstention_path = Path(abstention_output)
        abstention_path.parent.mkdir(parents=True, exist_ok=True)
        abstention_payload = {
            "uids": all_uids,
            "predicted_labels": predicted_labels,
            "confidences": confidences.tolist(),
            "thresholds": config.thresholds,
        }
        abstention_path.write_text(json.dumps(abstention_payload, indent=2))

    if config.metrics_path:
        path = Path(config.metrics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2))

    return metrics


def _compute_metrics(labels: Sequence[int], preds: Sequence[int], label_names: Sequence[str]) -> Dict[str, object]:
    num_classes = len(label_names)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for truth, pred in zip(labels, preds):
        confusion[int(truth), int(pred)] += 1

    per_class_f1: Dict[str, float] = {}
    f1_sum = 0.0
    for idx, label in enumerate(label_names):
        tp = confusion[idx, idx]
        fp = confusion[:, idx].sum() - tp
        fn = confusion[idx, :].sum() - tp
        denom = (2 * tp + fp + fn)
        score = (2 * tp / denom) if denom else 0.0
        per_class_f1[label] = float(score)
        f1_sum += score

    macro_f1 = f1_sum / num_classes if num_classes else 0.0
    accuracy = float(np.trace(confusion) / max(1, confusion.sum()))

    confusion_dict: Dict[str, Dict[str, int]] = {}
    for i, label in enumerate(label_names):
        confusion_dict[label] = {label_names[j]: int(confusion[i, j]) for j in range(num_classes)}

    return {
        "meta": {
            "label_names": list(label_names),
            "num_samples": len(labels),
        },
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": float(macro_f1),
            "per_class_f1": per_class_f1,
        },
        "confusion": confusion_dict,
    }


def _collate_examples(use_layout: bool):
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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the RVL classifier head.")
    parser.add_argument("--checkpoint", required=True, help="Path to classifier checkpoint.")
    parser.add_argument("--split", default="validation", help="Dataset split to evaluate.")
    parser.add_argument("--embedding-store", help="Embedding NPZ file (required unless dry-run).")
    parser.add_argument("--embedding-dim", type=int, default=1024, help="Embedding dimensionality.")
    parser.add_argument("--layout-store", help="Optional layout feature NPZ file.")
    parser.add_argument("--layout-head-checkpoint", help="Optional layout head checkpoint for inference.")
    parser.add_argument("--use-layout", action="store_true", help="Concatenate layout features.")
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic data for quick validation.")
    parser.add_argument("--dry-run-size", type=int, default=128, help="Synthetic dataset size.")
    parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--device", default="cpu", help="Computation device (cpu/cuda).")
    parser.add_argument("--save-logits", help="Optional NPZ path to store logits/probabilities.")
    parser.add_argument("--metrics-path", default="metrics/rvl_classifier.json", help="Destination JSON for metrics.")
    parser.add_argument("--temperature", help="Optional temperature calibration JSON.")
    parser.add_argument("--num-bins", type=int, default=10, help="Reliability diagram bins.")
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.90, 0.95, 0.97, 0.99], help="Thresholds for abstention analyses.")
    parser.add_argument("--abstention-output", help="Optional JSON to record per-sample confidences and labels.")
    parser.add_argument("--random-seed", type=int, default=13, help="Random seed for synthetic data.")
    parser.add_argument("--dataset-id", default="vaclavpechtor/rvl_cdip-small-200", help="HF dataset identifier.")
    parser.add_argument("--cache-dir", help="HF cache directory.")
    args = parser.parse_args(argv)
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    metrics = evaluate_classifier(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
