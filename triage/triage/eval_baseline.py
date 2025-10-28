# Location: triage/triage/eval_baseline.py
# Purpose: Evaluate the prompted baseline on RVL validation/test splits and log metrics.
# Why: Phase 2 requires reproducible baseline metrics persisted to metrics/baseline.json.
"""Baseline evaluation pipeline for the prompted VLM classifier."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from datasets import load_dataset

from triage.data import rvl_adapter
from triage.io.structured import allowed_labels
from triage.prompts import build_baseline_prompt
from triage_infer import BaselineInference, InferenceConfig


def main(args: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the baseline prompted classifier.")
    parser.add_argument("--output", default="metrics/baseline.json", help="Path for metrics JSON.")
    parser.add_argument("--model-id", default=BaselineInference().config.model_id, help="Baseline model identifier.")
    parser.add_argument("--calibration-id", default="baseline-temp-v0", help="Calibration identifier metadata.")
    parser.add_argument("--adapter-id", default="global", help="Adapter identifier metadata.")
    parser.add_argument(
        "--threshold-grid",
        nargs="*",
        type=float,
        default=[round(x * 0.05, 2) for x in range(0, 21)],
        help="Confidence thresholds to evaluate on validation split.",
    )
    parsed = parser.parse_args(args=args)

    predictor = BaselineInference(
        config=InferenceConfig(
            model_id=parsed.model_id,
            calibration_id=parsed.calibration_id,
            adapter_id=parsed.adapter_id,
        )
    )

    # Collect predictions for validation and test
    val_truth, val_preds = _score_split(predictor, "validation")
    test_truth, test_preds = _score_split(predictor, "test")

    # Sweep thresholds on validation
    best_threshold, val_metrics = _select_threshold(
        val_truth,
        val_preds,
        thresholds=parsed.threshold_grid,
    )

    # Apply chosen threshold to validation/test for reporting
    val_report = _compile_metrics(val_truth, val_preds, threshold=best_threshold)
    test_report = _compile_metrics(test_truth, test_preds, threshold=best_threshold)

    output = {
        "meta": {
            "model_id": predictor.config.model_id,
            "calibration_id": predictor.config.calibration_id,
            "adapter_id": predictor.config.adapter_id,
            "prompt_guidance": build_baseline_prompt().splitlines(),
        },
        "dataset": {
            "id": rvl_adapter.RVL_DATASET_ID,
            "splits": {
                "validation": len(val_truth),
                "test": len(test_truth),
            },
        },
        "metrics": {
            "validation": val_report,
            "test": test_report,
        },
        "selection": {
            "threshold": best_threshold,
            "sweep": val_metrics,
        },
    }

    metrics_path = Path(parsed.output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote metrics to {metrics_path}")


def _score_split(
    predictor: BaselineInference,
    split: str,
) -> Tuple[List[str], List[dict]]:
    dataset = load_dataset(
        rvl_adapter.RVL_DATASET_ID,
        split=rvl_adapter.RVL_SPLITS[split],
    )
    label_names = dataset.features["label"].names
    truths: List[str] = []
    preds: List[dict] = []

    for record in dataset:
        truths.append(label_names[int(record["label"])])
        result = predictor.predict(record["image"])
        preds.append(result)
    return truths, preds


def _select_threshold(
    truth: Sequence[str],
    predictions: Sequence[dict],
    thresholds: Iterable[float],
) -> Tuple[float, Dict[str, float]]:
    best_threshold = 0.0
    best_metric = -1.0
    sweep_scores: Dict[str, float] = {}

    for tau in thresholds:
        report = _compute_macro_f1(truth, predictions, threshold=tau)
        sweep_scores[f"{tau:.2f}"] = report
        if report > best_metric:
            best_metric = report
            best_threshold = tau
    return best_threshold, sweep_scores


def _compile_metrics(truth: Sequence[str], predictions: Sequence[dict], threshold: float) -> Dict[str, object]:
    final_labels = _apply_threshold(predictions, threshold)
    per_class_f1, macro_f1 = _compute_per_class_f1(truth, final_labels)
    confusion = _confusion_matrix(truth, final_labels)
    abstain_rate = final_labels.count("unknown") / len(final_labels)

    json_valid = sum(1 for pred in predictions if isinstance(pred, dict))

    return {
        "threshold": threshold,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "json_validity": json_valid / len(predictions),
        "abstain_rate": abstain_rate,
        "support": len(truth),
        "confusion": confusion,
    }


def _apply_threshold(predictions: Sequence[dict], threshold: float) -> List[str]:
    labels = []
    for pred in predictions:
        label = pred["label"]
        conf = float(pred["confidence"])
        if conf < threshold:
            label = "unknown"
        labels.append(label)
    return labels


def _compute_macro_f1(truth: Sequence[str], predictions: Sequence[dict], threshold: float) -> float:
    labels = _apply_threshold(predictions, threshold)
    _, macro = _compute_per_class_f1(truth, labels)
    return macro


def _compute_per_class_f1(truth: Sequence[str], predicted: Sequence[str]) -> Tuple[Dict[str, float], float]:
    classes = allowed_labels(include_unknown=False)
    per_class: Dict[str, float] = {}
    macro_sum = 0.0

    for class_name in classes:
        tp = sum(1 for t, p in zip(truth, predicted) if t == class_name and p == class_name)
        fp = sum(1 for t, p in zip(truth, predicted) if t != class_name and p == class_name)
        fn = sum(1 for t, p in zip(truth, predicted) if t == class_name and p != class_name)
        denom = (2 * tp + fp + fn)
        score = (2 * tp / denom) if denom else 0.0
        per_class[class_name] = score
        macro_sum += score

    macro_f1 = macro_sum / len(classes)
    return per_class, macro_f1


def _confusion_matrix(truth: Sequence[str], predicted: Sequence[str]) -> Dict[str, Dict[str, int]]:
    labels_with_unknown = list(allowed_labels(include_unknown=False)) + ["unknown"]
    matrix: Dict[str, Dict[str, int]] = {
        actual: {pred: 0 for pred in labels_with_unknown} for actual in labels_with_unknown
    }

    for actual, pred in zip(truth, predicted):
        matrix.setdefault(actual, {label: 0 for label in labels_with_unknown})
        if pred not in matrix[actual]:
            matrix[actual][pred] = 0
        matrix[actual][pred] += 1
    return matrix


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
