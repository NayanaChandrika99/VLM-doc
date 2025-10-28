"""
Guards to ensure dataset split integrity for RVL-CDIP-small and DocLayNet base.

Run this script in CI to detect template leakage across document-level splits
or mismatched sample counts compared to the authoritative specifications.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, Iterable, Set

from triage.data import doclaynet_adapter, rvl_adapter


def validate_rvl_counts(dataset_id: str = rvl_adapter.RVL_DATASET_ID) -> None:
    """Assert that RVL-CDIP-small splits match the published sample counts."""

    for split, expected in rvl_adapter.RVL_SPLIT_SIZES.items():
        dataset = rvl_adapter.load_rvl(split, dataset_id=dataset_id)
        actual = len(dataset)
        if actual != expected:
            raise AssertionError(
                f"RVL split '{split}' expected {expected} pages but found {actual} (dataset_id={dataset_id})."
            )


def validate_doclaynet_disjoint(dataset_id: str = doclaynet_adapter.DOCLAYNET_DATASET_ID) -> None:
    """Ensure document IDs do not appear in more than one DocLayNet split."""

    doc_ids: Dict[str, Set[str]] = defaultdict(set)
    for split in ("train", "validation", "test"):
        dataset = doclaynet_adapter.load_doclaynet(split, dataset_id=dataset_id)
        ids = {sample.document_id for sample in dataset.as_iterable()}
        if not ids:
            raise AssertionError(f"DocLayNet split '{split}' returned no document IDs.")
        doc_ids[split].update(ids)

    _assert_disjoint(doc_ids["train"], doc_ids["validation"], "train", "validation")
    _assert_disjoint(doc_ids["train"], doc_ids["test"], "train", "test")
    _assert_disjoint(doc_ids["validation"], doc_ids["test"], "validation", "test")


def _assert_disjoint(a: Set[str], b: Set[str], name_a: str, name_b: str) -> None:
    overlap = a.intersection(b)
    if overlap:
        raise AssertionError(
            f"DocLayNet document leakage detected between {name_a} and {name_b}: {sorted(list(overlap))[:5]}..."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RVL and DocLayNet split integrity.")
    parser.add_argument("--skip-rvl", action="store_true", help="Skip RVL split size checks.")
    parser.add_argument("--skip-doclaynet", action="store_true", help="Skip DocLayNet leakage checks.")
    parser.add_argument("--rvl-dataset-id", default=rvl_adapter.RVL_DATASET_ID, help="Override RVL dataset ID.")
    parser.add_argument(
        "--doclaynet-dataset-id",
        default=doclaynet_adapter.DOCLAYNET_DATASET_ID,
        help="Override DocLayNet dataset ID.",
    )
    args = parser.parse_args()

    if not args.skip_rvl:
        validate_rvl_counts(args.rvl_dataset_id)
    if not args.skip_doclaynet:
        validate_doclaynet_disjoint(args.doclaynet_dataset_id)


if __name__ == "__main__":
    main()
