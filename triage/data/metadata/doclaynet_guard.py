"""DocLayNet-specific leakage guard used in CI to ensure document-level splits are disjoint."""

from __future__ import annotations

import argparse

from triage.data import doclaynet_adapter
from triage.data.metadata.split_guard import validate_doclaynet_disjoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DocLayNet document-level split integrity.")
    parser.add_argument(
        "--dataset-id",
        default=doclaynet_adapter.DOCLAYNET_DATASET_ID,
        help="Override the DocLayNet dataset identifier.",
    )
    args = parser.parse_args()
    validate_doclaynet_disjoint(dataset_id=args.dataset_id)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
