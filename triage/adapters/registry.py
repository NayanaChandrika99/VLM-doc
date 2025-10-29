# Location: triage/adapters/registry.py
# Purpose: Manage LoRA adapter registry with simple CRUD helpers.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

REGISTRY_PATH = Path("triage/adapters/adapter_registry.json")


@dataclass
class AdapterRecord:
    adapter_id: str
    path: str
    metadata: Dict[str, object]


def _resolve_path(path: Path | None) -> Path:
    return path or REGISTRY_PATH


def _load_registry(path: Path | None = None) -> Dict[str, Dict[str, object]]:
    resolved = _resolve_path(path)
    if not resolved.exists():
        return {}
    try:
        data = json.loads(resolved.read_text())
    except json.JSONDecodeError:
        return {}
    return data


def _save_registry(data: Dict[str, Dict[str, object]], path: Path | None = None) -> None:
    resolved = _resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(data, indent=2))


def list_adapters(path: Path | None = None) -> List[AdapterRecord]:
    registry = _load_registry(path)
    records = []
    for adapter_id, payload in registry.items():
        if adapter_id == "default":
            continue
        records.append(
            AdapterRecord(
                adapter_id=adapter_id,
                path=str(payload.get("path", "")),
                metadata=payload.get("metadata", {}),
            )
        )
    return records


def register_adapter(adapter_id: str, adapter_path: Path, metadata: Optional[Dict[str, object]] = None, path: Path | None = None) -> None:
    registry = _load_registry(path)
    registry[adapter_id] = {
        "path": str(adapter_path),
        "metadata": metadata or {},
    }
    _save_registry(registry, path)


def set_default(adapter_id: str, path: Path | None = None) -> None:
    registry = _load_registry(path)
    if adapter_id not in registry:
        raise KeyError(f"Unknown adapter_id '{adapter_id}'")
    registry["default"] = {"adapter_id": adapter_id}
    _save_registry(registry, path)


def get_default(path: Path | None = None) -> Optional[str]:
    registry = _load_registry(path)
    default_entry = registry.get("default")
    if isinstance(default_entry, dict):
        return default_entry.get("adapter_id")
    return None


def get_adapter(adapter_id: str, path: Path | None = None) -> AdapterRecord:
    registry = _load_registry(path)
    if adapter_id not in registry:
        raise KeyError(f"Unknown adapter_id '{adapter_id}'")
    payload = registry[adapter_id]
    return AdapterRecord(
        adapter_id=adapter_id,
        path=str(payload.get("path", "")),
        metadata=payload.get("metadata", {}),
    )
