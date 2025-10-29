from __future__ import annotations

import json
from pathlib import Path

from triage.adapters.train_lora_adapter import AdapterConfig, REGISTRY_PATH, train_adapter


def test_train_adapter_dry_run(tmp_path: Path, monkeypatch) -> None:
    registry = tmp_path / "registry.json"
    monkeypatch.setattr("triage.adapters.train_lora_adapter.REGISTRY_PATH", registry)

    output_dir = tmp_path / "artifacts"
    config = AdapterConfig(
        adapter_id="tenant_demo",
        output_dir=output_dir,
        dry_run=True,
        random_seed=7,
    )

    metadata = train_adapter(config)
    adapter_file = output_dir / "tenant_demo.pt"
    metadata_file = output_dir / "tenant_demo.json"

    assert adapter_file.exists()
    assert metadata_file.exists()
    assert metadata["adapter_id"] == "tenant_demo"

    registry_data = json.loads(registry.read_text())
    assert "tenant_demo" in registry_data
