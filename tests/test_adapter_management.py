from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from triage.adapters.registry import register_adapter, set_default
from triage.serving.app import app


client = TestClient(app)


def test_adapter_endpoints(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "adapter_registry.json"
    monkeypatch.setattr("triage.adapters.registry.REGISTRY_PATH", registry_path, raising=False)
    monkeypatch.setattr("triage.serving.app._ACTIVE_ADAPTER", None, raising=False)

    # Seed registry with two adapters
    register_adapter("demo", tmp_path / "demo.pt", path=registry_path)
    register_adapter("demo2", tmp_path / "demo2.pt", path=registry_path)
    set_default("demo", path=registry_path)

    response = client.get("/adapters")
    assert response.status_code == 200
    data = response.json()
    assert "adapters" in data

    load_resp = client.post("/adapters/load", json={"adapter_id": "demo2"})
    assert load_resp.status_code == 200

    default_resp = client.post("/adapters/default", json={"adapter_id": "demo2"})
    assert default_resp.status_code == 200
