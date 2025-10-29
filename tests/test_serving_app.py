from __future__ import annotations

from fastapi.testclient import TestClient

from triage.serving.app import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_triage_stub_endpoint() -> None:
    response = client.post("/triage", json={"backend": "stub"})
    assert response.status_code == 200
    body = response.json()
    assert "label" in body
    assert "confidence" in body


def test_metrics_endpoint() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    if response.content:
        assert b"triage_requests_total" in response.content or b"triage_request_latency_seconds" in response.content
