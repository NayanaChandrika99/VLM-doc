# Location: triage/serving/app.py
# Purpose: FastAPI application exposing /triage, /healthz, /metrics endpoints with dry-run defaults.

from __future__ import annotations

import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import Response

try:
    from prometheus_client import Counter, Histogram, generate_latest
except ImportError:  # pragma: no cover - fallback when prometheus_client missing

    class _DummyMetric:
        def __init__(self, *_, **__):
            pass

        def observe(self, *_: object, **__: object) -> None:
            pass

        def inc(self, *_: object, **__: object) -> None:
            pass

        def labels(self, *_: object, **__: object) -> "_DummyMetric":
            return self

    def generate_latest() -> bytes:  # type: ignore
        return b""

    Counter = Histogram = _DummyMetric  # type: ignore


REQUEST_TIME = Histogram("triage_request_latency_seconds", "Triage request latency", buckets=(0.05, 0.1, 0.5, 1, 2, 5))
REQUEST_COUNTER = Counter("triage_requests_total", "Number of triage requests", ["status"])
ABSTAIN_COUNTER = Counter("triage_abstain_total", "Number of abstained responses")


class TriageRequest(BaseModel):
    image_path: Optional[str] = None
    backend: str = "stub"


class TriageResponse(BaseModel):
    label: str
    confidence: float
    meta: dict[str, str]


app = FastAPI(title="Triage Service", version="0.1.0")


def _stub_predict() -> dict[str, object]:
    return {
        "label": "memo",
        "confidence": 0.87,
        "meta": {"model_id": "stub", "calibration_id": "stub", "adapter_id": "global"},
    }


@app.post("/triage", response_model=TriageResponse)
def triage_endpoint(request: TriageRequest) -> TriageResponse:
    start = time.perf_counter()
    try:
        mode = os.getenv("TRIAGE_SERVING_MODE", request.backend)
        if mode != "stub":
            raise HTTPException(status_code=501, detail="Only stub backend is supported in dry-run mode.")
        payload = _stub_predict()
        if payload["confidence"] < 0.5:
            ABSTAIN_COUNTER.inc()
        return TriageResponse(**payload)
    finally:
        duration = time.perf_counter() - start
        REQUEST_TIME.observe(duration)
        REQUEST_COUNTER.labels(status="success").inc()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    content = generate_latest()
    return Response(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")
