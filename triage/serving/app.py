# Location: triage/serving/app.py
# Purpose: FastAPI application exposing /triage, /healthz, /metrics endpoints with dry-run defaults.

from __future__ import annotations

import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import Response

from triage.adapters.registry import AdapterRecord, get_adapter, get_default, list_adapters, set_default
from triage.serving.metrics import observe_request, generate_latest


class TriageRequest(BaseModel):
    image_path: Optional[str] = None
    backend: str = "stub"


class TriageResponse(BaseModel):
    label: str
    confidence: float
    meta: dict[str, str]


class AdapterLoadRequest(BaseModel):
    adapter_id: str


class AdapterDefaultRequest(BaseModel):
    adapter_id: str


app = FastAPI(title="Triage Service", version="0.1.0")

_ACTIVE_ADAPTER: Optional[str] = get_default()


def _stub_predict() -> dict[str, object]:
    return {
        "label": "memo",
        "confidence": 0.87,
        "meta": {"model_id": "stub", "calibration_id": "stub", "adapter_id": _ACTIVE_ADAPTER or "global"},
    }


@app.post("/triage", response_model=TriageResponse)
def triage_endpoint(request: TriageRequest) -> TriageResponse:
    start = time.perf_counter()
    try:
        mode = os.getenv("TRIAGE_SERVING_MODE", request.backend)
        if mode != "stub":
            raise HTTPException(status_code=501, detail="Only stub backend is supported in dry-run mode.")
        payload = _stub_predict()
        abstained = payload["confidence"] < 0.5
        observe_request(time.perf_counter() - start, "success", label=payload["label"], abstained=abstained)
        return TriageResponse(**payload)
    except HTTPException as exc:
        observe_request(time.perf_counter() - start, "error")
        raise exc
    except Exception:
        observe_request(time.perf_counter() - start, "error")
        raise
    finally:
        pass


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    content = generate_latest()
    return Response(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/adapters")
def list_registered_adapters() -> dict[str, object]:
    adapters = [
        {
            "adapter_id": record.adapter_id,
            "path": record.path,
            "metadata": record.metadata,
        }
        for record in list_adapters()
    ]
    return {
        "adapters": adapters,
        "default": get_default(),
        "active": _ACTIVE_ADAPTER,
    }


@app.post("/adapters/load")
def load_adapter(request: AdapterLoadRequest) -> dict[str, object]:
    global _ACTIVE_ADAPTER
    record: AdapterRecord = get_adapter(request.adapter_id)
    _ACTIVE_ADAPTER = record.adapter_id
    return {
        "status": "loaded",
        "adapter_id": record.adapter_id,
        "path": record.path,
    }


@app.post("/adapters/default")
def set_default_adapter(request: AdapterDefaultRequest) -> dict[str, object]:
    global _ACTIVE_ADAPTER
    set_default(request.adapter_id)
    if _ACTIVE_ADAPTER is None:
        _ACTIVE_ADAPTER = request.adapter_id
    return {"status": "default_set", "adapter_id": request.adapter_id}
