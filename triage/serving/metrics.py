# Location: triage/serving/metrics.py
# Purpose: Centralised Prometheus metric definitions for the serving app.

from __future__ import annotations

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest
except ImportError:  # pragma: no cover - fallback when prometheus_client unavailable
    class _DummyMetric:
        def __init__(self, *_, **__):
            pass

        def observe(self, *_: object, **__: object) -> None:
            pass

        def inc(self, *_: object, **__: object) -> None:
            pass

        def set(self, *_: object, **__: object) -> None:
            pass

        def labels(self, *_: object, **__: object) -> "_DummyMetric":
            return self

    def generate_latest() -> bytes:  # type: ignore
        return b""

    Counter = Gauge = Histogram = _DummyMetric  # type: ignore


REQUEST_LATENCY = Histogram(
    "triage_request_latency_seconds",
    "Latency of /triage requests",
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5),
)

REQUEST_COUNTER = Counter(
    "triage_requests_total",
    "Total triage requests",
    labelnames=["status"],
)

CLASS_COUNTER = Counter(
    "triage_predictions_total",
    "Predictions per class",
    labelnames=["label"],
)

ABSTAIN_COUNTER = Counter(
    "triage_abstentions_total",
    "Number of abstentions",
)

ECE_GAUGE = Gauge(
    "triage_calibration_ece",
    "Latest expected calibration error",
)


def observe_request(duration: float, status: str, label: str | None = None, abstained: bool = False) -> None:
    REQUEST_LATENCY.observe(duration)
    REQUEST_COUNTER.labels(status=status).inc()
    if label:
        CLASS_COUNTER.labels(label=label).inc()
    if abstained:
        ABSTAIN_COUNTER.inc()


def update_calibration_ece(ece: float | None) -> None:
    if ece is not None:
        try:
            ECE_GAUGE.set(ece)
        except AttributeError:  # dummy metric
            pass
