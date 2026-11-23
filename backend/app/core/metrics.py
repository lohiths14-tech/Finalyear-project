"""
Prometheus metrics + OpenTelemetry tracing for AgriPulse.
Exposes /metrics for Prometheus scraping.
"""

import time

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware

# ── Metrics ────────────────────────────────────

REQUEST_COUNT = Counter(
    "agrivision_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "agrivision_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

INFERENCE_COUNT = Counter(
    "agrivision_inference_total",
    "Total inference calls",
    ["scan_type", "disease"],
)

INFERENCE_LATENCY = Histogram(
    "agrivision_inference_duration_seconds",
    "Inference latency",
    ["scan_type"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

MODEL_CONFIDENCE = Gauge(
    "agrivision_model_confidence",
    "Last inference confidence score",
    ["scan_type"],
)

QUEUE_SIZE = Gauge(
    "agrivision_offline_queue_size",
    "Number of offline scans pending sync",
)

ACTIVE_MODELS_LOADED = Gauge(
    "agrivision_models_loaded",
    "Number of ML models loaded",
    ["model_name"],
)

DB_POOL_SIZE = Gauge(
    "agrivision_db_pool_size",
    "Database connection pool size",
)


# ── Middleware ──────────────────────────────────────


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in ("/metrics", "/api/v1/metrics"):
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        endpoint = request.url.path
        status = str(response.status_code)

        REQUEST_COUNT.labels(
            method=request.method, endpoint=endpoint, status=status
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method, endpoint=endpoint
        ).observe(duration)

        return response


# ── Endpoint ───────────────────────────────────


def register_metrics(app: FastAPI):
    app.add_middleware(MetricsMiddleware)

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.get("/api/v1/metrics", include_in_schema=False)
    async def api_metrics():
        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST,
        )


def record_inference(
    scan_type: str, disease: str, duration: float, confidence: float
):
    INFERENCE_COUNT.labels(scan_type=scan_type, disease=disease).inc()
    INFERENCE_LATENCY.labels(scan_type=scan_type).observe(duration)
    MODEL_CONFIDENCE.labels(scan_type=scan_type).set(confidence)
