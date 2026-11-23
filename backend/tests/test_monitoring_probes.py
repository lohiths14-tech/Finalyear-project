"""
Tests for AgriPulse 2.0 Kubernetes Health Probes and Prometheus Metrics.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.mark.asyncio
async def test_kubernetes_probes():
    """Verify Kubernetes liveness and readiness probes."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Liveness probe
        resp_liveness = await ac.get("/healthz")
        assert resp_liveness.status_code == 200
        assert resp_liveness.json()["status"] == "ok"

        # Readiness probe
        resp_ready = await ac.get("/readyz")
        assert resp_ready.status_code in [200, 503]
        ready_data = resp_ready.json()
        assert "database" in ready_data
        assert "models" in ready_data


@pytest.mark.asyncio
async def test_prometheus_metrics_endpoint():
    """Verify Prometheus text metrics format."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        body = resp.text
        assert "agripulse_system_info" in body
        assert "agripulse_models_loaded" in body
        assert "agripulse_service_ready" in body
