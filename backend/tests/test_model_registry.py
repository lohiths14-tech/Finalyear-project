"""
Tests for AgriPulse 2.0 Model Registry Manifest & Provenance.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.mark.asyncio
async def test_model_registry_manifest():
    """Verify cryptographic model manifest endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/model-registry/manifest")
        assert resp.status_code == 200
        manifest = resp.json()

        assert "architecture" in manifest
        assert "DINOv2" in manifest["architecture"]["foundation_backbone"]
        assert "verified_metrics" in manifest
        assert manifest["verified_metrics"]["macro_f1_score"] == 0.982
        assert manifest["verified_metrics"]["expected_calibration_error_ece"] == 0.038
        assert "cryptographic_weights_manifest" in manifest
        assert "ablation_lineage" in manifest
        assert len(manifest["ablation_lineage"]) == 4
