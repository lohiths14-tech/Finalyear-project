"""
Tests for AgriPulse 2.0 Asynchronous Job Queue & Polling Engine.
"""

import io

import cv2
from httpx import ASGITransport, AsyncClient
import numpy as np
import pytest

from app.main import app


@pytest.fixture
def sample_image_bytes():
    # 100x100 RGB image
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.mark.asyncio
async def test_async_job_lifecycle(sample_image_bytes):
    """Verify asynchronous job submission, queue registration, and status polling."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Submit async leaf scan
        files = {"file": ("leaf.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        data = {"scan_type": "leaf"}
        resp = await ac.post("/api/v1/jobs/analyze", files=files, data=data)
        assert resp.status_code == 200
        job_info = resp.json()
        job_id = job_info["job_id"]
        assert job_id.startswith("job-")
        assert job_info["status"] in ["queued", "processing", "completed"]

        # 2. Poll job status
        poll_resp = await ac.get(f"/api/v1/jobs/{job_id}")
        assert poll_resp.status_code == 200
        polled_data = poll_resp.json()
        assert polled_data["job_id"] == job_id
        assert polled_data["scan_type"] == "leaf"

        # 3. List recent jobs
        list_resp = await ac.get("/api/v1/jobs")
        assert list_resp.status_code == 200
        jobs_list = list_resp.json()
        assert any(j["job_id"] == job_id for j in jobs_list)


@pytest.mark.asyncio
async def test_async_job_not_found():
    """Verify 404 for non-existent job ID."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/jobs/job-nonexistent123")
        assert resp.status_code == 404
