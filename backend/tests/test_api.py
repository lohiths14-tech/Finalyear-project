"""
API integration tests — auth, health, history, leaf/field analysis.
"""
import io

from httpx import AsyncClient
import pytest

# ── Auth ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_register_and_login(client: AsyncClient):
    resp = await client.post("/api/v1/auth/register", json={
        "username": "testuser",
        "password": "TestPass123!",
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["username"] == "testuser"

    # Duplicate registration
    resp2 = await client.post("/api/v1/auth/register", json={
        "username": "testuser",
        "password": "OtherPass",
    })
    assert resp2.status_code == 409

    # Login
    resp3 = await client.post("/api/v1/auth/token", json={
        "username": "testuser",
        "password": "TestPass123!",
    })
    assert resp3.status_code == 200
    token = resp3.json()["access_token"]
    assert token
    assert resp3.json()["token_type"] == "bearer"

    # Wrong password
    resp4 = await client.post("/api/v1/auth/token", json={
        "username": "testuser",
        "password": "WrongPass",
    })
    assert resp4.status_code == 401


@pytest.mark.asyncio
async def test_get_me_authenticated(client: AsyncClient):
    await client.post("/api/v1/auth/register", json={"username": "me", "password": "Pass123!"})
    login = await client.post("/api/v1/auth/token", json={"username": "me", "password": "Pass123!"})
    token = login.json()["access_token"]

    resp = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["username"] == "me"


# ── Health ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_root(client: AsyncClient):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "Ready" in resp.json()["status"]


# ── History ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_history_empty(client: AsyncClient):
    resp = await client.get("/api/v1/history")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_history_with_scans(client: AsyncClient, green_image_bytes: bytes):
    # Create a leaf scan
    files = {"file": ("leaf.png", io.BytesIO(green_image_bytes), "image/png")}
    resp = await client.post("/api/v1/analyze/leaf", files=files)
    assert resp.status_code == 200
    scan_id = resp.json()["scan_id"]

    # List
    resp2 = await client.get("/api/v1/history")
    assert resp2.status_code == 200
    assert len(resp2.json()) >= 1

    # Filter
    resp3 = await client.get("/api/v1/history?scan_type=leaf")
    assert resp3.status_code == 200
    assert all(r["scan_type"] == "leaf" for r in resp3.json())

    # Get detail
    resp4 = await client.get(f"/api/v1/history/{scan_id}")
    assert resp4.status_code == 200
    assert resp4.json()["disease_type"]

    # Delete
    resp5 = await client.delete(f"/api/v1/history/{scan_id}")
    assert resp5.status_code == 204

    # Not found
    resp6 = await client.get(f"/api/v1/history/{scan_id}")
    assert resp6.status_code == 404


# ── Leaf Analysis ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_leaf_analysis(client: AsyncClient, green_image_bytes: bytes):
    files = {"file": ("leaf.png", io.BytesIO(green_image_bytes), "image/png")}
    resp = await client.post("/api/v1/analyze/leaf", files=files)
    assert resp.status_code == 200
    data = resp.json()

    assert "disease_type" in data
    assert "severity" in data
    assert "confidence" in data
    assert "treatment" in data
    assert "scan_id" in data
    assert "disclaimer" in data
    assert "farmer_friendly" in data


@pytest.mark.asyncio
async def test_leaf_non_image(client: AsyncClient):
    files = {"file": ("leaf.txt", io.BytesIO(b"not an image"), "text/plain")}
    resp = await client.post("/api/v1/analyze/leaf", files=files)
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_leaf_no_file(client: AsyncClient):
    resp = await client.post("/api/v1/analyze/leaf")
    assert resp.status_code == 422


# ── Field Analysis ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_field_analysis(client: AsyncClient, green_image_bytes: bytes):
    files = {"file": ("field.png", io.BytesIO(green_image_bytes), "image/png")}
    resp = await client.post("/api/v1/analyze/field", files=files)
    assert resp.status_code == 200
    data = resp.json()

    assert "disease_type" in data
    assert "average_severity" in data
    assert "infection_rate" in data
    assert "treatment" in data
    assert "grid_data" in data
    assert "scan_id" in data


@pytest.mark.asyncio
async def test_field_non_image(client: AsyncClient):
    files = {"file": ("field.txt", io.BytesIO(b"not an image"), "text/plain")}
    resp = await client.post("/api/v1/analyze/field", files=files)
    assert resp.status_code == 400


# ── Image Dimension Validation ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_oversized_image_rejected(client: AsyncClient):
    import numpy as np
    img = np.zeros((10, 5000, 3), dtype=np.uint8)
    import cv2
    _, buf = cv2.imencode(".jpg", img)
    files = {"file": ("huge.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")}
    resp = await client.post("/api/v1/analyze/leaf", files=files)
    assert resp.status_code == 422
    assert "dimensions" in resp.json()["detail"]


# ── Concurrent Requests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_concurrent_requests(client: AsyncClient, green_image_bytes: bytes):
    import asyncio
    tasks = [
        client.post("/api/v1/analyze/leaf",
                    files={"file": (f"test_{i}.jpg", io.BytesIO(green_image_bytes), "image/jpeg")})
        for i in range(4)
    ]
    responses = await asyncio.gather(*tasks)
    for r in responses:
        assert r.status_code == 200
        assert "disease_type" in r.json()
