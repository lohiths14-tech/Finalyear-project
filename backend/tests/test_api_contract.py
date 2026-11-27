import io

from httpx import AsyncClient
import pytest


@pytest.mark.asyncio
async def test_readiness_probe(client: AsyncClient):
    response = await client.get("/readyz")
    assert response.status_code in (200, 503)
    data = response.json()
    assert "status" in data
    assert "database" in data
    assert "models" in data


@pytest.mark.asyncio
async def test_liveness_probe(client: AsyncClient):
    response = await client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_metrics_endpoint(client: AsyncClient):
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "agripulse_system_info" in response.text


@pytest.mark.asyncio
async def test_api_leaf_contract_invalid_image(client: AsyncClient):
    # Send a text file instead of an image
    file_obj = io.BytesIO(b"This is not an image.")
    response = await client.post(
        "/api/v1/analyze/leaf",
        files={"file": ("test.txt", file_obj, "text/plain")},
    )
    # The image validator rejects this with 400
    assert response.status_code == 400
    assert "detail" in response.json()


@pytest.mark.asyncio
async def test_api_field_contract_invalid_image(client: AsyncClient):
    file_obj = io.BytesIO(b"This is not an image.")
    response = await client.post(
        "/api/v1/analyze/field",
        files={"file": ("test.txt", file_obj, "text/plain")},
    )
    assert response.status_code == 400
    assert "detail" in response.json()


@pytest.mark.asyncio
async def test_file_size_limit(client: AsyncClient):
    # Simulate a file larger than 10MB
    large_payload = b"0" * (10 * 1024 * 1024 + 100)
    response = await client.post(
        "/api/v1/analyze/leaf",
        content=large_payload,
        headers={"Content-Length": str(len(large_payload))},
    )
    assert response.status_code in (413, 422)


@pytest.mark.asyncio
async def test_api_field_stream_invalid_image(client: AsyncClient):
    file_obj = io.BytesIO(b"This is not an image.")
    response = await client.post(
        "/api/v1/analyze/field/stream",
        files={"file": ("test.txt", file_obj, "text/plain")},
    )
    assert response.status_code == 400
    assert "detail" in response.json()
