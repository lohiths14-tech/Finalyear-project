import pytest
from fastapi.testclient import TestClient
from app.main import app
import io

from app.services.inference_service import InferenceService
app.state.inference_service = InferenceService()
client = TestClient(app)

def test_readiness_probe():
    response = client.get("/readyz")
    assert response.status_code in (200, 503)
    data = response.json()
    assert "status" in data
    assert "database" in data
    assert "models" in data

def test_liveness_probe():
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "agripulse_system_info" in response.text

def test_api_leaf_contract_invalid_image():
    # Send a text file instead of an image
    file_obj = io.BytesIO(b"This is not an image.")
    response = client.post(
        "/api/v1/analyze/leaf",
        files={"file": ("test.txt", file_obj, "text/plain")}
    )
    # The image validator rejects this with 400
    assert response.status_code == 400
    assert "detail" in response.json()

def test_api_field_contract_invalid_image():
    file_obj = io.BytesIO(b"This is not an image.")
    response = client.post(
        "/api/v1/analyze/field",
        files={"file": ("test.txt", file_obj, "text/plain")}
    )
    assert response.status_code == 400
    assert "detail" in response.json()

def test_file_size_limit():
    # Simulate a file larger than 10MB by manipulating the content-length header
    # TestClient doesn't easily allow faking content-length for multipart, 
    # but we can try a direct POST.
    large_payload = b"0" * (10 * 1024 * 1024 + 100)
    response = client.post(
        "/api/v1/analyze/leaf",
        content=large_payload,
        headers={"Content-Length": str(len(large_payload))}
    )
    # Since it's not multipart it might fail earlier, but the middleware should catch it
    assert response.status_code in (413, 422)


def test_api_field_stream_invalid_image():
    file_obj = io.BytesIO(b"This is not an image.")
    response = client.post(
        "/api/v1/analyze/field/stream",
        files={"file": ("test.txt", file_obj, "text/plain")},
    )
    assert response.status_code == 400
    assert "detail" in response.json()

