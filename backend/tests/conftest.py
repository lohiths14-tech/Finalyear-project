"""
Pytest fixtures — async engine, test client, sample image.
"""
from collections.abc import AsyncGenerator
import os
import sys

from httpx import ASGITransport, AsyncClient
import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["DB_DIR"] = ":memory:"

from app.db.database import Base, get_db
from app.main import app

TEST_DB = "sqlite+aiosqlite:///:memory:"
_test_engine = create_async_engine(TEST_DB, echo=False)
_test_session = async_sessionmaker(_test_engine, expire_on_commit=False, class_=AsyncSession)


@pytest_asyncio.fixture
async def setup_db():
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client(setup_db) -> AsyncGenerator[AsyncClient, None]:
    async def _override_get_db():
        async with _test_session() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db

    from app.services.inference_service import InferenceService
    app.state.inference_service = InferenceService()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
def green_image_bytes() -> bytes:
    """128x128 synthetic green field image."""
    img = np.random.randint(0, 50, (128, 128, 3), dtype=np.uint8)
    img[:64, :, 1] = np.random.randint(150, 200, (64, 128), dtype=np.uint8)
    img[64:, :, :] = 0
    import cv2
    success, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()
