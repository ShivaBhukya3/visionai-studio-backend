import io
import pytest
import numpy as np
import cv2
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture
def sample_image_bytes():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw something for detection interest
    cv2.rectangle(img, (100, 100), (300, 300), (128, 128, 128), -1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_health_ping():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/health/ping")
    assert resp.status_code == 200
    assert resp.json()["pong"] is True


@pytest.mark.asyncio
async def test_root():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "app" in data


@pytest.mark.asyncio
async def test_detect_image(sample_image_bytes):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post(
            "/api/v1/detect/image",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            params={"confidence": 0.5, "return_annotated": True},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "detection_id" in data
    assert "detections" in data
    assert "summary" in data


@pytest.mark.asyncio
async def test_detect_history():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/api/v1/detect/history")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data


@pytest.mark.asyncio
async def test_analytics_summary():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/api/v1/analytics/summary")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_list_models():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/api/v1/analytics/models")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 5
