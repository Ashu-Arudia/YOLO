"""
API Integration tests for FastAPI endpoints:
- GET /
- GET /api/v1/health
- POST /api/v1/matches
- GET /api/v1/matches
- GET /api/v1/matches/{id}
"""
import pytest
from httpx import AsyncClient, ASGITransport
from backend.app.main import app
from backend.app.database.session import init_db


@pytest.mark.anyio
async def test_root():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/")
        assert res.status_code == 200
        data = res.json()
        assert data["name"] == "SportsVision AI"
        assert data["version"] == "0.1.0"


@pytest.mark.anyio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/api/v1/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "torch_version" in data
        assert "cuda_available" in data


@pytest.mark.anyio
async def test_matches_crud():
    await init_db()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Create match
        payload = {"name": "Test Match 1", "source_type": "file"}
        res = await client.post("/api/v1/matches", json=payload)
        assert res.status_code == 201
        match_data = res.json()
        assert match_data["name"] == "Test Match 1"
        assert match_data["status"] == "pending"
        match_id = match_data["id"]

        # List matches
        list_res = await client.get("/api/v1/matches")
        assert list_res.status_code == 200
        matches = list_res.json()
        assert any(m["id"] == match_id for m in matches)

        # Get match by id
        get_res = await client.get(f"/api/v1/matches/{match_id}")
        assert get_res.status_code == 200
        assert get_res.json()["id"] == match_id
