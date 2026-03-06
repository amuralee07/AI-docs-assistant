"""API endpoint tests."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.deps import get_embedding_service


class MockEmbeddingService:
    """Returns fixed-dimension dummy vectors so tests don't download a real model."""

    def encode(self, texts, normalize=True):
        if not texts:
            return []
        # MiniLM dimension
        return [[0.1] * 384 for _ in texts]


@pytest.fixture
def client_no_model():
    app.dependency_overrides[get_embedding_service] = lambda: MockEmbeddingService()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health():
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_root():
    r = TestClient(app).get("/")
    assert r.status_code == 200
    assert "docs" in r.json()


def test_query_empty_store(client_no_model):
    """Without ingesting, query returns empty chunks (mock embedding, no HF download)."""
    r = client_no_model.post("/query", json={"question": "What is the main topic?"})
    assert r.status_code == 200
    data = r.json()
    assert data["question"] == "What is the main topic?"
    assert data["chunks"] == []
