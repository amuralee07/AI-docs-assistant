"""Tests for in-memory vector store."""

import pytest
from app.services.vector_store import VectorStore


def test_add_and_search():
    store = VectorStore()
    # Normalized 2D vectors (cosine sim = dot product)
    texts = ["hello world", "foo bar"]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    store.add(texts, embeddings, source="test.txt")
    assert store.size == 2
    results = store.search([1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0][0] == "hello world"
    assert results[0][1] == pytest.approx(1.0)


def test_search_empty():
    store = VectorStore()
    assert store.search([1.0, 0.0], top_k=5) == []


def test_clear():
    store = VectorStore()
    store.add(["a"], [[1.0, 0.0]])
    store.clear()
    assert store.size == 0
    assert store.search([1.0, 0.0]) == []
