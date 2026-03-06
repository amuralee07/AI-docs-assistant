"""Shared dependencies (vector store, embedding service) for the app."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.core.config import get_settings, Settings
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStore

# Single global store and embedding service (in-memory app)
_vector_store: VectorStore | None = None
_embedding_service: EmbeddingService | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_embedding_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(model_name=settings.embedding_model)
    return _embedding_service
