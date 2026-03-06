"""Query: semantic search over ingested docs + optional LLM answer."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_embedding_service, get_vector_store
from app.core.config import get_settings, Settings
from app.schemas.rag import ChunkResult, QueryRequest, QueryResponse
from app.services.embedding import EmbeddingService
from app.services.llm import generate_answer
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def query(
    body: QueryRequest,
    store: Annotated[VectorStore, Depends(get_vector_store)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> QueryResponse:
    """Ask a question; returns relevant chunks and optionally an LLM-generated answer."""
    top_k = body.top_k or settings.top_k
    query_embedding = embedding_service.encode([body.question])
    if not query_embedding:
        return QueryResponse(question=body.question, chunks=[], answer=None)
    results = store.search(query_embedding[0], top_k=top_k)
    chunks = [
        ChunkResult(text=text, score=score, source=source, chunk_index=idx)
        for text, score, source, idx in results
    ]
    context_texts = [c.text for c in chunks]
    answer = generate_answer(
        body.question,
        context_texts,
        api_key=settings.openai_api_key,
        model=settings.openai_model,
    )
    return QueryResponse(question=body.question, chunks=chunks, answer=answer)
