"""Request and response schemas for RAG API."""

from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    question: str = Field(..., min_length=1, description="Natural language question")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Max number of chunks to return (default from config)")


class ChunkResult(BaseModel):
    """A single retrieved chunk with score and source."""

    text: str
    score: float
    source: str = ""
    chunk_index: int = 0


class QueryResponse(BaseModel):
    """Response for POST /query - retrieved chunks and optional answer."""

    question: str
    chunks: list[ChunkResult]
    answer: Optional[str] = Field(None, description="LLM-generated answer if configured")


class IngestResponse(BaseModel):
    """Response after ingesting a document."""

    filename: str
    chunks_created: int
    message: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "1.0.0"
