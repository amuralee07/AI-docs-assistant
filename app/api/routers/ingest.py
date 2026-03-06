"""Document ingestion: upload file, chunk, embed, store."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import get_embedding_service, get_vector_store
from app.core.config import get_settings, Settings
from app.schemas.rag import IngestResponse
from app.services.chunker import chunk_text, clean_text
from app.services.documents import extract_text_from_file
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
async def ingest_document(
    file: Annotated[UploadFile, File()],
    store: Annotated[VectorStore, Depends(get_vector_store)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> IngestResponse:
    """Upload a document (PDF or text). It will be chunked, embedded, and indexed for search."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        text = extract_text_from_file(content, file.filename or "document.txt")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")
    text = clean_text(text)
    chunks = chunk_text(
        text,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise HTTPException(status_code=400, detail="No text could be extracted from the document")
    embeddings = embedding_service.encode(chunks)
    store.add(chunks, embeddings, source=file.filename or "document")
    return IngestResponse(
        filename=file.filename or "document",
        chunks_created=len(chunks),
        message=f"Indexed {len(chunks)} chunks.",
    )
