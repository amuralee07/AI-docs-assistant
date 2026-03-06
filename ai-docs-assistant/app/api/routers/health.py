"""Health check endpoint."""

from fastapi import APIRouter
from app import __version__
from app.schemas.rag import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy", version=__version__)
