"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routers import health, ingest, query

app = FastAPI(
    title="AI Documentation Q&A",
    description="RAG API: ingest documents, ask questions, get answers from your docs.",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/")
def root() -> dict:
    return {
        "service": "AI Documentation Q&A",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }
