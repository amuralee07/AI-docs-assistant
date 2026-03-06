# AI Documentation Q&A (RAG API)

A **Retrieval-Augmented Generation (RAG)** API built with Python and FastAPI. Upload documents (PDF or text), then ask questions in natural language and get answers grounded in your content. Uses local embeddings (Sentence Transformers) and an optional LLM (OpenAI) for generated answers.

**Tech stack:** Python 3.10+, FastAPI, Sentence Transformers, NumPy (in-memory vector store), optional OpenAI.

---

## Features

- **Document ingestion** — Upload PDF or text files; automatic chunking and embedding
- **Semantic search** — Query with natural language; returns relevant chunks with similarity scores
- **Optional LLM answers** — Set `OPENAI_API_KEY` to get a short generated answer from retrieved context
- **Simple vector store** — In-memory NumPy-based search (cosine similarity); easy to swap for FAISS or pgvector later
- **REST API** — OpenAPI docs at `/docs`; health check at `/health`
- **Docker** — Single Dockerfile for deployment

---

## Quick start

### Prerequisites

- Python 3.10+
- (Optional) OpenAI API key for answer generation

### Install and run

```bash
cd ai-docs-assistant
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) to use the API.

### Example usage

1. **Ingest a document**

   ```bash
   curl -X POST http://localhost:8000/ingest -F "file=@README.md"
   ```

2. **Ask a question**

   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "How do I run this project?"}'
   ```

   Response includes `chunks` (relevant text + scores) and, if configured, `answer`.

3. **Health check**

   ```bash
   curl http://localhost:8000/health
   ```

---

## Configuration

Copy `.env.example` to `.env` and adjust:

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_MODEL` | Sentence Transformers model name | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Characters per chunk | 512 |
| `CHUNK_OVERLAP` | Overlap between chunks | 64 |
| `TOP_K` | Max chunks returned per query | 5 |
| `OPENAI_API_KEY` | Optional; enables LLM-generated answers | — |
| `OPENAI_MODEL` | Model for answers | `gpt-4o-mini` |

---

