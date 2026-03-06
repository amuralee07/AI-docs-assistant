"""Embedding service using Sentence Transformers."""

from typing import List

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Load and run a sentence-transformers model for encoding text."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """Encode texts to vectors. Optionally L2-normalize for cosine similarity via dot product."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return embeddings.tolist()
