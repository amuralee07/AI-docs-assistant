"""In-memory vector store using NumPy for cosine similarity search."""

from typing import List, Tuple

import numpy as np


class VectorStore:
    """
    Store text chunks and their embeddings; support similarity search.
    Uses normalized vectors so dot product = cosine similarity.
    """

    def __init__(self):
        self._texts: List[str] = []
        self._sources: List[str] = []
        self._indices: List[int] = []
        self._matrix: np.ndarray | None = None

    def add(self, texts: List[str], embeddings: List[List[float]], source: str = "") -> None:
        """Append chunks and their embedding matrix (each row one vector)."""
        if not texts or not embeddings:
            return
        n = len(texts)
        if n != len(embeddings):
            raise ValueError("texts and embeddings length must match")
        start = len(self._texts)
        self._texts.extend(texts)
        self._sources.extend([source] * n)
        self._indices.extend(range(n))
        arr = np.array(embeddings, dtype=np.float32)
        if self._matrix is None:
            self._matrix = arr
        else:
            self._matrix = np.vstack([self._matrix, arr])

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float, str, int]]:
        """
        Return top_k (text, score, source, chunk_index) by cosine similarity.
        query_embedding should be normalized if stored vectors are normalized.
        """
        if self._matrix is None or len(self._matrix) == 0:
            return []
        q = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        scores = np.dot(self._matrix, q.T).flatten()
        idx = np.argsort(scores)[::-1][:top_k]
        return [
            (self._texts[i], float(scores[i]), self._sources[i], self._indices[i])
            for i in idx
        ]

    def clear(self) -> None:
        """Remove all stored vectors and metadata."""
        self._texts = []
        self._sources = []
        self._indices = []
        self._matrix = None

    @property
    def size(self) -> int:
        return len(self._texts)
