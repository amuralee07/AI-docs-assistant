"""Split documents into overlapping chunks for embedding."""

import re
from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[str]:
    """
    Split text into overlapping chunks by character count.
    Tries to break on sentence boundaries when possible.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Prefer breaking at sentence end
        segment = text[start:end]
        last_period = segment.rfind(". ")
        last_newline = segment.rfind("\n")
        break_at = max(last_period + 1 if last_period >= 0 else 0, last_newline + 1 if last_newline >= 0 else 0)
        if break_at > chunk_size // 2:
            end = start + break_at
            chunk = text[start:end].strip()
        else:
            chunk = segment.strip()

        if chunk:
            chunks.append(chunk)
        start = end - overlap if overlap < end - start else end

    return chunks


def clean_text(raw: str) -> str:
    """Normalize whitespace and remove excessive newlines."""
    if not raw:
        return ""
    text = re.sub(r"\r\n", "\n", raw)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return " ".join(text.split()).replace(" \n ", "\n").strip()
