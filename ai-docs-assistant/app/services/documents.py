"""Extract text from uploaded files (plain text, PDF)."""

from pathlib import Path

from pypdf import PdfReader


def extract_text_from_file(content: bytes, filename: str) -> str:
    """Extract raw text from file bytes based on extension."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _pdf_to_text(content)
    if suffix in (".txt", ".md", ".markdown", ""):
        return content.decode("utf-8", errors="replace")
    return content.decode("utf-8", errors="replace")


def _pdf_to_text(content: bytes) -> str:
    """Extract text from PDF bytes."""
    import io

    reader = PdfReader(io.BytesIO(content))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n\n".join(parts)
