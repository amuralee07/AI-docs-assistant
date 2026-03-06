"""Tests for text chunking."""

import pytest
from app.services.chunker import chunk_text, clean_text


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_short():
    text = "One short sentence."
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == "One short sentence."


def test_chunk_text_splits_long():
    text = "First. Second. Third. Fourth. Fifth."
    chunks = chunk_text(text, chunk_size=15, overlap=2)
    assert len(chunks) >= 2
    assert all(len(c) for c in chunks)


def test_clean_text():
    assert clean_text("  a  b  \n\n  c  ") == "a b c"
    assert clean_text("") == ""
