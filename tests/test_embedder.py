# tests/test_embedder.py
# ─────────────────────────────────────────────────────────────
# Unit tests for rag/embedder.py.
#
# These tests mock ChromaDB and HuggingFaceEmbeddings to avoid
# loading the 90MB model during unit tests.
# ─────────────────────────────────────────────────────────────

from unittest.mock import MagicMock, patch

import pytest

from rag.embedder import embed_and_store_chunks, clear_collection


def _make_chunk(text: str, metadata: dict = None):
    """Helper to build a minimal Document-like chunk."""
    chunk = MagicMock()
    chunk.page_content = text
    chunk.metadata = metadata or {}
    return chunk


def test_embed_and_store_chunks_tags_metadata():
    """embed_and_store_chunks should add source_file and chunk_index."""
    mock_db = MagicMock()
    with patch("rag.embedder._get_db", return_value=mock_db):
        chunks = [_make_chunk("hello"), _make_chunk("world")]
        embed_and_store_chunks(chunks, source_filename="lecture.pdf")

    assert chunks[0].metadata["source_file"] == "lecture.pdf"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[1].metadata["source_file"] == "lecture.pdf"
    assert chunks[1].metadata["chunk_index"] == 1


def test_embed_and_store_chunks_calls_add_documents():
    """embed_and_store_chunks should call db.add_documents."""
    mock_db = MagicMock()
    with patch("rag.embedder._get_db", return_value=mock_db):
        chunks = [_make_chunk("test")]
        embed_and_store_chunks(chunks, source_filename="doc.pdf")

    mock_db.add_documents.assert_called_once()
    call_kwargs = mock_db.add_documents.call_args.kwargs
    assert call_kwargs["documents"] == chunks


def test_clear_collection_deletes():
    """clear_collection should call delete_collection and reset the singleton."""
    mock_db = MagicMock()
    with patch("rag.embedder._get_db", return_value=mock_db):
        clear_collection()

    mock_db.delete_collection.assert_called_once()


def test_clear_collection_graceful_on_error():
    """clear_collection should not raise if delete_collection fails."""
    mock_db = MagicMock()
    mock_db.delete_collection.side_effect = RuntimeError("boom")
    with patch("rag.embedder._get_db", return_value=mock_db):
        clear_collection()  # should not raise
