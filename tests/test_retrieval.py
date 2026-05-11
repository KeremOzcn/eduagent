# tests/test_retrieval.py
# ─────────────────────────────────────────────────────────────
# Tests for the Retrieval Agent and RAG pipeline search.
#
# Note: These tests require a real PDF at data/sample.pdf.
# If the file is missing, all tests are skipped automatically.
# ─────────────────────────────────────────────────────────────

import os
import pytest
from rag.pipeline import load_and_index_pdf, search, clear_index
from agents.retrieval_agent import retrieve


SAMPLE_PDF = "data/sample.pdf"


@pytest.fixture(scope="module", autouse=True)
def index_sample():
    """Index the sample PDF once before all tests in this module."""
    if not os.path.exists(SAMPLE_PDF):
        pytest.skip("No sample PDF found at data/sample.pdf")
    clear_index()
    load_and_index_pdf(SAMPLE_PDF)


# ── search() tests ───────────────────────────────────────────

def test_search_returns_results():
    """search() should return at least one chunk."""
    results = search("What is this document about?")
    assert len(results) > 0

def test_search_returns_dicts():
    """search() should return a list of dicts."""
    results = search("explain the main concept")
    assert all(isinstance(r, dict) for r in results)

def test_search_dict_has_required_keys():
    """Each dict must have text, source_file, page, chunk_index."""
    results = search("explain the main concept")
    for r in results:
        assert "text" in r
        assert "source_file" in r
        assert "page" in r
        assert "chunk_index" in r

def test_search_top_k():
    """search() should return at most k results."""
    results = search("any topic", k=5)
    assert len(results) <= 5

def test_search_default_k():
    """search() default k is 3, so at most 3 results."""
    results = search("any topic")
    assert len(results) <= 3

def test_search_source_file_is_string():
    """source_file metadata should be a string."""
    results = search("any topic")
    assert all(isinstance(r["source_file"], str) for r in results)

def test_search_page_is_int():
    """page metadata should be an integer."""
    results = search("any topic")
    assert all(isinstance(r["page"], int) for r in results)


# ── retrieve() tests ─────────────────────────────────────────

def test_retrieve_empty_question():
    """retrieve() should return empty list for blank question."""
    assert retrieve("") == []

def test_retrieve_whitespace_question():
    """retrieve() should return empty list for whitespace-only input."""
    assert retrieve("   ") == []

def test_retrieve_returns_results():
    """retrieve() should return chunks for a real question."""
    results = retrieve("What is this document about?")
    assert len(results) > 0

def test_retrieve_returns_dicts():
    """retrieve() should return a list of dicts."""
    results = retrieve("explain the main concept")
    assert all(isinstance(r, dict) for r in results)