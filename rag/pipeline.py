# rag/pipeline.py
# ─────────────────────────────────────────────────────────────
# This is the SINGLE ENTRY POINT for the entire RAG pipeline.
#
# Other agents (Monitor, Evaluator, Answer Agent, etc.) should
# ONLY import from this file — not from loader, embedder, or
# retriever directly. This keeps the internals easy to change
# without breaking other agents.
#
# Changes from Kerem's original:
#   1. load_and_index_pdf() now extracts the filename and passes
#      it to embed_and_store_chunks() as metadata.
#   2. search() now returns list[dict] instead of list[str].
#      Each dict has: text, source_file, page, chunk_index.
#   3. clear_index() added — wipes ChromaDB before a new upload.
#
# Quick-start for teammates:
#
#   from rag.pipeline import load_and_index_pdf, search
#
#   load_and_index_pdf("path/to/document.pdf")
#   chunks = search("What is Q-Learning?")
#   # chunks → [{"text": "...", "source_file": "doc.pdf", "page": 3, ...}]
# ─────────────────────────────────────────────────────────────

import os

from rag.loader   import load_and_split_pdf
from rag.embedder import embed_and_store_chunks, clear_collection
from rag.retriever import retrieve_top_chunks


def load_and_index_pdf(pdf_path: str) -> None:
    """
    Full ingestion pipeline: load a PDF, chunk it, embed it,
    and persist the vectors in ChromaDB with metadata.

    Args:
        pdf_path: Path to the PDF file to index.
    """
    source_filename = os.path.basename(pdf_path)
    chunks = load_and_split_pdf(pdf_path)
    embed_and_store_chunks(chunks, source_filename=source_filename)


def search(question: str, k: int = 3) -> list[dict]:
    """
    Retrieve the most relevant chunks for a student's question.

    Args:
        question: The student's question as a plain string.
        k:        How many chunks to return (default: 3).

    Returns:
        List of dicts with keys: text, source_file, page, chunk_index.
    """
    return retrieve_top_chunks(question, k=k)


def clear_index() -> None:
    """
    Delete all indexed data from ChromaDB.
    Call this before indexing a new PDF to avoid mixing old and new data.
    """
    clear_collection()