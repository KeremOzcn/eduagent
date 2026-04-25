# rag/retriever.py
# ─────────────────────────────────────────────────────────────
# Responsibility: Given a user question, find and return the most
# relevant text chunks from ChromaDB using vector similarity search.
#
# Changes from Kerem's original:
#   1. Singleton pattern — model loads once, reused every call.
#   2. retrieve_top_chunks() now returns list[dict] with full metadata
#      (text, source_file, page, chunk_index) so the UI can show
#      "Source: lecture3.pdf, page 5" alongside every answer.
# ─────────────────────────────────────────────────────────────

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "./chroma_db"
COLLECTION  = "eduagent_docs"

# ── Singleton ────────────────────────────────────────────────
_embeddings = None

def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return the shared embedding model, loading it on first call."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings


def retrieve_top_chunks(question: str, k: int = 3) -> list[dict]:
    """
    Return the top-k most relevant chunks with full metadata.

    Args:
        question: The student's question.
        k:        Number of chunks to return (default: 3).

    Returns:
        List of dicts with keys: text, source_file, page, chunk_index.

    Example:
        [
            {
                "text": "Q-Learning is a model-free algorithm...",
                "source_file": "lecture3.pdf",
                "page": 5,
                "chunk_index": 2
            },
            ...
        ]
    """
    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=_get_embeddings(),
        collection_name=COLLECTION,
    )
    results = db.similarity_search(question, k=k)
    return [
        {
            "text":        doc.page_content,
            "source_file": doc.metadata.get("source_file", "unknown"),
            "page":        doc.metadata.get("page", 0),
            "chunk_index": doc.metadata.get("chunk_index", 0),
        }
        for doc in results
    ]