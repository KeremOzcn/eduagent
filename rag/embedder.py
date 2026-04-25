# rag/embedder.py
# ─────────────────────────────────────────────────────────────

# Responsibility: Convert text chunks into vector embeddings and
# persist them in a local ChromaDB vector database.
#
# Changes from Kerem's original:
#   1. Singleton pattern — model loads once, reused every call.
#   2. Metadata — each chunk tagged with source_file, page, chunk_index.
#   3. clear_collection() — wipe all chunks before a new PDF upload.
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


# ── Store chunks ─────────────────────────────────────────────
def embed_and_store_chunks(chunks: list, source_filename: str = "unknown") -> None:
    """
    Embed a list of Document chunks and store them in ChromaDB.
    Each chunk is tagged with source file, page number, and chunk index.

    Args:
        chunks:          List of LangChain Document objects from loader.py.
        source_filename: Name of the PDF file (e.g. "lecture3.pdf").
    """
    for i, chunk in enumerate(chunks):
        chunk.metadata["source_file"] = source_filename
        chunk.metadata["chunk_index"] = i

    Chroma.from_documents(
        documents=chunks,
        embedding=_get_embeddings(),
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION,
    )

# ── Clear collection ─────────────────────────────────────────
def clear_collection() -> None:
    """
    Delete all stored chunks from ChromaDB.
    Call this before indexing a new PDF so old chunks do not pollute results.
    """
    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=_get_embeddings(),
        collection_name=COLLECTION,
    )
    try:
        db.delete_collection()
    except Exception:
        pass
