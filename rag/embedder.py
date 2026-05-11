# rag/embedder.py
# ─────────────────────────────────────────────────────────────

# Responsibility: Convert text chunks into vector embeddings and
# persist them in a local ChromaDB vector database.
#
# Changes from Kerem's original:
#   1. Singleton pattern — model loads once, reused every call.
#   2. Metadata — each chunk tagged with source_file, page, chunk_index.
#   3. clear_collection() — wipe all chunks before a new PDF upload.
#   4. Persistent client — uses add_documents() to append rather than
#      recreate, so multiple PDFs can be indexed together.
# ─────────────────────────────────────────────────────────────

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "./chroma_db"
COLLECTION  = "eduagent_docs"

# ── Singleton ────────────────────────────────────────────────
_embeddings = None
_db = None

def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return the shared embedding model, loading it on first call."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings


def _get_db() -> Chroma:
    """Return the shared Chroma client, creating it on first call."""
    global _db
    if _db is None:
        _db = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=_get_embeddings(),
            collection_name=COLLECTION,
        )
    return _db


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

    db = _get_db()
    db.add_documents(documents=chunks)


# ── Clear collection ─────────────────────────────────────────
def clear_collection() -> None:
    """
    Delete all stored chunks from ChromaDB.
    Call this before indexing a new PDF so old chunks do not pollute results.
    """
    global _db
    db = _get_db()
    try:
        db.delete_collection()
    except Exception:
        pass
    finally:
        _db = None
