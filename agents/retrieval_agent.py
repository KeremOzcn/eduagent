# agents/retrieval_agent.py
# ─────────────────────────────────────────────────────────────
# Responsibility: Act as the clean interface between the orchestrator
# and the RAG pipeline. The orchestrator never imports from rag/
# directly — it always goes through this agent.
#
# If Kerem's rag/pipeline.py ever changes internally, only this
# file needs to be updated — not the orchestrator.
# ─────────────────────────────────────────────────────────────

from rag.pipeline import search


def retrieve(question: str, k: int = 3) -> list[dict]:
    """
    Retrieval Agent — finds the most relevant document chunks
    for a given student question.

    Args:
        question: The student's question as a plain string.
        k:        Number of chunks to return (default: 3).

    Returns:
        List of dicts with keys: text, source_file, page, chunk_index.
        Returns empty list if question is blank or nothing is indexed.
    """
    # Guard: return empty list for blank or whitespace-only questions
    # instead of crashing or sending garbage to the LLM.
    if not question or not question.strip():
        return []

    return search(question, k=k)