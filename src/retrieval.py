"""
retrieval.py — Station 5: turn a question into relevant chunks.

This is the query-side orchestrator. It's short because Stations 3 & 4
did the heavy lifting: here we just embed the question and search the store.

Keeping this separate means we can upgrade HOW we retrieve later
(e.g. hybrid search in Phase 3) without changing the AI/generation code.
"""

from src.embedding import embed_query
from src.vectorstore import VectorStore
from src.config import config


def retrieve(query, store, top_k=None):
    """
    Given a question and a vector store, return the top_k most relevant chunks.
    Each returned chunk includes its text, source, page, and similarity score.
    """
    # Use the top_k from config unless the caller overrides it.
    if top_k is None:
        top_k = config.top_k

    # Step 1: turn the question into a vector (numbers).
    query_vec = embed_query(query)

    # Step 2: search the store for the closest chunks.
    return store.search(query_vec, top_k)