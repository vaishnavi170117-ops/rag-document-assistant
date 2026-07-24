"""
retrieval.py — Station 5: turn a question into relevant chunks.

This is the query-side orchestrator. It decides HOW to retrieve:
  - vector only (semantic search)
  - or hybrid (vector + BM25 keyword search, fused with RRF)

The choice is a config toggle, so we can compare both without code changes.
Keeping this decision here means generation.py never needs to know or care.
"""

from src.embedding import embed_query
from src.config import config

# Cache the hybrid retriever so we don't rebuild the BM25 index every question.
_hybrid_cache = {}


def _get_hybrid(store):
    """Build (or reuse) a HybridRetriever for this store."""
    # Import here so the app still works if rank-bm25 isn't installed.
    from src.hybrid import HybridRetriever

    key = id(store)                 # identify this particular store object
    if key not in _hybrid_cache:
        _hybrid_cache[key] = HybridRetriever(store)
    return _hybrid_cache[key]


def retrieve(query, store, top_k=None):
    """
    Given a question and a vector store, return the most relevant chunks.
    Each chunk includes its text, source, page, and score.
    """
    if top_k is None:
        top_k = config.top_k

    # Hybrid: vector + keyword, fused by rank.
    if config.use_hybrid:
        return _get_hybrid(store).search(query, top_k=top_k)

    # Vector only (the original Phase 1 behaviour).
    query_vec = embed_query(query)
    return store.search(query_vec, top_k)