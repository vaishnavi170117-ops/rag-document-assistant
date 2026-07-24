"""
hybrid.py — Phase 3: combine keyword search (BM25) with vector search.

WHY: vector search understands MEANING but is weak on exact terms —
product codes, names, acronyms like "SECL-4471". BM25 is the opposite:
great at exact matches, blind to paraphrase.

HOW WE FUSE — Reciprocal Rank Fusion (RRF):
The two searches produce scores on completely different scales, so adding
them directly is meaningless. Instead we compare RANKS. Each chunk earns
1/(k + rank) points from each list, and we sum those points.

IMPORTANT REFINEMENT: BM25 always returns SOMETHING, even when nothing
really matches — it will happily rank a chunk highly just for containing
"what" and "is". Feeding that noise into the fusion pollutes the results.
So we only fuse BM25 hits whose score clears a meaningful threshold.
"""

import numpy as np
from rank_bm25 import BM25Okapi
from src.embedding import embed_query
from src.config import config

# RRF constant. 60 is the standard value from the original paper.
RRF_K = 60

# BM25 scores below this fraction of the best score are treated as noise.
# Without this, common stop-words drag in irrelevant documents.
BM25_MIN_RATIO = 0.30


def tokenize(text):
    """Split text into lowercase words for BM25."""
    return text.lower().split()


class HybridRetriever:
    """
    Wraps a VectorStore and adds a BM25 keyword index over the same chunks.
    Both searches run over identical data, so their results can be fused.
    """

    def __init__(self, store):
        self.store = store
        corpus = [tokenize(c["text"]) for c in store.metadata]
        self.bm25 = BM25Okapi(corpus)

    def _vector_ranking(self, query, depth):
        """Run vector search; return chunk positions in ranked order."""
        query_vec = embed_query(query)
        _, indices = self.store.index.search(
            np.asarray([query_vec], dtype=np.float32), depth
        )
        return [int(i) for i in indices[0] if i != -1]

    def _bm25_ranking(self, query, depth):
        """
        Run BM25 keyword search; return chunk positions in ranked order.
        Only keeps hits that scored meaningfully — weak matches are noise.
        """
        scores = self.bm25.get_scores(tokenize(query))
        best = float(scores.max()) if len(scores) else 0.0

        # If nothing matched at all, contribute nothing to the fusion.
        if best <= 0:
            return []

        cutoff = best * BM25_MIN_RATIO
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # Keep only hits above the cutoff, up to `depth`.
        return [i for i in ranked[:depth] if scores[i] >= cutoff]

    def search(self, query, top_k=None, depth=20):
        """Run both searches and fuse them with RRF."""
        if top_k is None:
            top_k = config.top_k

        vector_ranked = self._vector_ranking(query, depth)
        bm25_ranked = self._bm25_ranking(query, depth)

        # Award RRF points based on position in each list.
        points = {}
        for rank, idx in enumerate(vector_ranked):
            points[idx] = points.get(idx, 0) + 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(bm25_ranked):
            points[idx] = points.get(idx, 0) + 1.0 / (RRF_K + rank + 1)

        fused = sorted(points.items(), key=lambda pair: pair[1], reverse=True)

        results = []
        for idx, score in fused[:top_k]:
            chunk = dict(self.store.metadata[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results