"""
vectorstore.py — Station 4: store chunk vectors and search them fast (FAISS).

FAISS only stores NUMBERS. It doesn't know your text or metadata.
So we keep a parallel list 'self.metadata' lined up by position:
   vector #i in FAISS  <->  self.metadata[i]  (the chunk's text/source/page)

We can also save the index to disk and load it back later, so we don't
have to re-read and re-embed the documents every single time.
"""

import pickle                # lets us save/load Python objects to disk
from pathlib import Path
import faiss                 # the fast vector search library
import numpy as np
from src.config import config


class VectorStore:
    def __init__(self, dim=384):
        # IndexFlatIP = a FAISS index that compares vectors by inner product.
        # Because we normalized our vectors, inner product = cosine similarity
        # = a clean measure of "how close in meaning".
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []   # chunk info, aligned 1-to-1 with the vectors

    def add(self, vectors, chunks):
        """Add vectors to FAISS and their matching chunk info to our list."""
        self.index.add(np.asarray(vectors, dtype=np.float32))
        self.metadata.extend(chunks)

    def search(self, query_vector, top_k):
        """Find the top_k chunks whose vectors are closest to the query."""
        q = np.asarray([query_vector], dtype=np.float32)
        scores, indices = self.index.search(q, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:          # FAISS uses -1 if there aren't enough vectors
                continue
            chunk = dict(self.metadata[idx])   # copy the matching chunk info
            chunk["score"] = float(score)      # attach how close it was
            results.append(chunk)
        return results

    def indexed_sources(self):
        """Return the set of filenames already stored in this index."""
        return {c["source"] for c in self.metadata}

    def has_source(self, filename):
        """Check whether a given file has already been indexed."""
        return filename in self.indexed_sources()

    def save(self, index_dir=None):
        """Save the FAISS index and the metadata list to disk."""
        index_dir = index_dir or config.index_dir
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "faiss.index"))
        with open(index_dir / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    @classmethod
    def load(cls, index_dir=None):
        """Load a previously saved index + metadata back from disk."""
        index_dir = index_dir or config.index_dir
        store = cls.__new__(cls)   # make an empty VectorStore without __init__
        store.index = faiss.read_index(str(index_dir / "faiss.index"))
        with open(index_dir / "metadata.pkl", "rb") as f:
            store.metadata = pickle.load(f)
        return store