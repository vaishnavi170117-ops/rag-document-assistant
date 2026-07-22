"""
embedding.py — Station 3: turn text into vectors (lists of numbers).

We use a local sentence-transformers model. It downloads once (~90MB),
then runs offline and free. Chunks with similar MEANING get similar
numbers — that's what lets us later search by meaning, not exact words.
"""

from sentence_transformers import SentenceTransformer
from src.config import config

# We load the model only ONCE and reuse it (loading is slow).
# Start as None; create it the first time it's actually needed.
_model = None


def get_model():
    """Load the embedding model once, then return the same one every time."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {config.embedding_model} ...")
        _model = SentenceTransformer(config.embedding_model)
    return _model


def embed_texts(texts):
    """
    Turn a LIST of strings into a 2D array of numbers.
    Shape = (number of texts, 384). Each row is one text's vector.
    """
    model = get_model()
    # normalize_embeddings=True scales vectors so we can compare them
    # using a simple 'closeness' measure later (cosine similarity).
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def embed_query(query):
    """Turn a SINGLE question string into one vector (a list of 384 numbers)."""
    return embed_texts([query])[0]   # wrap in a list, then take the first result