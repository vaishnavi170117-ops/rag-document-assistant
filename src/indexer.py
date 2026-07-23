"""
indexer.py — shared indexing logic used by both the CLI script and the web app.

The key feature here is INCREMENTAL indexing: instead of wiping the index
and rebuilding everything every time, we check which files are already
indexed and only process the new ones. That saves a lot of time once you
have many documents.
"""

from src.config import config
from src.ingestion import load_document
from src.chunking import chunk_pages
from src.embedding import embed_texts
from src.vectorstore import VectorStore

SUPPORTED = {".pdf", ".docx", ".txt"}


def load_or_create_store():
    """
    Try to load an existing index from disk.
    If there isn't one yet, return None (caller will create a fresh store).
    """
    try:
        return VectorStore.load()
    except Exception:
        return None


def index_files(paths, store=None, force=False):
    """
    Add documents to the index, skipping ones already indexed.

    paths : list of file paths to index
    store : an existing VectorStore (or None to start fresh)
    force : if True, re-index everything even if already present

    Returns (store, report) where report describes what happened.
    """
    # Start from the existing index unless we're forcing a rebuild.
    if store is None and not force:
        store = load_or_create_store()

    already = store.indexed_sources() if store else set()

    new_chunks = []
    skipped = []
    added = []

    for path in paths:
        name = path.name

        # Skip files we've already indexed (unless forcing).
        if not force and name in already:
            skipped.append(name)
            continue

        pages = load_document(path)
        chunks = chunk_pages(pages)
        if chunks:
            new_chunks.extend(chunks)
            added.append((name, len(chunks)))

    # Nothing new to do.
    if not new_chunks:
        return store, {"added": [], "skipped": skipped, "total_chunks":
                       len(store.metadata) if store else 0}

    # Embed only the NEW chunks — this is the time saved by going incremental.
    vectors = embed_texts([c["text"] for c in new_chunks])

    # Create the store if this is the very first index.
    if store is None:
        store = VectorStore(dim=vectors.shape[1])

    store.add(vectors, new_chunks)
    store.save()

    return store, {
        "added": added,
        "skipped": skipped,
        "total_chunks": len(store.metadata),
    }


def find_documents(folder):
    """Return every supported document file inside a folder."""
    files = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            files.append(path)
    return files