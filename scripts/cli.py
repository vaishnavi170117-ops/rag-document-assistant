"""
cli.py — Phase 1 end-to-end: read ONE document, build the index,
then answer questions about it in a terminal loop.

Run it like:
    python -m scripts.cli data/raw/drivemart_report.pdf
"""

import sys
from src.ingestion import load_document
from src.chunking import chunk_pages
from src.embedding import embed_texts
from src.vectorstore import VectorStore
from src.generation import answer_question


def build_index(file_path):
    """Read the document, chunk it, embed it, and load it into a vector store."""
    print(f"Loading {file_path} ...")
    pages = load_document(file_path)
    print(f"  {len(pages)} pages with text")

    chunks = chunk_pages(pages)
    print(f"  {len(chunks)} chunks")

    print("Embedding chunks (this may take a few seconds) ...")
    vectors = embed_texts([c["text"] for c in chunks])

    store = VectorStore(dim=vectors.shape[1])
    store.add(vectors, chunks)
    print("Index built!\n")
    return store


def main():
    # Expect the document path as a command-line argument.
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.cli <path-to-document>")
        sys.exit(1)

    store = build_index(sys.argv[1])

    print("Ask questions about the document. Type 'exit' to quit.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not query:
            continue

        result = answer_question(query, store)
        print(f"\nAssistant: {result['answer']}")
        if result["sources"]:
            cites = ", ".join(f"{s} (p.{p})" for s, p in result["sources"])
            print(f"Sources retrieved: {cites}")
        print()


if __name__ == "__main__":
    main()