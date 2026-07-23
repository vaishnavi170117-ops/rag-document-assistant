"""
build_index.py — read documents from data/raw/ and add them to the index.

INCREMENTAL by default: files that are already indexed are skipped, so only
new documents get embedded. That's the expensive step, so skipping it is the
whole point.

Run it like:
    python -m scripts.build_index            # incremental (default)
    python -m scripts.build_index --force    # rebuild everything from scratch
"""

import sys
from src.config import config
from src.indexer import index_files, find_documents


def main():
    # A "--force" flag on the command line means rebuild from scratch.
    force = "--force" in sys.argv

    raw_dir = config.raw_dir
    print(f"Looking for documents in: {raw_dir}")
    if force:
        print("FORCE mode: rebuilding the whole index from scratch.\n")
    else:
        print("Incremental mode: already-indexed files will be skipped.\n")

    files = find_documents(raw_dir)
    if not files:
        print("No supported documents found. Put PDF/DOCX/TXT files in data/raw/")
        return

    print(f"Found {len(files)} document(s) in the folder.")

    # Do the work (this handles skipping, embedding, and saving).
    store, report = index_files(files, force=force)

    # Report what happened.
    print()
    if report["skipped"]:
        print(f"Skipped (already indexed): {len(report['skipped'])}")
        for name in report["skipped"]:
            print(f"   - {name}")

    if report["added"]:
        print(f"Newly indexed: {len(report['added'])}")
        for name, count in report["added"]:
            print(f"   + {name}  ({count} chunks)")
    else:
        print("No new documents to index.")

    print(f"\nIndex now holds {report['total_chunks']} chunks total.")
    print(f"Saved to: {config.index_dir}")


if __name__ == "__main__":
    main()