"""
chunking.py — Station 2: split page text into small overlapping chunks.

Each chunk keeps its parent page's metadata (source + page), so a chunk
we retrieve later still knows which file and page it came from. That's
what makes citations possible.
"""

from src.config import config     # our settings (chunk_size, chunk_overlap)


def chunk_text(text, chunk_size, overlap):
    """
    Cut one big string into overlapping pieces.

    Example with size=500, overlap=100:
      chunk 1 = characters 0   to 500
      chunk 2 = characters 400 to 900   (starts 100 before chunk 1 ended)
      chunk 3 = characters 800 to 1300
    The overlap keeps sentences at the boundaries from being lost.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])       # grab characters from start to end
        start += chunk_size - overlap        # move forward, but leave overlap
    return chunks


def chunk_pages(pages):
    """
    Turn a list of page dicts into a list of chunk dicts.
    Each chunk carries the same source + page as its parent page.
    """
    all_chunks = []
    for page in pages:
        pieces = chunk_text(page["text"], config.chunk_size, config.chunk_overlap)
        for piece in pieces:
            if piece.strip():                # skip empty pieces
                all_chunks.append({
                    "text": piece,
                    "source": page["source"], # carry metadata forward
                    "page": page["page"],
                })
    return all_chunks