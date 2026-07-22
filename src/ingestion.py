"""
ingestion.py — Station 1: read a document and pull out its text.

For each file we return a list of "pages". Each page is a dictionary:
    {"text": <the text>, "source": <filename>, "page": <page number>}

That 'source' and 'page' info is METADATA — it's what lets us show
citations (filename + page) in the final answer later.
"""

from pathlib import Path          # safe file-path handling
from pypdf import PdfReader       # reads PDF files
from docx import Document         # reads Word (.docx) files


def load_pdf(path):
    """Read a PDF, one page at a time, keeping each page's number."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""     # get text; "" if the page is empty
        if text.strip():                     # skip blank / image-only pages
            pages.append({
                "text": text,
                "source": path.name,         # just the filename, e.g. "report.pdf"
                "page": i + 1,               # i starts at 0, so +1 for human page numbers
            })
    return pages


def load_docx(path):
    """Read a Word document. Word has no fixed pages, so we treat it as page 1."""
    doc = Document(str(path))
    # Join all non-empty paragraphs into one block of text.
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [{"text": text, "source": path.name, "page": 1}]


def load_txt(path):
    """Read a plain text file. Also treated as a single page."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [{"text": text, "source": path.name, "page": 1}]


def load_document(path):
    """
    Dispatcher: look at the file's extension and call the right reader.
    """
    path = Path(path)                 # turn the text path into a Path object
    suffix = path.suffix.lower()      # get the extension, e.g. ".pdf", lowercased

    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix == ".docx":
        return load_docx(path)
    elif suffix == ".txt":
        return load_txt(path)
    else:
        # If someone gives us an unsupported file, fail with a clear message.
        raise ValueError(f"Unsupported file type: {suffix}")