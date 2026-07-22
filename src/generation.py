"""
generation.py — Station 6b: build a GROUNDED prompt and return a cited answer.

This file enforces the rules you care about:
  1. Answer ONLY from the retrieved chunks (no outside knowledge).
  2. Say "I don't know" if the answer isn't in the chunks.
  3. Cite the source (filename + page) for every claim.

We label each chunk with its source so the AI can attribute claims to it.
"""

from src.llm import llm
from src.retrieval import retrieve

# The RULES we give the AI. Strict on purpose — this is what keeps answers
# grounded in your documents instead of made up.
SYSTEM_PROMPT = """You are a retrieval-augmented assistant. Follow these rules strictly:

1. Answer ONLY using the provided context below. Do not use any outside knowledge.
2. If the answer is not in the context, reply exactly: "I don't know based on the provided documents."
3. Cite your sources inline using the format (source: FILENAME, page N) after each claim.
4. Be concise and factual. Do not speculate or add information not in the context."""


def format_context(chunks):
    """
    Turn the retrieved chunks into a labeled text block.
    Each chunk gets a [Source: ...] header so the AI can cite it.
    """
    blocks = []
    for c in chunks:
        header = f"[Source: {c['source']}, page {c['page']}]"
        blocks.append(f"{header}\n{c['text']}")
    # Separate chunks clearly so the AI sees them as distinct sources.
    return "\n\n---\n\n".join(blocks)


def answer_question(query, store):
    """
    The full RAG answer step:
      1. Retrieve the most relevant chunks for the question.
      2. Build a grounded prompt (rules + labeled context + question).
      3. Ask the AI and return the answer plus the sources used.
    """
    # Step 1: get the most relevant chunks.
    chunks = retrieve(query, store)

    # If nothing was retrieved, don't even call the AI.
    if not chunks:
        return {"answer": "I don't know based on the provided documents.",
                "sources": []}

    # Step 2: build the prompt.
    context = format_context(chunks)
    user_prompt = f"Context:\n\n{context}\n\n---\n\nQuestion: {query}\n\nAnswer:"

    # Step 3: ask the AI.
    answer = llm.complete(SYSTEM_PROMPT, user_prompt)

    # Collect the unique sources used, for display under the answer.
    sources = sorted({(c["source"], c["page"]) for c in chunks})
    return {"answer": answer, "sources": sources}