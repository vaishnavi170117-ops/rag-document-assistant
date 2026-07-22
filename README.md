# RAG Document Assistant

A Retrieval-Augmented Generation (RAG) application that answers questions
from your documents and cites the exact source (filename + page). It only
answers from the provided documents and says "I don't know" when the answer
isn't there — no hallucinations.

## What it does

- Reads PDF, DOCX, and TXT documents
- Splits them into overlapping chunks and embeds them into a vector store
- Finds the most relevant chunks for any question (semantic search)
- Generates an answer grounded strictly in those chunks, with inline citations
- Refuses to answer when the information isn't in the documents

## Tech stack

- **Python** — core language
- **sentence-transformers** (`all-MiniLM-L6-v2`) — local embeddings (free, offline)
- **FAISS** — fast vector similarity search
- **Google Gemini API** — answer generation (via a swappable LLM layer)
- **PyYAML / python-dotenv** — configuration and secret management

## How it works (pipeline)

Document → Ingestion (extract text + metadata) → Chunking (overlapping pieces)
→ Embedding (text to vectors) → Vector store (FAISS) → Retrieval (top-k by meaning)
→ Generation (grounded, cited answer)

The design is modular: each stage is a separate module that passes plain data
to the next, so components (e.g. the LLM provider or retrieval method) can be
swapped without rewriting the pipeline.

## Setup

1. Clone the repository and create a virtual environment:

python -m venv .venv
.venv\Scripts\Activate.ps1 # Windows


2. Install dependencies:

pip install -r requirements.txt


3. Add your API key:

copy .env.example .env


   Then open `.env` and paste your free Gemini API key
   (get one at https://aistudio.google.com/apikey).

## Usage (Phase 1 — command line)

Put a document in `data/raw/`, then run:

python -m scripts.cli data/raw/your_document.pdf


Ask questions in the terminal. Type `exit` to quit.

## Configuration

All settings live in `config.yaml` — embedding model, chunk size, overlap,
top-k, and the LLM model. Change them there without touching any code.

## Project structure

rag-document-assistant/
├── config.yaml # all tunable settings
├── requirements.txt # dependencies
├── src/
│ ├── config.py # loads settings + secrets
│ ├── ingestion.py # reads PDF/DOCX/TXT into text + metadata
│ ├── chunking.py # splits text into overlapping chunks
│ ├── embedding.py # turns text into vectors
│ ├── vectorstore.py # FAISS: store and search vectors
│ ├── retrieval.py # finds the most relevant chunks
│ ├── llm.py # provider-agnostic LLM client
│ └── generation.py # builds grounded prompt, returns cited answer
├── scripts/
│ └── cli.py # Phase 1 command-line RAG loop
└── data/
├── raw/ # your source documents
└── index/ # saved vector index


## Roadmap

- **Phase 1 (done):** Command-line RAG with grounded, cited answers
- **Phase 2:** Multi-document ingestion, persistent index, Streamlit web UI
- **Phase 3:** Hybrid search, re-ranking, conversation memory, evaluation (RAGAS)