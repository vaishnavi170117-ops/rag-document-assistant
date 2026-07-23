"""
streamlit_app.py — Phase 2: the web UI.

Streamlit turns plain Python into a web page. Every time the user
interacts (types, clicks), Streamlit re-runs this whole script top to
bottom — so we use st.session_state to remember things between runs
(like the loaded index and the chat history).

Uploads are INCREMENTAL: new files are added to the existing index instead
of replacing it, and files already indexed are skipped.

Run it with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Make sure Python can find our 'src' package when Streamlit runs this file.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from src.config import config
from src.indexer import index_files, load_or_create_store
from src.generation import answer_question

# --- Page setup (title, icon, layout) ---
st.set_page_config(page_title="RAG Document Assistant", page_icon="📄")
st.title("📄 RAG Document Assistant")
st.caption("Ask questions about your documents. Every answer cites its source.")


# --- Session state: remember things across Streamlit's re-runs ---
if "store" not in st.session_state:
    st.session_state.store = load_or_create_store()
if "messages" not in st.session_state:
    st.session_state.messages = []   # chat history


# --- Sidebar: upload documents and add them to the index ---
with st.sidebar:
    st.header("Documents")

    uploaded = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded and st.button("Add to index"):
        with st.spinner("Reading, chunking and embedding new documents..."):
            raw_dir = config.raw_dir
            raw_dir.mkdir(parents=True, exist_ok=True)

            # Save the uploaded files to data/raw so we can read them normally.
            paths = []
            for file in uploaded:
                dest = raw_dir / file.name
                dest.write_bytes(file.getbuffer())
                paths.append(dest)

            # Incremental: adds new files, skips ones already in the index.
            store, report = index_files(paths, store=st.session_state.store)
            st.session_state.store = store

        # Tell the user exactly what happened.
        if report["added"]:
            names = ", ".join(f"{n} ({c} chunks)" for n, c in report["added"])
            st.success(f"Added: {names}")
        if report["skipped"]:
            st.info(f"Already indexed, skipped: {', '.join(report['skipped'])}")
        if not report["added"] and not report["skipped"]:
            st.error("No text could be extracted from those files.")

    # Show what's currently loaded.
    if st.session_state.store:
        n = len(st.session_state.store.metadata)
        files = sorted(st.session_state.store.indexed_sources())
        st.info(f"Index ready: {n} chunks from {len(files)} file(s).")
        for f in files:
            st.write(f"• {f}")
    else:
        st.warning("No index found. Upload documents to build one.")


# --- Main area: the chat ---

# Replay the conversation so far.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            cites = ", ".join(f"{s} (p.{p})" for s, p in msg["sources"])
            st.caption(f"Sources: {cites}")

# The input box at the bottom.
question = st.chat_input("Ask a question about your documents...")

if question:
    # Show the user's message immediately.
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate the answer.
    with st.chat_message("assistant"):
        if not st.session_state.store:
            reply = "No documents indexed yet. Upload some in the sidebar first."
            sources = []
            st.markdown(reply)
        else:
            with st.spinner("Searching your documents..."):
                result = answer_question(question, st.session_state.store)
            reply = result["answer"]
            sources = result["sources"]
            st.markdown(reply)
            if sources:
                cites = ", ".join(f"{s} (p.{p})" for s, p in sources)
                st.caption(f"Sources: {cites}")

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources}
    )