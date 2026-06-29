from __future__ import annotations

import tempfile
import streamlit as st

from pipeline import SmartDocPipeline

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="SmartDocAI",
    page_icon="📄",
    layout="wide",
)

st.title("📄 SmartDocAI")
st.write("Explainable RAG using Gemini")

# --------------------------------------------------
# Session State
# --------------------------------------------------

if "pipeline" not in st.session_state:
    st.session_state.pipeline = SmartDocPipeline()

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if "result" not in st.session_state:
    st.session_state.result = None

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("Upload PDF")

    uploaded_pdf = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
    )

    if uploaded_pdf:

        if st.button("Build Index"):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
            ) as temp:

                temp.write(uploaded_pdf.read())
                pdf_path = temp.name

            with st.spinner("Building document index..."):

                stats = st.session_state.pipeline.index_document(pdf_path)

            st.session_state.indexed = True

            st.success("Document indexed!")

            st.write("### Statistics")

            st.write(f"Pages : {stats['pages']}")
            st.write(f"Chunks : {stats['chunks']}")
            st.write(f"Words : {stats['total_words']}")
            st.write(f"Index Time : {stats['index_time_seconds']} sec")

# --------------------------------------------------
# Question
# --------------------------------------------------

if st.session_state.indexed:

    question = st.text_input(
        "Ask a question"
    )

    if st.button("Generate Answer"):

        if question.strip() == "":
            st.warning("Enter a question.")

        else:

            with st.spinner("Gemini is thinking..."):

                result = st.session_state.pipeline.ask(question)

            st.session_state.result = result

else:

    st.info("Upload a PDF first.")

# --------------------------------------------------
# Result
# --------------------------------------------------

if st.session_state.result:

    result = st.session_state.result

    st.divider()

    st.header("Answer")

    st.write(result.answer)

    # BUG FIX 1: The original code had broken syntax — col2.metric() was opened,
    # then Python logic ran inside it, then col2.metric() was called again and
    # the outer one was never closed. Fixed by computing chunk_name BEFORE the
    # st.columns / metric calls, then passing it in cleanly.
    # BUG FIX 2: best.chunk.chunk_id → best.chunk_id
    # (best_overall() returns the ScoredChunk directly, not a wrapper with .chunk)
    # BUG FIX 3: col4 was declared but never used; removed the unused column.

    comparison = result.retrieval_comparison
    best = comparison.best_overall()

    if best:
        chunk_name = best.chunk_id.split("::")[-1]
    else:
        chunk_name = "Unknown"

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Confidence",
        result.confidence,
    )

    col2.metric(
        "Chunk",
        chunk_name,
    )

    col3.metric(
        "Time",
        f"{result.total_time:.2f} sec",
    )

    st.divider()

    st.subheader("Reasoning")

    st.write(result.reasoning)

    # --------------------------------------------------
    # Retrieved Chunks
    # --------------------------------------------------

    st.divider()

    st.header("Retrieved Chunks")

    methods = [
        ("Embedding", comparison.embedding),
        ("TF-IDF", comparison.tfidf),
        ("BoW", comparison.bow),
    ]

    for method_name, results in methods:

        st.subheader(method_name)

        if not results:
            st.write("No results.")
            continue

        for r in results:

            with st.expander(
                f"{r.chunk_id} | Score = {r.similarity_score:.4f}"
            ):

                st.write(f"Pages : {r.page_range()}")

                st.write(r.chunk.raw_text)