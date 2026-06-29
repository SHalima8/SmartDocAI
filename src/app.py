from __future__ import annotations

import tempfile
import pandas as pd
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
            # Reset previous result when a new doc is indexed
            st.session_state.result = None

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

    question = st.text_input("Ask a question")

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
    comparison = result.retrieval_comparison
    best = comparison.best_overall()

    # Clean chunk name helper
    def short_id(chunk_id: str) -> str:
        return chunk_id.split("::")[-1] if "::" in chunk_id else chunk_id

    chunk_name = short_id(best.chunk_id) if best else "Unknown"

    # ── Answer ────────────────────────────────────────────────────────────────
    st.divider()
    st.header("🤖 Answer")
    st.write(result.answer)

    col1, col2, col3 = st.columns(3)
    col1.metric("Confidence", result.confidence)
    col2.metric("Chunk", chunk_name)
    col3.metric("Time", f"{result.total_time:.2f} sec")

    if result.reasoning:
        st.subheader("Reasoning")
        st.write(result.reasoning)

    # ── Similarity Chart ──────────────────────────────────────────────────────
    st.divider()
    st.header("📊 Similarity Scores by Method")

    rows = []
    method_chunks = [
        ("Embedding", comparison.embedding),
        ("TF-IDF",    comparison.tfidf),
        ("BoW",       comparison.bow),
    ]
    for method_name, scored_list in method_chunks:
        if scored_list:
            for r in scored_list:
                rows.append({
                    "Chunk":      short_id(r.chunk_id),
                    "Method":     method_name,
                    "Similarity": round(r.similarity_score, 4),
                })

    if rows:
        chart_df = pd.DataFrame(rows)

        # One bar chart per method side-by-side
        c1, c2, c3 = st.columns(3)
        for col, method_name in zip([c1, c2, c3], ["Embedding", "TF-IDF", "BoW"]):
            with col:
                st.subheader(method_name)
                method_df = chart_df[chart_df["Method"] == method_name][["Chunk", "Similarity"]].set_index("Chunk")
                st.bar_chart(method_df)

    # ── Retrieval Comparison Table ────────────────────────────────────────────
    st.divider()
    st.header("📋 Retrieval Comparison Table")

    table_rows = []
    for method_name, scored_list in method_chunks:
        if scored_list:
            for rank, r in enumerate(scored_list, start=1):
                table_rows.append({
                    "Method":     method_name,
                    "Rank":       rank,
                    "Chunk":      short_id(r.chunk_id),
                    "Pages":      r.page_range(),
                    "Similarity": round(r.similarity_score, 4),
                })

    if table_rows:
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    # ── Retrieved Chunks (expandable) ─────────────────────────────────────────
    st.divider()
    st.header("📄 Retrieved Chunks")

    for method_name, scored_list in method_chunks:
        st.subheader(method_name)

        if not scored_list:
            st.write("No results.")
            continue

        for rank, r in enumerate(scored_list, start=1):
            label = f"Rank {rank} — {short_id(r.chunk_id)} | Score = {r.similarity_score:.4f} | Pages: {r.page_range()}"
            with st.expander(label):
                st.write(r.chunk.raw_text)

    # ── Pipeline Timeline ─────────────────────────────────────────────────────
    st.divider()
    st.header("⏱ Pipeline Timeline")

    trace = result.trace
    timeline_data = {
        "Stage": ["PDF Loader", "Preprocessing", "Chunking", "Retrieval", "Prompt Builder", "Gemini", "Total"],
        "Time (sec)": [
            trace.get("pdf_loader",     {}).get("time", 0),
            trace.get("preprocessing",  {}).get("time", 0),
            trace.get("chunking",       {}).get("time", 0),
            trace.get("retrieval",      {}).get("time", 0),
            trace.get("prompt_builder", {}).get("time", 0),
            trace.get("llm",            {}).get("time", 0),
            result.total_time,
        ],
    }
    timeline_df = pd.DataFrame(timeline_data).set_index("Stage")
    st.bar_chart(timeline_df)
    st.dataframe(timeline_df, use_container_width=True)

    # ── Prompt Preview ────────────────────────────────────────────────────────
    st.divider()
    st.header("📝 Prompt Sent to Gemini")

    prompt_text = result.prompt_package.prompt
    max_chars   = 1200
    truncated   = len(prompt_text) > max_chars
    display     = prompt_text[:max_chars].rstrip() + ("\n\n...(truncated)..." if truncated else "")

    st.caption(f"Total length: {len(prompt_text):,} characters")
    st.code(display, language="text")
    if truncated:
        st.info("Only the first 1,200 characters are shown.")

    # ── LLM Info ─────────────────────────────────────────────────────────────
    st.divider()
    st.header("🤖 Gemini Info")

    llm_trace = trace.get("llm", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Model",      llm_trace.get("model",   "Unknown"))
    c2.metric("Latency",    f"{llm_trace.get('time', 0):.2f} sec")
    c3.metric("Confidence", result.confidence)

    # ── How It Works ─────────────────────────────────────────────────────────
    st.divider()
    st.header("🎓 How SmartDocAI Produced This Answer")

    steps = [
        "1. Your question was converted into an embedding vector.",
        "2. Every document chunk already had its own embedding.",
        "3. Cosine similarity compared your question against every chunk.",
        "4. TF-IDF and BoW scores were computed alongside embeddings.",
        "5. The highest-ranking chunks were selected across all methods.",
        "6. Those chunks were inserted into a carefully designed prompt.",
        "7. Gemini answered using only the retrieved context.",
        "8. The final answer, confidence and source information were returned.",
    ]
    for step in steps:
        st.write(step)

st.divider()
st.caption("SmartDocAI • Explainable Retrieval-Augmented Question Answering using Gemini")