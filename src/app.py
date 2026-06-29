"""
app.py
======

SmartDocAI Streamlit Application

This is the only file responsible for the user interface.

Architecture
------------

User
    ↓
Streamlit UI
    ↓
SmartDocPipeline
    ↓
Visualizer
    ↓
Display Results

The application itself contains almost no NLP logic.

All heavy processing is delegated to:

    pipeline.py
    visualizer.py
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import streamlit as st

from pipeline import SmartDocPipeline

from visualizer import (
    format_answer,
    build_confidence_badge,
    build_source_summary,
    build_chunk_cards,
    build_similarity_table,
    build_similarity_chart,
    build_retrieval_summary,
    build_pipeline_timeline,
    build_pipeline_flow,
    build_prompt_preview,
    build_llm_summary,
    build_how_it_works,
)

# ==========================================================
# Streamlit Configuration
# ==========================================================

st.set_page_config(

    page_title="SmartDocAI",

    page_icon="📄",

    layout="wide",

)

st.title("📄 SmartDocAI")

st.caption(
    "Explainable Retrieval-Augmented Question Answering using Gemini"
)

st.divider()
# ==========================================================
# Session State
# ==========================================================

if "pipeline" not in st.session_state:

    st.session_state.pipeline = SmartDocPipeline()

if "indexed" not in st.session_state:

    st.session_state.indexed = False

if "document_stats" not in st.session_state:

    st.session_state.document_stats = None

if "result" not in st.session_state:

    st.session_state.result = None
# ==========================================================
# Sidebar
# ==========================================================

with st.sidebar:

    st.header("📂 Document")

    uploaded_pdf = st.file_uploader(

        "Upload a PDF",

        type=["pdf"],

    )

    if uploaded_pdf:

        if st.button("Build Document Index"):

            with st.spinner("Indexing document..."):

                with tempfile.NamedTemporaryFile(

                    delete=False,

                    suffix=".pdf",

                ) as temp_pdf:

                    temp_pdf.write(uploaded_pdf.read())

                    temp_path = temp_pdf.name

                stats = st.session_state.pipeline.index_document(

                    temp_path

                )

                st.session_state.indexed = True

                st.session_state.document_stats = stats

            st.success("Document indexed successfully!")

    st.divider()

    if st.session_state.document_stats:

        stats = st.session_state.document_stats

        st.subheader("Statistics")

        st.write(f"**Pages:** {stats['pages']}")

        st.write(f"**Chunks:** {stats['chunks']}")

        st.write(f"**Words:** {stats['total_words']}")

        st.write(f"**Index Time:** {stats['index_time_seconds']} sec")

    # ==========================================================
# Ask Questions
# ==========================================================

st.header("💬 Ask Questions")

if not st.session_state.indexed:

    st.info("Upload a PDF and build the document index first.")

else:

    question = st.text_input(

        "Ask something about the uploaded document",

        placeholder="Example: What is positional encoding?",

    )

    ask_button = st.button(

        "Generate Answer",

        use_container_width=True,

    )

    if ask_button:

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            with st.spinner("Thinking..."):

                result = st.session_state.pipeline.ask(

                    question=question,

                    top_k=3,

                )

                st.session_state.result = result

    # ==========================================================
# Answer
# ==========================================================

if st.session_state.result:

    result = st.session_state.result

    answer = format_answer(result)

    badge = build_confidence_badge(result)

    st.divider()

    st.header("🤖 Answer")

    st.write(answer["answer"])

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(

            "Confidence",

            badge["label"],

        )

    with col2:

        st.metric(

            "Source Chunk",

            answer["source_chunk"],

        )

    with col3:

        st.metric(

            "Pipeline Time",

            f"{answer['total_time']} sec",

        )

    if answer["reasoning"]:

        st.subheader("Reasoning")

        st.write(answer["reasoning"])

    # ==========================================================
# Retrieval Summary
# ==========================================================

summary = build_retrieval_summary(result)

st.divider()

st.header("🔍 How SmartDocAI Found the Answer")

st.write(summary["explanation"])

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Method",
        summary["method"],
    )

with col2:
    st.metric(
        "Selected Chunk",
        summary["selected_chunk"],
    )

with col3:
    st.metric(
        "Similarity",
        summary["similarity"],
    )

st.caption(
    f"Pages: {summary['pages']}"
)

# ==========================================================
# Retrieved Chunks
# ==========================================================

cards = build_chunk_cards(result)

st.divider()

st.header("📄 Retrieved Chunks")

for card in cards:

    with st.expander(

        f"Rank {card['rank']} — {card['chunk_id']}"

    ):

        st.write(card["preview"])

        c1, c2, c3 = st.columns(3)

        c1.metric(

            "Similarity",

            card["similarity"],

        )

        c2.metric(

            "Pages",

            card["pages"],

        )

        c3.metric(

            "Words",

            card["word_count"],

        )

# ==========================================================
# Similarity Table
# ==========================================================

table = build_similarity_table(result)

st.divider()

st.header("📋 Retrieval Comparison")

st.dataframe(

    table,

    use_container_width=True,

)

# ==========================================================
# Similarity Chart
# ==========================================================

chart = build_similarity_chart(result)

st.divider()

st.header("📊 Similarity Scores")

chart_df = pd.DataFrame({

    "Chunk": chart["labels"],

    "Similarity": chart["scores"],

})

st.bar_chart(

    chart_df.set_index("Chunk")

)

# ==========================================================
# Source Information
# ==========================================================

source = build_source_summary(result)

st.divider()

st.header("📚 Source Information")

st.write(f"**Document:** {source['document']}")

st.write(f"**Chunk:** {source['chunk_id']}")

st.write(f"**Pages:** {source['pages']}")

st.write(f"**Words:** {source['words']}")

st.write(f"**Sentences:** {source['sentences']}")

# ==========================================================
# Pipeline Timeline
# ==========================================================

timeline = build_pipeline_timeline(result)

st.divider()

st.header("⏱ Pipeline Timeline")

timeline_df = pd.DataFrame(timeline)

st.dataframe(
    timeline_df,
    use_container_width=True,
)

# ==========================================================
# Pipeline Flow
# ==========================================================

flow = build_pipeline_flow()

st.divider()

st.header("⚙️ How SmartDocAI Works")

for step in flow:

    st.write(step)

# ==========================================================
# Prompt Preview
# ==========================================================

preview = build_prompt_preview(result)

st.divider()

st.header("📝 Prompt Sent to Gemini")

st.caption(
    f"Prompt Length : {preview['length']} characters"
)

st.code(
    preview["prompt"],
    language="text",
)

if preview["truncated"]:

    st.info(
        "Only the beginning of the prompt is shown."
    )

# ==========================================================
# LLM Information
# ==========================================================

llm = build_llm_summary(result)

st.divider()

st.header("🤖 Gemini Information")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Model",
    llm["model"],
)

col2.metric(
    "Latency",
    f"{llm['latency']} sec",
)

col3.metric(
    "Confidence",
    llm["confidence"],
)

# ==========================================================
# How SmartDocAI Answered
# ==========================================================

steps = build_how_it_works(result)

st.divider()

st.header("🎓 How SmartDocAI Produced This Answer")

for step in steps:

    st.write(step)

st.divider()

st.caption(
    "SmartDocAI • Explainable Retrieval-Augmented Question Answering using Gemini"
)