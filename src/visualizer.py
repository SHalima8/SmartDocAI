"""
visualizer.py
==============

SmartDocAI Explainability Module

This module does NOT perform retrieval or generate answers.

Its only responsibility is to convert a PipelineResult into
human-friendly visual information.

The returned dictionaries are later rendered by app.py using
Streamlit.

Responsibilities
----------------

✓ Answer formatting
✓ Retrieval summaries
✓ Pipeline timeline
✓ Chunk cards
✓ Similarity tables
✓ Confidence indicators
✓ Prompt preview
✓ Source information

This separation keeps the architecture clean:

Pipeline
    ↓
PipelineResult
    ↓
Visualizer
    ↓
Streamlit UI
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any


# ==========================================================
# Answer Formatting
# ==========================================================

def format_answer(result) -> dict:
    """
    Convert the PipelineResult into a clean answer block.
    """

    return {

        "question": result.question,

        "answer": result.answer,

        "confidence": result.confidence,

        "reasoning": result.reasoning,

        "source_chunk": result.source_chunk,

        "total_time": result.total_time,

    }


# ==========================================================
# Confidence
# ==========================================================

def build_confidence_badge(result) -> dict:
    """
    Return colour + label for confidence.

    This is UI-independent.

    Streamlit later decides how to display it.
    """

    confidence = str(result.confidence).lower()

    if confidence == "high":

        colour = "green"

        icon = "🟢"

    elif confidence == "medium":

        colour = "orange"

        icon = "🟡"

    elif confidence == "low":

        colour = "red"

        icon = "🔴"

    else:

        colour = "grey"

        icon = "⚪"

    return {

        "label": result.confidence,

        "colour": colour,

        "icon": icon,

    }


# ==========================================================
# Source Information
# ==========================================================

def build_source_summary(result) -> dict:
    """
    Small source panel.

    Example

    Document
        attention.pdf

    Source
        chunk_0012

    Pages
        6–7
    """

    retrieved = result.retrieved_chunks.hybrid[0]

    return {

        "document": retrieved.source_document,

        "chunk_id": retrieved.chunk_id,

        "pages": f"{retrieved.page_start}–{retrieved.page_end}",

        "words": retrieved.word_count,

        "sentences": retrieved.sentence_count,

    }

# ==========================================================
# Retrieved Chunk Cards
# ==========================================================

def build_chunk_cards(result) -> list[dict]:
    """
    Convert the retrieved chunks into clean cards.

    Each card contains only the information that is useful
    for visualisation.

    Returns
    -------
    list[dict]
    """

    cards = []

    # Hybrid retrieval is the final ranking used by SmartDocAI
    retrieved = result.retrieved_chunks.hybrid

    for rank, chunk in enumerate(retrieved, start=1):

        preview = chunk.clean_text.strip()

        if len(preview) > 220:
            preview = preview[:220].rstrip() + "..."

        cards.append({

            "rank": rank,

            "chunk_id": chunk.chunk_id,

            "pages": f"{chunk.page_start}–{chunk.page_end}",

            "word_count": chunk.word_count,

            "sentence_count": chunk.sentence_count,

            "similarity": round(chunk.similarity_score, 3),

            "preview": preview,

        })

    return cards


# ==========================================================
# Similarity Table
# ==========================================================

def build_similarity_table(result) -> list[dict]:
    """
    Build a table that compares the retrieved chunks.

    Later this becomes a Streamlit dataframe.

    Returns
    -------
    list[dict]
    """

    table = []

    for rank, chunk in enumerate(result.retrieved_chunks.hybrid, start=1):

        table.append({

            "Rank": rank,

            "Chunk": chunk.chunk_id,

            "Pages": f"{chunk.page_start}-{chunk.page_end}",

            "Similarity": round(chunk.similarity_score, 3),

            "Words": chunk.word_count,

        })

    return table


# ==========================================================
# Similarity Chart
# ==========================================================

def build_similarity_chart(result) -> dict:
    """
    Return x/y values for a similarity bar chart.

    Streamlit later turns this into a chart.

    Returns
    -------
    dict
    """

    labels = []

    scores = []

    for chunk in result.retrieved_chunks.hybrid:

        labels.append(chunk.chunk_id)

        scores.append(round(chunk.similarity_score, 3))

    return {

        "labels": labels,

        "scores": scores,

    }


# ==========================================================
# Retrieval Summary
# ==========================================================

def build_retrieval_summary(result) -> dict:
    """
    Explain retrieval in plain English.

    This is displayed under:

        'How SmartDocAI found the answer'
    """

    top_chunk = result.retrieved_chunks.hybrid[0]

    return {

        "method": "Hybrid Retrieval",

        "selected_chunk": top_chunk.chunk_id,

        "pages": f"{top_chunk.page_start}-{top_chunk.page_end}",

        "similarity": round(top_chunk.similarity_score, 3),

        "explanation":

            (
                "SmartDocAI compared your question with every chunk "
                "inside the document using semantic similarity. "
                "The chunk with the highest relevance score was sent "
                "to Gemini along with your question."
            )

    }

# ==========================================================
# Pipeline Timeline
# ==========================================================

def build_pipeline_timeline(result) -> list[dict]:
    """
    Convert pipeline timings into a timeline.

    Returns
    -------
    list[dict]

    Example
    -------
    [
        {"stage":"PDF Loader","time":0.21},
        {"stage":"Preprocessing","time":0.54},
        ...
    ]
    """

    trace = result.trace

    timeline = [

        {
            "stage": "PDF Loader",
            "time": trace.get("pdf_loader", {}).get("time", 0),
        },

        {
            "stage": "Preprocessing",
            "time": trace.get("preprocessing", {}).get("time", 0),
        },

        {
            "stage": "Chunking",
            "time": trace.get("chunking", {}).get("time", 0),
        },

        {
            "stage": "Retrieval",
            "time": trace.get("retrieval", {}).get("time", 0),
        },

        {
            "stage": "Prompt Builder",
            "time": trace.get("prompt_builder", {}).get("time", 0),
        },

        {
            "stage": "Gemini",
            "time": trace.get("llm", {}).get("time", 0),
        },

        {
            "stage": "Total",
            "time": result.total_time,
        },

    ]

    return timeline


# ==========================================================
# Pipeline Flow
# ==========================================================

def build_pipeline_flow() -> list[str]:
    """
    Return the logical flow of SmartDocAI.

    This is purely educational.
    """

    return [

        "📄 PDF Loaded",

        "🧹 Text Preprocessed",

        "✂️ Semantic Chunking",

        "🔍 Retrieval",

        "📝 Prompt Construction",

        "🤖 Gemini",

        "✅ Final Answer",

    ]


# ==========================================================
# Prompt Preview
# ==========================================================

def build_prompt_preview(result, max_chars: int = 1200) -> dict:
    """
    Build a shortened prompt preview.

    The full prompt can be very long,
    so we show only the beginning.
    """

    prompt = result.prompt_package.prompt

    truncated = False

    if len(prompt) > max_chars:

        prompt = prompt[:max_chars].rstrip()

        prompt += "\n\n...(prompt truncated)..."

        truncated = True

    return {

        "prompt": prompt,

        "length": len(result.prompt_package.prompt),

        "truncated": truncated,

    }


# ==========================================================
# LLM Summary
# ==========================================================

def build_llm_summary(result) -> dict:
    """
    Build information about the LLM response.
    """

    return {

        "model": result.trace.get(

            "llm",

            {}

        ).get(

            "model",

            "Unknown",

        ),

        "latency": result.trace.get(

            "llm",

            {}

        ).get(

            "time",

            0,

        ),

        "confidence": result.confidence,

        "source_chunk": result.source_chunk,

    }


# ==========================================================
# Educational Explanation
# ==========================================================

def build_how_it_works(result) -> list[str]:
    """
    Produce a beginner-friendly explanation of
    how SmartDocAI answered the question.
    """

    explanation = [

        "1. Your question was converted into an embedding vector.", 

        "2. Every document chunk already had its own embedding.",

        "3. Cosine similarity compared your question against every chunk.",

        "4. The highest-ranking chunks were selected.",

        "5. Those chunks were inserted into a carefully designed prompt.",

        "6. Gemini answered using only the retrieved context.",

        "7. The final answer, confidence and source information were returned.",

    ]

    return explanation

# ==========================================================
# Complete Visual Report
# ==========================================================

def build_visual_report(result) -> dict:
    """
    Build one complete report for the UI.

    Instead of app.py calling 10 different functions,
    it calls this one function once and receives
    everything needed to render the interface.
    """

    return {

        "answer": format_answer(result),

        "confidence": build_confidence_badge(result),

        "source": build_source_summary(result),

        "chunk_cards": build_chunk_cards(result),

        "similarity_table": build_similarity_table(result),

        "similarity_chart": build_similarity_chart(result),

        "retrieval_summary": build_retrieval_summary(result),

        "pipeline_timeline": build_pipeline_timeline(result),

        "pipeline_flow": build_pipeline_flow(),

        "prompt_preview": build_prompt_preview(result),

        "llm_summary": build_llm_summary(result),

        "how_it_works": build_how_it_works(result),

    }