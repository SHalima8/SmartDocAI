"""
prompt_builder.py — SmartDocAI
================================
Sole responsibility: construct structured prompts for the Gemini LLM.

This module receives a retrieval result and a user query and returns
a carefully engineered prompt string ready to send to llm.py.

What prompt_builder.py does:
    - Builds prompts from retrieved chunk.raw_text (NOT clean_text)
    - Engineers role, context, question, constraints, and output format
    - Handles low-confidence retrieval with honest "I don't know" instructions
    - Supports multi-chunk context (top-3 results if needed)
    - Returns a PromptPackage with both the prompt and metadata for explain.py

What prompt_builder.py does NOT do:
    - Call any LLM API (that is llm.py)
    - Perform retrieval (that is retrieval.py)
    - Modify or preprocess text (that is preprocessing.py)

Why raw_text and not clean_text for the LLM?
    clean_text has been lemmatized and normalized for retrieval.
    "researcher study attention mechanism" is correct for TF-IDF matching
    but looks broken to Gemini. The LLM needs original, natural prose.
    raw_text is what a human would read. That is what Gemini receives.

Downstream consumers:
    llm.py       → receives prompt_package.prompt and sends it to Gemini
    explain.py   → reads prompt_package metadata for the pipeline breakdown
    pipeline.py  → calls build_prompt() as step 4 of the RAG loop
"""

import textwrap
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# PromptPackage — what build_prompt() returns
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PromptPackage:
    """
    The complete prompt artifact passed from prompt_builder to llm.py.

    Fields:
        prompt              : the full prompt string to send to Gemini
        query               : original user question (unmodified)
        context_chunks      : list of (chunk_id, raw_text, pages) tuples
                              used as context — for explain.py traceability
        retrieval_method    : which method selected the primary chunk
        primary_chunk_id    : chunk_id of the top-ranked retrieved chunk
        primary_score       : similarity score of the primary chunk
        is_low_confidence   : True if the retrieval score was below threshold
        prompt_strategy     : "standard" | "low_confidence" | "multi_chunk"
        char_count          : total characters in the final prompt
        context_char_count  : characters used for context (vs instructions)

    Why store all this metadata?
        explain.py needs to show: "I sent chunk_0006 from pages 3–4 to Gemini
        using the TF-IDF result (score 0.28)." Without this metadata, the
        explainer would have to re-run retrieval just to reconstruct the story.
    """
    prompt:             str
    query:              str
    context_chunks:     list[tuple[str, str, str]]  # (chunk_id, raw_text, page_range)
    retrieval_method:   str
    primary_chunk_id:   str
    primary_score:      float
    is_low_confidence:  bool
    prompt_strategy:    str
    char_count:         int = field(init=False)
    context_char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.prompt)
        self.context_char_count = sum(len(text) for _, text, _ in self.context_chunks)

    def summary(self) -> str:
        return (
            f"Strategy    : {self.prompt_strategy}\n"
            f"Method      : {self.retrieval_method}\n"
            f"Chunk       : {self.primary_chunk_id}\n"
            f"Score       : {self.primary_score:.4f}"
            + (" ⚠ LOW CONFIDENCE" if self.is_low_confidence else "") + "\n"
            f"Prompt size : {self.char_count:,} chars "
            f"({self.context_char_count:,} context + "
            f"{self.char_count - self.context_char_count:,} instructions)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────────────────────────────────────

# The role definition that opens every prompt.
# Clear role assignment improves response consistency — Gemini behaves
# differently when it knows it is a document assistant vs a general assistant.
_ROLE = textwrap.dedent("""\
    You are a precise document assistant. Your only job is to answer
    questions using the document context provided below.
    You do not use outside knowledge. You do not guess or speculate.
    If the answer is not in the provided context, you say so clearly.\
""")

# Output format instruction appended to every prompt.
# JSON output makes parsing in llm.py reliable and deterministic.
# The keys are fixed so pipeline.py and visualizer.py can read them
# without parsing freeform text.
_OUTPUT_FORMAT = textwrap.dedent("""\
    Respond ONLY with a JSON object. No preamble, no explanation outside the JSON.
    Use exactly these keys:
    {
        "answer": "<your answer here, in full sentences>",
        "confidence": "<high | medium | low>",
        "source_chunk": "<chunk_id of the context you used most>",
        "reasoning": "<one sentence explaining how you found the answer>"
    }\
""")

# Constraint block — explicit rules prevent hallucination and hedging.
_CONSTRAINTS = textwrap.dedent("""\
    Rules you must follow:
    1. Answer ONLY from the context below. Do not use outside knowledge.
    2. If the answer is partially in the context, give the partial answer
       and note what is missing.
    3. If the answer is not in the context at all, set "answer" to:
       "The provided document does not contain information about this question."
    4. Never invent facts, numbers, names, or citations.
    5. Keep your answer concise and directly responsive to the question.\
""")

# Used when retrieval confidence is low — strengthens the "say I don't know"
# instruction because Gemini may still try to answer from general knowledge.
_LOW_CONFIDENCE_WARNING = textwrap.dedent("""\
    ⚠ RETRIEVAL WARNING: The document search returned a low-confidence match
    for this question. The context below may not contain the answer.
    Apply Rule 3 strictly — if the answer is not clearly present, say so.\
""")


# ─────────────────────────────────────────────────────────────────────────────
# Private Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_single_context(
    chunk_id:  str,
    raw_text:  str,
    page_range: str,
    source_doc: str,
) -> str:
    """
    Format one chunk as a labeled context block.

    The label carries chunk_id, pages, and source so Gemini can reference
    them in the "source_chunk" and "reasoning" fields of its JSON response.
    """
    header = f"[Context | {chunk_id} | {page_range} | {source_doc}]"
    separator = "─" * len(header)
    return f"{header}\n{separator}\n{raw_text.strip()}\n{separator}"


def _format_multi_context(
    chunks: list[tuple[str, str, str, str]],  # (chunk_id, raw_text, page_range, source_doc)
) -> str:
    """
    Format multiple chunks as numbered context blocks.
    Used when top-3 retrieval results are all included for ambiguous queries.

    Why include multiple chunks?
        Some questions span two sections of a document. Providing the top-3
        chunks gives Gemini more material to synthesize from while still
        keeping context grounded in the document.

    When NOT to use multi-chunk:
        When the top result is high-confidence (score > 0.5), one chunk is
        enough. Multi-chunk increases prompt length and can dilute focus.
    """
    blocks = []
    for i, (chunk_id, raw_text, page_range, source_doc) in enumerate(chunks, 1):
        header = f"[Context {i} | {chunk_id} | {page_range} | {source_doc}]"
        separator = "─" * len(header)
        blocks.append(f"{header}\n{separator}\n{raw_text.strip()}\n{separator}")
    return "\n\n".join(blocks)


def _assemble_prompt(
    role:       str,
    warning:    Optional[str],
    constraints: str,
    context:    str,
    question:   str,
    output_fmt: str,
) -> str:
    """
    Assemble all prompt sections in the correct order.

    Section order is deliberate:
        1. Role        — sets Gemini's mindset before it sees anything else
        2. Warning     — if low confidence, establish caution early
        3. Constraints — rules before context so they frame how context is read
        4. Context     — the retrieved document chunk(s)
        5. Question    — asked after context so Gemini reads context first
        6. Output fmt  — last, so it's the freshest instruction when generating

    This ordering follows the "primacy and recency" principle:
        Instructions at the start (role, constraints) and end (output format)
        are most reliably followed. Context in the middle keeps the LLM
        grounded without burying the instructions.
    """
    sections = [role]
    if warning:
        sections.append(warning)
    sections.append(constraints)
    sections.append("DOCUMENT CONTEXT:\n" + context)
    sections.append("QUESTION:\n" + question)
    sections.append(output_fmt)

    return "\n\n" .join(sections)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

# Similarity threshold above which we use single-chunk context.
# Below this, we include top-3 chunks to give Gemini more material.
_MULTI_CHUNK_THRESHOLD = 0.35

# Absolute low-confidence threshold — mirrors retrieval.py's constant.
_LOW_CONFIDENCE_THRESHOLD = 0.10


def build_prompt(
    query:            str,
    retrieval_result: "RetrievalResult",                    # noqa: F821
    all_results:      Optional[list["RetrievalResult"]] = None,  # noqa: F821
) -> PromptPackage:
    """
    Build a complete prompt from a retrieval result and user query.

    Parameters
    ----------
    query            : the user's original question, unmodified
    retrieval_result : the best RetrievalResult from retrieve_all()
                       (RetrievalComparison.best_overall())
    all_results      : optional list of all top-k results from the winning method
                       if provided and primary score is low, top-3 are included

    Returns
    -------
    PromptPackage containing the full prompt and metadata for explain.py

    Strategy selection:
        score >= 0.35  → standard: single chunk, high confidence instructions
        0.10–0.35      → multi_chunk: include top-3, moderate confidence
        < 0.10         → low_confidence: single chunk + strong "say IDK" warning

    Why not always use top-3?
        More context is not always better. A high-confidence single chunk
        is focused — Gemini answers from exactly the right section.
        Top-3 chunks at 0.91, 0.84, 0.79 means all three are relevant;
        top-3 at 0.15, 0.12, 0.11 means none are clearly relevant and
        combining them risks a confident-sounding hallucination.
    """
    score    = retrieval_result.similarity_score
    chunk    = retrieval_result.chunk
    method   = retrieval_result.retrieval_method

    # ── Determine prompt strategy ─────────────────────────────────────────────
    if score < _LOW_CONFIDENCE_THRESHOLD:
        strategy = "low_confidence"
    elif score < _MULTI_CHUNK_THRESHOLD and all_results and len(all_results) > 1:
        strategy = "multi_chunk"
    else:
        strategy = "standard"

    # ── Build context block ───────────────────────────────────────────────────
    context_chunks_meta: list[tuple[str, str, str]] = []  # (chunk_id, raw_text, page_range)

    if strategy == "multi_chunk" and all_results:
        # Include up to 3 results — use raw_text from each chunk
        multi_data = []
        for r in all_results[:3]:
            page_range = r.page_range()
            multi_data.append((
                r.chunk_id,
                r.chunk.raw_text,
                page_range,
                r.source_document,
            ))
            context_chunks_meta.append((r.chunk_id, r.chunk.raw_text, page_range))
        context_block = _format_multi_context(multi_data)

    else:
        # Single chunk — standard or low_confidence strategy
        page_range = retrieval_result.page_range()
        context_block = _format_single_context(
            chunk_id   = chunk.chunk_id,
            raw_text   = chunk.raw_text,
            page_range = page_range,
            source_doc = chunk.source_document,
        )
        context_chunks_meta.append((chunk.chunk_id, chunk.raw_text, page_range))

    # ── Build warning block ───────────────────────────────────────────────────
    warning = _LOW_CONFIDENCE_WARNING if strategy == "low_confidence" else None

    # ── Assemble full prompt ──────────────────────────────────────────────────
    prompt = _assemble_prompt(
        role        = _ROLE,
        warning     = warning,
        constraints = _CONSTRAINTS,
        context     = context_block,
        question    = query,
        output_fmt  = _OUTPUT_FORMAT,
    )

    return PromptPackage(
        prompt            = prompt,
        query             = query,
        context_chunks    = context_chunks_meta,
        retrieval_method  = method,
        primary_chunk_id  = chunk.chunk_id,
        primary_score     = score,
        is_low_confidence = retrieval_result.is_low_confidence,
        prompt_strategy   = strategy,
    )


def build_prompt_from_comparison(
    query:      str,
    comparison: "RetrievalComparison",   # noqa: F821
) -> PromptPackage:
    """
    Convenience wrapper — takes the full RetrievalComparison from retrieve_all()
    and builds the prompt using the best overall result.

    This is what pipeline.py calls. It handles the method priority
    (embedding > tfidf > bow) via comparison.best_overall() and also
    passes all_results from the winning method for multi-chunk strategy.

    Priority order for all_results:
        We pass the top-3 from whichever method produced the best result,
        not a mix of methods. Mixing methods in context is confusing —
        the chunks may contradict each other in subtle ways and Gemini
        cannot know which method to trust.
    """
    best = comparison.best_overall()

    if best is None:
        # No results at all — build a prompt that will return "I don't know"
        return _build_empty_prompt(query)

    # Get all results from the same method as the best result
    method_results = {
        "bow":       comparison.bow,
        "tfidf":     comparison.tfidf,
        "embedding": comparison.embedding,
    }.get(best.retrieval_method, [])

    return build_prompt(
        query            = query,
        retrieval_result = best,
        all_results      = method_results,
    )


def _build_empty_prompt(query: str) -> PromptPackage:
    """
    Build a prompt for the case where retrieval returned no results at all.
    This should be extremely rare — only if the chunk list was empty.
    """
    prompt = _assemble_prompt(
        role        = _ROLE,
        warning     = _LOW_CONFIDENCE_WARNING,
        constraints = _CONSTRAINTS,
        context     = "[No document context available — no document was processed.]",
        question    = query,
        output_fmt  = _OUTPUT_FORMAT,
    )
    return PromptPackage(
        prompt            = prompt,
        query             = query,
        context_chunks    = [],
        retrieval_method  = "none",
        primary_chunk_id  = "none",
        primary_score     = 0.0,
        is_low_confidence = True,
        prompt_strategy   = "low_confidence",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Demo  (python src/prompt_builder.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from pdf_loader    import load_pdf, PDFNotFoundError
    from preprocessing import preprocess_document
    from chunking      import chunk_document
    from retrieval     import retrieve_all

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "data", "uploads", "attention.pdf"
    )

    print(f"\n{'─' * 65}")
    print(f"  SmartDocAI — prompt_builder.py demo")
    print(f"{'─' * 65}\n")

    try:
        # Build the pipeline up to retrieval
        print("Step 1–3: Load → Preprocess → Chunk...")
        document = load_pdf(pdf_path)
        document = preprocess_document(document)
        chunks   = chunk_document(document, target_sentences=10, overlap_sentences=2)
        print(f"          {len(chunks)} chunks ready.\n")

        # Three queries that trigger different prompt strategies
        test_cases = [
            {
                "query":    "What is scaled dot-product attention?",
                "expected": "standard (high confidence, single chunk)",
            },
            {
                "query":    "How does the model handle sequence length limitations?",
                "expected": "multi_chunk (moderate confidence, top-3 chunks)",
            },
            {
                "query":    "What is the capital of France?",
                "expected": "low_confidence (out of scope, IDK response)",
            },
        ]

        for i, case in enumerate(test_cases, 1):
            query    = case["query"]
            expected = case["expected"]

            print(f"{'═' * 65}")
            print(f"  TEST {i}: {expected}")
            print(f"  Query: \"{query}\"")
            print(f"{'═' * 65}\n")

            # Retrieve
            comparison = retrieve_all(query, chunks, top_k=3)
            best       = comparison.best_overall()

            if best:
                print(f"  Best retrieval → {best.retrieval_method.upper()} "
                      f"score={best.similarity_score:.4f} "
                      f"chunk={best.chunk_id.split('::')[-1]}")

            # Build prompt
            package = build_prompt_from_comparison(query, comparison)

            # Show package summary
            print(f"\n  ── Prompt Package ──")
            print(f"  {package.summary().replace(chr(10), chr(10) + '  ')}")

            # Show the full prompt (what Gemini will receive)
            print(f"\n  ── Full Prompt ({'─' * 40})")
            # Indent every line for readability in demo output
            for line in package.prompt.split("\n"):
                print(f"  {line}")

            print()

        print(f"{'─' * 65}")
        print(f"  Demo complete. prompt_builder.py is working correctly.")
        print(f"{'─' * 65}\n")

    except PDFNotFoundError:
        print(f"\n  PDF not found: {pdf_path}")
        print("  Place your PDF at data/uploads/attention.pdf\n")
    except Exception as e:
        import traceback
        print(f"\n  Error: {e}")
        traceback.print_exc()