"""
pipeline.py — SmartDocAI
=========================
Sole responsibility: wire every module together into one callable pipeline.

This is the orchestrator. It does not implement any logic itself —
it calls pdf_loader, preprocessing, chunking, retrieval, prompt_builder,
and llm in the correct order and returns a structured PipelineResult.

Usage:
    # Option 1 — class-based (reuse index across multiple questions)
    pipeline = SmartDocPipeline()
    pipeline.index_document("data/uploads/attention.pdf")
    result = pipeline.ask("What is scaled dot-product attention?")

    # Option 2 — one-shot convenience function
    result = run_pipeline("data/uploads/attention.pdf", "What is attention?")
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdf_loader     import load_pdf
from preprocessing  import preprocess_document
from chunking       import chunk_document
from retrieval      import retrieve_all
from prompt_builder import build_prompt_from_comparison
from llm            import generate_answer


# ─────────────────────────────────────────────────────────────────────────────
# PipelineResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Everything produced by one question-answering run.

    Storing the full objects (retrieval comparison, prompt package)
    rather than just strings means:
        - visualizer.py can build charts without re-running retrieval
        - explain.py can reconstruct the full pipeline story
        - the Streamlit UI can display scores, pages, method names
        - debugging is trivial — print result.retrieval_comparison.bow[0].score

    Fields
    ------
    question          : original user question, unmodified
    answer            : LLM answer string
    confidence        : "high" | "medium" | "low" | "unknown"
    source_chunk      : chunk_id Gemini says it used most
    reasoning         : one-sentence explanation from Gemini
    retrieval_comparison : full RetrievalComparison from retrieve_all()
    prompt_package    : full PromptPackage from build_prompt_from_comparison()
    llm_response      : full dict returned by generate_answer()
    trace             : timing and step metadata for explain.py
    total_time        : wall-clock seconds for the full ask() call
    success           : False if the LLM call failed
    """
    question:              str
    answer:                str
    confidence:            str
    source_chunk:          str
    reasoning:             str
    retrieval_comparison:  Any          # RetrievalComparison
    prompt_package:        Any          # PromptPackage
    llm_response:          dict[str, Any]
    trace:                 dict[str, Any]
    total_time:            float
    success:               bool = True

    def best_chunk(self):
        """Convenience — the top-ranked chunk used as LLM context."""
        return self.retrieval_comparison.best_overall()

    def summary(self) -> str:
        best = self.best_chunk()
        chunk_info = (
            f"{best.chunk_id.split('::')[-1]} "
            f"({best.retrieval_method.upper()}, "
            f"score={best.similarity_score:.3f}, "
            f"{best.page_range()})"
            if best else "none"
        )
        return (
            f"Q : {self.question}\n"
            f"A : {self.answer[:200]}\n"
            f"Confidence : {self.confidence}\n"
            f"Chunk used : {chunk_info}\n"
            f"Total time : {self.total_time}s"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SmartDocPipeline
# ─────────────────────────────────────────────────────────────────────────────

class SmartDocPipeline:
    """
    Stateful pipeline that indexes a document once and answers many questions.

    State
    -----
    document     : preprocessed DocumentResult (pdf_loader + preprocessing)
    chunks       : list[Chunk] produced by chunking.py
    index_ready  : bool — True after index_document() succeeds
    last_result  : most recent PipelineResult (useful for debugging in Streamlit)

    Why stateful?
        Loading, preprocessing, and chunking a PDF takes 2–5 seconds.
        For a multi-turn UI where the user asks several questions about
        the same document, doing that work once and caching the chunks
        makes every subsequent question answer in <2 seconds.
    """

    def __init__(self) -> None:
        self.document      = None
        self.chunks        = None
        self.index_ready   = False
        self.document_name = ""
        self.statistics    = {}
        self.trace         = {}
        self.last_result   = None

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Clear all cached state.
        Call this before indexing a new document in the same session.
        """
        self.document      = None
        self.chunks        = None
        self.index_ready   = False
        self.document_name = ""
        self.statistics    = {}
        self.trace         = {}
        self.last_result   = None
        print("[pipeline] Pipeline reset.")

    # ── Index Document ────────────────────────────────────────────────────────

    def index_document(
        self,
        pdf_path: str,
        target_sentences:  int = 10,
        overlap_sentences: int = 2,
    ) -> dict:
        """
        Load, preprocess, and chunk a PDF — the expensive one-time setup.

        Parameters
        ----------
        pdf_path          : path to the PDF file
        target_sentences  : sentences per chunk (default 10)
        overlap_sentences : overlap between consecutive chunks (default 2)

        Returns
        -------
        dict — summary statistics about the indexed document
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: '{pdf_path}'")

        self.reset()
        self.document_name = os.path.basename(pdf_path)

        print("\n" + "=" * 65)
        print("  SmartDocAI — Building Document Index")
        print("=" * 65)

        total_start = time.perf_counter()

        # Step 1 — Load PDF
        print("\nStep 1/3 : Loading PDF...")
        t = time.perf_counter()
        self.document = load_pdf(pdf_path)
        pdf_time = round(time.perf_counter() - t, 3)
        self.trace["pdf_loader"] = {
            "pages": len(self.document.pages),
            "time":  pdf_time,
        }

        # Step 2 — Preprocess
        print("\nStep 2/3 : Preprocessing...")
        t = time.perf_counter()
        preprocess_document(self.document)
        preprocessing_time = round(time.perf_counter() - t, 3)
        self.trace["preprocessing"] = {"time": preprocessing_time}

        # Step 3 — Chunk
        print("\nStep 3/3 : Chunking...")
        t = time.perf_counter()
        self.chunks = chunk_document(
            self.document,
            target_sentences  = target_sentences,
            overlap_sentences = overlap_sentences,
        )
        chunk_time = round(time.perf_counter() - t, 3)
        self.trace["chunking"] = {
            "chunks": len(self.chunks),
            "time":   chunk_time,
        }

        # Summary statistics
        total_time  = round(time.perf_counter() - total_start, 3)
        total_words = sum(c.word_count for c in self.chunks)
        avg_words   = round(total_words / len(self.chunks), 1) if self.chunks else 0

        self.statistics = {
            "document":           self.document_name,
            "pages":              len(self.document.pages),
            "chunks":             len(self.chunks),
            "total_words":        total_words,
            "avg_chunk_words":    avg_words,
            "index_time_seconds": total_time,
        }
        self.trace["indexing"] = {"total_time": total_time}
        self.index_ready = True

        print("\n" + "=" * 65)
        print("  Document Indexed Successfully")
        print("=" * 65)
        print(f"  Document      : {self.document_name}")
        print(f"  Pages         : {self.statistics['pages']}")
        print(f"  Chunks        : {self.statistics['chunks']}")
        print(f"  Total words   : {total_words:,}")
        print(f"  Avg chunk     : {avg_words} words")
        print(f"  Index time    : {total_time}s")
        print("=" * 65)

        return self.statistics

    # ── Ask ───────────────────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        top_k:    int = 3,
    ) -> PipelineResult:
        """
        Answer a question using the indexed document.

        Pipeline:
            question → retrieve → build prompt → Gemini → PipelineResult

        Parameters
        ----------
        question : the user's natural language question
        top_k    : number of chunks to retrieve per method (default 3)

        Returns
        -------
        PipelineResult with answer, scores, timings, and full trace
        """
        if not self.index_ready:
            raise RuntimeError(
                "No document indexed.\n"
                "Call index_document(pdf_path) first."
            )
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        print("\n" + "=" * 65)
        print("  SmartDocAI — Answering Question")
        print("=" * 65)
        print(f"\n  Q: {question}\n")

        total_start = time.perf_counter()
        trace: dict[str, Any] = {}

        # Step 4 — Retrieval
        print("Step 4/6 : Retrieving relevant chunks...")
        t = time.perf_counter()
        retrieval_comparison = retrieve_all(
            query  = question,
            chunks = self.chunks,
            top_k  = top_k,
        )
        retrieval_time = round(time.perf_counter() - t, 3)
        trace["retrieval"] = {
            "time":  retrieval_time,
            "top_k": top_k,
            "bow_top_score":       retrieval_comparison.bow[0].similarity_score       if retrieval_comparison.bow       else None,
            "tfidf_top_score":     retrieval_comparison.tfidf[0].similarity_score     if retrieval_comparison.tfidf     else None,
            "embedding_top_score": retrieval_comparison.embedding[0].similarity_score if retrieval_comparison.embedding else None,
        }

        # Step 5 — Build prompt
        print("Step 5/6 : Building prompt...")
        t = time.perf_counter()
        prompt_package = build_prompt_from_comparison(
            query      = question,
            comparison = retrieval_comparison,
        )
        prompt_time = round(time.perf_counter() - t, 3)
        trace["prompt_builder"] = {
            "time":            prompt_time,
            "strategy":        prompt_package.prompt_strategy,
            "prompt_chars":    prompt_package.char_count,
            "context_chunks":  len(prompt_package.context_chunks),
            "is_low_confidence": prompt_package.is_low_confidence,
        }

        # Step 6 — Generate answer
        print("Step 6/6 : Sending to Gemini...")
        llm_response = generate_answer(prompt_package.prompt)
        llm_time = llm_response.get("latency", 0.0)
        trace["llm"] = {
            "time":        llm_time,
            "model":       llm_response.get("model", MODEL_NAME if 'MODEL_NAME' in dir() else "gemini"),
            "json_parsed": llm_response.get("json_parsed", False),
            "success":     llm_response.get("success", False),
        }

        total_time = round(time.perf_counter() - total_start, 3)
        trace["pipeline"] = {"total_time": total_time}

        # Assemble result
        result = PipelineResult(
            question             = question,
            answer               = llm_response.get("answer", ""),
            confidence           = llm_response.get("confidence", "unknown"),
            source_chunk         = llm_response.get("source_chunk", "unknown"),
            reasoning            = llm_response.get("reasoning", ""),
            retrieval_comparison = retrieval_comparison,
            prompt_package       = prompt_package,
            llm_response         = llm_response,
            trace                = trace,
            total_time           = total_time,
            success              = llm_response.get("success", False),
        )
        self.last_result = result

        print("\n" + "=" * 65)
        print("  Answer")
        print("=" * 65)
        print(f"  {result.answer[:300]}")
        print(f"\n  Confidence : {result.confidence}")
        print(f"  Source     : {result.source_chunk}")
        print(f"  Total time : {total_time}s")
        print("=" * 65)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Function
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    pdf_path: str,
    question: str,
    top_k:    int = 3,
) -> PipelineResult:
    """
    Complete SmartDocAI pipeline in one function call.

    Useful for scripts and testing. For multi-question sessions,
    use SmartDocPipeline directly so the index is built only once.

    Parameters
    ----------
    pdf_path : path to PDF file
    question : user's question
    top_k    : chunks to retrieve per method

    Returns
    -------
    PipelineResult
    """
    pipeline = SmartDocPipeline()
    pipeline.index_document(pdf_path)
    return pipeline.ask(question=question, top_k=top_k)


# ─────────────────────────────────────────────────────────────────────────────
# Demo  (python src/pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "data", "uploads", "attention.pdf"
    )

    print(f"\n{'─' * 65}")
    print(f"  SmartDocAI — pipeline.py demo")
    print(f"{'─' * 65}")

    # Build index once
    pipeline = SmartDocPipeline()
    try:
        pipeline.index_document(pdf_path)
    except FileNotFoundError as e:
        print(f"\n  {e}")
        print("  Place your PDF at data/uploads/attention.pdf\n")
        sys.exit(1)

    # Ask three questions against the same index
    questions = [
        "What is scaled dot-product attention and how is it computed?",
        "Why does the transformer use positional encoding?",
        "What is the capital of France?",   # out-of-scope
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n{'─' * 65}")
        print(f"  Question {i}/{len(questions)}")
        print(f"{'─' * 65}")
        try:
            result = pipeline.ask(q)
            print(f"\n  Summary:\n  {result.summary()}")
        except Exception as e:
            import traceback
            print(f"  Error: {e}")
            traceback.print_exc()

    print(f"\n{'─' * 65}")
    print(f"  Demo complete.")
    print(f"{'─' * 65}\n")