"""
test_pipeline.py
================
SmartDocAI End-to-End Pipeline Tester

Terminal interface for testing the full pipeline before building the
Streamlit UI. Indexes the document once, then answers as many questions
as you want without reloading.

Usage
-----
    python src/test_pipeline.py
    python src/test_pipeline.py data/uploads/my_document.pdf

Commands inside the loop
    q / quit / exit   → exit
    stats             → show document index statistics
    trace             → show timing trace from last answer
    chunks            → show all retrieved chunks from last answer
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import SmartDocPipeline, PipelineResult


# ─────────────────────────────────────────────────────────────────────────────
# PDF Path
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_PDF = os.path.join(ROOT_DIR, "data", "uploads", "attention.pdf")

pdf_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF

if not os.path.exists(pdf_path):
    print(f"\n  PDF not found: {pdf_path}")
    print("  Usage: python src/test_pipeline.py path/to/your.pdf\n")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Helper Printers
# ─────────────────────────────────────────────────────────────────────────────

def print_answer(result: PipelineResult) -> None:
    print("\n" + "=" * 65)
    print("  ANSWER")
    print("=" * 65)
    print(f"\n  {result.answer}\n")
    print(f"  Confidence   : {result.confidence}")
    print(f"  Source chunk : {result.source_chunk}")
    print(f"  Reasoning    : {result.reasoning[:120]}")
    print(f"  Total time   : {result.total_time:.3f}s")
    print("=" * 65)


def print_chunks(result: PipelineResult) -> None:
    """Show top chunks from the method that won (best_overall)."""
    best = result.retrieval_comparison.best_overall()
    if not best:
        print("  No retrieval results available.")
        return

    # Get all results from the winning method
    method = best.retrieval_method
    results = {
        "bow":       result.retrieval_comparison.bow,
        "tfidf":     result.retrieval_comparison.tfidf,
        "embedding": result.retrieval_comparison.embedding,
    }.get(method, [])

    print(f"\n  Retrieved Chunks — {method.upper()} (winning method)\n")
    print(f"  {'#':<4} {'Score':>7}  {'Pages':>10}  Preview")
    print(f"  {'─' * 60}")

    for r in results:
        preview = r.clean_text[:80].replace("\n", " ")
        print(f"  #{r.rank:<3} {r.similarity_score:>7.4f}  {r.page_range():>10}  {preview}...")

    # Also show what each method chose as its top result
    print(f"\n  Comparison — top chunk per method:\n")
    print(f"  {'Method':<12} {'Chunk':<16} {'Score':>7}  {'Pages':>10}")
    print(f"  {'─' * 50}")
    for method_name, res_list in [
        ("BoW",       result.retrieval_comparison.bow),
        ("TF-IDF",    result.retrieval_comparison.tfidf),
        ("Embedding", result.retrieval_comparison.embedding),
    ]:
        if res_list:
            r   = res_list[0]
            cid = r.chunk_id.split("::")[-1]
            print(f"  {method_name:<12} {cid:<16} {r.similarity_score:>7.4f}  {r.page_range():>10}")
        else:
            print(f"  {method_name:<12} {'—':<16} {'—':>7}  {'—':>10}")


def print_trace(result: PipelineResult) -> None:
    print(f"\n  Pipeline Timings\n")
    print(f"  {'Stage':<20} {'Time':>8}  Notes")
    print(f"  {'─' * 55}")

    stage_labels = {
        "pdf_loader":     "PDF Loading",
        "preprocessing":  "Preprocessing",
        "chunking":       "Chunking",
        "retrieval":      "Retrieval",
        "prompt_builder": "Prompt Building",
        "llm":            "Gemini API",
        "pipeline":       "TOTAL",
    }

    for stage, info in result.trace.items():
        if not isinstance(info, dict):
            continue
        label = stage_labels.get(stage, stage)
        t     = info.get("time", info.get("total_time", None))
        if t is None:
            continue

        # Extra notes per stage
        notes = ""
        if stage == "retrieval":
            scores = []
            for k in ("bow_top_score", "tfidf_top_score", "embedding_top_score"):
                v = info.get(k)
                if v is not None:
                    scores.append(f"{k.split('_')[0]}={v:.3f}")
            notes = "  " + " | ".join(scores) if scores else ""
        elif stage == "chunking":
            notes = f"  {info.get('chunks', '')} chunks"
        elif stage == "prompt_builder":
            notes = f"  strategy={info.get('strategy', '')}  low_conf={info.get('is_low_confidence', '')}"
        elif stage == "llm":
            notes = f"  json_parsed={info.get('json_parsed', '')}  model={info.get('model', '')}"

        marker = "▶" if stage == "pipeline" else " "
        print(f"  {marker} {label:<18} {t:>7.3f}s{notes}")


def print_stats(pipeline: SmartDocPipeline) -> None:
    stats = pipeline.statistics
    if not stats:
        print("  No document indexed yet.")
        return
    print(f"\n  Document Index Statistics\n")
    print(f"  Document    : {stats.get('document', '—')}")
    print(f"  Pages       : {stats.get('pages', '—')}")
    print(f"  Chunks      : {stats.get('chunks', '—')}")
    print(f"  Total words : {stats.get('total_words', '—'):,}")
    print(f"  Avg chunk   : {stats.get('avg_chunk_words', '—')} words")
    print(f"  Index time  : {stats.get('index_time_seconds', '—')}s")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  SmartDocAI — Pipeline Tester")
print("=" * 65)
print(f"\n  Document : {os.path.basename(pdf_path)}")
print(f"  Commands : stats | trace | chunks | quit\n")

# Build the index ONCE
pipeline = SmartDocPipeline()
try:
    pipeline.index_document(pdf_path)
except Exception as e:
    print(f"\n  Failed to index document: {e}\n")
    sys.exit(1)

last_result: PipelineResult | None = None

# ── Interactive loop ──────────────────────────────────────────────────────────
while True:

    try:
        question = input("\n  Question: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n\n  Goodbye.\n")
        break

    if not question:
        continue

    # ── Special commands ──────────────────────────────────────────────────────
    if question.lower() in ("q", "quit", "exit"):
        print("\n  Goodbye.\n")
        break

    if question.lower() == "stats":
        print_stats(pipeline)
        continue

    if question.lower() == "trace":
        if last_result:
            print_trace(last_result)
        else:
            print("  No question answered yet.")
        continue

    if question.lower() == "chunks":
        if last_result:
            print_chunks(last_result)
        else:
            print("  No question answered yet.")
        continue

    # ── Answer the question ───────────────────────────────────────────────────
    try:
        last_result = pipeline.ask(question)
        print_answer(last_result)

    except ValueError as e:
        print(f"\n  Input error: {e}")
    except Exception as e:
        import traceback
        print(f"\n  Pipeline error: {e}")
        traceback.print_exc()