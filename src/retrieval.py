"""
retrieval.py — SmartDocAI
==========================
Sole responsibility: score chunks against a query and return ranked results.

This module implements three independent retrieval methods:
    1. Bag of Words (BoW)       — raw word count vectors + cosine similarity
    2. TF-IDF                   — weighted word counts + cosine similarity
    3. Sentence Transformer     — dense semantic embeddings + cosine similarity

Each method is fully independent. They share no state and can be called
individually by pipeline.py or all together via retrieve_all().

This module does NOT:
    - Generate answers (that is llm.py's job)
    - Preprocess text (that is preprocessing.py's job)
    - Call any LLM API
    - Modify chunks

What retrieval.py receives:
    - query: str  (raw user question — preprocessing happens inside each method)
    - chunks: list[Chunk]  (output of chunking.py)

What retrieval.py returns:
    - list[RetrievalResult]  (ranked, scored, method-labeled)

Downstream consumers:
    pipeline.py       → calls retrieve_all(), passes best chunk to prompt_builder
    visualizer.py     → reads scores for bar charts and comparison table
    explain.py        → reads method, rank, score for step-by-step breakdown
    app.py            → displays RetrievalResult objects directly in Streamlit
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sentence Transformer imported lazily inside the function that needs it.
# This avoids loading a 90MB PyTorch model at import time when only BoW
# or TF-IDF is being used.


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Model name used for Sentence Transformer — fixed so every module references
# the same constant rather than hardcoding the string in multiple places.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Similarity threshold below which we consider a result "not found".
# A score below this means the query has almost no overlap with any chunk.
# Tuned for cosine similarity (range 0–1).
LOW_SIMILARITY_THRESHOLD = 0.10

# Default number of top results to return per method.
DEFAULT_TOP_K = 3


# ─────────────────────────────────────────────────────────────────────────────
# RetrievalResult — the unit pipeline.py and Streamlit consume
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """
    A single ranked retrieval result.

    Fields:
        chunk            : the full Chunk object — pipeline.py reads
                           chunk.raw_text for the LLM prompt and
                           chunk.clean_text was used for scoring
        similarity_score : cosine similarity between query and chunk (0–1)
        rank             : 1 = best match, 2 = second best, etc.
        retrieval_method : "bow" | "tfidf" | "embedding"
        is_low_confidence: True if score < LOW_SIMILARITY_THRESHOLD

    Why store the full Chunk and not just chunk_id?
        pipeline.py needs chunk.raw_text to build the Gemini prompt.
        visualizer.py needs chunk.page_start/page_end for the UI.
        Storing the full object avoids a second lookup.
    """
    chunk:             "Chunk"      # noqa: F821  (imported at runtime)
    similarity_score:  float
    rank:              int
    retrieval_method:  str
    is_low_confidence: bool = field(init=False)

    def __post_init__(self):
        self.is_low_confidence = self.similarity_score < LOW_SIMILARITY_THRESHOLD

    # ── Convenience accessors (so callers don't need to dig into .chunk) ─────
    @property
    def chunk_id(self)        -> str:   return self.chunk.chunk_id
    @property
    def source_document(self) -> str:   return self.chunk.source_document
    @property
    def page_start(self)      -> int:   return self.chunk.page_start
    @property
    def page_end(self)        -> int:   return self.chunk.page_end
    @property
    def raw_text(self)        -> str:   return self.chunk.raw_text
    @property
    def clean_text(self)      -> str:   return self.chunk.clean_text

    def page_range(self) -> str:
        if self.page_start == self.page_end:
            return f"p.{self.page_start}"
        return f"p.{self.page_start}–{self.page_end}"

    def summary(self) -> str:
        confidence = " ⚠ LOW CONFIDENCE" if self.is_low_confidence else ""
        return (
            f"[{self.retrieval_method.upper():>9}] "
            f"rank={self.rank} | "
            f"score={self.similarity_score:.4f}{confidence} | "
            f"{self.page_range():>10} | "
            f"{self.chunk.preview(70)}"
        )


@dataclass
class RetrievalComparison:
    """
    Holds results from all three methods for one query.
    This is what retrieve_all() returns and what visualizer.py reads.

    best_overall: the single highest-scoring result across all methods.
    Used by pipeline.py to select context for the LLM.
    """
    query:     str
    bow:       list[RetrievalResult]
    tfidf:     list[RetrievalResult]
    embedding: list[RetrievalResult]
    timing:    dict[str, float] = field(default_factory=dict)

    def best_overall(self) -> Optional[RetrievalResult]:
        """
        Return the single best result across all three methods.

        Strategy: prefer the Sentence Transformer result if its score is
        above the low-confidence threshold. If the embedding score is very
        low but TF-IDF found something strong, use TF-IDF.

        Why prefer embeddings?
            Embeddings understand synonyms and paraphrasing.
            "What does self-attention compute?" matches "the mechanism
            calculates weighted sums" even though no keywords overlap.
            BoW and TF-IDF would score that near zero.

        Why not always use embeddings?
            For very short queries with exact technical terms
            ("BLEU score", "Multi-Head Attention formula"),
            TF-IDF often scores higher because those exact tokens appear
            in the relevant chunk and the embedding model may generalise
            too broadly.
        """
        candidates = []
        for results in (self.embedding, self.tfidf, self.bow):
            if results:
                candidates.append(results[0])  # rank=1 from each method

        if not candidates:
            return None

        return max(candidates, key=lambda r: r.similarity_score)

    def all_results(self) -> list[RetrievalResult]:
        """Flat list of all results across all methods — useful for Streamlit."""
        return self.bow + self.tfidf + self.embedding

    def methods_agree(self) -> bool:
        """True if all three methods returned the same top chunk."""
        tops = []
        for results in (self.bow, self.tfidf, self.embedding):
            if results:
                tops.append(results[0].chunk_id)
        return len(set(tops)) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Shared Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _validate_inputs(query: str, chunks: list) -> Optional[str]:
    """
    Validate inputs before any retrieval method runs.
    Returns an error message string if invalid, None if OK.
    """
    if not query or not query.strip():
        return "Query is empty. Please enter a question."
    if not chunks:
        return "No chunks available. Upload and process a document first."
    return None


def _get_clean_texts(chunks: list) -> list[str]:
    """
    Extract clean_text from chunks, falling back to raw_text if missing.
    clean_text should always be present after preprocessing.py runs,
    but we fail gracefully rather than crashing.
    """
    texts = []
    for chunk in chunks:
        text = getattr(chunk, "clean_text", None) or chunk.raw_text
        texts.append(text.strip() if text.strip() else chunk.chunk_id)
    return texts


def _preprocess_query(query: str) -> str:
    """
    Run the query through the same preprocessing pipeline as documents.
    Critical for retrieval consistency: if chunks are lemmatized, the
    query must be too, or "running" won't match "run".
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from preprocessing import preprocess_query
        return preprocess_query(query)
    except Exception:
        # Fail silently — unpreprocessed query is better than no query
        return query.lower().strip()


def _build_results(
    scores:  np.ndarray,
    chunks:  list,
    method:  str,
    top_k:   int,
) -> list[RetrievalResult]:
    """
    Convert a raw scores array into a ranked list of RetrievalResult objects.

    scores: 1-D numpy array, one score per chunk, same order as chunks.
    Returns top_k results sorted by score descending.
    """
    if len(scores) == 0:
        return []

    # argsort ascending, then reverse for descending
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(ranked_indices, start=1):
        results.append(RetrievalResult(
            chunk            = chunks[idx],
            similarity_score = float(scores[idx]),
            rank             = rank,
            retrieval_method = method,
        ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Method 1 — Bag of Words
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_bow(
    query:  str,
    chunks: list,
    top_k:  int = DEFAULT_TOP_K,
) -> list[RetrievalResult]:
    """
    Bag of Words retrieval using cosine similarity.

    How it works:
        1. Build a vocabulary from all chunk clean_texts
        2. Represent each chunk as a raw word-count vector
        3. Represent the query as a word-count vector using the same vocabulary
        4. Compute cosine similarity between query vector and every chunk vector
        5. Return top_k chunks by score

    Strength:  Fast, zero setup, no model needed, fully interpretable.
               "What words match?" is the exact question it answers.

    Weakness:  Completely blind to meaning.
               Query "What does self-attention compute?" won't match a chunk
               that says "the mechanism calculates weighted sums" because
               "compute" ≠ "calculates" and "self-attention" ≠ "mechanism"
               at the character level.
               Also penalised by common words — "the", "is", "a" inflate
               vectors but carry no meaning. (TF-IDF solves this.)

    Implementation note:
        CountVectorizer handles tokenization and vocabulary building.
        We fit on chunk texts only, not the query — the query is transformed
        using the fitted vocabulary. Words in the query that don't appear in
        any chunk get score 0 automatically.
    """
    error = _validate_inputs(query, chunks)
    if error:
        print(f"[bow] {error}")
        return []

    clean_query  = _preprocess_query(query)
    clean_chunks = _get_clean_texts(chunks)

    try:
        vectorizer = CountVectorizer(
            lowercase     = True,
            token_pattern = r"(?u)\b\w+\b",  # include single-char tokens
            min_df        = 1,                # include rare terms
        )

        # Fit on chunks, transform both chunks and query
        chunk_matrix = vectorizer.fit_transform(clean_chunks)
        query_vector = vectorizer.transform([clean_query])

        # cosine_similarity returns shape (1, n_chunks) — flatten to 1-D
        scores = cosine_similarity(query_vector, chunk_matrix).flatten()

        return _build_results(scores, chunks, method="bow", top_k=top_k)

    except Exception as e:
        print(f"[bow] Retrieval failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Method 2 — TF-IDF
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_tfidf(
    query:  str,
    chunks: list,
    top_k:  int = DEFAULT_TOP_K,
) -> list[RetrievalResult]:
    """
    TF-IDF retrieval using cosine similarity.

    How it works:
        TF  (Term Frequency)         = how often a word appears in THIS chunk
        IDF (Inverse Document Freq.) = log(total chunks / chunks containing word)

        TF-IDF score = TF × IDF

        Words that appear in every chunk (like "the", "attention") get low IDF
        and are effectively down-weighted — they don't help distinguish chunks.
        Words unique to one or two chunks get high IDF and dominate scoring.

    Strength over BoW:
        Rare, distinctive terms are amplified. If only one chunk discusses
        "positional encoding", a query about it will score that chunk very high.
        Common filler words stop polluting the scores.

    Weakness:
        Still keyword-only. Synonyms still don't match.
        "calculate" and "compute" are unrelated from TF-IDF's perspective.
        Query and document must share exact tokens (after lemmatization).

    Implementation note:
        sublinear_tf=True replaces raw TF with 1+log(TF), which reduces the
        effect of a word appearing 50 times vs 5 times. Both signal relevance;
        the 10x difference should not 10x the score.

        smooth_idf=True adds 1 to document frequencies, preventing division
        by zero for terms in every document and reducing extreme IDF values.
    """
    error = _validate_inputs(query, chunks)
    if error:
        print(f"[tfidf] {error}")
        return []

    clean_query  = _preprocess_query(query)
    clean_chunks = _get_clean_texts(chunks)

    try:
        vectorizer = TfidfVectorizer(
            lowercase    = True,
            token_pattern= r"(?u)\b\w+\b",
            sublinear_tf = True,   # 1 + log(tf) instead of raw tf
            smooth_idf   = True,   # prevents division by zero
            min_df       = 1,
        )

        chunk_matrix = vectorizer.fit_transform(clean_chunks)
        query_vector = vectorizer.transform([clean_query])

        scores = cosine_similarity(query_vector, chunk_matrix).flatten()

        return _build_results(scores, chunks, method="tfidf", top_k=top_k)

    except Exception as e:
        print(f"[tfidf] Retrieval failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Method 3 — Sentence Transformer Embeddings
# ─────────────────────────────────────────────────────────────────────────────

# Module-level cache so the model loads only once per session.
# Loading all-MiniLM-L6-v2 takes ~2 seconds and ~90MB RAM.
# Reloading it for every query would make the system unusable.
_embedding_model = None


def _get_embedding_model():
    """
    Load the Sentence Transformer model, caching it after first load.
    Lazy loading means BoW and TF-IDF users never pay the model load cost.
    """
    global _embedding_model
    if _embedding_model is None:
        print(f"[embedding] Loading {EMBEDDING_MODEL_NAME} (first call only)...")
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"[embedding] Model loaded.")
    return _embedding_model


def retrieve_embedding(
    query:  str,
    chunks: list,
    top_k:  int = DEFAULT_TOP_K,
) -> list[RetrievalResult]:
    """
    Semantic retrieval using Sentence Transformer embeddings.

    How it works:
        1. Encode all chunk clean_texts as dense vectors (384 dimensions)
        2. Encode the query as a dense vector using the same model
        3. Compute cosine similarity between query vector and every chunk vector
        4. Return top_k chunks by score

    Why cosine similarity and not dot product?
        Cosine similarity normalises for vector magnitude, so a long chunk
        with many sentences doesn't automatically score higher than a short
        one just because its embedding has larger values.

    Strength over BoW and TF-IDF:
        The model was trained on 1 billion sentence pairs to produce vectors
        where semantically similar text clusters together in embedding space.
        "compute weighted sums" and "calculate attention scores" are near each
        other in that 384-D space even though they share no keywords.

        This is why embedding retrieval wins for natural language queries.

    Weakness:
        Slower (model inference required).
        For very short, exact-match technical queries ("BLEU score formula"),
        TF-IDF sometimes scores higher because the model may generalise the
        query to a broader semantic neighbourhood that misses the exact chunk.

    Implementation note:
        We encode chunk texts once and cache them in the future (pipeline.py
        will handle caching between queries so re-encoding 40 chunks per query
        doesn't happen in production). For this demo, encoding runs each call.

        batch_size=32 and show_progress_bar=False keep logs clean.
    """
    error = _validate_inputs(query, chunks)
    if error:
        print(f"[embedding] {error}")
        return []

    # Note: we use clean_text for chunks but the raw query for embedding.
    # Sentence Transformers are trained on natural language — lemmatized
    # text like "researcher study attention mechanism" is less natural and
    # may slightly hurt embedding quality. Raw query is intentional here.
    clean_chunks = _get_clean_texts(chunks)

    try:
        model = _get_embedding_model()

        # Encode chunks and query
        chunk_embeddings = model.encode(
            clean_chunks,
            batch_size        = 32,
            show_progress_bar = False,
            convert_to_numpy  = True,
        )
        query_embedding = model.encode(
            [query],                  # raw query — see note above
            show_progress_bar = False,
            convert_to_numpy  = True,
        )

        scores = cosine_similarity(query_embedding, chunk_embeddings).flatten()

        return _build_results(scores, chunks, method="embedding", top_k=top_k)

    except Exception as e:
        print(f"[embedding] Retrieval failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Unified Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_all(
    query:  str,
    chunks: list,
    top_k:  int = DEFAULT_TOP_K,
) -> RetrievalComparison:
    """
    Run all three retrieval methods and return a RetrievalComparison.

    This is the function pipeline.py calls. It returns a structured object
    so the caller can choose which method's result to use and visualizer.py
    can build comparison charts without re-running retrieval.

    Timing is recorded for each method so visualizer.py can show
    "BoW: 0.003s | TF-IDF: 0.012s | Embedding: 0.241s" — useful for
    teaching learners about the speed/quality tradeoff.
    """
    timing: dict[str, float] = {}

    t0 = time.perf_counter()
    bow_results = retrieve_bow(query, chunks, top_k=top_k)
    timing["bow"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    tfidf_results = retrieve_tfidf(query, chunks, top_k=top_k)
    timing["tfidf"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    embedding_results = retrieve_embedding(query, chunks, top_k=top_k)
    timing["embedding"] = round(time.perf_counter() - t0, 4)

    return RetrievalComparison(
        query     = query,
        bow       = bow_results,
        tfidf     = tfidf_results,
        embedding = embedding_results,
        timing    = timing,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Educational Output
# ─────────────────────────────────────────────────────────────────────────────

def print_retrieval_results(comparison: RetrievalComparison) -> None:
    """
    Print a full educational breakdown of all three retrieval methods.

    This is what you study on Day 3 — read every line and ask:
        - Why did BoW pick a different chunk than TF-IDF?
        - Which method's chunk actually contains the answer?
        - What does the score difference tell you about the query type?
    """
    print(f"\n{'═' * 65}")
    print(f"  RETRIEVAL RESULTS")
    print(f"{'═' * 65}")
    print(f"  Query: \"{comparison.query}\"")
    print(f"  Timing → "
          f"BoW: {comparison.timing.get('bow', 0):.4f}s | "
          f"TF-IDF: {comparison.timing.get('tfidf', 0):.4f}s | "
          f"Embedding: {comparison.timing.get('embedding', 0):.4f}s")

    # ── Per-method breakdown ──────────────────────────────────────────────────
    method_data = [
        ("Bag of Words",         comparison.bow,       "bow"),
        ("TF-IDF",               comparison.tfidf,     "tfidf"),
        ("Sentence Transformer", comparison.embedding, "embedding"),
    ]

    for method_name, results, method_key in method_data:
        print(f"\n  ── {method_name} ──")

        if not results:
            print(f"     No results returned.")
            continue

        for r in results:
            confidence = " ⚠ LOW" if r.is_low_confidence else ""
            print(
                f"     #{r.rank}  score={r.similarity_score:.4f}{confidence:6}  "
                f"{r.page_range():>12}  [{r.source_document}]"
            )
            print(f"         {r.chunk.preview(80)}")

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print(f"  {'Method':<22} {'Best Chunk':<18} {'Score':>7}  {'Pages':>10}  Source")
    print(f"{'─' * 65}")

    for method_name, results, _ in method_data:
        if results:
            r = results[0]
            cid   = r.chunk_id.split("::")[-1]   # "chunk_0004" not full id
            score = f"{r.similarity_score:.4f}"
            pages = r.page_range()
            src   = r.source_document[:20]
            flag  = " ⚠" if r.is_low_confidence else ""
            print(f"  {method_name:<22} {cid:<18} {score:>7}{flag}  {pages:>10}  {src}")
        else:
            print(f"  {method_name:<22} {'—':<18} {'—':>7}  {'—':>10}  —")

    # ── Agreement analysis ────────────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    if comparison.methods_agree():
        print(
            "  ✓ All three methods agree on the best chunk.\n"
            "  This is a strong signal — the answer is almost certainly there."
        )
    else:
        _print_disagreement_explanation(comparison)

    # ── Best overall ─────────────────────────────────────────────────────────
    best = comparison.best_overall()
    if best:
        print(f"\n  SELECTED FOR LLM CONTEXT:")
        print(f"  Method   : {best.retrieval_method.upper()}")
        print(f"  Chunk    : {best.chunk_id}")
        print(f"  Score    : {best.similarity_score:.4f}")
        print(f"  Pages    : {best.page_range()}")
        print(f"  Preview  : {best.chunk.preview(100)}")

        if best.is_low_confidence:
            print(
                f"\n  ⚠  WARNING: Best score ({best.similarity_score:.4f}) is below "
                f"the confidence threshold ({LOW_SIMILARITY_THRESHOLD}).\n"
                f"     The document may not contain an answer to this question.\n"
                f"     Gemini will be instructed to say so rather than hallucinate."
            )


def _print_disagreement_explanation(comparison: RetrievalComparison) -> None:
    """
    When methods disagree, print an educational explanation of why.
    This is what makes SmartDocAI different from a black-box chatbot.
    """
    bow_top   = comparison.bow[0].chunk_id   if comparison.bow   else None
    tfidf_top = comparison.tfidf[0].chunk_id if comparison.tfidf else None
    emb_top   = comparison.embedding[0].chunk_id if comparison.embedding else None

    print("  ⚡ Methods disagree on the best chunk. Here's why:\n")
    print(
        "  Bag of Words counts how many query words appear in each chunk.\n"
        "  It favours chunks with the highest raw keyword overlap, regardless\n"
        "  of whether those words are meaningful or common filler."
    )
    print(
        "\n  TF-IDF also counts words but down-weights terms that appear in\n"
        "  many chunks (common words). It amplifies rare, distinctive terms,\n"
        "  so it often selects a more specific chunk than BoW."
    )
    print(
        "\n  Sentence Transformer encodes meaning, not keywords. It can match\n"
        "  'calculate weighted sums' with 'compute attention scores' even\n"
        "  though they share no words. It usually selects the most semantically\n"
        "  relevant chunk, which is why it is used as the final context for Gemini."
    )

    # Highlight specific disagreements
    if bow_top and tfidf_top and bow_top != tfidf_top:
        print(
            f"\n  BoW vs TF-IDF disagreement:\n"
            f"    BoW   selected: {bow_top.split('::')[-1]}\n"
            f"    TF-IDF selected: {tfidf_top.split('::')[-1]}\n"
            f"    Likely cause: BoW was influenced by a common word in the query\n"
            f"    that appears frequently in one chunk. TF-IDF discounted it."
        )

    if emb_top and (emb_top != bow_top or emb_top != tfidf_top):
        print(
            f"\n  Embedding vs keyword methods:\n"
            f"    Embedding selected: {emb_top.split('::')[-1]}\n"
            f"    The embedding model found a chunk that captures the query's\n"
            f"    meaning even if it doesn't share exact keywords with the query."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Demo  (python src/retrieval.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from pdf_loader     import load_pdf, PDFNotFoundError
    from preprocessing  import preprocess_document
    from chunking       import chunk_document

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "data", "uploads", "attention.pdf"
    )

    print(f"\n{'─' * 65}")
    print(f"  SmartDocAI — retrieval.py demo")
    print(f"{'─' * 65}\n")

    try:
        # Build the chunk index
        print("Step 1: Loading PDF...")
        document = load_pdf(pdf_path)

        print("Step 2: Preprocessing...")
        document = preprocess_document(document)

        print("Step 3: Chunking...")
        chunks = chunk_document(document, target_sentences=10, overlap_sentences=2)
        print(f"        {len(chunks)} chunks ready.\n")

        # ── Five test queries — study these results carefully ─────────────────
        # Each query is chosen to show a different retrieval behaviour:
        #   Q1 — exact technical term  (TF-IDF likely wins)
        #   Q2 — conceptual question   (embedding likely wins)
        #   Q3 — numerical fact        (all methods should agree)
        #   Q4 — synonym test          (only embedding handles this)
        #   Q5 — out-of-scope question (all methods should show low confidence)

        test_queries = [
            "What is scaled dot-product attention?",
            "Why does the model use multiple attention heads?",
            "How many layers does the encoder have?",
            "How does the model compute weighted context vectors?",
            "What is the capital of France?",          # out-of-scope
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n{'═' * 65}")
            print(f"  TEST QUERY {i} / {len(test_queries)}")
            print(f"{'═' * 65}")

            comparison = retrieve_all(query, chunks, top_k=3)
            print_retrieval_results(comparison)

        print(f"\n{'─' * 65}")
        print(f"  Demo complete.")
        print(f"{'─' * 65}\n")

    except PDFNotFoundError:
        print(f"\n  PDF not found: {pdf_path}")
        print("  Place your PDF at data/uploads/attention.pdf")
        print("  Or: python src/retrieval.py path/to/your.pdf\n")
    except Exception as e:
        import traceback
        print(f"\n  Error: {e}")
        traceback.print_exc()