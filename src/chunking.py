"""
chunking.py — SmartDocAI
=========================
Sole responsibility: split a preprocessed DocumentResult into retrievable
Chunk objects that retrieval.py can score against a query.

Design philosophy:
    Chunk boundaries follow sentence boundaries — never mid-sentence.
    Overlap of 2 sentences between consecutive chunks preserves context
    so answers that straddle a boundary are not lost.

    raw_text  → preserved exactly for the LLM (Gemini receives this)
    clean_text → generated via preprocessing.py for BoW/TF-IDF/Embeddings

Why sentence-based rather than character-based chunking?
    Character windows are fast but semantically blind.
    "...the attention weight is computed as softmax(QK^T /\n√dk)V\nwhere Q" —
    a character window splits this mid-formula. A sentence window keeps it
    together because the sentence ends after the formula is complete.

Downstream consumers:
    retrieval.py  → scores chunks against query using clean_text
    embeddings.py → encodes chunks using clean_text
    prompt_builder.py → sends chunk.raw_text to Gemini as context
    pipeline.py   → calls chunk_document() as step 3 of the RAG loop
"""

import re
import sys
import os
from dataclasses import dataclass, field
from typing import Optional

import nltk

# Ensure sentence tokenizer is available
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize


# ─────────────────────────────────────────────────────────────────────────────
# Chunk Dataclass — the unit retrieval.py works with
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A single retrievable unit of document text.

    Every field exists for a reason:

    chunk_id       → globally unique across all documents in a session
                     format: "{filename}::chunk_{index:04d}"
    raw_text       → original extracted text — sent to Gemini as context
    clean_text     → preprocessed text — used by retrieval methods
    source_document→ filename this chunk came from (multi-doc support)
    page_start     → first page this chunk's text came from (1-indexed)
    page_end       → last page this chunk's text came from (1-indexed)
    word_count     → quick size signal; used in summary stats
    sentence_count → how many sentences in this chunk
    chunk_index    → position in the ordered list of chunks (0-indexed)

    page_start / page_end:
        A chunk can span two pages when overlap carries sentences from
        the bottom of page N into the top of page N+1. Both are recorded
        so the explainer can show "this answer came from pages 4–5".
    """
    chunk_id:        str
    raw_text:        str
    clean_text:      str
    source_document: str
    page_start:      int
    page_end:        int
    word_count:      int
    sentence_count:  int
    chunk_index:     int

    def preview(self, chars: int = 120) -> str:
        """Short preview of raw_text for logging and the explainer UI."""
        text = self.raw_text.replace("\n", " ").strip()
        return text[:chars] + "..." if len(text) > chars else text


# ─────────────────────────────────────────────────────────────────────────────
# Sentence-level representation (internal — not exposed to retrieval.py)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Sentence:
    """
    Internal unit. Carries a sentence plus the page it came from.
    Page mapping survives through chunking so every Chunk knows
    which pages it spans.
    """
    text:        str
    page_number: int


# ─────────────────────────────────────────────────────────────────────────────
# Special Block Detection
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that mark the START of a block that should not be split internally.
# We treat everything until the matching end marker as one atomic unit.
_BLOCK_START_PATTERNS = [
    # Tables inserted by pdf_loader  (kept together — table rows are related)
    re.compile(r"^\[Table \d+\]", re.MULTILINE),
    # Tables on this page header
    re.compile(r"^\[Tables on this page\]", re.MULTILINE),
    # Hyperlinks section
    re.compile(r"^\[Hyperlinks on this page\]", re.MULTILINE),
    # Figure captions — "Figure 3:" or "Fig. 3:"
    re.compile(r"^(?:Figure|Fig\.)\s+\d+[:\.]", re.MULTILINE | re.IGNORECASE),
]

# A line that looks like a section heading (short, ends without period,
# possibly numbered). Used to avoid breaking headings from their content.
_HEADING_PATTERN = re.compile(
    r"^(?:\d+\.?\d*\.?\s+)?[A-Z][A-Za-z\s\-]{3,60}$"
)


def _is_special_block_start(line: str) -> bool:
    """Return True if this line begins a block that should stay together."""
    for pattern in _BLOCK_START_PATTERNS:
        if pattern.match(line.strip()):
            return True
    return False


def _is_heading(line: str) -> bool:
    """Return True if this line looks like a section heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    return bool(_HEADING_PATTERN.match(stripped))


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Merge pages into a flat sentence list with page tracking
# ─────────────────────────────────────────────────────────────────────────────

def _extract_sentences_from_pages(pages: list) -> list[_Sentence]:
    """
    Merge all pages into a flat list of _Sentence objects.
    Each sentence carries the page number it came from.

    Special handling:
        - Table blocks: treated as a single "sentence" so they stay together.
          A table split across chunks is unreadable.
        - Headings: kept attached to the first sentence that follows them.
          A heading orphaned at the end of a chunk is confusing.
        - Empty pages: skipped silently.

    Why raw_text and not clean_text here?
        We build raw sentences here so each _Sentence.text is original prose.
        clean_text is generated AFTER chunking — we run preprocessing.py
        on the assembled chunk text, not on individual sentences.
        This preserves preprocessing context (header/footer detection works
        better on full chunks than single sentences).
    """
    all_sentences: list[_Sentence] = []

    for page in pages:
        if not page.has_text or not page.raw_text.strip():
            continue

        page_num  = page.page_number
        page_text = page.raw_text

        # Split page text into lines to detect special blocks first
        lines = page_text.split("\n")
        buffer: list[str] = []
        in_special_block = False
        pending_heading:  Optional[str] = None

        for line in lines:

            # Detect special block boundaries
            if _is_special_block_start(line):
                # Flush any buffered normal text first
                if buffer:
                    normal_text = " ".join(buffer).strip()
                    if normal_text:
                        for sent in _split_into_sentences(normal_text, page_num, pending_heading):
                            all_sentences.append(sent)
                        pending_heading = None
                    buffer = []
                in_special_block = True
                buffer.append(line)
                continue

            if in_special_block:
                # End of special block: blank line or new heading
                if not line.strip() or _is_heading(line):
                    block_text = "\n".join(buffer).strip()
                    if block_text:
                        all_sentences.append(_Sentence(text=block_text, page_number=page_num))
                    buffer = []
                    in_special_block = False
                    if _is_heading(line):
                        pending_heading = line.strip()
                else:
                    buffer.append(line)
                continue

            # Capture headings to attach to the next sentence
            if _is_heading(line) and not buffer:
                if pending_heading:
                    # Two consecutive headings — emit first as standalone
                    all_sentences.append(_Sentence(text=pending_heading, page_number=page_num))
                pending_heading = line.strip()
                continue

            buffer.append(line)

        # Flush remaining buffer
        if buffer:
            remaining = " ".join(buffer).strip()
            if remaining:
                for sent in _split_into_sentences(remaining, page_num, pending_heading):
                    all_sentences.append(sent)

        # If page ended with a pending heading and no following text
        if pending_heading and (
            not all_sentences or all_sentences[-1].text != pending_heading
        ):
            all_sentences.append(_Sentence(text=pending_heading, page_number=page_num))

    return all_sentences


def _split_into_sentences(
    text: str,
    page_number: int,
    heading_prefix: Optional[str] = None,
) -> list[_Sentence]:
    """
    Split a text block into individual _Sentence objects using NLTK.

    heading_prefix: if set, it is prepended to the first sentence.
    This keeps "3 Attention Mechanisms\nAttention is..." together
    rather than making the heading an orphan.

    Edge cases handled:
        - Very short "sentences" (<= 3 words): merged into the next sentence.
          PDF extraction frequently leaves page numbers, lone labels ("a)", "i.")
          as isolated tokens that would pollute chunk boundaries.
        - Mathematical expressions: NLTK sometimes splits at periods inside
          formulas (e.g. "0.1"). We re-merge sentences that end with a number
          and the next begins with a number or operator.
    """
    raw_sentences = sent_tokenize(text)

    # Re-merge fragments
    merged: list[str] = []
    i = 0
    while i < len(raw_sentences):
        sent = raw_sentences[i].strip()
        if not sent:
            i += 1
            continue

        # Merge very short fragments into the next sentence
        word_count = len(sent.split())
        if word_count <= 3 and i + 1 < len(raw_sentences):
            raw_sentences[i + 1] = sent + " " + raw_sentences[i + 1]
            i += 1
            continue

        # Re-merge math splits: "...= 0." + "1 * d_k..." → keep together
        if (
            merged
            and re.search(r"\d\.$", merged[-1])
            and re.match(r"^\d", sent)
        ):
            merged[-1] = merged[-1] + sent
            i += 1
            continue

        merged.append(sent)
        i += 1

    # Prepend heading to first sentence
    if heading_prefix and merged:
        merged[0] = heading_prefix + "\n" + merged[0]

    return [_Sentence(text=s, page_number=page_number) for s in merged if s.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Group sentences into overlapping chunks
# ─────────────────────────────────────────────────────────────────────────────

def _group_into_chunks(
    sentences:      list[_Sentence],
    source_doc:     str,
    target_sentences: int = 10,
    overlap_sentences: int = 2,
    min_words:      int = 30,
    max_words:      int = 600,
) -> list[tuple[list[_Sentence], int]]:
    """
    Group a flat sentence list into overlapping windows.

    Parameters
    ----------
    target_sentences  : aim for this many sentences per chunk (default 10)
    overlap_sentences : sentences shared between consecutive chunks (default 2)
    min_words         : chunks smaller than this merge forward into the next
    max_words         : chunks larger than this split even mid-window
                        (only happens with very long single sentences, e.g. formulas)

    Returns list of (sentence_list, chunk_index) tuples.

    Overlap mechanics:
        Chunk 0: sentences [0 .. 9]
        Chunk 1: sentences [8 .. 17]   ← sentences 8–9 are the overlap
        Chunk 2: sentences [16 .. 25]  ← sentences 16–17 are the overlap

    Why overlap?
        If a question's answer straddles a boundary — the question is in
        sentence 9 and the answer in sentence 10 — neither chunk alone
        contains the full context. Overlap ensures at least one chunk
        captures both sides of every boundary.

    Why not larger overlap?
        Overlap = target means every sentence appears in two chunks.
        TF-IDF scores become noisy because the same terms inflate IDF.
        2 sentences is the minimum meaningful overlap — enough to carry
        the semantic thread without poisoning the retrieval scores.
    """
    if not sentences:
        return []

    chunks: list[tuple[list[_Sentence], int]] = []
    chunk_index = 0
    i = 0
    step = target_sentences - overlap_sentences  # advance by this each iteration

    while i < len(sentences):
        window = sentences[i : i + target_sentences]

        # Word count check — split oversized windows
        word_count = sum(len(s.text.split()) for s in window)

        if word_count > max_words and len(window) > 1:
            # Binary search for a split point that keeps both halves under max
            mid = len(window) // 2
            # Emit first half now, leave second half for next iteration
            first_half = window[:mid]
            chunks.append((first_half, chunk_index))
            chunk_index += 1
            # Next iteration starts at i + mid - overlap_sentences
            i += max(1, mid - overlap_sentences)
            continue

        # Merge tiny trailing chunks forward — avoids 1-sentence orphan chunks
        if (
            word_count < min_words
            and chunks
            and i + target_sentences >= len(sentences)
        ):
            # Append these sentences to the last chunk instead of making a new one
            prev_sents, prev_idx = chunks[-1]
            chunks[-1] = (prev_sents + window, prev_idx)
            break

        chunks.append((window, chunk_index))
        chunk_index += 1
        i += step

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Build Chunk objects with raw + clean text
# ─────────────────────────────────────────────────────────────────────────────

def _build_chunk(
    sentences:   list[_Sentence],
    chunk_index: int,
    source_doc:  str,
    preprocess_fn,          # callable: str → str  (preprocessing.preprocess_query)
) -> Chunk:
    """
    Assemble a Chunk from a list of _Sentence objects.

    raw_text  : sentences joined with single newlines — natural paragraph flow
    clean_text: raw_text run through preprocess_fn (same pipeline as documents)

    Page range: min and max page numbers across all sentences in the window.
    This correctly handles chunks that span a page boundary.
    """
    raw_text = "\n".join(s.text for s in sentences).strip()
    clean_text = preprocess_fn(raw_text)

    page_numbers = [s.page_number for s in sentences]
    page_start   = min(page_numbers)
    page_end     = max(page_numbers)

    # chunk_id is human-readable and globally unique within a session
    safe_name = re.sub(r"[^\w\-]", "_", source_doc)
    chunk_id  = f"{safe_name}::chunk_{chunk_index:04d}"

    return Chunk(
        chunk_id        = chunk_id,
        raw_text        = raw_text,
        clean_text      = clean_text,
        source_document = source_doc,
        page_start      = page_start,
        page_end        = page_end,
        word_count      = len(raw_text.split()),
        sentence_count  = len(sentences),
        chunk_index     = chunk_index,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def chunk_document(
    document,                           # DocumentResult from pdf_loader
    target_sentences:   int = 10,
    overlap_sentences:  int = 2,
    min_words:          int = 30,
    max_words:          int = 600,
) -> list[Chunk]:
    """
    Convert a preprocessed DocumentResult into a list of Chunk objects.

    Parameters
    ----------
    document          : DocumentResult — must have been run through
                        preprocessing.preprocess_document() so that
                        page.clean_text is available. If clean_text is
                        missing, we fall back to preprocessing raw_text
                        on the fly (slower but safe).
    target_sentences  : aim for this many sentences per chunk (8–12 recommended)
    overlap_sentences : sentences shared between adjacent chunks (1–3 recommended)
    min_words         : chunks below this threshold are merged into the previous
    max_words         : chunks above this threshold are forcibly split

    Returns
    -------
    list[Chunk]
        Ordered list of chunks ready for retrieval.py to consume.
        Chunks are ordered by their position in the original document.

    Raises
    ------
    ValueError  if overlap_sentences >= target_sentences (infinite loop)
    """
    if overlap_sentences >= target_sentences:
        raise ValueError(
            f"overlap_sentences ({overlap_sentences}) must be less than "
            f"target_sentences ({target_sentences}). "
            f"Overlap >= target would cause an infinite loop."
        )

    # Import preprocessing here to avoid circular imports at module level
    # (preprocessing.py imports nothing from chunking.py, but being explicit
    #  about the dependency direction keeps the architecture clear)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocessing import preprocess_query as _preprocess_fn

    source_doc = document.metadata.filename
    pages      = [p for p in document.pages if p.has_text]

    if not pages:
        print(f"[chunking] No pages with text found in '{source_doc}'.")
        return []

    print(f"[chunking] '{source_doc}' — {len(pages)} pages → extracting sentences...")

    # Step 1 — flatten pages to sentences with page tracking
    sentences = _extract_sentences_from_pages(pages)
    print(f"[chunking] Extracted {len(sentences)} sentences.")

    if not sentences:
        print(f"[chunking] No sentences found — returning empty chunk list.")
        return []

    # Step 2 — group sentences into overlapping windows
    grouped = _group_into_chunks(
        sentences,
        source_doc,
        target_sentences  = target_sentences,
        overlap_sentences = overlap_sentences,
        min_words         = min_words,
        max_words         = max_words,
    )

    # Step 3 — build Chunk objects
    chunks: list[Chunk] = []
    for sentence_list, chunk_index in grouped:
        chunk = _build_chunk(
            sentences    = sentence_list,
            chunk_index  = chunk_index,
            source_doc   = source_doc,
            preprocess_fn= _preprocess_fn,
        )
        chunks.append(chunk)

    print(
        f"[chunking] Done — {len(chunks)} chunks | "
        f"target={target_sentences} sentences | "
        f"overlap={overlap_sentences} sentences"
    )
    return chunks


def chunk_multiple_documents(
    documents: list,
    target_sentences:  int = 10,
    overlap_sentences: int = 2,
    min_words:         int = 30,
    max_words:         int = 600,
) -> list[Chunk]:
    """
    Chunk multiple DocumentResult objects.
    Returns a single flat list with all chunks from all documents,
    each carrying its source_document field for traceability.
    """
    all_chunks: list[Chunk] = []
    for doc in documents:
        chunks = chunk_document(
            doc,
            target_sentences  = target_sentences,
            overlap_sentences = overlap_sentences,
            min_words         = min_words,
            max_words         = max_words,
        )
        all_chunks.extend(chunks)

    print(f"\n[chunking] Total across {len(documents)} documents: {len(all_chunks)} chunks.")
    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Educational Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_chunk_summary(chunks: list[Chunk]) -> None:
    """
    Print a detailed educational summary of the chunking results.

    This is what you study on Day 2 — read every line of this output
    and ask yourself: does each chunk look like a coherent unit?
    If chunk 3 starts mid-sentence, something is wrong upstream.
    """
    if not chunks:
        print("[chunking summary] No chunks to summarize.")
        return

    word_counts = [c.word_count for c in chunks]
    sent_counts = [c.sentence_count for c in chunks]

    total_words  = sum(word_counts)
    avg_words    = total_words / len(chunks)
    avg_sents    = sum(sent_counts) / len(chunks)
    min_chunk    = min(chunks, key=lambda c: c.word_count)
    max_chunk    = max(chunks, key=lambda c: c.word_count)

    # Page coverage
    all_pages = set()
    for c in chunks:
        all_pages.update(range(c.page_start, c.page_end + 1))

    print(f"\n{'═' * 60}")
    print(f"  CHUNKING SUMMARY")
    print(f"{'═' * 60}")
    print(f"  Total chunks          : {len(chunks)}")
    print(f"  Total words chunked   : {total_words:,}")
    print(f"  Avg words / chunk     : {avg_words:.0f}")
    print(f"  Avg sentences / chunk : {avg_sents:.1f}")
    print(f"  Pages covered         : {min(all_pages)}–{max(all_pages)}")
    print(f"")
    print(f"  Smallest chunk  → chunk_{min_chunk.chunk_index:04d} "
          f"({min_chunk.word_count} words, {min_chunk.sentence_count} sentences, "
          f"pages {min_chunk.page_start}–{min_chunk.page_end})")
    print(f"  Largest chunk   → chunk_{max_chunk.chunk_index:04d} "
          f"({max_chunk.word_count} words, {max_chunk.sentence_count} sentences, "
          f"pages {max_chunk.page_start}–{max_chunk.page_end})")

    # Word count distribution buckets
    buckets = {"< 100": 0, "100–200": 0, "200–350": 0, "350–500": 0, "> 500": 0}
    for w in word_counts:
        if w < 100:          buckets["< 100"]   += 1
        elif w < 200:        buckets["100–200"]  += 1
        elif w < 350:        buckets["200–350"]  += 1
        elif w <= 500:       buckets["350–500"]  += 1
        else:                buckets["> 500"]    += 1

    print(f"\n  Word count distribution:")
    for label, count in buckets.items():
        bar = "█" * count
        print(f"    {label:>10}  {bar} ({count})")

    # Per-chunk table
    print(f"\n{'─' * 60}")
    print(f"  {'#':>4}  {'Words':>6}  {'Sents':>5}  {'Pages':>8}  Preview")
    print(f"{'─' * 60}")
    for chunk in chunks:
        page_range = (
            f"p.{chunk.page_start}"
            if chunk.page_start == chunk.page_end
            else f"p.{chunk.page_start}–{chunk.page_end}"
        )
        print(
            f"  {chunk.chunk_index:>4}  "
            f"{chunk.word_count:>6}  "
            f"{chunk.sentence_count:>5}  "
            f"{page_range:>8}  "
            f"{chunk.preview(60)}"
        )

    # Deep preview of first 3 chunks
    print(f"\n{'═' * 60}")
    print(f"  FIRST 3 CHUNKS — FULL PREVIEW")
    print(f"{'═' * 60}")
    for chunk in chunks[:3]:
        print(f"\n  ── Chunk {chunk.chunk_index} "
              f"({chunk.word_count} words, pages {chunk.page_start}–{chunk.page_end}) ──")
        print(f"\n  RAW TEXT:")
        print(f"  {chunk.raw_text[:400].replace(chr(10), chr(10) + '  ')}")
        print(f"\n  CLEAN TEXT:")
        print(f"  {chunk.clean_text[:400].replace(chr(10), chr(10) + '  ')}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo  (python src/chunking.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from pdf_loader import load_pdf, PDFNotFoundError
    from preprocessing import preprocess_document

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "data", "uploads", "attention.pdf"
    )

    print(f"\n{'─' * 60}")
    print(f"  SmartDocAI — chunking.py demo")
    print(f"{'─' * 60}\n")

    try:
        # Full pipeline: load → preprocess → chunk
        print("Step 1: Loading PDF...")
        document = load_pdf(pdf_path)

        print("\nStep 2: Preprocessing...")
        document = preprocess_document(document)

        print("\nStep 3: Chunking...")
        chunks = chunk_document(
            document,
            target_sentences  = 10,
            overlap_sentences = 2,
        )

        # Educational summary
        print_chunk_summary(chunks)

        # Overlap verification — show sentences shared between chunk 0 and chunk 1
        if len(chunks) >= 2:
            print(f"\n{'═' * 60}")
            print(f"  OVERLAP VERIFICATION — Chunks 0 and 1")
            print(f"{'═' * 60}")
            sents_0 = chunks[0].raw_text.split("\n")
            sents_1 = chunks[1].raw_text.split("\n")
            shared = [s for s in sents_0 if s in sents_1 and s.strip()]
            if shared:
                print(f"\n  {len(shared)} shared sentence(s) between chunk 0 and chunk 1:")
                for s in shared:
                    print(f"  ↔  {s[:100]}")
            else:
                print("  No exact line overlap detected (sentences may span line boundaries).")

        print(f"\n{'─' * 60}")
        print(f"  Demo complete. {len(chunks)} chunks ready for retrieval.py.")
        print(f"{'─' * 60}\n")

    except PDFNotFoundError:
        print(f"\n  PDF not found: {pdf_path}")
        print("  Place your PDF at data/uploads/attention.pdf")
        print("  Or: python src/chunking.py path/to/your.pdf\n")
    except Exception as e:
        import traceback
        print(f"\n  Error: {e}")
        traceback.print_exc()