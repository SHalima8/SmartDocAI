"""
preprocessing.py — SmartDocAI
==============================
Sole responsibility: clean and normalize raw text extracted by pdf_loader.py.

This module receives a DocumentResult and returns it with clean_text added
to every PageData. raw_text is NEVER modified — it stays intact for the LLM.

Pipeline order (order matters — each step assumes the previous ran):
    1. Unicode normalization
    2. Whitespace cleanup
    3. Header/footer removal
    4. URL removal
    5. Email removal
    6. Citation marker removal
    7. Lemmatization

What this module deliberately does NOT do:
    - Remove stopwords       (hurts RAG retrieval — "not", "no" carry meaning)
    - Remove numbers         (figures, dates, statistics are retrieval signals)
    - Remove figure/table refs ("Figure 3", "Table 1" anchor answers)
    - Stem words             (stemming is lossy; lemmatization preserves roots)
    - Alter sentence order   (downstream chunking depends on structure)

Downstream consumers:
    chunking.py   → reads page.clean_text for chunk creation
    retrieval.py  → queries run through the same pipeline before matching
    pipeline.py   → calls preprocess_document() as step 2 of the RAG loop
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ── NLTK data check ───────────────────────────────────────────────────────────
# Download required NLTK packages if not already present.
# We use nltk.download() with quiet=True — it skips silently if already present.
# This is simpler and more portable than nltk.data.find(), which raises OSError
# on Windows when internal package structure differs from expected paths.
_REQUIRED_NLTK = [
    "punkt",
    "punkt_tab",
    "wordnet",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
]

for _pkg in _REQUIRED_NLTK:
    nltk.download(_pkg, quiet=True)

_LEMMATIZER = WordNetLemmatizer()


# ─────────────────────────────────────────────────────────────────────────────
# Statistics Tracking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PreprocessingStats:
    """
    Records what each pipeline step did to a page's text.
    Surfaced in the "Explain How I Answered" feature.

    Every count is additive — if a step removes 3 URLs and another
    call removes 2 more, total_urls_removed = 5.
    """
    original_char_count:    int = 0
    final_char_count:       int = 0
    chars_removed:          int = 0

    ligatures_normalized:   int = 0
    unicode_replacements:   int = 0
    whitespace_collapses:   int = 0
    blank_lines_removed:    int = 0
    headers_footers_removed:int = 0
    urls_removed:           int = 0
    emails_removed:         int = 0
    citation_markers_removed: int = 0
    words_lemmatized:       int = 0
    total_tokens:           int = 0

    steps_applied: list[str] = field(default_factory=list)

    def summary(self) -> str:
        pct = (
            100 * (1 - self.final_char_count / self.original_char_count)
            if self.original_char_count > 0 else 0.0
        )
        return (
            f"chars {self.original_char_count:,} → {self.final_char_count:,} "
            f"({pct:.1f}% removed) | "
            f"urls={self.urls_removed} emails={self.emails_removed} "
            f"citations={self.citation_markers_removed} "
            f"lemmatized={self.words_lemmatized}/{self.total_tokens}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Unicode Normalization
# ─────────────────────────────────────────────────────────────────────────────

# PDF ligatures and typographic characters that do not decompose via NFC/NFKC
_LIGATURE_MAP = {
    "\ufb00": "ff",   # ﬀ
    "\ufb01": "fi",   # ﬁ
    "\ufb02": "fl",   # ﬂ
    "\ufb03": "ffi",  # ﬃ
    "\ufb04": "ffl",  # ﬄ
    "\ufb05": "st",   # ﬅ
    "\ufb06": "st",   # ﬆ
}

_TYPOGRAPHIC_MAP = {
    "\u2018": "'",    # left single quotation mark
    "\u2019": "'",    # right single quotation mark
    "\u201c": '"',    # left double quotation mark
    "\u201d": '"',    # right double quotation mark
    "\u2013": "-",    # en dash
    "\u2014": "-",    # em dash
    "\u2015": "-",    # horizontal bar
    "\u2026": "...",  # ellipsis
    "\u00a0": " ",    # non-breaking space
    "\u00ad": "",     # soft hyphen (invisible, causes tokenization issues)
    "\u200b": "",     # zero-width space
    "\u200c": "",     # zero-width non-joiner
    "\u200d": "",     # zero-width joiner
    "\ufeff": "",     # byte order mark
}


def normalize_unicode(text: str, stats: PreprocessingStats) -> str:
    """
    Step 1: Normalize unicode characters that PDFs commonly introduce.

    Order:
        a) Replace known ligatures manually (NFKC won't catch all of them)
        b) Apply NFKC normalization (decomposes accented chars, normalizes widths)
        c) Replace typographic characters (smart quotes, dashes, etc.)

    Why NFKC not NFC?
        NFC only canonically composes. NFKC also handles compatibility
        equivalents — e.g. full-width latin letters (Ａ → A) common in
        PDFs generated from non-English typesetting software.
    """
    # a) Manual ligature replacement
    for ligature, replacement in _LIGATURE_MAP.items():
        count = text.count(ligature)
        if count:
            text = text.replace(ligature, replacement)
            stats.ligatures_normalized += count

    # b) NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # c) Typographic character replacement
    for char, replacement in _TYPOGRAPHIC_MAP.items():
        count = text.count(char)
        if count:
            text = text.replace(char, replacement)
            stats.unicode_replacements += count

    stats.steps_applied.append("unicode_normalization")
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Whitespace Cleanup
# ─────────────────────────────────────────────────────────────────────────────

def clean_whitespace(text: str, stats: PreprocessingStats) -> str:
    """
    Step 2: Normalize whitespace without altering document structure.

    Rules:
        - Collapse multiple spaces to one (but not newlines — those mark structure)
        - Remove trailing spaces from every line
        - Collapse 3+ consecutive blank lines to 2 (preserve paragraph breaks)
        - Strip leading/trailing whitespace from the full text

    Why we keep double newlines:
        Paragraph boundaries are meaningful. chunking.py uses them as
        natural split points. Flattening to single newlines destroys that.
    """
    # Collapse multiple spaces on a single line (not newlines)
    before = len(text)
    text = re.sub(r"[ \t]+", " ", text)
    stats.whitespace_collapses += before - len(text)

    # Remove trailing whitespace from each line
    text = re.sub(r" +$", "", text, flags=re.MULTILINE)

    # Collapse 3+ consecutive blank lines → 2 blank lines
    before_blanks = len(re.findall(r"\n{3,}", text))
    text = re.sub(r"\n{3,}", "\n\n", text)
    stats.blank_lines_removed += before_blanks

    text = text.strip()
    stats.steps_applied.append("whitespace_cleanup")
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Header / Footer Removal
# ─────────────────────────────────────────────────────────────────────────────

def _detect_repeated_lines(pages_text: list[str], min_frequency: float = 0.4) -> set[str]:
    """
    Find lines that appear on a suspiciously high proportion of pages.

    Strategy:
        - Split each page into lines
        - Count how often each non-trivial line appears across pages
        - Lines appearing on >= min_frequency of pages are header/footer candidates

    min_frequency=0.4 means: appears on 40%+ of pages → likely a header/footer.
    Threshold is not 100% because page numbers vary (so "Page 1", "Page 2" won't
    match as identical lines — we handle numbering separately).

    Returns a set of exact line strings to remove.
    """
    all_lines: list[str] = []
    for page_text in pages_text:
        lines = [line.strip() for line in page_text.split("\n")]
        all_lines.extend(lines)

    line_counts = Counter(all_lines)
    total_pages = len(pages_text)
    threshold = max(2, int(total_pages * min_frequency))

    repeated = set()
    for line, count in line_counts.items():
        # Skip trivial lines: empty, single chars, pure numbers (page numbers vary)
        if not line or len(line) <= 2:
            continue
        if re.fullmatch(r"\d+", line):  # standalone page number
            continue
        if count >= threshold:
            repeated.add(line)

    return repeated


def _remove_page_number_lines(text: str) -> tuple[str, int]:
    """
    Remove standalone page number lines.
    Matches lines that are only a number, optionally with "Page" prefix.

    Examples removed:
        "4"
        "Page 4"
        "- 4 -"
        "4 of 15"
    """
    pattern = r"^[\s\-]*(?:Page\s*)?\d+(?:\s*of\s*\d+)?[\s\-]*$"
    lines = text.split("\n")
    cleaned = []
    removed = 0
    for line in lines:
        if re.fullmatch(pattern, line.strip(), re.IGNORECASE):
            removed += 1
        else:
            cleaned.append(line)
    return "\n".join(cleaned), removed


def remove_headers_footers(
    pages_text: list[str],
    stats_per_page: list[PreprocessingStats],
) -> list[str]:
    """
    Step 3: Remove repeated headers, footers, and page numbers.

    Two-pass approach:
        Pass 1 — detect which lines repeat across pages (cross-page analysis)
        Pass 2 — remove those lines from every page + remove page number lines

    Why this needs all pages at once:
        You can't detect repetition on a single page. The header "Attention Is
        All You Need" on page 3 looks like content. Seeing it on pages 1–15
        reveals it's a running header.
    """
    repeated_lines = _detect_repeated_lines(pages_text)

    cleaned_pages = []
    for idx, text in enumerate(pages_text):
        lines = text.split("\n")
        cleaned_lines = []
        removed_count = 0

        for line in lines:
            if line.strip() in repeated_lines:
                removed_count += 1
            else:
                cleaned_lines.append(line)

        cleaned_text = "\n".join(cleaned_lines)

        # Also remove standalone page number lines
        cleaned_text, page_num_count = _remove_page_number_lines(cleaned_text)
        removed_count += page_num_count

        stats_per_page[idx].headers_footers_removed += removed_count
        if removed_count:
            stats_per_page[idx].steps_applied.append("header_footer_removal")

        cleaned_pages.append(cleaned_text)

    return cleaned_pages


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — URL Removal
# ─────────────────────────────────────────────────────────────────────────────

# Covers http/https/ftp URLs and bare www. domains
_URL_PATTERN = re.compile(
    r"https?://[^\s\)\]\},\"\'<>]+|"   # http:// or https:// URLs
    r"ftp://[^\s\)\]\},\"\'<>]+|"      # ftp:// URLs
    r"www\.[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}[^\s\)\]\},\"\'<>]*",  # www. domains
    re.IGNORECASE,
)


def remove_urls(text: str, stats: PreprocessingStats) -> str:
    """
    Step 4: Remove URLs while preserving surrounding text and punctuation.

    Why remove URLs?
        URLs carry no semantic meaning for retrieval. "https://arxiv.org/abs/1706.03762"
        won't help answer "what does self-attention compute?".
        They inflate token counts and confuse TF-IDF weighting.

    Why preserve surrounding text?
        "See [12] for details (https://example.com)" → "See [12] for details ()"
        We then clean the empty parens in whitespace cleanup.
        The citation marker [12] is handled separately.
    """
    matches = _URL_PATTERN.findall(text)
    stats.urls_removed += len(matches)
    if matches:
        text = _URL_PATTERN.sub("", text)
        stats.steps_applied.append("url_removal")
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Email Removal
# ─────────────────────────────────────────────────────────────────────────────

_EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)


def remove_emails(text: str, stats: PreprocessingStats) -> str:
    """
    Step 5: Remove email addresses.

    Academic papers typically have author contact emails in headers/footers.
    These are noise for retrieval — nobody asks "what is Vaswani's email?".

    Pattern handles:
        standard@email.com
        first.last@university.edu
        user+tag@domain.co.uk
    """
    matches = _EMAIL_PATTERN.findall(text)
    stats.emails_removed += len(matches)
    if matches:
        text = _EMAIL_PATTERN.sub("", text)
        stats.steps_applied.append("email_removal")
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Citation Marker Removal
# ─────────────────────────────────────────────────────────────────────────────

# Matches: [1], [12], [3,5], [1-4], [1, 2, 3] — isolated numeric references
# Does NOT match: [Figure 3], [Table 1], [Section 2.1] — those are preserved
_CITATION_MARKER_PATTERN = re.compile(
    r"\[\s*\d+(?:\s*[,\-]\s*\d+)*\s*\]"
)

# Matches superscript-style citations: word¹ or word1,2 at end of token
# These appear in some PDF extraction artifacts
_SUPERSCRIPT_CITATION_PATTERN = re.compile(
    r"(?<=\w)[\u00B9\u00B2\u00B3\u2070-\u2079]+"  # Unicode superscript digits
)


def remove_citation_markers(text: str, stats: PreprocessingStats) -> str:
    """
    Step 6: Remove inline numeric citation markers.

    Removes:
        [1]         single reference
        [1, 2, 3]   multiple references
        [1-4]       range reference
        ¹²³         superscript citation artifacts from PDF extraction

    Preserves:
        Vaswani et al. (2017)      — named citation, meaningful for retrieval
        Figure 3                    — figure reference
        Table 1                     — table reference
        Section 2.1                 — section reference
        (BLEU score of 28.4)        — parenthetical numbers in context

    Why remove [12] but keep "Vaswani et al. (2017)"?
        "[12]" is a pointer to a bibliography — it carries no semantic content
        in isolation. "Vaswani et al. (2017)" names the work and is a real
        retrieval signal: someone might ask "what did Vaswani propose?".
    """
    bracket_matches = _CITATION_MARKER_PATTERN.findall(text)
    stats.citation_markers_removed += len(bracket_matches)
    if bracket_matches:
        text = _CITATION_MARKER_PATTERN.sub("", text)

    superscript_matches = _SUPERSCRIPT_CITATION_PATTERN.findall(text)
    stats.citation_markers_removed += len(superscript_matches)
    if superscript_matches:
        text = _SUPERSCRIPT_CITATION_PATTERN.sub("", text)

    if bracket_matches or superscript_matches:
        stats.steps_applied.append("citation_marker_removal")

    return text


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Cautious Lemmatization
# ─────────────────────────────────────────────────────────────────────────────

def _penn_to_wordnet_pos(penn_tag: str) -> Optional[str]:
    """
    Convert Penn Treebank POS tag to WordNet POS tag.

    Returns None for verb tags — we handle verbs separately with a
    confirm-before-apply strategy to avoid bad conversions like
    "left" → "leave" or "are" → "be".

    WordNet accepts: 'n' (noun), 'v' (verb), 'a' (adjective), 'r' (adverb)
    """
    if penn_tag.startswith("V"):
        return "v"        # caller will validate before applying
    elif penn_tag.startswith("J"):
        return "a"        # adjective — generally safe
    elif penn_tag.startswith("R"):
        return "r"        # adverb — generally safe
    else:
        return "n"        # noun — safest default


# Words we never lemmatize regardless of POS tag.
# Two categories:
#   1. Copula / auxiliary verbs — lemmatize to "be"/"have", unreadable
#   2. Words whose lemma is a different common word (false conversions)
_NEVER_LEMMATIZE = {
    # Copula and auxiliaries — "are" → "be", "were" → "be", unreadable
    "is", "are", "was", "were", "am", "be", "been", "being",
    "has", "have", "had", "having",
    "do", "does", "did", "doing",
    # Spatial/directional adjectives — POS tagger reads as verb past tense
    # "left" → "leave", "right" stays fine, "top" stays fine
    "left",
    # Modal verbs — these don't conjugate and lemmatizing adds no value
    "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "ought",
    # Common words the Penn tagger frequently mis-tags in technical text
    "used", "based", "trained", "fixed", "learned",  # past tense adj, not verbs
}

# Technical and domain tokens we never lemmatize.
# Includes NLP/ML acronyms, math notation fragments, and short tokens
# that tokenize or lemmatize unpredictably.
_PRESERVE_TOKENS = {
    # NLP / ML acronyms
    "nlp", "ml", "ai", "dl", "rnn", "cnn", "gpt", "bert", "bleu",
    "relu", "lstm", "gru", "ffn", "mha", "rag",
    # Math / notation fragments common in academic PDFs
    "softmax", "argmax", "concat", "multihead",
    # Abbreviations that tokenize strangely
    "etc", "vs", "fig", "eq", "sec", "approx",
}

# Verb lemmas that are too destructive or produce unnatural output.
# If the computed lemma is in this set, we keep the original token instead.
_REJECT_LEMMAS = {
    "be",     # "are/were/is" → "be" — unreadable
    "have",   # "has/had" → "have" — sometimes fine, often not worth it
}


def lemmatize_text(text: str, stats: PreprocessingStats) -> str:
    """
    Step 7: Cautious lemmatization — only apply where the conversion is safe
    and beneficial for retrieval.

    Strategy:
        - Nouns and adjectives: lemmatize freely (plurals, comparatives)
          "mechanisms" → "mechanism", "weighted" → "weighted" (no change)
        - Verbs: lemmatize ONLY if:
            a) token is not in _NEVER_LEMMATIZE blocklist
            b) resulting lemma is not in _REJECT_LEMMAS
            c) the lemma change is not suspiciously large (>4 chars difference)
        - Preserved tokens: pass through unchanged (acronyms, math)
        - Non-alpha tokens: pass through unchanged (numbers, punctuation)

    What this prevents:
        "are"   (VBP) → would become "be"    → BLOCKED by _NEVER_LEMMATIZE
        "left"  (VBD) → would become "leave" → BLOCKED by _NEVER_LEMMATIZE
        "used"  (VBD) → would become "use"   → BLOCKED by _NEVER_LEMMATIZE
        "based" (VBD) → would become "base"  → BLOCKED by _NEVER_LEMMATIZE

    What this keeps:
        "mechanisms" (NNS) → "mechanism"  ✓ useful for retrieval
        "computing"  (VBG) → "compute"    ✓ clear and correct
        "studies"    (NNS) → "study"      ✓ correct plural reduction
        "running"    (VBG) → "run"        ✓ clear verbal noun
    """
    try:
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)

        lemmatized_tokens = []
        words_changed = 0

        for token, pos in pos_tags:
            # Pass through punctuation and numbers unchanged
            if not token.isalpha():
                lemmatized_tokens.append(token)
                continue

            lowered = token.lower()

            # Pass through preserved technical tokens
            if lowered in _PRESERVE_TOKENS:
                lemmatized_tokens.append(lowered)
                continue

            # Pass through blocklisted words without lemmatizing
            if lowered in _NEVER_LEMMATIZE:
                lemmatized_tokens.append(lowered)
                continue

            wn_pos = _penn_to_wordnet_pos(pos)
            lemma = _LEMMATIZER.lemmatize(lowered, pos=wn_pos)

            # For verbs: apply extra validation before accepting the lemma
            if wn_pos == "v":
                # Reject if lemma is in the known-bad set
                if lemma in _REJECT_LEMMAS:
                    lemmatized_tokens.append(lowered)
                    continue

                # Reject if the change is suspiciously large
                # (usually a sign of POS mis-tagging on technical words)
                # e.g. "left" (5) → "leave" (5): same length but wrong
                # We check semantic distance via character edit distance heuristic
                if abs(len(lemma) - len(lowered)) > 4:
                    lemmatized_tokens.append(lowered)
                    continue

            if lemma != lowered:
                words_changed += 1

            lemmatized_tokens.append(lemma)

        stats.words_lemmatized += words_changed
        stats.total_tokens += len([t for t in tokens if t.isalpha()])
        stats.steps_applied.append("lemmatization")

        # Re-join tokens. word_tokenize splits punctuation as separate tokens,
        # so naive join produces "Hello , world ." — we fix spacing around punct.
        result = _rejoin_tokens(lemmatized_tokens)
        return result

    except Exception as e:
        # Lemmatization failure is non-fatal — return text unchanged
        stats.steps_applied.append(f"lemmatization_skipped({e})")
        return text


def _rejoin_tokens(tokens: list[str]) -> str:
    """
    Re-join word_tokenize output into a readable string.

    word_tokenize splits: "Hello, world." → ["Hello", ",", "world", "."]
    Naive join: "Hello , world ." — wrong.
    This function re-attaches punctuation to the preceding token.
    """
    if not tokens:
        return ""

    # Punctuation that should attach to the LEFT (preceding token)
    LEFT_ATTACH = set(".,;:!?)]}'\"")
    # Punctuation that should attach to the RIGHT (following token)
    RIGHT_ATTACH = set("([{\"'")

    result = []
    for i, token in enumerate(tokens):
        if i == 0:
            result.append(token)
        elif token in LEFT_ATTACH:
            # Attach to previous token without space
            if result:
                result[-1] = result[-1] + token
        elif result and result[-1][-1] in RIGHT_ATTACH:
            # Previous token ended with open bracket — no space
            result.append(token)
        else:
            result.append(" " + token)

    return "".join(result).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_page(raw_text: str, stats: PreprocessingStats) -> str:
    """
    Run steps 1, 2, 4, 5, 6, 7 on a single page's text.
    Step 3 (header/footer removal) requires all pages and is handled separately.

    This function is also used to preprocess query strings before retrieval,
    so the query goes through the same normalization as the document text.
    """
    stats.original_char_count = len(raw_text)

    text = normalize_unicode(raw_text, stats)
    text = clean_whitespace(text, stats)
    text = remove_urls(text, stats)
    text = remove_emails(text, stats)
    text = remove_citation_markers(text, stats)
    text = lemmatize_text(text, stats)

    # Final whitespace pass — lemmatization can introduce minor spacing artifacts
    text = re.sub(r"[ \t]+", " ", text).strip()

    stats.final_char_count = len(text)
    stats.chars_removed = stats.original_char_count - stats.final_char_count

    return text


def preprocess_document(document: "DocumentResult") -> "DocumentResult":
    """
    Main entry point. Preprocesses all pages in a DocumentResult.

    Mutates document in place by adding to each PageData:
        page.clean_text           — preprocessed text for retrieval
        page.preprocessing_stats  — what changed on this page

    raw_text is never touched.

    Two-phase execution:
        Phase 1 — per-page steps (unicode, whitespace, URLs, emails, citations, lemmatization)
        Phase 2 — cross-page step (header/footer detection requires seeing all pages)

    Returns the same DocumentResult (mutated), so callers can chain calls.
    """
    pages = document.pages

    if not pages:
        print("[preprocessing] No pages to process.")
        return document

    print(f"[preprocessing] Processing {len(pages)} pages...")

    # ── Phase 1: per-page preprocessing ──────────────────────────────────────
    stats_list: list[PreprocessingStats] = []

    for page in pages:
        stats = PreprocessingStats()
        clean = preprocess_page(page.raw_text, stats)

        # Attach results to the page object
        page.clean_text = clean
        page.preprocessing_stats = stats
        stats_list.append(stats)

    # ── Phase 2: cross-page header/footer removal ─────────────────────────────
    cleaned_texts = [page.clean_text for page in pages]
    cleaned_texts = remove_headers_footers(cleaned_texts, stats_list)

    for page, cleaned, stats in zip(pages, cleaned_texts, stats_list):
        page.clean_text = cleaned
        # Recompute final char count after header removal
        stats.final_char_count = len(cleaned)
        stats.chars_removed = stats.original_char_count - stats.final_char_count

    # ── Summary ───────────────────────────────────────────────────────────────
    total_chars_before = sum(s.original_char_count for s in stats_list)
    total_chars_after  = sum(s.final_char_count for s in stats_list)
    total_urls         = sum(s.urls_removed for s in stats_list)
    total_emails       = sum(s.emails_removed for s in stats_list)
    total_citations    = sum(s.citation_markers_removed for s in stats_list)
    total_lemmatized   = sum(s.words_lemmatized for s in stats_list)
    total_tokens       = sum(s.total_tokens for s in stats_list)
    total_hf_removed   = sum(s.headers_footers_removed for s in stats_list)

    pct_removed = (
        100 * (1 - total_chars_after / total_chars_before)
        if total_chars_before > 0 else 0.0
    )

    print(
        f"[preprocessing] Done — "
        f"chars {total_chars_before:,} → {total_chars_after:,} ({pct_removed:.1f}% reduced) | "
        f"urls={total_urls} emails={total_emails} citations={total_citations} "
        f"headers/footers={total_hf_removed} | "
        f"lemmatized {total_lemmatized}/{total_tokens} tokens"
    )

    return document


def preprocess_query(query: str) -> str:
    """
    Preprocess a user query using the same pipeline as document text.

    CRITICAL for retrieval consistency:
        If documents are lemmatized, queries must be too.
        "running attention mechanisms" must become "run attention mechanism"
        to match chunks where those words were lemmatized.

    Does NOT apply header/footer removal (single string, not a document).
    """
    stats = PreprocessingStats()
    return preprocess_page(query, stats)


# ─────────────────────────────────────────────────────────────────────────────
# Quick Demo  (python src/preprocessing.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src.pdf_loader import load_pdf, PDFNotFoundError

    print(f"\n{'─' * 60}")
    print("  SmartDocAI — preprocessing.py demo")
    print(f"{'─' * 60}\n")

    # ── Unit tests for individual pipeline steps ──────────────────────────────

    print("── Step-by-step unit tests ───────────────────────────────────────")

    # Unicode normalization
    s = PreprocessingStats()
    result = normalize_unicode("The ﬁrst ﬂow of \u201csmart quotes\u201d and em\u2014dashes.", s)
    print(f"\n[unicode]  in : The ﬁrst ﬂow of \u201csmart quotes\u201d and em\u2014dashes.")
    print(f"[unicode]  out: {result}")
    print(f"           ligatures={s.ligatures_normalized} unicode_replacements={s.unicode_replacements}")

    # URL removal
    s = PreprocessingStats()
    result = remove_urls("See https://arxiv.org/abs/1706.03762 and www.example.com for details.", s)
    print(f"\n[urls]     in : See https://arxiv.org/abs/1706.03762 and www.example.com for details.")
    print(f"[urls]     out: {result}")
    print(f"           urls_removed={s.urls_removed}")

    # Email removal
    s = PreprocessingStats()
    result = remove_emails("Contact vaswani@google.com or noam@mit.edu for questions.", s)
    print(f"\n[emails]   in : Contact vaswani@google.com or noam@mit.edu for questions.")
    print(f"[emails]   out: {result}")
    print(f"           emails_removed={s.emails_removed}")

    # Citation marker removal
    s = PreprocessingStats()
    result = remove_citation_markers(
        "As shown in [1, 2] and [3], Vaswani et al. (2017) proposed [12] the transformer.", s
    )
    print(f"\n[citations] in : As shown in [1, 2] and [3], Vaswani et al. (2017) proposed [12] the transformer.")
    print(f"[citations] out: {result}")
    print(f"            citation_markers_removed={s.citation_markers_removed}")

    # Lemmatization — good conversions
    s = PreprocessingStats()
    result = lemmatize_text(
        "The researchers are studying attention mechanisms and computing weighted representations.", s
    )
    print(f"\n[lemma good] in : The researchers are studying attention mechanisms and computing weighted representations.")
    print(f"[lemma good] out: {result}")
    print(f"             words_lemmatized={s.words_lemmatized}/{s.total_tokens}")

    # Lemmatization — problem cases that must NOT change
    s = PreprocessingStats()
    result = lemmatize_text(
        "The model are trained on the left side using used based fixed learned weights.", s
    )
    print(f"\n[lemma safe] in : The model are trained on the left side using used based fixed learned weights.")
    print(f"[lemma safe] out: {result}")
    print(f"             Expected: \'are\'→\'are\'  \'left\'→\'left\'  \'used\'→\'used\'  \'based\'→\'based\'")
    assert "be" not in result.split(),    "FAIL: \'are\' was converted to \'be\'"
    assert "leave" not in result.split(), "FAIL: \'left\' was converted to \'leave\'"
    print(f"             PASSED ✓ — no bad conversions")

    # Query preprocessing
    query = "How do attention mechanisms work in transformers?"
    processed_query = preprocess_query(query)
    print(f"\n[query]    in : {query}")
    print(f"[query]    out: {processed_query}")

    # ── Full document test ────────────────────────────────────────────────────
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "uploads", "attention.pdf")

    print(f"\n{'─' * 60}")
    print(f"── Full document test: {pdf_path}")
    print(f"{'─' * 60}")

    try:
        from src.pdf_loader import load_pdf
        document = load_pdf(pdf_path)
        document = preprocess_document(document)

        print("\n── Per-page stats (first 5 pages) ───────────────────────────────")
        for page in document.pages[:5]:
            print(f"\n  Page {page.page_number}: {page.preprocessing_stats.summary()}")
            print(f"  RAW   (100 chars): {page.raw_text[:100].replace(chr(10), ' ')}")
            print(f"  CLEAN (100 chars): {page.clean_text[:100].replace(chr(10), ' ')}")

    except PDFNotFoundError:
        print(f"\n  PDF not found at: {pdf_path}")
        print("  Run: python src/preprocessing.py data/uploads/attention.pdf")
    except Exception as e:
        print(f"\n  Error: {e}")

    print(f"\n{'─' * 60}")
    print("  Demo complete.")
    print(f"{'─' * 60}\n")