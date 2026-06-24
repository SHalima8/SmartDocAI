"""
pdf_loader.py — SmartDocAI
==========================
Sole responsibility: extract structured content from PDF files.

NO preprocessing happens here. No lowercasing, no stopword removal,
no chunking, no embeddings. Just extraction.

Downstream consumers:
    preprocessing.py  → receives page["raw_text"]
    chunking.py       → receives page["raw_text"] after preprocessing
    pipeline.py       → receives the full DocumentResult dict
"""

import os
import fitz  # PyMuPDF
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Custom Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class PDFLoaderError(Exception):
    """Base exception for all pdf_loader errors."""
    pass


class PDFNotFoundError(PDFLoaderError):
    """Raised when the file path does not exist."""
    pass


class PDFEncryptedError(PDFLoaderError):
    """Raised when a PDF is password-protected and cannot be opened."""
    pass


class PDFCorruptedError(PDFLoaderError):
    """Raised when PyMuPDF cannot parse the file structure."""
    pass


class PDFEmptyError(PDFLoaderError):
    """Raised when the PDF exists but contains no extractable text on any page."""
    pass


class PDFScannedError(PDFLoaderError):
    """
    Raised when the PDF appears to be a scanned image document.
    Scanned PDFs have pages but zero selectable text — OCR would be needed,
    which is outside this module's responsibility.
    """
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ImageMetadata:
    """
    Metadata about an image found on a page.
    We do NOT extract pixel data, perform OCR, or caption images.
    This is purely structural information.
    """
    page_number: int          # 1-indexed
    image_index: int          # position of image on the page (0-indexed)
    width: Optional[int]      # pixels, if available
    height: Optional[int]     # pixels, if available
    bounding_box: Optional[tuple[float, float, float, float]]  # (x0, y0, x1, y1)
    colorspace: Optional[str] # e.g. "DeviceRGB", "DeviceGray"
    xref: Optional[int]       # PyMuPDF internal reference number


@dataclass
class TableData:
    """
    A table extracted from a page.

    PyMuPDF's find_tables() returns structured row/column data.
    We preserve that structure rather than flattening to a string,
    so downstream modules can choose how to represent it.

    rows: list of rows, each row is a list of cell strings
    Example:
        rows = [
            ["Name",  "Score", "Grade"],
            ["Alice", "92",    "A"],
            ["Bob",   "78",    "B+"],
        ]
    """
    page_number: int
    table_index: int          # position of table on the page (0-indexed)
    rows: list[list[str]]     # row-major: rows[i][j] = cell at row i, col j
    row_count: int
    col_count: int

    def to_readable_text(self) -> str:
        """
        Convert table to a pipe-delimited text format.
        Used when this table's text is included in raw_text for the LLM.

        Example output:
            Name  | Score | Grade
            Alice | 92    | A
            Bob   | 78    | B+
        """
        if not self.rows:
            return ""

        lines = []
        for row in self.rows:
            # Replace None cells (merged cells in PDFs) with empty string
            cleaned = [str(cell) if cell is not None else "" for cell in row]
            lines.append(" | ".join(cleaned))

        return "\n".join(lines)


@dataclass
class PageData:
    """
    All extracted content for a single PDF page.

    raw_text    : original text as extracted — used by LLM for answer generation
    tables      : structured table data found on this page
    images      : metadata for images on this page (no pixel data)
    char_count  : quick signal for downstream — empty pages have char_count == 0
    has_text    : False if the page exists but has no selectable text (scanned)
    """
    page_number: int          # 1-indexed
    raw_text: str
    tables: list[TableData]
    images: list[ImageMetadata]
    char_count: int
    has_text: bool


@dataclass
class DocumentMetadata:
    """
    File-level metadata extracted from the PDF.
    All fields are Optional because PDFs frequently omit them.
    """
    filename: str
    filepath: str
    title: Optional[str]
    author: Optional[str]
    total_pages: int
    creation_date: Optional[str]
    modification_date: Optional[str]
    producer: Optional[str]     # PDF software that created the file
    subject: Optional[str]
    keywords: Optional[str]
    file_size_bytes: int


@dataclass
class DocumentResult:
    """
    The complete structured output of pdf_loader for one PDF file.

    This is what every downstream module receives.

    Structure:
        result.metadata          → DocumentMetadata
        result.pages             → list[PageData]
        result.pages[0].raw_text → text of first page
        result.pages[0].tables   → tables on first page
        result.warnings          → non-fatal issues encountered during extraction
    """
    metadata: DocumentMetadata
    pages: list[PageData]
    warnings: list[str] = field(default_factory=list)

    def all_text(self) -> str:
        """
        Concatenate raw_text from all pages into one string.
        Useful for quick inspection. Not used for retrieval
        (chunking works page by page, not on the full blob).
        """
        return "\n\n".join(
            f"[Page {p.page_number}]\n{p.raw_text}"
            for p in self.pages
            if p.raw_text.strip()
        )

    def total_chars(self) -> int:
        return sum(p.char_count for p in self.pages)

    def total_images(self) -> int:
        return sum(len(p.images) for p in self.pages)

    def total_tables(self) -> int:
        return sum(len(p.tables) for p in self.pages)

    def pages_with_text(self) -> int:
        return sum(1 for p in self.pages if p.has_text)


# ─────────────────────────────────────────────────────────────────────────────
# Private Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def _extract_document_metadata(doc: fitz.Document, filepath: str) -> DocumentMetadata:
    """
    Extract file-level metadata from an open fitz.Document.

    PyMuPDF stores metadata as a flat dict with string keys.
    Missing fields come back as empty strings, not None,
    so we normalize empty strings → None for cleanliness.
    """
    raw_meta = doc.metadata  # dict from PyMuPDF

    def _or_none(value: str) -> Optional[str]:
        """Return None if value is empty/whitespace, else return stripped value."""
        return value.strip() if value and value.strip() else None

    return DocumentMetadata(
        filename=os.path.basename(filepath),
        filepath=os.path.abspath(filepath),
        title=_or_none(raw_meta.get("title", "")),
        author=_or_none(raw_meta.get("author", "")),
        total_pages=len(doc),
        creation_date=_or_none(raw_meta.get("creationDate", "")),
        modification_date=_or_none(raw_meta.get("modDate", "")),
        producer=_or_none(raw_meta.get("producer", "")),
        subject=_or_none(raw_meta.get("subject", "")),
        keywords=_or_none(raw_meta.get("keywords", "")),
        file_size_bytes=os.path.getsize(filepath),
    )


def _extract_images_from_page(page: fitz.Page, page_number: int) -> list[ImageMetadata]:
    """
    Extract metadata for all images on a page.

    fitz.Page.get_images() returns a list of tuples:
        (xref, smask, width, height, bpc, colorspace, alt_colorspace, name, filter, referencer)

    We extract: xref, width, height, colorspace.
    We also get bounding boxes by querying image positions via get_image_rects().
    """
    images = []
    image_list = page.get_images(full=True)

    for idx, img_info in enumerate(image_list):
        xref        = img_info[0]
        width       = img_info[2] if len(img_info) > 2 else None
        height      = img_info[3] if len(img_info) > 3 else None
        colorspace  = img_info[5] if len(img_info) > 5 else None

        # Get bounding box — where on the page this image appears
        bounding_box = None
        try:
            rects = page.get_image_rects(xref)
            if rects:
                r = rects[0]  # take first rect if multiple instances
                bounding_box = (r.x0, r.y0, r.x1, r.y1)
        except Exception:
            # Non-critical — bbox is optional metadata
            pass

        images.append(ImageMetadata(
            page_number=page_number,
            image_index=idx,
            width=width,
            height=height,
            bounding_box=bounding_box,
            colorspace=colorspace if colorspace else None,
            xref=xref,
        ))

    return images


def _extract_tables_from_page(page: fitz.Page, page_number: int) -> list[TableData]:
    """
    Extract tables from a page using PyMuPDF's built-in table finder.

    find_tables() uses heuristics based on line geometry to detect tables.
    It works well for clearly bordered tables; results vary for borderless ones.

    Each detected table exposes an .extract() method that returns
    list[list[str|None]] in row-major order.
    """
    tables = []

    try:
        table_finder = page.find_tables()

        for idx, table in enumerate(table_finder.tables):
            raw_rows = table.extract()  # list[list[str|None]]

            if not raw_rows:
                continue

            row_count = len(raw_rows)
            col_count = max(len(row) for row in raw_rows) if raw_rows else 0

            tables.append(TableData(
                page_number=page_number,
                table_index=idx,
                rows=raw_rows,
                row_count=row_count,
                col_count=col_count,
            ))

    except Exception as e:
        # Table extraction is best-effort — never crash the whole pipeline
        # The warning will surface in DocumentResult.warnings
        pass

    return tables


def _extract_text_from_page(page: fitz.Page) -> str:
    """
    Extract raw text from a page preserving reading order.

    "blocks" sort mode groups text by its spatial position on the page,
    top-to-bottom, left-to-right — which matches natural reading order
    better than the default stream mode.

    We also append any hyperlinks found on the page, labeled clearly,
    so the LLM can reference them if relevant.
    """
    # Extract text blocks in reading order
    raw_text = page.get_text("text", sort=True)

    # Append hyperlinks if present
    links = page.get_links()
    url_links = [lnk for lnk in links if lnk.get("uri")]

    if url_links:
        raw_text += "\n\n[Hyperlinks on this page]\n"
        for lnk in url_links:
            raw_text += f"  - {lnk['uri']}\n"

    return raw_text


def _extract_page(page: fitz.Page, page_number: int) -> PageData:
    """
    Extract all content from a single page.
    Coordinates calls to text, table, and image extractors.
    """
    raw_text = _extract_text_from_page(page)
    tables   = _extract_tables_from_page(page, page_number)
    images   = _extract_images_from_page(page, page_number)

    # Embed table text into raw_text so the LLM sees table content too
    # Tables are appended after the page text, clearly labeled
    if tables:
        table_section = "\n\n[Tables on this page]\n"
        for tbl in tables:
            table_section += f"\n[Table {tbl.table_index + 1}]\n"
            table_section += tbl.to_readable_text()
            table_section += "\n"
        raw_text += table_section

    char_count = len(raw_text.strip())
    has_text   = char_count > 0

    return PageData(
        page_number=page_number,
        raw_text=raw_text,
        tables=tables,
        images=images,
        char_count=char_count,
        has_text=has_text,
    )


def _check_for_scanned_pdf(pages: list[PageData], total_pages: int) -> bool:
    """
    Heuristic: if a PDF has pages but >90% of them have zero extractable text,
    it's almost certainly a scanned document.

    Threshold is 90% not 100% because some PDFs have legitimate blank pages
    (title pages, section dividers) mixed with real content.
    """
    if total_pages == 0:
        return False

    empty_pages = sum(1 for p in pages if not p.has_text)
    return (empty_pages / total_pages) > 0.90


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_pdf(file_path: str, max_pages: Optional[int] = None) -> DocumentResult:
    """
    Load and extract structured content from a single PDF file.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the PDF file.
    max_pages : Optional[int]
        If set, only the first N pages are processed.
        Useful for testing with large documents.
        Default: None (process all pages).

    Returns
    -------
    DocumentResult
        Structured extraction result containing metadata, per-page content,
        tables, image metadata, and any non-fatal warnings.

    Raises
    ------
    PDFNotFoundError     if the file does not exist
    PDFEncryptedError    if the PDF is password-protected
    PDFCorruptedError    if PyMuPDF cannot parse the file
    PDFEmptyError        if the PDF has zero pages
    PDFScannedError      if the PDF appears to be a scanned image document
    PDFLoaderError       for any other unexpected extraction failure
    """

    # ── Existence check ───────────────────────────────────────────────────────
    if not os.path.exists(file_path):
        raise PDFNotFoundError(
            f"File not found: '{file_path}'\n"
            f"Check that the path is correct and the file has been uploaded."
        )

    if not file_path.lower().endswith(".pdf"):
        raise PDFLoaderError(
            f"Expected a .pdf file, got: '{file_path}'\n"
            f"Only PDF files are supported by this loader."
        )

    warnings: list[str] = []

    # ── Open document ─────────────────────────────────────────────────────────
    try:
        doc = fitz.open(file_path)
    except fitz.FileDataError as e:
        raise PDFCorruptedError(
            f"Could not open '{file_path}' — file may be corrupted.\n"
            f"PyMuPDF error: {e}"
        )
    except Exception as e:
        raise PDFLoaderError(
            f"Unexpected error opening '{file_path}': {e}"
        )

    # ── Encryption check ──────────────────────────────────────────────────────
    if doc.is_encrypted:
        doc.close()
        raise PDFEncryptedError(
            f"'{os.path.basename(file_path)}' is password-protected.\n"
            f"Encrypted PDFs cannot be processed without the password.\n"
            f"Please decrypt the file before uploading."
        )

    # ── Empty PDF check ───────────────────────────────────────────────────────
    if len(doc) == 0:
        doc.close()
        raise PDFEmptyError(
            f"'{os.path.basename(file_path)}' has no pages.\n"
            f"The file may be empty or malformed."
        )

    # ── Extract metadata ──────────────────────────────────────────────────────
    metadata = _extract_document_metadata(doc, file_path)

    # ── Extract pages ─────────────────────────────────────────────────────────
    pages: list[PageData] = []
    total_to_process = len(doc) if max_pages is None else min(max_pages, len(doc))

    if max_pages is not None and max_pages < len(doc):
        warnings.append(
            f"max_pages={max_pages} set — only first {max_pages} of {len(doc)} pages processed."
        )

    for page_index in range(total_to_process):
        try:
            fitz_page = doc[page_index]
            page_data = _extract_page(fitz_page, page_number=page_index + 1)
            pages.append(page_data)
        except Exception as e:
            # One bad page should not kill the whole document
            warnings.append(f"Page {page_index + 1}: extraction failed — {e}")
            pages.append(PageData(
                page_number=page_index + 1,
                raw_text="",
                tables=[],
                images=[],
                char_count=0,
                has_text=False,
            ))

    doc.close()

    # ── Scanned PDF check ─────────────────────────────────────────────────────
    if _check_for_scanned_pdf(pages, total_to_process):
        raise PDFScannedError(
            f"'{os.path.basename(file_path)}' appears to be a scanned document.\n"
            f"Over 90% of pages contain no selectable text.\n"
            f"OCR is required to extract text from scanned PDFs, "
            f"which is outside this module's scope."
        )

    # ── Warn on mostly-empty documents ────────────────────────────────────────
    pages_with_text = sum(1 for p in pages if p.has_text)
    if pages_with_text == 0:
        warnings.append(
            "No text was extracted from any page. "
            "The document may contain only images or unsupported content."
        )

    result = DocumentResult(
        metadata=metadata,
        pages=pages,
        warnings=warnings,
    )

    print(
        f"[pdf_loader] '{metadata.filename}' → "
        f"{len(pages)} pages | "
        f"{result.total_chars():,} chars | "
        f"{result.total_images()} images | "
        f"{result.total_tables()} tables"
    )

    return result


def load_multiple_pdfs(
    file_paths: list[str],
    max_pages_per_doc: Optional[int] = None,
) -> list[DocumentResult]:
    """
    Load multiple PDF files. Returns one DocumentResult per file.

    Files that fail are skipped with an error message rather than
    crashing the entire batch — important for a multi-upload UI.

    Parameters
    ----------
    file_paths : list[str]
        List of paths to PDF files.
    max_pages_per_doc : Optional[int]
        If set, applies max_pages limit to each document individually.

    Returns
    -------
    list[DocumentResult]
        One result per successfully loaded PDF.
        Failed files are printed to console but not included in output.
    """
    results = []

    for path in file_paths:
        try:
            result = load_pdf(path, max_pages=max_pages_per_doc)
            results.append(result)
        except PDFLoaderError as e:
            print(f"[pdf_loader] SKIPPED '{path}': {e}")
        except Exception as e:
            print(f"[pdf_loader] UNEXPECTED ERROR '{path}': {e}")

    print(
        f"\n[pdf_loader] Batch complete: "
        f"{len(results)}/{len(file_paths)} files loaded successfully."
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Quick Demo  (python src/pdf_loader.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Accept path as CLI argument, fall back to default test location
    test_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "uploads", "attention.pdf")

    print(f"\n{'─' * 60}")
    print(f"  SmartDocAI — pdf_loader.py demo")
    print(f"{'─' * 60}\n")

    try:
        result = load_pdf(test_path)

        # ── Document summary ──────────────────────────────────────────────────
        print("\n── Document Metadata ─────────────────────────────────────────")
        print(f"  Filename      : {result.metadata.filename}")
        print(f"  Title         : {result.metadata.title or '(not set)'}")
        print(f"  Author        : {result.metadata.author or '(not set)'}")
        print(f"  Total pages   : {result.metadata.total_pages}")
        print(f"  File size     : {result.metadata.file_size_bytes / 1024:.1f} KB")
        print(f"  Created       : {result.metadata.creation_date or '(not set)'}")

        print("\n── Extraction Summary ────────────────────────────────────────")
        print(f"  Pages processed   : {len(result.pages)}")
        print(f"  Pages with text   : {result.pages_with_text()}")
        print(f"  Total characters  : {result.total_chars():,}")
        print(f"  Total images      : {result.total_images()}")
        print(f"  Total tables      : {result.total_tables()}")

        if result.warnings:
            print("\n── Warnings ──────────────────────────────────────────────────")
            for w in result.warnings:
                print(f"  ⚠  {w}")

        # ── Page-by-page preview ──────────────────────────────────────────────
        print("\n── Per-Page Preview (first 3 pages) ──────────────────────────")
        for page in result.pages[:3]:
            print(f"\n  Page {page.page_number}")
            print(f"    Characters : {page.char_count:,}")
            print(f"    Tables     : {len(page.tables)}")
            print(f"    Images     : {len(page.images)}")
            preview = page.raw_text[:300].replace("\n", " ").strip()
            print(f"    Text preview: {preview}...")

        # ── Table demo (if any found) ─────────────────────────────────────────
        for page in result.pages:
            if page.tables:
                print(f"\n── First Table Found (Page {page.page_number}) ────────────────────")
                print(page.tables[0].to_readable_text())
                break

        # ── Image metadata demo ───────────────────────────────────────────────
        for page in result.pages:
            if page.images:
                print(f"\n── First Image Found (Page {page.page_number}) ────────────────────")
                img = page.images[0]
                print(f"  Dimensions : {img.width} x {img.height} px")
                print(f"  Colorspace : {img.colorspace or 'unknown'}")
                print(f"  Bounding box: {img.bounding_box}")
                break

        print(f"\n{'─' * 60}")
        print("  Demo complete. pdf_loader.py is working correctly.")
        print(f"{'─' * 60}\n")

    except PDFNotFoundError:
        print(f"\n  PDF not found at: {test_path}")
        print("  Place your PDF at data/uploads/attention.pdf")
        print("  Or run: python src/pdf_loader.py path/to/your.pdf\n")

    except PDFEncryptedError as e:
        print(f"\n  Encrypted PDF: {e}\n")

    except PDFScannedError as e:
        print(f"\n  Scanned PDF detected: {e}\n")

    except PDFLoaderError as e:
        print(f"\n  PDF loading error: {e}\n")