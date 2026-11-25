"""Reusable, model-free building blocks of the RAG project (also defined inline in the notebook).

Contains: PDF validation/extraction, text cleaning, chunking and retrieval metrics.
Heavy components (embedding model, FAISS index, LLM) live in the notebook.
"""
import re
from pathlib import Path

try:
    import pymupdf as fitz  # PyMuPDF (current module name)
except ImportError:
    import fitz  # older releases

def validate_pdf(path):
    """Return (is_valid, message) for a candidate PDF file."""
    path = Path(path)
    try:
        if path.stat().st_size == 0:
            return False, "empty file (0 bytes)"
        with fitz.open(path) as pdf:
            if pdf.page_count == 0:
                return False, "PDF contains no pages"
            if pdf.needs_pass:
                return False, "PDF is password-protected"
        return True, "ok"
    except Exception as exc:  # corrupted / not a PDF
        return False, f"cannot be opened as a PDF ({exc})"


def extract_pdf_pages(pdf_path):
    """Extract text page by page with PyMuPDF.

    Returns
    -------
    pages  : list of dicts {text, source, page}  (pages with selectable text only)
    report : dict with per-document extraction statistics (used to flag possible OCR needs)
    """
    pdf_path = Path(pdf_path)
    pages, empty_pages, empty_with_images = [], 0, 0
    with fitz.open(pdf_path) as pdf:
        total_pages = pdf.page_count
        for page_number, page in enumerate(pdf, start=1):   # physical page index, starting at 1
            text = page.get_text("text").strip()
            if text:
                pages.append({"text": text, "source": pdf_path.name, "page": page_number})
            else:
                empty_pages += 1
                if page.get_images(full=True):
                    empty_with_images += 1
    empty_ratio = empty_pages / total_pages if total_pages else 1.0
    report = {
        "source": pdf_path.name,
        "total_pages": total_pages,
        "pages_with_text": len(pages),
        "empty_pages": empty_pages,
        "empty_pages_with_images": empty_with_images,
        "characters": sum(len(p["text"]) for p in pages),
        "possible_ocr_needed": bool(empty_ratio >= 0.3 or (total_pages > 0 and not pages)),
    }
    return pages, report


_LIGATURES = {"\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff", "\ufb03": "ffi", "\ufb04": "ffl"}


def clean_text(text):
    """Light-weight cleaning that preserves punctuation, numbers, code and paragraph structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")          # normalise line breaks
    for bad, good in _LIGATURES.items():                            # typographic ligatures
        text = text.replace(bad, good)
    text = text.replace("\u00a0", " ").replace("\u200b", "")       # non-breaking / zero-width spaces
    text = re.sub(r"(?<=[A-Za-z])-\n(?=[a-z])", "", text)          # re-join words hyphenated at line end
    text = re.sub(r"[ \t]+", " ", text)                             # collapse runs of spaces/tabs
    text = re.sub(r" *\n *", "\n", text)                            # strip spaces around line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)                          # at most one blank line
    return text.strip()


def clean_pages(pages):
    """Apply clean_text to every page and drop pages that become empty."""
    cleaned = []
    for p in pages:
        new_text = clean_text(p["text"])
        if new_text:
            cleaned.append({**p, "text": new_text})
    return cleaned


def split_into_chunks(text, chunk_size=900, overlap=150):
    """Character-based sliding-window chunking (course baseline: 900 / 150)."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def chunk_pages(pages, chunk_size=900, overlap=150):
    """Chunk every page and keep source / page / chunk_id metadata (+ a unique id)."""
    all_chunks = []
    for page in pages:
        for chunk_id, chunk_text in enumerate(split_into_chunks(page["text"], chunk_size, overlap)):
            all_chunks.append({
                "text": chunk_text,
                "source": page["source"],
                "page": page["page"],
                "chunk_id": chunk_id,
                "uid": f"{page['source']}|p{page['page']}|c{chunk_id}",
            })
    return all_chunks


def precision_at_k(retrieved_ids, relevant_ids, k):
    """Precision@K = (# relevant items among the top K) / K."""
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = retrieved_ids[:k]
    return sum(x in relevant_ids for x in top_k) / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    """Recall@K = (# relevant items found in the top K) / (# known relevant items).
    Returns None when the question has no known relevant item (recall is undefined)."""
    if not relevant_ids:
        return None
    top_k = retrieved_ids[:k]
    return sum(x in relevant_ids for x in set(top_k)) / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    """RR = 1 / rank of the first relevant item (0 if none is retrieved)."""
    for rank, x in enumerate(retrieved_ids, start=1):
        if x in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_ranking(retrieved_page_keys, relevant_page_keys, k):
    """Page-level evaluation of one ranked list of chunks.

    retrieved_page_keys : [(source, page), ...] one entry per retrieved chunk, best first
    relevant_page_keys  : set of (source, page) labelled as relevant
    * Precision@K counts chunks whose page is relevant (denominator K).
    * Recall@K counts distinct relevant pages found (denominator = # relevant pages).
    * RR uses the rank of the first chunk lying on a relevant page.
    """
    relevant = set(relevant_page_keys)
    top = list(retrieved_page_keys[:k])
    unique_pages = list(dict.fromkeys(top))
    return {
        "precision": precision_at_k(top, relevant, k),
        "recall": recall_at_k(unique_pages, relevant, len(unique_pages)),
        "rr": reciprocal_rank(top, relevant),
    }
