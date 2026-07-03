# backend/document_processor.py
import os
from typing import List


def extract_text(filepath: str) -> str:
    """
    Dispatch to the correct extractor based on file extension.

    Supported: .txt, .pdf, .docx
    Raises ValueError for anything else.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".txt":
        return _from_txt(filepath)
    elif ext == ".pdf":
        return _from_pdf(filepath)
    elif ext == ".docx":
        return _from_docx(filepath)
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            "Only .txt, .pdf, and .docx files are accepted."
        )


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping character-based chunks.

    chunk_size : maximum characters per chunk
    overlap    : characters of shared content between consecutive chunks

    Example with chunk_size=10, overlap=3:
      text = "ABCDEFGHIJKLMNOP"
      chunks = ["ABCDEFGHIJ", "HIJKLMNOP", ...]
    """
    chunks = []
    text = text.strip()
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# ─── Internal extractors ──────────────────────────────────────────────────────

def _from_txt(filepath: str) -> str:
    """Read a plain text file as UTF-8."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def _from_pdf(filepath: str) -> str:
    """
    Extract text from a PDF using PyMuPDF.
    PyMuPDF is imported as 'fitz' — this is correct, not a mistake.
    Install it as: pip install PyMuPDF
    """
    import fitz  # PyMuPDF
    doc = fitz.open(filepath)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


def _from_docx(filepath: str) -> str:
    """
    Extract text from a .docx file using python-docx.
    Only extracts paragraph text (not tables or headers — add those if needed).
    """
    from docx import Document
    doc = Document(filepath)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)
