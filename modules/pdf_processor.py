"""
modules/pdf_processor.py
========================
Loads PDFs page by page using PyMuPDF and tags each page with metadata.
"""

import logging
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def get_subject_from_path(pdf_path: Path, docs_folder: Path) -> str:
    """Infer the subject from the top-level subfolder name."""
    try:
        relative = pdf_path.relative_to(docs_folder)
        parts = relative.parts
        return parts[0] if len(parts) > 1 else "General"
    except ValueError:
        return "General"


def iter_pages(
    pdf_path: Path,
    docs_folder: Path,
) -> Generator[dict, None, None]:
    """
    Open a PDF and yield one dict per page containing:
      - doc: fitz.Document (open)
      - page: fitz.Page
      - page_num: int (0-indexed)
      - subject: str
      - source_file: str
      - pdf_path: Path
    Caller is responsible for closing the doc after use.
    """
    subject = get_subject_from_path(pdf_path, docs_folder)
    try:
        doc = fitz.open(str(pdf_path))
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            yield {
                "doc": doc,
                "page": page,
                "page_num": page_num,
                "subject": subject,
                "source_file": pdf_path.name,
                "pdf_path": pdf_path,
            }
        doc.close()
    except Exception as e:
        logger.error(f"Failed to open PDF {pdf_path.name}: {e}")


def extract_page_text(page: fitz.Page) -> str:
    """Extract plain text from a PDF page."""
    try:
        return page.get_text("text").strip()
    except Exception:
        return ""


def find_all_pdfs(docs_folder: Path) -> list[Path]:
    """Recursively find all PDF files under docs_folder."""
    return sorted(docs_folder.rglob("*.pdf"))
