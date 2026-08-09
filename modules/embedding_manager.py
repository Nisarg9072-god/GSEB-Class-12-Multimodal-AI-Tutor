"""
modules/embedding_manager.py
=============================
Manages ChromaDB collections for both text chunks and image records.
Handles incremental processing — skips already-processed PDF files.
"""

import json
import logging
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PROCESSED FILES TRACKER  (for incremental updates)
# ─────────────────────────────────────────────────────────────────────────────

def load_tracker(tracker_file: Path) -> set[str]:
    """Load the set of already-processed PDF filenames."""
    if tracker_file.exists():
        try:
            with open(tracker_file, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()


def save_tracker(tracker_file: Path, processed: set[str]) -> None:
    """Persist the set of processed PDF filenames."""
    try:
        with open(tracker_file, "w", encoding="utf-8") as f:
            json.dump(sorted(processed), f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save tracker: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CHUNK STORAGE
# ─────────────────────────────────────────────────────────────────────────────

def store_text_chunks(
    chunks: list[Document],
    vector_store: Chroma,
) -> int:
    """
    Add text chunks to the ChromaDB text collection.
    Returns number of chunks stored.
    """
    if not chunks:
        return 0
    try:
        vector_store.add_documents(chunks)
        return len(chunks)
    except Exception as e:
        logger.error(f"Failed to store text chunks: {e}")
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE RECORD STORAGE
# ─────────────────────────────────────────────────────────────────────────────

def build_image_embedding_text(record: dict) -> str:
    """
    Compose the text that will be embedded to represent this image.
    Includes: caption, OCR text, nearby paragraph, subject.
    """
    parts = []
    if record.get("caption"):
        parts.append(f"Figure: {record['caption']}")
    if record.get("ocr_text"):
        parts.append(f"Labels: {record['ocr_text']}")
    if record.get("nearby_text"):
        parts.append(record["nearby_text"][:400])
    if record.get("subject"):
        parts.append(f"Subject: {record['subject']}")
    return " | ".join(parts) if parts else "diagram"


def store_image_records(
    image_records: list[dict],
    vector_store: Chroma,
) -> int:
    """
    Convert image records to Documents and store in the image ChromaDB collection.
    Returns number of records stored.
    """
    if not image_records:
        return 0

    documents = []
    for rec in image_records:
        embedding_text = build_image_embedding_text(rec)

        # Metadata stored alongside the embedding
        metadata: dict[str, Any] = {
            "chunk_type":   "image",
            "subject":      rec.get("subject", ""),
            "source_file":  rec.get("source_file", ""),
            "page":         rec.get("page", 0),
            "caption":      rec.get("caption", ""),
            "image_path":   rec.get("image_path", ""),
            "ocr_text":     rec.get("ocr_text", "")[:500],
            "nearby_text":  rec.get("nearby_text", "")[:500],
        }

        documents.append(Document(page_content=embedding_text, metadata=metadata))

    try:
        vector_store.add_documents(documents)
        return len(documents)
    except Exception as e:
        logger.error(f"Failed to store image records: {e}")
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# COLLECTION FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def get_or_create_collection(
    collection_name: str,
    embedding_function: Any,
    chroma_dir: Path,
) -> Chroma:
    """Return a Chroma vector store for the given collection (creates if not exists)."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=str(chroma_dir),
    )
