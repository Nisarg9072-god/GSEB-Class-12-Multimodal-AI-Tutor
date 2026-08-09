"""
create_database.py
==================
Run this script ONCE (or when new PDFs are added) to:
  1. Recursively scan all PDFs in the Document Loaders folder
  2. Extract text → split into chunks → embed → store in ChromaDB (text collection)
  3. Extract images from every page → run OCR → embed text description → store in ChromaDB (image collection)
  4. Save extracted images to database_images/<subject>/
  5. Track processed files to support incremental updates

Usage:
    python create_database.py
    python create_database.py --rebuild    # force rebuild even if already processed
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── Setup logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("create_database")

# ── Imports ───────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

import os
import config

from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from modules.pdf_processor   import find_all_pdfs, iter_pages, extract_page_text
from modules.image_extractor import extract_images_from_page
from modules.ocr_processor   import run_ocr, is_ocr_available
from modules.embedding_manager import (
    get_or_create_collection,
    store_text_chunks,
    store_image_records,
    load_tracker,
    save_tracker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the multimodal ChromaDB database.")
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force reprocessing of all PDFs even if already tracked."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Check OCR availability ─────────────────────────────────────────────
    ocr_available = config.ENABLE_OCR and is_ocr_available()
    if config.ENABLE_OCR and not ocr_available:
        print("Tesseract not found — OCR disabled.")
    elif ocr_available:
        print("Tesseract OCR available.")
    else:
        print("OCR disabled in config.")

    # ── If rebuild: wipe BEFORE opening any connections ────────────────────
    if args.rebuild:
        print("\nRebuild mode — clearing old data...")
        if config.CHROMA_DIR.exists():
            shutil.rmtree(config.CHROMA_DIR)
            print(f"  Cleared ChromaDB : {config.CHROMA_DIR}")
        if config.IMAGES_DIR.exists():
            shutil.rmtree(config.IMAGES_DIR)
            print(f"  Cleared images   : {config.IMAGES_DIR}")
        if config.TRACKER_FILE.exists():
            config.TRACKER_FILE.unlink()
        print("  Done. Building fresh database...")

    # ── Create output directories ──────────────────────────────────────────
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Setup embeddings + ChromaDB collections ────────────────────────────
    print("\nConnecting to ChromaDB...")
    embedding_model = MistralAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=os.getenv("MISTRAL_API_KEY"),
    )

    text_store  = get_or_create_collection(config.TEXT_COLLECTION,  embedding_model, config.CHROMA_DIR)
    image_store = get_or_create_collection(config.IMAGE_COLLECTION, embedding_model, config.CHROMA_DIR)
    print(f"Collections ready: '{config.TEXT_COLLECTION}' + '{config.IMAGE_COLLECTION}'")

    # ── Text splitter ──────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    # ── Incremental tracker ────────────────────────────────────────────────
    processed: set[str] = set() if args.rebuild else load_tracker(config.TRACKER_FILE)

    # ── Find all PDFs ──────────────────────────────────────────────────────
    pdf_files = find_all_pdfs(config.DOCS_FOLDER)
    if not pdf_files:
        print(f"❌ No PDFs found in: {config.DOCS_FOLDER}")
        sys.exit(1)

    new_pdfs = [p for p in pdf_files if p.name not in processed]
    print(f"\n📂 Found {len(pdf_files)} PDF(s). {len(new_pdfs)} new / unprocessed.")
    if not new_pdfs:
        print("✅ All PDFs already processed. Use --rebuild to force reprocessing.")
        return

    # ── Process each PDF ───────────────────────────────────────────────────
    total_text_chunks  = 0
    total_image_chunks = 0
    total_pages        = 0

    for pdf_path in new_pdfs:
        rel = pdf_path.relative_to(config.DOCS_FOLDER)
        print(f"\n📄 Processing: {rel}")

        page_texts   : list[Document] = []
        image_records: list[dict]     = []

        try:
            for page_info in iter_pages(pdf_path, config.DOCS_FOLDER):
                page       = page_info["page"]
                doc_handle = page_info["doc"]
                page_num   = page_info["page_num"]
                subject    = page_info["subject"]
                src_file   = page_info["source_file"]

                total_pages += 1

                # ── Extract text ───────────────────────────────────────────
                text = extract_page_text(page)
                if text:
                    page_texts.append(Document(
                        page_content=text,
                        metadata={
                            "chunk_type":  "text",
                            "subject":     subject,
                            "source_file": src_file,
                            "page":        page_num + 1,
                        },
                    ))

                # ── Extract images ─────────────────────────────────────────
                imgs = extract_images_from_page(
                    doc=doc_handle,
                    page=page,
                    page_num=page_num,
                    subject=subject,
                    source_file=src_file,
                    images_dir=config.IMAGES_DIR,
                    min_width=config.MIN_IMAGE_WIDTH,
                    min_height=config.MIN_IMAGE_HEIGHT,
                )

                # ── OCR each extracted image ───────────────────────────────
                for img_rec in imgs:
                    ocr_text = run_ocr(Path(img_rec["image_path"]), enable_ocr=ocr_available)
                    img_rec["ocr_text"] = ocr_text
                    image_records.append(img_rec)

        except Exception as e:
            logger.error(f"Failed processing {rel}: {e}")
            continue

        # ── Store text chunks ──────────────────────────────────────────────
        if page_texts:
            chunks = splitter.split_documents(page_texts)
            n = store_text_chunks(chunks, text_store)
            total_text_chunks += n
            print(f"   ✅ Text  : {len(page_texts)} pages → {n} chunks stored.")

        # ── Store image records ────────────────────────────────────────────
        if image_records:
            n = store_image_records(image_records, image_store)
            total_image_chunks += n
            print(f"   ✅ Images: {len(image_records)} extracted → {n} records stored.")
        else:
            print(f"   ℹ️  Images: none extracted (or all below size threshold).")

        # ── Mark as processed ──────────────────────────────────────────────
        processed.add(pdf_path.name)
        save_tracker(config.TRACKER_FILE, processed)

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("✅ DATABASE BUILD COMPLETE")
    print("=" * 62)
    print(f"   PDFs processed  : {len(new_pdfs)}")
    print(f"   Pages read      : {total_pages}")
    print(f"   Text chunks     : {total_text_chunks}")
    print(f"   Image records   : {total_image_chunks}")
    print(f"   Images saved to : {config.IMAGES_DIR}")
    print(f"   ChromaDB at     : {config.CHROMA_DIR}")
    print("\n🚀 Run main.py to start the multimodal tutor!")


if __name__ == "__main__":
    main()
