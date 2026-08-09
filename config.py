"""
config.py
=========
Central configuration for the Multimodal Educational AI Tutor.
All paths, model names, and feature flags are defined here.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# BASE PATHS — all relative to the project folder (works on any machine)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent          # project root (wherever this file lives)
DOCS_FOLDER = BASE_DIR / "Document Loaders"
CHROMA_DIR  = BASE_DIR / "chroma_db"
IMAGES_DIR  = BASE_DIR / "database_images"
TRACKER_FILE = BASE_DIR / "processed_files.json"

# ─────────────────────────────────────────────────────────────────────────────
# CHROMADB COLLECTIONS
# ─────────────────────────────────────────────────────────────────────────────
TEXT_COLLECTION  = "tuition_assistant"   # existing text chunks
IMAGE_COLLECTION = "tuition_images"      # new image records

# ─────────────────────────────────────────────────────────────────────────────
# MISTRAL AI MODELS
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "mistral-embed"
LLM_MODEL        = "mistral-small-2506"
VISION_LLM_MODEL = "pixtral-12b-2409"   # Optional: Mistral vision model

# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
TEXT_K             = 4    # number of text chunks to retrieve
IMAGE_K            = 2    # number of image records to retrieve for diagram queries
CHAT_HISTORY_LIMIT = 20   # max messages kept in memory (10 turns)

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE EXTRACTION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
MIN_IMAGE_WIDTH  = 100    # pixels — skip tiny/decorative images
MIN_IMAGE_HEIGHT = 100    # pixels
IMAGE_FORMAT     = "png"

# Caption detection: lines to scan above/below each image bounding box
CAPTION_SEARCH_LINES = 3

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE FLAGS
# ─────────────────────────────────────────────────────────────────────────────
ENABLE_OCR        = True   # set False if Tesseract is not installed
ENABLE_VISION_LLM = False  # set True when pixtral model is available

# Keywords that trigger image/diagram retrieval
DIAGRAM_KEYWORDS = [
    "diagram", "figure", "show", "label", "structure", "parts",
    "draw", "illustrate", "image", "illustration", "sketch",
    "identify", "mark", "picture", "photograph", "chart",
    "table", "graph", "plot", "explain using", "using figure",
]

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
