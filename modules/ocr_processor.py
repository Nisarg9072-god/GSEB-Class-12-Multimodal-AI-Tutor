"""
modules/ocr_processor.py
========================
Runs OCR on extracted images using pytesseract.
Gracefully disabled if Tesseract is not installed.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Attempt to import pytesseract — optional dependency
try:
    import pytesseract
    from PIL import Image as PILImage
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not available. OCR will be skipped.")


def run_ocr(image_path: Path, enable_ocr: bool = True) -> str:
    """
    Run OCR on the image at image_path.
    Returns extracted text string, or empty string if OCR is disabled/fails.
    """
    if not enable_ocr or not TESSERACT_AVAILABLE:
        return ""

    try:
        img = PILImage.open(str(image_path)).convert("L")  # grayscale

        # Basic threshold for cleaner OCR
        img = img.point(lambda x: 0 if x < 140 else 255, "1")

        text = pytesseract.image_to_string(img, lang="eng").strip()
        return text
    except Exception as e:
        logger.warning(f"OCR failed for {image_path.name}: {e}")
        return ""


def is_ocr_available() -> bool:
    """Return True if pytesseract + Tesseract binary are available."""
    if not TESSERACT_AVAILABLE:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False
