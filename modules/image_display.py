"""
modules/image_display.py
========================
Displays retrieved images using PIL.
Modular: swap PIL for Streamlit/OpenCV by replacing show_image().
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

logger = logging.getLogger(__name__)


def autocrop_diagram(img: PILImage.Image, threshold: int = 245, padding: int = 12) -> PILImage.Image:
    """
    Crop white/near-white borders from the image so only the diagram is shown.
    Works for NCERT-style textbook images where diagrams sit on a white page.

    Args:
        img:       PIL Image (RGB)
        threshold: pixels brighter than this on all channels are treated as background
        padding:   pixels of padding to keep around the cropped region

    Returns:
        Cropped PIL Image (or original if no white border was detected)
    """
    try:
        arr  = np.array(img.convert("RGB"))
        # Mask: True where any channel is darker than threshold (= content pixel)
        mask = np.any(arr < threshold, axis=2)

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any() or not cols.any():
            return img   # Fully white — return as-is

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        h, w = arr.shape[:2]
        rmin = max(0, rmin - padding)
        rmax = min(h, rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(w, cmax + padding)

        cropped = img.crop((cmin, rmin, cmax, rmax))

        # Only return crop if it reduced the image by at least 10%
        orig_area    = img.width  * img.height
        crop_area    = cropped.width * cropped.height
        if crop_area < orig_area * 0.9:
            return cropped
        return img
    except Exception as e:
        logger.debug(f"Autocrop failed: {e}")
        return img


def strip_colored_bands(
    img: PILImage.Image,
    white_threshold: int = 240,
    min_band_px: int = 20,
) -> PILImage.Image:
    """
    Remove solid coloured header/footer bands from textbook page images
    (e.g. the NCERT green/orange chapter-header bar at the top of the page).

    Scans row-by-row from the top until a row with ≥ 25 % near-white pixels
    is found, indicating that diagram content has started, and mirrors the
    same scan from the bottom to strip footer bands.
    """
    try:
        arr  = np.array(img.convert("RGB"))
        h, w = arr.shape[:2]

        # ── Top band ─────────────────────────────────────────────────────────
        top_cut = 0
        for i in range(h):
            white_frac = float(np.mean(np.all(arr[i] >= white_threshold, axis=1)))
            if white_frac >= 0.25:      # first row with meaningful white content
                top_cut = i
                break

        # ── Bottom band ───────────────────────────────────────────────────────
        bottom_cut = h
        for i in range(h - 1, -1, -1):
            white_frac = float(np.mean(np.all(arr[i] >= white_threshold, axis=1)))
            if white_frac >= 0.25:
                bottom_cut = i + 1
                break

        if top_cut >= min_band_px:
            img = img.crop((0, top_cut, w, bottom_cut))
        elif bottom_cut <= h - min_band_px:
            img = img.crop((0, 0, w, bottom_cut))

        return img
    except Exception as e:
        logger.debug(f"strip_colored_bands failed: {e}")
        return img


def load_and_crop(image_path: str) -> PILImage.Image | None:
    """
    Load an image from disk, strip coloured page-header/footer bands,
    then auto-crop remaining white margins.
    Returns a PIL Image, or None if loading fails.
    """
    path = Path(image_path)
    if not path.exists():
        return None
    try:
        img = PILImage.open(str(path)).convert("RGB")
        img = strip_colored_bands(img)   # remove NCERT chapter-header bars first
        return autocrop_diagram(img)     # then trim remaining white margins
    except Exception as e:
        logger.error(f"Could not load image {image_path}: {e}")
        return None


def show_image(image_path: str, caption: str = "") -> bool:
    """
    Display an image from the given path (auto-cropped to diagram region).
    Returns True if displayed successfully, False otherwise.
    """
    img = load_and_crop(image_path)

    if img is None:
        logger.warning(f"Image not found: {image_path}")
        print(f"Image file not found: {image_path}")
        return False

    try:
        title = caption if caption else Path(image_path).name
        print(f"\nDisplaying: {title}")
        print(f"   Path: {image_path}  |  Cropped size: {img.size}")
        img.show(title=title)
        return True
    except Exception as e:
        logger.error(f"Could not display image {image_path}: {e}")
        print(f"Could not display image: {e}")
        return False


def print_image_info(record: dict) -> None:
    """Print a formatted summary of an image record."""
    caption     = record.get("caption", "No caption")
    source_file = record.get("source_file", "?")
    page        = record.get("page", "?")
    subject     = record.get("subject", "?")
    ocr_text    = record.get("ocr_text", "")

    print(f"\n{'─' * 62}")
    print(f"📷  Diagram Retrieved")
    print(f"{'─' * 62}")
    print(f"   📖 Subject   : {subject}")
    print(f"   📄 Source    : {source_file}  (Page {page})")
    if caption:
        print(f"   🏷️  Caption   : {caption}")
    if ocr_text:
        print(f"   🔍 OCR Labels: {ocr_text[:150]}{'...' if len(ocr_text) > 150 else ''}")
    print(f"{'─' * 62}")
