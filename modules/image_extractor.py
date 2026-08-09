"""
modules/image_extractor.py
===========================
Extracts diagrams from PDF pages using PyMuPDF (fitz).

Strategy (most accurate for NCERT-style PDFs):
  1. PRIMARY — Figure-caption guided extraction
     Scans each page for "Fig/Figure/Diagram" caption text,
     then renders the region ABOVE the caption (where the actual diagram is).
     This gives exactly the labelled diagram, not random page regions.

  2. FALLBACK — Embedded image XObject extraction
     For pages without detectable figure captions but with large embedded images.
     Applies autocrop to remove white borders.
"""

import logging
import re
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image as PILImage
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CAPTION DETECTION PATTERNS
# ─────────────────────────────────────────────────────────────────────────────
# Only actual caption lines — must START with Figure/Fig/Plate/Chart/Graph + number
CAPTION_RE = re.compile(
    r'^(fig(ure)?[\.\s]*\d+[\d\.]*'
    r'|plate\s*\d+'
    r'|chart\s*\d+'
    r'|graph\s*\d+'
    r'|scheme\s*\d+)',
    re.IGNORECASE | re.MULTILINE,
)


def _is_caption_text(text: str) -> bool:
    """Return True only for standalone figure caption lines (starts with Fig X.X)."""
    text = text.strip()
    return bool(CAPTION_RE.match(text))


def _autocrop(img: PILImage.Image, threshold: int = 245, pad: int = 15) -> PILImage.Image:
    """Trim near-white borders so only diagram content remains."""
    try:
        arr  = np.array(img.convert("RGB"))
        mask = np.any(arr < threshold, axis=2)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return img
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        h, w = arr.shape[:2]
        crop = img.crop((
            max(0, cmin - pad),
            max(0, rmin - pad),
            min(w, cmax + pad),
            min(h, rmax + pad),
        ))
        # Only keep crop if meaningful reduction (>10%)
        if crop.width * crop.height < img.width * img.height * 0.9:
            return crop
        return img
    except Exception:
        return img


def _render_rect(page: fitz.Page, rect: fitz.Rect, scale: float = 2.0) -> PILImage.Image | None:
    """Render a PDF page region to a PIL Image."""
    try:
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, clip=rect, colorspace=fitz.csRGB)
        return PILImage.frombytes("RGB", (pix.width, pix.height), pix.samples)
    except Exception as e:
        logger.debug(f"Render failed: {e}")
        return None


def _save_pil(img: PILImage.Image, path: Path) -> bool:
    """Save PIL image to disk. Returns True if saved and non-trivial."""
    try:
        img.save(str(path), format="PNG")
        return path.stat().st_size > 1000
    except Exception as e:
        logger.warning(f"Save failed {path.name}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 1: FIGURE-CAPTION GUIDED EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_by_captions(
    page: fitz.Page,
    page_num: int,
    subject: str,
    source_file: str,
    images_dir: Path,
    min_width: int,
    min_height: int,
) -> list[dict]:
    """
    Find figure caption blocks and render the diagram region around each.
    NCERT captions are usually BELOW the figure.
    We render the region above the caption + include caption text.
    """
    subject_dir = images_dir / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    pdf_stem  = Path(source_file).stem
    page_rect = page.rect
    records   = []
    fig_idx   = 0

    try:
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    except Exception:
        return []

    for block in blocks:
        if block.get("type") != 0:   # 0 = text block
            continue

        # Collect full text of this block
        block_text = " ".join(
            span.get("text", "")
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ).strip()

        if not _is_caption_text(block_text):
            continue

        # Found a figure caption
        cap_rect = fitz.Rect(block["bbox"])
        fig_idx += 1

        # Diagram is ABOVE the caption — render from page top to caption bottom
        # Constrain width to the caption's horizontal extent ± margin
        margin_x = 60
        x0 = max(0,            cap_rect.x0 - margin_x)
        x1 = min(page_rect.width, cap_rect.x1 + margin_x)

        # Look from page start to just below caption
        # (captures everything on the page above this caption)
        # But limit to the column section containing the caption
        fig_rect = fitz.Rect(x0, 0, x1, cap_rect.y1 + 10)

        # If region is too tall (captures too much page), limit height
        max_height = 420   # ~15cm in points — typical diagram height
        if fig_rect.height > max_height:
            fig_rect = fitz.Rect(x0, max(0, cap_rect.y0 - max_height), x1, cap_rect.y1 + 10)

        if fig_rect.width < min_width or fig_rect.height < min_height:
            continue

        # Render the region
        img = _render_rect(page, fig_rect)
        if img is None:
            continue

        # Autocrop to just the diagram content
        img = _autocrop(img)

        # Must be large enough after crop
        if img.width < min_width or img.height < min_height:
            continue

        img_filename = f"{pdf_stem}_page{page_num + 1}_fig{fig_idx}.png"
        img_path     = subject_dir / img_filename

        if not _save_pil(img, img_path):
            continue

        # Nearby text for embedding (text around caption)
        nearby_clip = fitz.Rect(x0, max(0, cap_rect.y0 - 200), x1, min(page_rect.height, cap_rect.y1 + 100))
        try:
            nearby_text = page.get_text("text", clip=nearby_clip).strip()[:500]
        except Exception:
            nearby_text = ""

        records.append({
            "image_path":  str(img_path),
            "source_file": source_file,
            "subject":     subject,
            "page":        page_num + 1,
            "img_index":   fig_idx,
            "caption":     block_text[:200],
            "nearby_text": nearby_text,
            "width":       img.width,
            "height":      img.height,
            "chunk_type":  "image",
            "method":      "caption-guided",
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 2: EMBEDDED IMAGE XOBJECT FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def _extract_by_xobjects(
    doc: fitz.Document,
    page: fitz.Page,
    page_num: int,
    subject: str,
    source_file: str,
    images_dir: Path,
    min_width: int,
    min_height: int,
) -> list[dict]:
    """
    Fallback: extract embedded image XObjects.
    Filters out full-page background images.
    Applies autocrop to remove white borders.
    """
    subject_dir = images_dir / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    pdf_stem  = Path(source_file).stem
    page_area = page.rect.get_area()
    records   = []

    try:
        image_list = page.get_images(full=True)
    except Exception:
        return []

    img_idx = 0
    for img_info in image_list:
        xref = img_info[0]

        # Get bounding rect
        img_rect = None
        try:
            rects = page.get_image_rects(xref)
            if rects:
                img_rect = rects[0]
        except Exception:
            pass

        # Get dimensions
        try:
            base   = doc.extract_image(xref)
            width  = base.get("width", 0)
            height = base.get("height", 0)
        except Exception:
            width  = int(img_rect.width)  if img_rect else 0
            height = int(img_rect.height) if img_rect else 0

        if width < min_width or height < min_height:
            continue

        # Skip images covering > 55% of page area (background/full-page scans)
        if img_rect:
            coverage = img_rect.get_area() / page_area
            if coverage > 0.55:
                continue

            # Skip images wider than 80% of page (NCERT header bars, page templates)
            if img_rect.width > page.rect.width * 0.80:
                continue

            # Skip very thin images (decorative lines / dividers)
            if img_rect.height < page.rect.height * 0.05:
                continue

        img_idx += 1
        img_filename = f"{pdf_stem}_page{page_num + 1}_img{img_idx}.png"
        img_path     = subject_dir / img_filename

        # Render via page region if we have a rect, else raw extraction
        if img_rect and not img_rect.is_empty:
            rendered = _render_rect(page, img_rect)
            if rendered is None:
                continue
            rendered = _autocrop(rendered)
        else:
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                if pix.alpha:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                rendered = PILImage.frombytes("RGB", (pix.width, pix.height), pix.samples)
                rendered = _autocrop(rendered)
            except Exception as e:
                logger.debug(f"XObject extraction failed xref {xref}: {e}")
                continue

        # Minimum rendered size (250px = ~125pt at 2x zoom ≈ 4.4cm — real diagrams are bigger)
        if rendered.width < 250 or rendered.height < 250:
            continue

        if not _save_pil(rendered, img_path):
            continue

        fallback_rect = img_rect or fitz.Rect(0, 0, width, height)
        try:
            nearby_text = page.get_text("text", clip=fallback_rect.expand(80)).strip()[:500]
        except Exception:
            nearby_text = ""

        # Caption near the image
        cap_clip = fitz.Rect(fallback_rect.x0, fallback_rect.y1, fallback_rect.x1, fallback_rect.y1 + 60)
        try:
            caption = page.get_text("text", clip=cap_clip).strip()[:200]
        except Exception:
            caption = ""

        records.append({
            "image_path":  str(img_path),
            "source_file": source_file,
            "subject":     subject,
            "page":        page_num + 1,
            "img_index":   img_idx,
            "caption":     caption,
            "nearby_text": nearby_text,
            "width":       rendered.width,
            "height":      rendered.height,
            "chunk_type":  "image",
            "method":      "xobject-fallback",
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def extract_images_from_page(
    doc: fitz.Document,
    page: fitz.Page,
    page_num: int,
    subject: str,
    source_file: str,
    images_dir: Path,
    min_width: int = 100,
    min_height: int = 100,
) -> list[dict]:
    """
    Extract diagrams from a PDF page.
    Tries caption-guided extraction first; falls back to XObject extraction.
    """
    # Strategy 1: find actual figure captions and render around them
    records = _extract_by_captions(
        page, page_num, subject, source_file, images_dir, min_width, min_height
    )

    # Strategy 2: fallback for pages without detectable captions
    if not records:
        records = _extract_by_xobjects(
            doc, page, page_num, subject, source_file, images_dir, min_width, min_height
        )

    logger.debug(
        f"Page {page_num + 1} of {source_file}: "
        f"{len(records)} image(s) extracted "
        f"({'caption' if records and records[0].get('method') == 'caption-guided' else 'fallback'})"
    )
    return records
