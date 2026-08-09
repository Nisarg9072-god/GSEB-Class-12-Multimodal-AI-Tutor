"""
modules/retriever.py
=====================
Unified retriever that searches both text and image ChromaDB collections.
Detects whether a question needs diagram retrieval based on keywords.
Infers subject from question to pre-filter retrieval for accuracy.
"""

import logging
from typing import Any, Optional

from langchain_chroma import Chroma

import config

logger = logging.getLogger(__name__)

# Keywords that indicate each subject
BIOLOGY_KEYWORDS = [
    "cell", "neuron", "nerve", "heart", "blood", "lung", "kidney", "liver",
    "dna", "rna", "gene", "chromosome", "mitosis", "meiosis", "photosynthesis",
    "respiration", "digestion", "hormone", "enzyme", "plant", "animal", "tissue",
    "organ", "bacteria", "virus", "evolution", "ecology", "reproduction",
    "ovary", "pollen", "embryo", "seed", "flower", "leaf", "root", "stem",
    "human body", "anatomy", "physiology", "biology",
]
PHYSICS_KEYWORDS = [
    "ray", "refraction", "reflection", "lens", "mirror", "light", "optic",
    "electric", "magnetic", "force", "motion", "velocity", "acceleration",
    "wave", "sound", "heat", "thermodynamic", "circuit", "current", "voltage",
    "resistance", "capacitor", "inductor", "semiconductor", "diode", "transistor",
    "nuclear", "radioactive", "quantum", "photon", "electron", "proton",
    "gravitation", "momentum", "energy", "work", "power", "pressure", "physics",
]
CHEMISTRY_KEYWORDS = [
    "molecule", "atom", "bond", "reaction", "acid", "base", "salt", "solution",
    "concentration", "molarity", "mole", "compound", "element", "periodic",
    "oxidation", "reduction", "electrochemistry", "polymer", "organic",
    "carbon", "benzene", "alkane", "alkene", "alcohol", "ester", "aldehyde",
    "ketone", "amine", "protein", "carbohydrate", "lipid", "coordination",
    "crystal", "solid", "gas", "equilibrium", "entropy", "enthalpy", "chemistry",
]


def infer_subject(question: str) -> Optional[str]:
    """
    Infer the most likely subject from the question text.
    Returns 'biology', 'Physics', 'chemisty', or None.
    (Folder names match the actual subject folder names in the project.)
    """
    q = question.lower()
    bio_score  = sum(1 for kw in BIOLOGY_KEYWORDS  if kw in q)
    phy_score  = sum(1 for kw in PHYSICS_KEYWORDS  if kw in q)
    chem_score = sum(1 for kw in CHEMISTRY_KEYWORDS if kw in q)

    if max(bio_score, phy_score, chem_score) == 0:
        return None   # no clear subject — search all

    scores = {"biology": bio_score, "Physics": phy_score, "chemisty": chem_score}
    return max(scores, key=scores.get)


def is_diagram_query(question: str) -> bool:
    """Return True if the question likely needs a diagram/image."""
    q_lower = question.lower()
    return any(keyword in q_lower for keyword in config.DIAGRAM_KEYWORDS)


def retrieve(
    question: str,
    text_store: Chroma,
    image_store: Chroma,
    text_k: int = None,
    image_k: int = None,
) -> dict[str, list]:
    """
    Retrieve relevant text chunks and (optionally) image records.

    Image retrieval strategy:
      1. PRIMARY — find images from the SAME pages as the top text chunks
         (much more accurate than independent vector search)
      2. FALLBACK — if no page-matched images, fall back to vector search

    Returns:
        {
            "text":   [list of LangChain Documents],
            "images": [list of LangChain Documents with image metadata],
        }
    """
    text_k  = text_k  or config.TEXT_K
    image_k = image_k or config.IMAGE_K

    results: dict[str, list] = {"text": [], "images": []}

    # ── Infer subject to pre-filter text retrieval ────────────────────────────
    subject = infer_subject(question)

    # ── Text retrieval (subject-filtered when possible) ───────────────────────
    try:
        if subject:
            filtered = text_store.similarity_search(
                question, k=text_k, filter={"subject": subject}
            )
            if filtered:
                results["text"] = filtered
                logger.info(f"Subject filter '{subject}' applied — {len(filtered)} chunks.")
            else:
                # No results with filter → fall back to unfiltered
                results["text"] = text_store.similarity_search(question, k=text_k)
        else:
            results["text"] = text_store.similarity_search(question, k=text_k)
    except Exception as e:
        logger.error(f"Text retrieval failed: {e}")

    # ── Image retrieval (only for diagram queries) ────────────────────────────
    if not is_diagram_query(question):
        return results

    logger.info("Diagram query detected — retrieving images.")

    # Strategy 1: find images on the same pages (±2) as relevant text chunks
    page_matched_images = []
    seen_paths = set()

    for text_doc in results["text"]:
        m    = text_doc.metadata
        src  = m.get("source_file", "")
        page = m.get("page", 0)
        if not src or not page:
            continue

        # Search exact page first, then expand window ±1, ±2
        for offset in [0, 1, -1, 2, -2]:
            target_page = page + offset
            if target_page < 1:
                continue
            try:
                hits = image_store.similarity_search(
                    question,
                    k=image_k,
                    filter={"source_file": src, "page": target_page},
                )
                for hit in hits:
                    ip = hit.metadata.get("image_path", "")
                    if ip and ip not in seen_paths:
                        page_matched_images.append(hit)
                        seen_paths.add(ip)
            except Exception:
                pass

        if len(page_matched_images) >= image_k:
            break

    if page_matched_images:
        results["images"] = page_matched_images[:image_k]
        return results

    # Strategy 2: fallback — direct vector search (deduplicated)
    try:
        fallback = image_store.similarity_search(question, k=image_k * 3)
        seen_fb  = set()
        deduped  = []
        for doc in fallback:
            ip = doc.metadata.get("image_path", "")
            if ip and ip not in seen_fb:
                deduped.append(doc)
                seen_fb.add(ip)
            if len(deduped) >= image_k:
                break
        results["images"] = deduped
    except Exception as e:
        logger.error(f"Image vector search failed: {e}")

    return results


def build_text_context(text_docs: list) -> str:
    """Format retrieved text documents into a context string."""
    if not text_docs:
        return ""
    parts = []
    for doc in text_docs:
        m = doc.metadata
        header = (
            f"[Subject: {m.get('subject', '?')} | "
            f"File: {m.get('source_file', '?')} | "
            f"Page {m.get('page', '?')}]"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)


def build_image_context(image_docs: list) -> str:
    """Format retrieved image records into a text description for the LLM."""
    if not image_docs:
        return ""
    parts = []
    for doc in image_docs:
        m = doc.metadata
        lines = [f"[Diagram from {m.get('source_file', '?')} | Page {m.get('page', '?')}]"]
        if m.get("caption"):
            lines.append(f"Caption: {m['caption']}")
        if m.get("ocr_text"):
            lines.append(f"Labels visible in diagram: {m['ocr_text']}")
        if m.get("nearby_text"):
            lines.append(f"Surrounding text: {m['nearby_text'][:300]}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)
