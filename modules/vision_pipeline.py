"""
modules/vision_pipeline.py
===========================
Optional vision LLM integration.
When ENABLE_VISION_LLM=True in config.py and a vision model is available,
sends (image + text context + question) to the vision LLM.
Falls back to text-only mode if unavailable.
"""

import base64
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def encode_image_base64(image_path: str) -> Optional[str]:
    """Encode an image file to a base64 string."""
    path = Path(image_path)
    if not path.exists():
        logger.warning(f"Image not found for vision encoding: {image_path}")
        return None
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return None


def run_vision_query(
    image_path: str,
    question: str,
    text_context: str,
    llm_model: str,
    api_key: str,
    enable_vision: bool = False,
) -> Optional[str]:
    """
    Send image + context + question to a vision-capable LLM.

    Returns:
        str: answer from vision LLM
        None: if vision is disabled or unavailable (caller uses text-only path)
    """
    if not enable_vision:
        return None

    encoded = encode_image_base64(image_path)
    if not encoded:
        return None

    try:
        # Determine MIME type from extension
        ext = Path(image_path).suffix.lower().lstrip(".")
        mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png", "gif": "image/gif",
                    "webp": "image/webp"}
        mime_type = mime_map.get(ext, "image/png")

        # Try Mistral vision API
        from langchain_mistralai import ChatMistralAI
        from langchain_core.messages import HumanMessage

        vision_llm = ChatMistralAI(model=llm_model, api_key=api_key)

        message = HumanMessage(content=[
            {
                "type": "text",
                "text": (
                    f"You are an expert Class 12 science tutor. "
                    f"Use the diagram provided and the following context to answer the student's question.\n\n"
                    f"Context from textbook:\n{text_context[:1000]}\n\n"
                    f"Student's question: {question}\n\n"
                    f"Please explain every labelled part visible in the diagram, "
                    f"provide important exam points, and give a clear explanation."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            },
        ])

        response = vision_llm.invoke([message])
        return response.content

    except Exception as e:
        logger.warning(f"Vision LLM failed ({e}). Falling back to text-only mode.")
        return None
