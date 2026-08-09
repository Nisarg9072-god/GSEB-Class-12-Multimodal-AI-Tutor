"""
main.py
=======
GSEB Class 12 Multimodal Tutor — Physics, Chemistry, Biology
-------------------------------------------------------------
IMPORTANT: Run create_database.py first to build the ChromaDB vector store.

Features:
  - Text + diagram retrieval from ChromaDB
  - Displays original textbook diagrams when relevant
  - Conversation memory (last 10 turns)
  - Follow-up question reformulation
  - Optional vision LLM support (set ENABLE_VISION_LLM=True in config.py)
"""

import logging
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.WARNING,   # suppress INFO logs in chat mode
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from dotenv import load_dotenv
load_dotenv()

import config
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

from modules.embedding_manager import get_or_create_collection
from modules.retriever         import retrieve, build_text_context, build_image_context, is_diagram_query
from modules.image_display     import show_image, print_image_info
from modules.vision_pipeline   import run_vision_query
from modules.answer_generator  import (
    reformulate_question,
    generate_text_answer,
    generate_diagram_answer,
)


def check_database() -> bool:
    """Return True if the ChromaDB directory exists and is non-empty."""
    return config.CHROMA_DIR.exists() and any(config.CHROMA_DIR.iterdir())


def load_stores(embedding_model: MistralAIEmbeddings) -> tuple[Chroma, Chroma]:
    """Load both ChromaDB collections from disk."""
    text_store  = get_or_create_collection(config.TEXT_COLLECTION,  embedding_model, config.CHROMA_DIR)
    image_store = get_or_create_collection(config.IMAGE_COLLECTION, embedding_model, config.CHROMA_DIR)
    return text_store, image_store


def main() -> None:
    # ── Preflight check ───────────────────────────────────────────────────────
    if not check_database():
        print("❌ ChromaDB not found! Please run create_database.py first.")
        sys.exit(1)

    # ── Load embedding model + stores ─────────────────────────────────────────
    print("⚡ Loading ChromaDB from disk...")
    embedding_model = MistralAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=os.getenv("MISTRAL_API_KEY"),
    )
    text_store, image_store = load_stores(embedding_model)
    print("✅ ChromaDB loaded successfully.")

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm = ChatMistralAI(
        model=config.LLM_MODEL,
        api_key=os.getenv("MISTRAL_API_KEY"),
    )

    # ── Chat loop ──────────────────────────────────────────────────────────────
    chat_history: list = []

    print("\n" + "=" * 62)
    print("   🎓  GSEB Class 12 Tutor — Physics, Chemistry, Biology   ")
    print("=" * 62)
    print("   Ask any question from your textbooks.")
    print("   Ask for diagrams: 'show me the diagram of...'")
    print("   The bot remembers your previous questions!")
    print("   Type  'exit'  or  'quit'  to stop.")
    print("=" * 62)

    while True:
        # ── Get input ─────────────────────────────────────────────────────────
        try:
            question = input("\n📚 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye! Good luck with your studies!")
            break

        if question.lower() in ("exit", "quit", "q", "bye"):
            print("👋 Goodbye! Good luck with your studies!")
            break

        if not question:
            print("⚠️  Please type a question.")
            continue

        # ── Step 1: Reformulate follow-up questions ────────────────────────────
        standalone = reformulate_question(question, chat_history, llm)

        # ── Step 2: Retrieve text + images from ChromaDB ──────────────────────
        results     = retrieve(standalone, text_store, image_store)
        text_docs   = results["text"]
        image_docs  = results["images"]

        text_context  = build_text_context(text_docs)
        image_context = build_image_context(image_docs)

        # ── Step 3: Display retrieved images ──────────────────────────────────
        if image_docs:
            for img_doc in image_docs:
                m = img_doc.metadata
                print_image_info(m)
                show_image(m.get("image_path", ""), m.get("caption", ""))

        # ── Step 4: Generate answer ────────────────────────────────────────────
        is_diagram = is_diagram_query(standalone)

        if is_diagram and image_docs:
            # Try vision LLM first (if enabled)
            img_path = image_docs[0].metadata.get("image_path", "")
            answer = run_vision_query(
                image_path=img_path,
                question=question,
                text_context=text_context,
                llm_model=config.VISION_LLM_MODEL,
                api_key=os.getenv("MISTRAL_API_KEY", ""),
                enable_vision=config.ENABLE_VISION_LLM,
            )
            if answer is None:
                # Vision unavailable — use diagram-aware text prompt
                answer = generate_diagram_answer(
                    question=question,
                    text_context=text_context,
                    image_context=image_context,
                    chat_history=chat_history,
                    llm=llm,
                )
        else:
            # Standard text-only answer
            answer = generate_text_answer(
                question=question,
                text_context=text_context,
                chat_history=chat_history,
                llm=llm,
            )

        # ── Step 5: Update conversation memory ────────────────────────────────
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))
        if len(chat_history) > config.CHAT_HISTORY_LIMIT:
            chat_history = chat_history[-config.CHAT_HISTORY_LIMIT:]

        # ── Step 6: Print answer ───────────────────────────────────────────────
        print("\n" + "─" * 62)
        print("🤖 Tutor:")
        print("─" * 62)
        print(answer)
        print("─" * 62)


if __name__ == "__main__":
    main()