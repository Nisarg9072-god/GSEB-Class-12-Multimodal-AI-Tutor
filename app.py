"""
app.py — Streamlit Web Interface
=================================
GSEB Class 12 Multimodal Tutor — Physics, Chemistry, Biology

Students open this in any browser (PC or phone).
Images from textbooks are displayed INLINE in the chat.

Run with:
    streamlit run app.py
"""

import os
import sys
import logging

import streamlit as st
from dotenv import load_dotenv
from PIL import Image as PILImage
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# Suppress noisy logs in the UI
logging.basicConfig(level=logging.WARNING)

import config
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_chroma import Chroma
from modules.embedding_manager import get_or_create_collection
from modules.retriever import retrieve, build_text_context, build_image_context, is_diagram_query
from modules.answer_generator import reformulate_question, generate_text_answer, generate_diagram_answer
from modules.vision_pipeline import run_vision_query
from modules.image_display import load_and_crop

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GSEB Class 12 AI Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — premium look
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(102,126,234,0.3);
}
.main-header h1 { font-size: 2rem; font-weight: 700; margin: 0; }
.main-header p  { font-size: 1rem; opacity: 0.85; margin: 0.5rem 0 0; }

.diagram-box {
    background: linear-gradient(135deg, #f8f9ff 0%, #e8ecff 100%);
    border: 2px solid #667eea;
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
}
.diagram-label {
    font-weight: 600;
    color: #667eea;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
}

.stChatMessage { border-radius: 12px !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] h2 { color: #667eea !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CACHED RESOURCE LOADING — only loads once per session
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI Tutor...")
def load_resources():
    if not config.CHROMA_DIR.exists() or not any(config.CHROMA_DIR.iterdir()):
        return None, None, None

    embedding_model = MistralAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=os.getenv("MISTRAL_API_KEY"),
    )
    text_store  = get_or_create_collection(config.TEXT_COLLECTION,  embedding_model, config.CHROMA_DIR)
    image_store = get_or_create_collection(config.IMAGE_COLLECTION, embedding_model, config.CHROMA_DIR)
    llm = ChatMistralAI(
        model=config.LLM_MODEL,
        api_key=os.getenv("MISTRAL_API_KEY"),
    )
    return text_store, image_store, llm


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 GSEB Class 12 AI Tutor</h1>
    <p>Physics · Chemistry · Biology &nbsp;|&nbsp; Powered by Mistral AI</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📖 How to Use")
    st.markdown("""
    **Ask anything from your textbook:**
    - `What is Molarity?`
    - `Explain Huygens' Principle`
    - `Show me the diagram of human heart`
    - `What are the parts of a plant cell?`

    ---
    **Diagram queries** automatically retrieve the original textbook image!

    ---
    """)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.chat_history  = []
        st.rerun()

    st.markdown("---")
    st.markdown("**Subjects Available:**")
    st.markdown("- ⚡ Physics (Part 1 & 2)")
    st.markdown("- ⚗️ Chemistry (Part 1 & 2)")
    st.markdown("- 🔬 Biology")

# ─────────────────────────────────────────────────────────────────────────────
# INIT SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "messages"     not in st.session_state:
    st.session_state.messages     = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─────────────────────────────────────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
text_store, image_store, llm = load_resources()

if text_store is None:
    st.error("❌ ChromaDB not found! Please run `create_database.py` first.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# RENDER CHAT HISTORY
# ─────────────────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    role    = msg["role"]
    content = msg["content"]
    images  = msg.get("images", [])

    with st.chat_message(role, avatar="📚" if role == "user" else "🤖"):
        # Show images inline (for assistant messages)
        if images:
            for img_info in images:
                st.markdown(f'<div class="diagram-box"><div class="diagram-label">📷 Diagram: {img_info.get("caption", "Textbook Figure")}</div></div>', unsafe_allow_html=True)
                img_path = img_info.get("image_path", "")
                if img_path and Path(img_path).exists():
                    try:
                        img = load_and_crop(img_path)
                        if img:
                            st.image(img, caption=img_info.get("caption", ""), use_container_width=True)
                    except Exception:
                        st.warning(f"Could not display image: {img_path}")
                src = img_info.get("source_file", "?")
                pg  = img_info.get("page", "?")
                sub = img_info.get("subject", "?")
                st.caption(f"📖 {sub.title()} | {src} | Page {pg}")

        st.markdown(content)

# ─────────────────────────────────────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────────────────────────────────────
question = st.chat_input("Ask any question from Physics, Chemistry or Biology...")

if question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="📚"):
        st.markdown(question)

    # Process question
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                # Step 1: Reformulate follow-up if needed
                standalone = reformulate_question(question, st.session_state.chat_history, llm)

                # Step 2: Retrieve from ChromaDB
                results      = retrieve(standalone, text_store, image_store)
                text_docs    = results["text"]
                image_docs   = results["images"]
                text_context = build_text_context(text_docs)
                image_context= build_image_context(image_docs)

                # Step 3: Display retrieved images inline
                displayed_images = []
                if image_docs:
                    for img_doc in image_docs:
                        m        = img_doc.metadata
                        img_path = m.get("image_path", "")
                        caption  = m.get("caption", "Textbook Figure")
                        if img_path and Path(img_path).exists():
                            st.markdown(f'<div class="diagram-box"><div class="diagram-label">📷 {caption}</div></div>', unsafe_allow_html=True)
                            try:
                                img = load_and_crop(img_path)
                                if img:
                                    st.image(img, caption=caption, use_container_width=True)
                            except Exception as img_err:
                                st.warning(f"Image unavailable: {img_err}")
                            src = m.get("source_file", "?")
                            pg  = m.get("page", "?")
                            sub = m.get("subject", "?")
                            st.caption(f"📖 {sub.title()} | {src} | Page {pg}")
                            displayed_images.append(m)

                # Step 4: Generate answer
                is_diagram = is_diagram_query(standalone)
                if is_diagram and image_docs:
                    answer = run_vision_query(
                        image_path=image_docs[0].metadata.get("image_path", ""),
                        question=question,
                        text_context=text_context,
                        llm_model=config.VISION_LLM_MODEL,
                        api_key=os.getenv("MISTRAL_API_KEY", ""),
                        enable_vision=config.ENABLE_VISION_LLM,
                    )
                    if not answer:
                        answer = generate_diagram_answer(
                            question=question,
                            text_context=text_context,
                            image_context=image_context,
                            chat_history=st.session_state.chat_history,
                            llm=llm,
                        )
                else:
                    answer = generate_text_answer(
                        question=question,
                        text_context=text_context,
                        chat_history=st.session_state.chat_history,
                        llm=llm,
                    )

                if not answer:
                    answer = "Sorry, I couldn't generate an answer. Please try rephrasing your question."

            except Exception as e:
                answer = f"⚠️ An error occurred: {str(e)}"
                displayed_images = []
                import traceback
                print(traceback.format_exc())   # logs to terminal for debugging

            st.markdown(answer)

    # Update memory
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=answer))
    if len(st.session_state.chat_history) > config.CHAT_HISTORY_LIMIT:
        st.session_state.chat_history = st.session_state.chat_history[-config.CHAT_HISTORY_LIMIT:]

    # Save to message history for display
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "images":  displayed_images,
    })
