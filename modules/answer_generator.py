"""
modules/answer_generator.py
============================
Generates the final answer using the LLM.
Handles both text-only and multimodal (text + image metadata) queries.
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

logger = logging.getLogger(__name__)

# ── Prompt for standard text-only answers ────────────────────────────────────
TEXT_SYSTEM_PROMPT = (
    "You are an expert tutor for GSEB Class 12 students covering Physics, Chemistry, and Biology. "
    "Answer the student's question using ONLY the context retrieved from the textbooks below. "
    "Be clear, accurate, and student-friendly. Include formulas or key terms where relevant. "
    "If the answer is not in the context, say: "
    "'I don't have enough information on that topic in the loaded textbook.'\n\n"
    "Context from Textbook:\n{context}"
)

# ── Prompt for diagram/image answers ────────────────────────────────────────
DIAGRAM_SYSTEM_PROMPT = (
    "You are an expert tutor for GSEB Class 12 students covering Physics, Chemistry, and Biology. "
    "A diagram has been retrieved from the textbook and is being displayed to the student. "
    "Use the diagram information and the textbook context below to provide a comprehensive explanation.\n\n"
    "Your answer MUST include ALL of the following sections:\n"
    "1. **Overview** — brief introduction to the diagram/topic\n"
    "2. **Labelled Parts** — explain every labelled component visible in the diagram\n"
    "3. **How It Works** — functional explanation\n"
    "4. **Important Exam Points** — key facts that frequently appear in board exams\n"
    "5. **Common Mistakes** — errors students often make\n"
    "6. **Quick Revision Notes** — 3–5 bullet points for fast revision\n\n"
    "Diagram Information:\n{image_context}\n\n"
    "Textbook Context:\n{text_context}"
)

# ── Reformulation prompt ──────────────────────────────────────────────────────
REFORMULATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a question reformulator. Given a conversation history and a new user question, "
     "rewrite the question as a fully standalone question that does not rely on the conversation "
     "history to be understood. Do NOT answer the question — only rewrite it if needed, "
     "otherwise return it exactly as-is. Output ONLY the reformulated question."),
    MessagesPlaceholder("chat_history"),
    ("human", "Reformulate this question: {question}"),
])


def reformulate_question(
    question: str,
    chat_history: list,
    llm: Any,
) -> str:
    """
    If there is chat history, reformulate the question into a standalone query.
    Returns the original question if no history exists.
    """
    if not chat_history:
        return question
    try:
        chain    = REFORMULATE_PROMPT | llm
        result   = chain.invoke({"chat_history": chat_history, "question": question})
        reformed = result.content.strip()
        if reformed:
            return reformed
    except Exception as e:
        logger.warning(f"Question reformulation failed: {e}. Using original.")
    return question


def generate_text_answer(
    question: str,
    text_context: str,
    chat_history: list,
    llm: Any,
) -> str:
    """Generate a text-only answer."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", TEXT_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])
    try:
        chain  = prompt | llm
        result = chain.invoke({
            "context":      text_context,
            "chat_history": chat_history,
            "question":     question,
        })
        return result.content
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return "⚠️ An error occurred while generating the answer. Please try again."


def generate_diagram_answer(
    question: str,
    text_context: str,
    image_context: str,
    chat_history: list,
    llm: Any,
) -> str:
    """Generate an answer that incorporates retrieved diagram metadata."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", DIAGRAM_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])
    try:
        chain  = prompt | llm
        result = chain.invoke({
            "text_context":  text_context,
            "image_context": image_context,
            "chat_history":  chat_history,
            "question":      question,
        })
        return result.content
    except Exception as e:
        logger.error(f"Diagram answer generation failed: {e}")
        # Fall back to text-only
        return generate_text_answer(question, text_context, chat_history, llm)
