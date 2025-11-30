# app/chat_logic.py
"""
Handles:
- intent detection (booking vs rag vs smalltalk)
- short-term memory (20-25 messages) kept in st.session_state['chat_memory']
- routing queries to rag_pipeline and LLM (via chat_completion)
- minimal sanitization & error handling
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, List
import streamlit as st

from models.llm import chat_completion
from app import booking_flow, rag_pipeline

# ---------------- Memory helpers ----------------
MAX_MEMORY = 25


def _add_to_memory(role: str, text: str):
    mem = st.session_state.setdefault("chat_memory", [])
    mem.append({"role": role, "content": text})
    if len(mem) > MAX_MEMORY:
        mem.pop(0)
    st.session_state["chat_memory"] = mem


def get_memory_messages() -> List[Dict[str, str]]:
    """Return a copy of recent memory as list of dicts {"role","content"}."""
    return st.session_state.get("chat_memory", []).copy()


# ---------------- Intent detection ----------------
BOOKING_KEYWORDS = [
    "book",
    "appointment",
    "reserve",
    "schedule",
    "consultation",
    "meeting",
    "trip",
    "seat",
]
RAG_KEYWORDS = [
    "pdf",
    "document",
    "policy",
    "package",
    "itinerary",
    "visa",
    "insurance",
    "included",
    "inclusion",
]


def detect_intent(text: str) -> str:
    txt = text.lower()
    if any(k in txt for k in BOOKING_KEYWORDS):
        return "booking"
    if any(k in txt for k in RAG_KEYWORDS):
        return "rag"
    if any(g in txt for g in ["hi", "hello", "hey", "thanks", "thank you"]):
        return "smalltalk"
    return "rag"


# ---------------- RAG + LLM integration ----------------
def handle_rag_query(user_text: str) -> str:
    """Use Chroma index + LLM to answer from PDFs."""
    index_obj = st.session_state.get("index_obj")
    if not index_obj:
        return "No documents indexed yet. Please upload PDFs and try again."

    try:
        rag_res = rag_pipeline.answer_with_rag(
            query=user_text,
            index_obj=index_obj,
            llm_fn=chat_completion,
            top_k=4,
        )
        answer = rag_res.get("answer", "")
        if not answer:
            answer = "I could not generate a concise answer from the documents."
    except Exception as e:
        answer = f"RAG processing failed: {e}"

    _add_to_memory("user", user_text)
    _add_to_memory("assistant", answer)
    return answer


# ---------------- Booking routing ----------------
def handle_booking_intent(user_text: str) -> str:
    """Route message into the booking slot-filling flow."""
    reply = booking_flow.booking_step(user_text, st.session_state)
    _add_to_memory("user", user_text)
    _add_to_memory("assistant", reply)
    return reply


# ---------------- Public entry ----------------
def handle_user_message(user_text: str) -> str:
    """
    Top-level router for incoming user messages.
    Call this from main.py when the user sends a message.
    """
    # 1) If a booking is already in progress, ALWAYS stay in booking flow
    if st.session_state.get("booking_started"):
        return handle_booking_intent(user_text)

    # 2) Otherwise use intent detection
    intent = detect_intent(user_text)
    if intent == "booking":
        return handle_booking_intent(user_text)
    elif intent == "smalltalk":
        _add_to_memory("user", user_text)
        reply = (
            "Hi! I can help with travel package questions and book consultation appointments. "
            "What would you like to do?"
        )
        _add_to_memory("assistant", reply)
        return reply
    else:
        return handle_rag_query(user_text)
