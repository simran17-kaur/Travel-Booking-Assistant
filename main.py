import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import shutil
import streamlit as st
from typing import List, Dict, Any

# local imports
from app import chat_logic, rag_pipeline, booking_flow
try:
    from models.llm import get_chatgroq_model
except Exception:
    get_chatgroq_model = None

# Optional admin dashboard (if present)
try:
    from app import admin_dashboard
except Exception:
    admin_dashboard = None

# Paths
DOCS_DIR = "./docs"
PERSIST_DIR = "./db/chroma_store"

# Ensure directories exist
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(PERSIST_DIR), exist_ok=True)

# ---------- Styling ----------
CHAT_CSS = """
/* Chat bubble styles */
.chat-container {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 8px;
}
.user-bubble {
  align-self: flex-end;
  background: linear-gradient(135deg,#a7f3d0,#67f1b9);
  color: #03241b;
  padding: 10px 14px;
  border-radius: 16px;
  max-width: 78%;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.assistant-bubble {
  align-self: flex-start;
  background: linear-gradient(135deg,#e6eeff,#c7d7ff);
  color: #022248;
  padding: 10px 14px;
  border-radius: 16px;
  max-width: 78%;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.meta {
  font-size: 12px;
  color: #6b7280;
  margin-top: 4px;
}
.small-muted { font-size:12px; color:#94a3b8; }
"""

# ---------- Helpers ----------
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "chat_memory" not in st.session_state:
        st.session_state["chat_memory"] = []
    if "index_obj" not in st.session_state:
        st.session_state["index_obj"] = None
    if "index_loaded" not in st.session_state:
        st.session_state["index_loaded"] = False
    if "booking_slots" not in st.session_state:
        st.session_state["booking_slots"] = {}

def save_uploaded_files_local(uploaded_files, target_dir=DOCS_DIR) -> List[str]:
    os.makedirs(target_dir, exist_ok=True)
    saved = []
    for f in uploaded_files:
        filename = os.path.basename(f.name)
        dest = os.path.join(target_dir, filename)
        if os.path.exists(dest):
            name, ext = os.path.splitext(filename)
            dest = os.path.join(target_dir, f"{name}_{int(time.time())}{ext}")
        with open(dest, "wb") as fh:
            fh.write(f.getbuffer())
        saved.append(dest)
    return saved

def build_index_and_store(docs_dir: str, index_dir: str, force_rebuild: bool = False) -> Any:
    try:
        st.session_state["index_status"] = "Indexing... (this may take a while for large PDFs)"
    except Exception:
        pass

    try:
        idx = rag_pipeline.build_or_load_index(docs_dir=docs_dir, index_dir=index_dir, force_rebuild=force_rebuild)
        st.session_state["index_obj"] = idx
        st.session_state["index_loaded"] = True
        st.session_state["index_status"] = "Index ready."
        return idx
    except Exception as e:
        st.session_state["index_obj"] = None
        st.session_state["index_loaded"] = False
        st.session_state["index_status"] = f"Indexing failed: {e}"
        st.error(f"Indexing failed: {e}")
        return None

# ---------- LLM helper ----------
def get_chat_response_adapter(chat_model, messages: List[Dict[str, str]], system_prompt: str) -> str:
    if chat_model is None:
        return "LLM not configured."

    try:
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        formatted = []
        formatted.append(SystemMessage(content=system_prompt))
        for m in messages:
            if m.get("role") == "user":
                formatted.append(HumanMessage(content=m.get("content", "")))
            elif m.get("role") == "assistant":
                formatted.append(AIMessage(content=m.get("content", "")))
            else:
                formatted.append(HumanMessage(content=m.get("content", "")))
        resp = chat_model.invoke(formatted)
        return getattr(resp, "content", str(resp))
    except Exception:
        try:
            formatted = [{"role": "system", "content": system_prompt}]
            formatted += [{"role": m.get("role"), "content": m.get("content")} for m in messages]
            resp = chat_model.invoke(formatted)
            return getattr(resp, "content", str(resp))
        except Exception as e:
            return f"LLM invocation failed: {e}"

# ---------- UI pages ----------
def show_instructions():
    st.title("Instructions")
    st.markdown(
        """
        Welcome to the Oceania Travel Assistant demo.
        - Upload PDF packages (Australia / New Zealand) using the + button in the chat.
        - Build index, then ask travel questions.
        - Book a consultation using natural language (example: "I want to book a consultation for Australia package on Dec 10").
        """
    )
    st.markdown("System uses RAG (PDF retrieval) + an LLM to give factual, sourced answers.")

def show_admin():
    st.title("Admin Dashboard")
    if admin_dashboard:
        try:
            if hasattr(admin_dashboard, "admin_page"):
                admin_dashboard.admin_page()
                return
            elif hasattr(admin_dashboard, "show_admin_page"):
                admin_dashboard.show_admin_page()
                return
        except Exception:
            st.info("Admin dashboard exists but couldn't be displayed; falling back to simple view.")
    st.markdown("## Bookings (from DB) - fallback view")
    try:
        from db.database import SessionLocal
        from db.models import Booking
        db = SessionLocal()
        rows = db.query(Booking).order_by(Booking.id.desc()).limit(100).all()
        table = []
        for b in rows:
            table.append({
                "id": b.id,
                "customer_id": b.customer_id,
                "date": b.date,
                "time": b.time,
                "status": b.status,
                "type": b.booking_type,
                "customer_email": getattr(b.customer, "email", None),
                "customer_name": getattr(b.customer, "name", None)
            })
        st.write(table)
    except Exception as e:
        st.warning(f"Could not load bookings: {e}")

from app import booking_flow

def show_booking_page():
    st.title("Book a Travel Consultation")

    with st.form("booking_form"):
        name = st.text_input("Full name *", help="3–100 characters.")
        email = st.text_input("Email *")
        phone = st.text_input("Phone number *")

        destination = st.selectbox(
            "Package / destination *",
            options=["Australia", "New Zealand"],
        )

        date = st.date_input("Preferred date *")      # calendar
        time = st.time_input("Preferred time *")      # clock

        notes = st.text_area("Other information (optional)")

        submitted = st.form_submit_button("Confirm booking")

    if submitted:
        user_text = (
    f"My name is {name}. "
    f"My email is {email}. "
    f"My phone number is {phone}. "
    f"I want to book the {destination} package on {date.isoformat()} "
    f"at {time.strftime('%H:%M')}. "
    f"Additional info: {notes}"
)

        reply = booking_flow.booking_step(user_text, st.session_state)
        st.success(reply)
        
def render_chat_messages():
    st.markdown("<style>" + CHAT_CSS + "</style>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state["messages"]:
            role = msg.get("role")
            content = msg.get("content")
            meta = msg.get("meta", {})
            if role == "user":
                st.markdown(f'<div class="user-bubble">{content}</div>', unsafe_allow_html=True)
            else:
                footer = ""
                if meta.get("sources"):
                    srcs = ", ".join(meta["sources"])
                    footer = f'<div class="meta small-muted">Sources: {srcs}</div>'
                st.markdown(f'<div class="assistant-bubble">{content}{footer}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

def handle_user_input(user_text: str):
    st.session_state["messages"].append({"role": "user", "content": user_text})
    index_obj = st.session_state.get("index_obj")
    try:
        response = chat_logic.handle_user_message(user_text, index_obj=index_obj)
    except TypeError:
        response = chat_logic.handle_user_message(user_text)

    if isinstance(response, dict):
        reply_text = response.get("text") or str(response)
        meta = response.get("meta", {})
    else:
        reply_text = str(response)
        meta = {}

    st.session_state["messages"].append({"role": "assistant", "content": reply_text, "meta": meta})

# ---------- Main app ----------
def main():
    st.set_page_config(page_title="Oceania Travel Assistant", page_icon="🌏", layout="wide")
    init_session_state()
    # Comment out automatic index loading
    # load_index_on_start()

    with st.sidebar:
        st.title("Oceania Travel Assistant")
        page = st.radio("Go to", ["Chat",  "Admin", "Instructions"])
        st.markdown("---")
        if st.button("Clear Chat"):
            st.session_state["messages"] = []
            st.session_state["chat_memory"] = []
            st.session_state["booking_slots"] = {}
            st.rerun()
        st.markdown("### Index status")
        if st.session_state.get("index_loaded"):
            st.success("Index loaded")
        else:
            st.warning("No index loaded")
        st.markdown("---")
        st.markdown("**Tips**")
        st.markdown("- Ask package questions (e.g., inclusions, itinerary).")
        st.markdown("- Say 'I want to book' to start booking flow.")
        st.markdown("- Upload policy/insurance PDFs for better answers.")

    if page == "Instructions":
        show_instructions()
        return
    

    if page == "Admin":
        show_admin()
        return

    # Chat page
    st.header("Oceania Travel Assistant")
    render_chat_messages()

    # Upload PDF in chat
    uploaded = st.file_uploader("Upload PDFs (click +)", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        with st.spinner("Saving files and building index..."):
            try:
                # Clear previous docs to ensure only uploaded PDFs are used
                if os.path.exists(DOCS_DIR):
                    shutil.rmtree(DOCS_DIR)
                os.makedirs(DOCS_DIR, exist_ok=True)

                saved = save_uploaded_files_local(uploaded, DOCS_DIR)
                idx_obj = build_index_and_store(docs_dir=DOCS_DIR, index_dir=PERSIST_DIR, force_rebuild=True)
                if idx_obj:
                    st.success(f"Indexed {len(saved)} files. You can now ask questions.")
                else:
                    st.error("Indexing failed; see messages above.")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

    # Chat input
    user_input = st.chat_input("Ask travel questions or say 'I want to book'...")
    if user_input:
        handle_user_input(user_input)
        time.sleep(0.2)
        st.rerun()

if __name__ == "__main__":
    main()
