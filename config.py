# app/config.py
"""
Central configuration for the Travel Booking Assistant.

- File/directory constants used across the app
- Helpers to read SMTP config from streamlit secrets or environment variables
- Lightweight validation helper for SMTP settings
"""

import os
from typing import Dict, Any, Optional

try:
    import streamlit as st
except Exception:
    st = None  # avoid hard dependency in non-Streamlit contexts

# ---------- Paths & constants ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
CHROMA_INDEX_DIR = os.path.join(PROJECT_ROOT, "db", "chroma_store")
CHROMA_COLLECTION_NAME = "travel_docs"

# RAG defaults
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 60

# DB filename (sqlite)
SQLITE_DB_FILE = os.path.join(PROJECT_ROOT, "app.db")

# ---------- SMTP helpers ----------
def _secrets_dict() -> Dict[str, Any]:
    """
    Return a dict of streamlit secrets (if available), otherwise {}.
    """
    if st is None:
        return {}
    try:
        return st.secrets.to_dict()
    except Exception:
        # older streamlit versions or offline usage
        try:
            return dict(st.secrets)  # fallback
        except Exception:
            return {}

def get_smtp_config() -> Dict[str, Any]:
    """
    Read SMTP config from streamlit secrets under the key 'smtp' OR from environment variables.

    Supported secret keys (preferred):
      .streamlit/secrets.toml
      [smtp]
      smtp_server = "smtp.example.com"
      smtp_port = 465
      username = "user@example.com"
      password = "secret"
      from_email = "noreply@example.com"

    Environment variable fallbacks (uppercase):
      SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM
    """
    s = _secrets_dict().get("smtp", {}) if _secrets_dict() else {}

    cfg = {
        "smtp_server": s.get("smtp_server") or os.getenv("SMTP_SERVER"),
        "smtp_port": int(s.get("smtp_port") or os.getenv("SMTP_PORT") or 0) or None,
        "username": s.get("username") or os.getenv("SMTP_USERNAME"),
        "password": s.get("password") or os.getenv("SMTP_PASSWORD"),
        "from_email": s.get("from_email") or os.getenv("SMTP_FROM"),
        # optional TLS vs SSL flag
        "use_ssl": bool(s.get("use_ssl")) if s.get("use_ssl") is not None else None
    }

    # Normalize types & sensible defaults
    if cfg["smtp_port"] is None:
        # common default (SSL)
        cfg["smtp_port"] = 465

    if cfg["use_ssl"] is None:
        # default to SSL if port 465, else TLS for 587
        cfg["use_ssl"] = cfg["smtp_port"] == 465

    return cfg

def validate_smtp_config(cfg: Optional[Dict[str, Any]] = None) -> (bool, str):
    """
    Validate an SMTP configuration dict. Returns (ok, message).
    If cfg is None, it will read from get_smtp_config().
    """
    if cfg is None:
        cfg = get_smtp_config()

    required = ["smtp_server", "smtp_port", "username", "password"]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        return False, f"Missing SMTP config: {', '.join(missing)}"

    # basic sanity checks
    server = cfg.get("smtp_server")
    port = cfg.get("smtp_port")
    if not isinstance(port, int) or port <= 0 or port > 65535:
        return False, f"Invalid SMTP port: {port}"

    return True, "SMTP config looks OK."

# ---------- Utility for other modules ----------
def get_rag_paths() -> Dict[str, str]:
    """
    Return dictionary with main RAG-related paths for other modules.
    """
    return {
        "docs_dir": DOCS_DIR,
        "index_dir": CHROMA_INDEX_DIR,
        "collection_name": CHROMA_COLLECTION_NAME
    }

# Example usage:
# from app.config import get_smtp_config, validate_smtp_config
# ok, msg = validate_smtp_config()
# if not ok:
#     st.warning(msg)
