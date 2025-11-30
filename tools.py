# app/tools.py
"""
Utility helpers for the Travel Booking Assistant.

Functions:
- ensure_docs_dir
- save_uploaded_files
- list_pdfs_in_docs
- list_indexed_sources
- format_retrieved_for_display

These are small, defensive helpers used by the Streamlit UI and RAG pipeline.
"""

from typing import List, Dict, Any, Iterable, Tuple
import os
from pathlib import Path
import shutil
import time

# Allowed upload types
ALLOWED_EXTENSIONS = {".pdf"}

# Default docs directory (relative to project root via app package)
DEFAULT_DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


def ensure_docs_dir(docs_dir: str = DEFAULT_DOCS_DIR) -> None:
    """Ensure the docs directory exists."""
    Path(docs_dir).mkdir(parents=True, exist_ok=True)


def _safe_filename(orig_name: str) -> str:
    """
    Create a filesystem-safe filename that avoids collisions by appending a timestamp
    if the file already exists.
    """
    base = os.path.basename(orig_name)
    name, ext = os.path.splitext(base)
    name = "".join(ch for ch in name if ch.isalnum() or ch in (" ", "_", "-")).rstrip()
    ext = ext.lower()
    if not ext:
        ext = ".pdf"
    safe = f"{name}{ext}"
    return safe


def save_uploaded_files(uploaded_files: Iterable, docs_dir: str = DEFAULT_DOCS_DIR) -> List[str]:
    """
    Save an iterable of Streamlit UploadedFile objects (or any file-like object with .name and .read())
    into docs_dir. Returns list of saved absolute file paths.

    - Validates extension (.pdf)
    - Avoids overwriting by adding a timestamp suffix if necessary
    - Raises ValueError for invalid files
    """
    ensure_docs_dir(docs_dir)
    saved_paths: List[str] = []

    for f in uploaded_files:
        # Streamlit UploadedFile has .name and .read()
        try:
            fname = getattr(f, "name", None) or getattr(f, "filename", None)
            if not fname:
                raise ValueError("Uploaded file missing name attribute.")
            _, ext = os.path.splitext(fname)
            if ext.lower() not in ALLOWED_EXTENSIONS:
                raise ValueError(f"Unsupported file type: {ext}. Only PDF files are allowed.")
            safe = _safe_filename(fname)
            dest = os.path.join(docs_dir, safe)

            # avoid collisions: append timestamp if file exists
            if os.path.exists(dest):
                ts = int(time.time())
                name_only, ext_only = os.path.splitext(safe)
                dest = os.path.join(docs_dir, f"{name_only}_{ts}{ext_only}")

            # If the object has .read(), write bytes
            if hasattr(f, "read"):
                content = f.read()
                # Some frameworks give text strings; ensure bytes
                if isinstance(content, str):
                    content = content.encode("utf-8")
                with open(dest, "wb") as out:
                    out.write(content)
            else:
                # fallback: if it's a path-like or fileobject with .name
                shutil.copyfile(fname, dest)

            saved_paths.append(os.path.abspath(dest))
        except Exception as exc:
            # Raise the error upward so UI can catch and show to user
            raise ValueError(f"Failed to save uploaded file '{getattr(f,'name',str(f))}': {exc}")

    return saved_paths


def list_pdfs_in_docs(docs_dir: str = DEFAULT_DOCS_DIR) -> List[str]:
    """Return sorted list of PDF filenames (absolute paths) in docs_dir."""
    ensure_docs_dir(docs_dir)
    files = [
        os.path.join(docs_dir, f)
        for f in sorted(os.listdir(docs_dir))
        if os.path.isfile(os.path.join(docs_dir, f)) and os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS
    ]
    return files


def list_indexed_sources(index_obj: Dict[str, Any]) -> List[str]:
    """
    Given an index_obj returned from rag_pipeline.load_index() or build_or_load_index(),
    return a list of unique source filenames found in index_obj["metadatas"].

    If index_obj is None or missing metadatas, returns an empty list.
    """
    if not index_obj:
        return []
    metas = index_obj.get("metadatas") or []
    sources = []
    for m in metas:
        # metadata entries may be dicts or strings (defensive)
        if isinstance(m, dict):
            s = m.get("source") or m.get("file") or None
        else:
            s = None
        if s and s not in sources:
            sources.append(s)
    return sources


def format_retrieved_for_display(retrieved: List[Dict[str, Any]], snippet_len: int = 300) -> List[Dict[str, Any]]:
    """
    Format a list of retrieved chunks (from rag_pipeline.retrieve) into a display-friendly list.

    Each returned dict contains:
      - snippet: truncated chunk text (snippet_len chars)
      - source: source filename
      - page: page number (if available)
      - score: similarity score (float)
    """
    out = []
    for r in retrieved or []:
        chunk = r.get("chunk", "") or ""
        snippet = chunk if len(chunk) <= snippet_len else chunk[:snippet_len].rsplit(" ", 1)[0] + "..."
        out.append({
            "snippet": snippet,
            "source": r.get("source"),
            "page": r.get("page"),
            "score": float(r.get("score", 0.0)) if r.get("score") is not None else 0.0,
        })
    return out


# Example usage (for your Streamlit pages):
# from app.tools import save_uploaded_files, list_indexed_sources, format_retrieved_for_display
# saved = save_uploaded_files(st.file_uploader(..., accept_multiple_files=True))
# files_in_docs = list_pdfs_in_docs()
# sources = list_indexed_sources(st.session_state.get("index_obj"))
# formatted = format_retrieved_for_display(retrieved_hits)