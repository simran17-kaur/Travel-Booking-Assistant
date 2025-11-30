# app/rag_pipeline.py
"""
Robust Chroma-backed RAG pipeline (version-compatible).

Features:
- Works across multiple chromadb versions (PersistentClient or Client)
- Avoids deprecated configuration warnings
- Tolerant collection.get / collection.query / collection.delete differences
- Exposes: load_index(index_dir=...), build_or_load_index(...), retrieve(...)
"""

import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer

from models.llm import chat_completion

# Try to import modern PersistentClient, otherwise fallback to chromadb.Client
try:
    from chromadb import PersistentClient as ChromaPersistentClient
except Exception:
    ChromaPersistentClient = None
try:
    import chromadb
    from chromadb import Client as ChromaClient
except Exception:
    chromadb = None
    ChromaClient = None

# ---------------- CONFIG ----------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 60

DOCS_DIR_DEFAULT = "./docs"
INDEX_DIR_DEFAULT = "./db/chroma_store"
COLLECTION_NAME = "travel_docs"

_EMBED_MODEL = None


def _get_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBED_MODEL


# ---------------- PDF extraction ----------------
def extract_text_from_pdf(path: str) -> List[Tuple[int, str]]:
    pages = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()
                text = " ".join(text.split())
                if text:
                    pages.append((i, text))
    except Exception as e:
        print(f"[rag_pipeline] PDF read error in {path}: {e}")
    return pages


# ---------------- Chunking ----------------
def chunk_text(text: str) -> List[str]:
    if not text:
        return []
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - CHUNK_OVERLAP
    return chunks


# ---------------- Chroma client helper ----------------
def _ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def _create_client(persist_directory: str):
    """
    Create a chroma client in a version-tolerant way.
    Returns a client instance and a flag `is_persistent` indicating whether it supports persistence API.
    """
    _ensure_dir(persist_directory)
    # Try PersistentClient first (modern)
    if ChromaPersistentClient is not None:
        try:
            client = ChromaPersistentClient(path=persist_directory)
            return client, True
        except Exception:
            pass
    # Fallback to older Client
    if ChromaClient is not None:
        try:
            client = ChromaClient()
            return client, False
        except Exception:
            pass
    # last resort - try chromadb.Client() dynamically
    try:
        import chromadb as _ch
        client_cls = getattr(_ch, "Client", None)
        if client_cls:
            return client_cls(), False
    except Exception:
        pass
    raise RuntimeError("[rag_pipeline] Could not construct a chroma client. Install chromadb.")


# ---------------- Build index (in-memory) ----------------
def build_index_from_docs(docs_dir: str = DOCS_DIR_DEFAULT) -> Dict[str, Any]:
    docs_dir = os.path.abspath(docs_dir)
    pdf_files = [
        os.path.join(docs_dir, f) for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {docs_dir}")

    all_ids, all_texts, all_meta = [], [], []
    for fp in pdf_files:
        file_name = os.path.basename(fp)
        pages = extract_text_from_pdf(fp)
        for page_no, content in pages:
            chunks = chunk_text(content)
            for ch in chunks:
                all_texts.append(ch)
                all_meta.append({"source": file_name, "page": page_no, "text": ch})
                all_ids.append(str(uuid.uuid4()))

    model = _get_model()
    embeddings = model.encode(all_texts, convert_to_numpy=True).astype("float32")

    return {"ids": all_ids, "documents": all_texts, "metadatas": all_meta, "embeddings": embeddings}


# ---------------- Save index ----------------
def save_index(index_obj: Dict[str, Any], index_dir: str = INDEX_DIR_DEFAULT, collection_name: str = COLLECTION_NAME):
    """
    Persist embeddings/metadatas into a Chroma collection, robust across versions.
    """
    client, is_persistent = _create_client(index_dir)

    # create or get collection
    try:
        try:
            collection = client.get_collection(collection_name)
        except Exception:
            collection = client.create_collection(collection_name)
    except Exception as e:
        try:
            if hasattr(client, "shutdown"):
                client.shutdown()
        except Exception:
            pass
        raise RuntimeError(f"[rag_pipeline] Could not get/create collection: {e}")

    # Attempt to fetch existing ids in a safe way
    ids_to_delete = []
    try:
        try:
            existing = collection.get(include=["documents", "metadatas"])
        except Exception:
            try:
                existing = collection.get()
            except Exception:
                existing = {}
        ids_to_delete = existing.get("ids", [])
    except Exception:
        ids_to_delete = []

    # Attempt deletion only if we have ids
    if ids_to_delete:
        try:
            try:
                collection.delete(ids_to_delete)
            except TypeError:
                collection.delete(ids=ids_to_delete)
        except Exception:
            pass

    # Add vectors (convert embeddings to list)
    try:
        embeds = index_obj["embeddings"]
        if hasattr(embeds, "tolist"):
            embeds_to_add = embeds.tolist()
        else:
            embeds_to_add = [list(e) for e in embeds]
        collection.add(
            ids=index_obj["ids"],
            documents=index_obj["documents"],
            metadatas=index_obj["metadatas"],
            embeddings=embeds_to_add,
        )
    except Exception as e:
        try:
            if hasattr(client, "shutdown"):
                client.shutdown()
        except Exception:
            pass
        raise RuntimeError(f"[rag_pipeline] Failed to add vectors: {e}")

    # persist/shutdown in a version-tolerant way
    try:
        if is_persistent and hasattr(client, "persist"):
            client.persist()
        elif hasattr(client, "shutdown"):
            client.shutdown()
    except Exception as e:
        print(f"[rag_pipeline] Warning: client persist/shutdown failed: {e}")


# ---------------- Load index ----------------
def load_index(index_dir: str = INDEX_DIR_DEFAULT, collection_name: str = COLLECTION_NAME) -> Optional[Dict[str, Any]]:
    if not os.path.exists(index_dir):
        return None

    try:
        client, _ = _create_client(index_dir)
        collection = client.get_collection(collection_name)
    except Exception:
        return None

    try:
        try:
            data = collection.get(include=["documents", "metadatas"])
        except Exception:
            data = collection.get()
    except Exception:
        data = {}

    docs = data.get("documents", [])
    mds = data.get("metadatas", [])
    ids = data.get("ids", [])

    metadatas = []
    count = max(len(docs), len(mds), len(ids))
    for i in range(count):
        md = mds[i] if i < len(mds) else {}
        if "text" not in md and i < len(docs):
            md["text"] = docs[i]
        metadatas.append(md)

    try:
        if hasattr(client, "shutdown"):
            client.shutdown()
    except Exception:
        pass

    return {"persist_directory": index_dir, "collection_name": collection_name, "metadatas": metadatas}


# ---------------- Build or load ----------------
def build_or_load_index(docs_dir: str = DOCS_DIR_DEFAULT, index_dir: str = INDEX_DIR_DEFAULT, force_rebuild: bool = False, collection_name: str = COLLECTION_NAME):
    if not force_rebuild:
        loaded = load_index(index_dir=index_dir, collection_name=collection_name)
        if loaded:
            print("[rag_pipeline] Loaded existing Chroma index.")
            return loaded

    built = build_index_from_docs(docs_dir)
    save_index(built, index_dir=index_dir, collection_name=collection_name)
    return load_index(index_dir=index_dir, collection_name=collection_name)


# ---------------- Retrieval ----------------
def retrieve(query: str, index_obj: Dict[str, Any], top_k: int = 4) -> List[Dict[str, Any]]:
    if index_obj is None:
        raise ValueError("Index object is None. Build or load an index first.")

    persist_dir = index_obj.get("persist_directory") or INDEX_DIR_DEFAULT
    collection_name = index_obj.get("collection_name") or COLLECTION_NAME

    client, _ = _create_client(persist_dir)
    collection = client.get_collection(collection_name)

    model = _get_model()
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
    query_vec = q_emb[0].tolist()

    try:
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        try:
            results = collection.query(
                query_embeddings=[query_vec],
                n_results=top_k,
                include=["documents", "metadatas"],
            )
        except Exception:
            results = collection.query(query_embeddings=[query_vec], n_results=top_k)

    ids = []
    docs = []
    mds = []
    dists = []
    try:
        ids = results.get("ids", [[]])[0]
    except Exception:
        ids = results.get("ids", []) or []
    try:
        docs = results.get("documents", [[]])[0]
    except Exception:
        docs = results.get("documents", []) or []
    try:
        mds = results.get("metadatas", [[]])[0]
    except Exception:
        mds = results.get("metadatas", []) or []
    try:
        dists = results.get("distances", [[]])[0]
    except Exception:
        dists = [0.0] * max(len(ids), len(docs), len(mds))

    hits = []
    count = max(len(ids), len(docs), len(mds), len(dists))
    for i in range(count):
        hit_id = ids[i] if i < len(ids) else None
        doc_text = docs[i] if i < len(docs) else ""
        meta = mds[i] if i < len(mds) else {}
        score = float(dists[i]) if i < len(dists) else 0.0
        hits.append(
            {
                "score": score,
                "chunk_id": hit_id,
                "chunk": doc_text,
                "source": meta.get("source"),
                "page": meta.get("page"),
            }
        )

    try:
        if hasattr(client, "shutdown"):
            client.shutdown()
    except Exception:
        pass

    return hits


# ---------------- Prompt assembly ----------------
def assemble_prompt(query: str, retrieved: List[Dict[str, Any]], max_chars: int = 2000) -> str:
    parts = []
    total = 0
    for r in retrieved:
        seg = f"Source: {r.get('source')} (page {r.get('page')})\n{r.get('chunk')}\n---\n"
        if total + len(seg) > max_chars:
            break
        parts.append(seg)
        total += len(seg)
    ctx = "\n".join(parts)
    return (
        "You are a travel assistant. Use only the context below to answer the question.\n\n"
        f"Context:\n{ctx}\nQuestion:\n{query}\n"
        "Answer concisely in 2–4 sentences and mention file names when relevant."
    )


# ---------------- RAG entrypoint ----------------
def answer_with_rag(query: str, index_obj: Dict[str, Any], llm_fn=None, top_k: int = 4) -> Dict[str, Any]:
    if index_obj is None:
        raise ValueError("Index object is None. Build or load an index first.")

    if llm_fn is None:
        llm_fn = chat_completion

    retrieved = retrieve(query, index_obj, top_k=top_k)
    prompt = assemble_prompt(query, retrieved)

    try:
        answer = llm_fn(prompt)
    except Exception as e:
        files = {r.get("source") for r in retrieved if r.get("source")}
        flist = ", ".join(files) if files else "your documents"
        answer = f"I found relevant information in {flist} but could not call the LLM ({e})."

    if not answer:
        files = {r.get("source") for r in retrieved if r.get("source")}
        flist = ", ".join(files) if files else "your documents"
        answer = f"I found relevant information in {flist}, but could not generate a detailed answer."

    return {"retrieved": retrieved, "prompt": prompt, "answer": answer}