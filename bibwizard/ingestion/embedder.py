"""Chunk paper text and push embeddings into ChromaDB.

Chunking strategy:
  - Whitespace-tokenize (proxy for tokens).
  - Sliding window of CHUNK_SIZE tokens with CHUNK_OVERLAP tokens of overlap.
  - One chunk = one ChromaDB document with metadata {paper_id, title, page_hint}.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from bibwizard.llm.client import OllamaClient, get_client
from bibwizard.utils.config import settings


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_hint: int | None
    paper_id: int
    title: str


# ---------- chunking ----------

_WORD_RE = re.compile(r"\S+")


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Sliding-window word chunking. Approximates token chunking well enough."""
    if not text or not text.strip():
        return []
    cs = chunk_size or settings.chunk_size
    ov = overlap if overlap is not None else settings.chunk_overlap
    if cs <= 0:
        raise ValueError("chunk_size must be positive")
    if ov >= cs:
        raise ValueError("overlap must be smaller than chunk_size")

    words = _WORD_RE.findall(text)
    if not words:
        return []
    chunks: list[str] = []
    step = cs - ov
    for start in range(0, len(words), step):
        window = words[start : start + cs]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + cs >= len(words):
            break
    return chunks


def chunk_pages(
    pages: Sequence[tuple[int, str]],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[tuple[str, int]]:
    """Chunk per-page so we can attach a page_hint to each chunk."""
    out: list[tuple[str, int]] = []
    for page_no, text in pages:
        for c in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
            out.append((c, page_no))
    return out


# ---------- ChromaDB ----------

_chroma_client: chromadb.api.ClientAPI | None = None


def get_chroma_client() -> chromadb.api.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        settings.vectors_dir.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=str(settings.vectors_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
        )
    return _chroma_client


def get_collection(name: str | None = None):
    client = get_chroma_client()
    coll_name = name or settings.chroma_collection
    return client.get_or_create_collection(
        name=coll_name,
        metadata={"hnsw:space": "cosine"},
    )


# ---------- ingest / query ----------

def ingest_paper_chunks(
    paper_id: int,
    title: str,
    pages: Sequence[tuple[int, str]],
    *,
    ollama: OllamaClient | None = None,
    collection_name: str | None = None,
) -> int:
    """Embed all chunks of a paper and persist them to ChromaDB. Returns count."""
    client = ollama or get_client()
    client.ensure_ready(need_llm=False, need_embed=True)

    chunked = chunk_pages(pages)
    if not chunked:
        return 0

    coll = get_collection(collection_name)

    # Drop any existing chunks for this paper, so re-ingest is idempotent.
    try:
        coll.delete(where={"paper_id": paper_id})
    except Exception:
        # Some Chroma versions are picky about empty deletes — swallow.
        pass

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    embeddings: list[list[float]] = []

    for i, (text, page_no) in enumerate(chunked):
        ids.append(f"p{paper_id}-c{i:05d}")
        docs.append(text)
        metas.append(
            {
                "paper_id": int(paper_id),
                "title": title,
                "page": int(page_no) if page_no is not None else -1,
                "chunk_index": i,
            }
        )
        embeddings.append(client.embed(text))

    coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    return len(ids)


def query_chunks(
    query: str,
    *,
    top_k: int | None = None,
    paper_ids: Sequence[int] | None = None,
    ollama: OllamaClient | None = None,
    collection_name: str | None = None,
) -> list[dict]:
    """Embed `query` and return top-k chunks as a list of dicts."""
    client = ollama or get_client()
    client.ensure_ready(need_llm=False, need_embed=True)
    coll = get_collection(collection_name)
    k = top_k or settings.rag_top_k

    where = None
    if paper_ids:
        where = {"paper_id": {"$in": [int(p) for p in paper_ids]}}

    q_emb = client.embed(query)
    res = coll.query(
        query_embeddings=[q_emb],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    out: list[dict] = []
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for d, m, dist in zip(docs, metas, dists):
        score = 1.0 - float(dist) if dist is not None else 0.0
        out.append(
            {
                "text": d,
                "metadata": m or {},
                "score": score,
            }
        )
    return out


def all_chunk_embeddings() -> tuple[list[int], list[list[float]], list[str]]:
    """Pull every chunk embedding (for content map clustering)."""
    coll = get_collection()
    result = coll.get(include=["embeddings", "metadatas", "documents"])
    # Newer ChromaDB versions return numpy arrays for `embeddings`; the old
    # `value or []` short-circuit raises on numpy truthiness. Use explicit
    # None checks instead.
    embs_raw = result.get("embeddings")
    embs = list(embs_raw) if embs_raw is not None else []
    metas = result.get("metadatas") or []
    docs = result.get("documents") or []
    paper_ids = [int(m.get("paper_id", -1)) for m in metas]
    return paper_ids, embs, docs


def delete_paper_chunks(paper_id: int, collection_name: str | None = None) -> int:
    """Delete all ChromaDB chunks for a single paper. Returns deleted count."""
    coll = get_collection(collection_name)
    try:
        existing = coll.get(where={"paper_id": int(paper_id)})
    except Exception:
        return 0
    n = len(existing.get("ids") or [])
    if n:
        try:
            coll.delete(where={"paper_id": int(paper_id)})
        except Exception:
            return 0
    return n


def reset_collection(collection_name: str | None = None) -> None:
    """Drop and recreate the ChromaDB collection (wipes every chunk)."""
    name = collection_name or settings.chroma_collection
    client = get_chroma_client()
    try:
        client.delete_collection(name)
    except Exception:
        # Collection may not exist yet — fine.
        pass
    client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def save_pdf_to_library(src: Path, target_dir: Path | None = None) -> Path:
    """Copy/move a PDF into the library's papers/ directory."""
    target_dir = target_dir or settings.papers_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    src = Path(src).expanduser().resolve()
    dest = (target_dir / src.name).resolve()
    if dest != src:
        dest.write_bytes(src.read_bytes())
    return dest
