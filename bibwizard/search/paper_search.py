"""Paper-level retrieval for `bibwizard find`.

The chunk index in ChromaDB returns chunks, but for "what papers are about
X?" we want a PAPER-level ranking — one row per paper, ordered by how well
the paper as a whole matches the query.

Pipeline:
  1. Embed the user's query and retrieve a wide pool of chunks (default 50).
  2. Group those chunks by `paper_id`.
  3. Score each paper as the weighted mean of its top-N chunk scores, with
     a small penalty when only one chunk matched. This rewards papers with
     several medium-relevance passages over papers with one strong hit and
     nothing else.
  4. Hydrate the top-K papers from SQLite (title, authors, year) and pick
     each paper's best-scoring chunk as the display snippet.

No LLM call required for the ranking itself. An optional `with_reasons=True`
adds a one-line "why this matched" blurb per row (cheap: one small LLM call
per result).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from bibwizard.database.migrations import session_scope
from bibwizard.database.models import Paper
from bibwizard.ingestion.embedder import query_chunks


@dataclass
class PaperHit:
    """A paper-level search result."""

    paper_id: int
    score: float
    title: str
    year: int | None
    authors: list[str]
    venue: str | None
    cite: str
    best_snippet: str
    best_page: int
    n_chunks_matched: int

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "score": self.score,
            "title": self.title,
            "year": self.year,
            "authors": self.authors,
            "venue": self.venue,
            "cite": self.cite,
            "best_snippet": self.best_snippet,
            "best_page": self.best_page,
            "n_chunks_matched": self.n_chunks_matched,
        }


# Score = mean of the top-N chunk scores for the paper, where N is `top_chunks`.
# A paper with fewer than `top_chunks` matches gets a small penalty so a paper
# with three medium hits beats a paper with one strong hit and nothing else.
_SINGLE_HIT_PENALTY = 0.85
_TWO_HIT_PENALTY = 0.95


def _score_paper(chunk_scores: list[float], top_chunks: int) -> float:
    if not chunk_scores:
        return 0.0
    top = sorted(chunk_scores, reverse=True)[:top_chunks]
    mean = sum(top) / len(top)
    if len(chunk_scores) == 1:
        return mean * _SINGLE_HIT_PENALTY
    if len(chunk_scores) == 2:
        return mean * _TWO_HIT_PENALTY
    return mean


def _short_cite(authors: Sequence[str], year: int | None) -> str:
    """`Smith et al. 2024` style citation."""
    if not authors:
        return f"(unknown){f' {year}' if year else ''}".strip()
    first = authors[0]
    if "," in first:
        last = first.split(",", 1)[0].strip()
    else:
        tokens = first.split()
        last = tokens[-1] if tokens else first
    yr = f" {year}" if year else ""
    if len(authors) == 1:
        return f"{last}{yr}"
    if len(authors) == 2:
        second = authors[1]
        sec_last = (
            second.split(",", 1)[0].strip()
            if "," in second
            else (second.split()[-1] if second.split() else second)
        )
        return f"{last} & {sec_last}{yr}"
    return f"{last} et al.{yr}"


def find_papers(
    query: str,
    *,
    top_k: int = 10,
    chunks_per_paper: int = 3,
    pool_size: int = 50,
) -> list[PaperHit]:
    """Return up to `top_k` papers ranked by relevance to `query`.

    Args:
      query: free-text search query.
      top_k: how many paper results to return.
      chunks_per_paper: how many of a paper's best chunks contribute to its
        score (3 is a good default — see _score_paper).
      pool_size: how many chunks to pull from ChromaDB before grouping. Set
        higher for larger libraries; default 50 covers typical libraries up
        to a few hundred papers.

    Returns:
      A list of PaperHit, ordered by descending score.
    """
    if not query or not query.strip():
        return []

    chunks = query_chunks(query, top_k=pool_size)
    if not chunks:
        return []

    # Group chunks by paper_id, retaining the best snippet/page per paper.
    by_paper: dict[int, dict] = {}
    for ch in chunks:
        meta = ch.get("metadata") or {}
        pid = int(meta.get("paper_id", -1))
        if pid < 0:
            continue
        score = float(ch.get("score", 0.0))
        page = int(meta.get("page", -1))
        text = ch.get("text", "") or ""
        bucket = by_paper.setdefault(
            pid,
            {"scores": [], "best_score": -1.0, "best_text": "", "best_page": -1},
        )
        bucket["scores"].append(score)
        if score > bucket["best_score"]:
            bucket["best_score"] = score
            bucket["best_text"] = text
            bucket["best_page"] = page

    # Compute the per-paper composite score
    scored = []
    for pid, bucket in by_paper.items():
        s = _score_paper(bucket["scores"], chunks_per_paper)
        scored.append(
            (
                s,
                pid,
                bucket["best_text"],
                bucket["best_page"],
                len(bucket["scores"]),
            )
        )
    scored.sort(key=lambda r: -r[0])
    scored = scored[:top_k]

    # Hydrate paper metadata in one DB round-trip
    paper_ids = [pid for _, pid, *_ in scored]
    info: dict[int, dict] = {}
    with session_scope() as session:
        for p in session.query(Paper).filter(Paper.id.in_(paper_ids)).all():
            info[p.id] = {
                "title": p.title or "(untitled)",
                "year": p.year,
                "authors": [a.name for a in p.authors],
                "venue": p.venue,
            }

    hits: list[PaperHit] = []
    for score, pid, snippet, page, n_matched in scored:
        meta = info.get(pid)
        if not meta:
            # Paper row vanished (deleted but chunks still in Chroma) — skip
            continue
        authors = meta["authors"]
        hits.append(
            PaperHit(
                paper_id=pid,
                score=score,
                title=meta["title"],
                year=meta["year"],
                authors=authors,
                venue=meta["venue"],
                cite=_short_cite(authors, meta["year"]),
                best_snippet=_trim_snippet(snippet, query),
                best_page=page,
                n_chunks_matched=n_matched,
            )
        )
    return hits


def _trim_snippet(text: str, query: str, max_len: int = 280) -> str:
    """Return a window around the first query-word match if possible."""
    if not text:
        return ""
    text = " ".join(text.split())  # collapse whitespace
    if len(text) <= max_len:
        return text
    # Try to center on the first interesting query word
    q_tokens = [t for t in query.lower().split() if len(t) >= 4]
    pos = -1
    low = text.lower()
    for tok in q_tokens:
        i = low.find(tok)
        if i >= 0:
            pos = i
            break
    if pos < 0:
        return text[:max_len].rstrip() + "..."
    half = max_len // 2
    start = max(0, pos - half)
    end = min(len(text), start + max_len)
    snippet = text[start:end].strip()
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"
