"""CRUD + search helpers around the SQLAlchemy models."""

from __future__ import annotations

from typing import Iterable, Sequence

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from .models import Author, Citation, Paper, Tag


# ---------- Authors / Tags upsert helpers ----------

def get_or_create_author(session: Session, name: str) -> Author:
    name = name.strip()
    if not name:
        raise ValueError("Empty author name")
    obj = session.scalar(select(Author).where(Author.name == name))
    if obj is None:
        obj = Author(name=name)
        session.add(obj)
        session.flush()
    return obj


def get_or_create_tag(session: Session, name: str) -> Tag:
    name = name.strip().lower()
    if not name:
        raise ValueError("Empty tag name")
    obj = session.scalar(select(Tag).where(Tag.name == name))
    if obj is None:
        obj = Tag(name=name)
        session.add(obj)
        session.flush()
    return obj


# ---------- Paper CRUD ----------

def find_paper_by_identity(
    session: Session,
    *,
    doi: str | None = None,
    arxiv_id: str | None = None,
    sha256: str | None = None,
) -> Paper | None:
    clauses = []
    if doi:
        clauses.append(Paper.doi == doi)
    if arxiv_id:
        clauses.append(Paper.arxiv_id == arxiv_id)
    if sha256:
        clauses.append(Paper.sha256 == sha256)
    if not clauses:
        return None
    return session.scalar(select(Paper).where(or_(*clauses)))


def create_paper(
    session: Session,
    *,
    title: str,
    authors: Sequence[str] = (),
    year: int | None = None,
    doi: str | None = None,
    arxiv_id: str | None = None,
    abstract: str | None = None,
    venue: str | None = None,
    file_path: str | None = None,
    sha256: str | None = None,
    tags: Sequence[str] = (),
) -> Paper:
    paper = Paper(
        title=title.strip() or "(untitled)",
        year=year,
        doi=doi,
        arxiv_id=arxiv_id,
        abstract=abstract,
        venue=venue,
        file_path=file_path,
        sha256=sha256,
    )
    # Add the paper to the session BEFORE appending authors/tags so the
    # relationship cascade has both sides registered when we flush.
    session.add(paper)
    for a in authors:
        paper.authors.append(get_or_create_author(session, a))
    for t in tags:
        paper.tags.append(get_or_create_tag(session, t))
    session.flush()
    return paper


def get_paper(session: Session, paper_id: int) -> Paper | None:
    return session.get(Paper, paper_id)


def list_papers(
    session: Session,
    *,
    year: int | None = None,
    tag: str | None = None,
    limit: int | None = None,
) -> list[Paper]:
    stmt = select(Paper).order_by(Paper.year.desc().nullslast(), Paper.id.desc())
    if year is not None:
        stmt = stmt.where(Paper.year == year)
    if tag:
        stmt = stmt.join(Paper.tags).where(Tag.name == tag.lower())
    if limit:
        stmt = stmt.limit(limit)
    return list(session.scalars(stmt))


def list_recent_papers(
    session: Session,
    *,
    since,  # datetime
    limit: int | None = None,
) -> list[Paper]:
    """Return papers whose `created_at` is at or after `since`, newest first."""
    stmt = (
        select(Paper)
        .where(Paper.created_at >= since)
        .order_by(Paper.created_at.desc(), Paper.id.desc())
    )
    if limit:
        stmt = stmt.limit(limit)
    return list(session.scalars(stmt))


def add_tags(session: Session, paper_id: int, tag_names: Iterable[str]) -> Paper:
    paper = session.get(Paper, paper_id)
    if paper is None:
        raise LookupError(f"No paper with id {paper_id}")
    existing = {t.name for t in paper.tags}
    for name in tag_names:
        n = name.strip().lower()
        if not n or n in existing:
            continue
        paper.tags.append(get_or_create_tag(session, n))
        existing.add(n)
    session.flush()
    return paper


def update_summary(
    session: Session, paper_id: int, summary_json: str, summary_path: str | None
) -> None:
    paper = session.get(Paper, paper_id)
    if paper is None:
        raise LookupError(f"No paper with id {paper_id}")
    paper.summary_json = summary_json
    if summary_path is not None:
        paper.summary_path = summary_path
    session.flush()


def _looks_weak_title(t: str | None) -> bool:
    """A title is 'weak' if it's empty, the placeholder, or has obvious PDF
    extraction artifacts (multiple consecutive spaces, no spaces at all in a
    long string, etc.)."""
    if not t:
        return True
    s = t.strip()
    if not s or s == "(untitled)":
        return True
    # Triple spaces or wider — typical PyMuPDF font-spacing artifact
    if "   " in s:
        return True
    # Single token that's too long — no word breaks at all
    if len(s) > 25 and " " not in s:
        return True
    return False


def backfill_paper_from_summary(
    session: Session,
    paper_id: int,
    *,
    title: str | None = None,
    authors: Sequence[str] | None = None,
    year: int | None = None,
    abstract: str | None = None,
    doi: str | None = None,
    arxiv_id: str | None = None,
) -> tuple[Paper, list[str]]:
    """Fill in Paper fields using values from a structured summary.

    Returns (paper, filled_fields) so callers can report what changed.

    Rules:
      - title: replace if existing one is empty / placeholder / shows obvious
        PyMuPDF whitespace artifacts (triple spaces, no spaces in long string)
      - year/doi/arxiv_id/abstract: only write if currently NULL/empty
      - authors: only attach if the paper currently has no authors
    """
    paper = session.get(Paper, paper_id)
    if paper is None:
        raise LookupError(f"No paper with id {paper_id}")

    filled: list[str] = []

    if title and _looks_weak_title(paper.title):
        new_title = " ".join(title.split())  # collapse weird whitespace
        if new_title != paper.title:
            paper.title = new_title
            filled.append("title")

    if year is not None and paper.year is None:
        paper.year = year
        filled.append("year")

    if abstract and not paper.abstract:
        paper.abstract = abstract.strip()
        filled.append("abstract")

    if doi and not paper.doi:
        paper.doi = doi.strip()
        filled.append("doi")

    if arxiv_id and not paper.arxiv_id:
        paper.arxiv_id = arxiv_id.strip()
        filled.append("arxiv_id")

    if authors and not paper.authors:
        n_added = 0
        for name in authors:
            n = " ".join((name or "").split())
            if not n:
                continue
            paper.authors.append(get_or_create_author(session, n))
            n_added += 1
        if n_added:
            filled.append(f"authors[{n_added}]")

    session.flush()
    return paper, filled


# ---------- Deletion ----------

def delete_paper(session: Session, paper_id: int) -> bool:
    """Delete a paper row and its dependent rows (citations, M:N joins).

    Citations targeting this paper get their `target_paper_id` nulled out
    rather than the row deleted, so other papers' bibliographies stay intact.
    Returns True if a row was removed.
    """
    paper = session.get(Paper, paper_id)
    if paper is None:
        return False
    # Null out incoming citation links pointing at this paper
    for cit in session.query(Citation).filter(Citation.target_paper_id == paper_id).all():
        cit.target_paper_id = None
    session.delete(paper)
    session.flush()
    return True


def delete_all_papers(session: Session) -> int:
    """Delete every paper row (and all dependent rows). Returns count."""
    papers = session.query(Paper).all()
    n = 0
    for p in papers:
        session.delete(p)
        n += 1
    session.flush()
    return n


def delete_orphan_authors(session: Session) -> int:
    """Delete authors that no longer link to any paper."""
    orphans = session.query(Author).filter(~Author.papers.any()).all()
    for a in orphans:
        session.delete(a)
    session.flush()
    return len(orphans)


def delete_orphan_tags(session: Session) -> int:
    """Delete tags that no longer link to any paper."""
    orphans = session.query(Tag).filter(~Tag.papers.any()).all()
    for t in orphans:
        session.delete(t)
    session.flush()
    return len(orphans)


def reset_summary_columns(session: Session) -> int:
    """Null out summary_json + summary_path on every paper. Returns rows touched."""
    papers = session.query(Paper).all()
    for p in papers:
        p.summary_json = None
        p.summary_path = None
    session.flush()
    return len(papers)


def reset_chunk_counts(session: Session) -> int:
    """Set n_chunks=0 on every paper. Returns rows touched."""
    papers = session.query(Paper).all()
    for p in papers:
        p.n_chunks = 0
    session.flush()
    return len(papers)


def set_chunk_count(session: Session, paper_id: int, n: int) -> None:
    paper = session.get(Paper, paper_id)
    if paper is None:
        raise LookupError(f"No paper with id {paper_id}")
    paper.n_chunks = n
    session.flush()


# ---------- Citations ----------

def add_citation(
    session: Session,
    *,
    source_paper_id: int,
    raw_text: str,
    target_title: str | None = None,
    target_doi: str | None = None,
    target_arxiv_id: str | None = None,
    target_year: int | None = None,
) -> Citation | None:
    raw = raw_text.strip()
    if not raw:
        return None
    # de-dup
    existing = session.scalar(
        select(Citation).where(
            Citation.source_paper_id == source_paper_id, Citation.raw_text == raw
        )
    )
    if existing is not None:
        return existing
    target_id = None
    if target_doi or target_arxiv_id:
        target = find_paper_by_identity(session, doi=target_doi, arxiv_id=target_arxiv_id)
        if target is not None and target.id != source_paper_id:
            target_id = target.id
    cit = Citation(
        source_paper_id=source_paper_id,
        target_paper_id=target_id,
        raw_text=raw[:4000],
        target_title=target_title,
        target_doi=target_doi,
        target_arxiv_id=target_arxiv_id,
        target_year=target_year,
    )
    session.add(cit)
    session.flush()
    return cit


def all_citations(session: Session) -> list[Citation]:
    return list(session.scalars(select(Citation)))


# ---------- Duplicate detection ----------

def _normalize_title_for_dedup(s: str | None) -> str:
    import re as _re
    s = (s or "").lower()
    # collapse ligatures + unicode noise
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff")
    # keep only alnum
    s = _re.sub(r"[^a-z0-9]+", " ", s)
    s = _re.sub(r"\s+", " ", s).strip()
    return s


def _first_surname(authors: list[str]) -> str:
    if not authors:
        return ""
    first = authors[0]
    if "," in first:
        return first.split(",", 1)[0].strip().lower()
    tokens = first.split()
    return (tokens[-1] if tokens else first).strip().lower()


def find_duplicate_groups(
    session: Session,
    *,
    content_threshold: float = 0.97,
) -> list[dict]:
    """Group papers that are likely duplicates of each other.

    Three signal tiers, ordered strongest to weakest:
      - 'doi': same non-null DOI
      - 'arxiv': same non-null arXiv id (version-stripped)
      - 'title': same normalized title + same first-author surname
      - 'content': mean-embedding cosine similarity >= threshold

    Returns a list of groups, each dict with:
      {tier, members: [paper info dicts], reason}
    """
    import re as _re

    papers = session.query(Paper).all()
    info_by_id = {}
    for p in papers:
        info_by_id[p.id] = {
            "id": p.id,
            "title": p.title or "",
            "authors": [a.name for a in p.authors],
            "year": p.year,
            "doi": (p.doi or "").lower().strip() or None,
            "arxiv_id": _re.sub(r"v\d+$", "", (p.arxiv_id or "").strip(), flags=_re.IGNORECASE) or None,
            "sha256": p.sha256,
            "file_path": p.file_path,
            "n_chunks": p.n_chunks,
        }

    # Union-find over paper ids
    parent: dict[int, int] = {pid: pid for pid in info_by_id}
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Track reason for each merge so we can label tiers
    reasons: dict[tuple[int, int], str] = {}

    # 1) DOI exact match
    by_doi: dict[str, list[int]] = {}
    for pid, info in info_by_id.items():
        if info["doi"]:
            by_doi.setdefault(info["doi"], []).append(pid)
    for doi, ids in by_doi.items():
        if len(ids) > 1:
            for i in ids[1:]:
                union(ids[0], i)
                reasons[(min(ids[0], i), max(ids[0], i))] = f"doi={doi}"

    # 2) arXiv id exact match (version-stripped)
    by_arx: dict[str, list[int]] = {}
    for pid, info in info_by_id.items():
        if info["arxiv_id"]:
            by_arx.setdefault(info["arxiv_id"], []).append(pid)
    for aid, ids in by_arx.items():
        if len(ids) > 1:
            for i in ids[1:]:
                union(ids[0], i)
                reasons[(min(ids[0], i), max(ids[0], i))] = f"arxiv={aid}"

    # 3) Normalized title + first-author surname
    by_title: dict[tuple[str, str], list[int]] = {}
    for pid, info in info_by_id.items():
        nt = _normalize_title_for_dedup(info["title"])
        sa = _first_surname(info["authors"])
        if nt and len(nt) >= 12:
            by_title.setdefault((nt, sa), []).append(pid)
    for (nt, sa), ids in by_title.items():
        if len(ids) > 1:
            for i in ids[1:]:
                union(ids[0], i)
                if (min(ids[0], i), max(ids[0], i)) not in reasons:
                    reasons[(min(ids[0], i), max(ids[0], i))] = (
                        f"title+author: {nt[:50]!r}+{sa!r}"
                    )

    # 4) Content similarity (cosine of mean chunk embeddings)
    try:
        from bibwizard.ingestion.embedder import all_chunk_embeddings
        from collections import defaultdict as _dd
        import numpy as _np
        from sklearn.metrics.pairwise import cosine_similarity as _cs

        paper_ids, embs, _docs = all_chunk_embeddings()
        bucket: dict[int, list] = _dd(list)
        for pid, e in zip(paper_ids, embs):
            if pid and pid > 0 and pid in info_by_id:
                bucket[pid].append(e)
        if bucket:
            pids = sorted(bucket.keys())
            centroids = _np.array([_np.mean(_np.array(bucket[pid]), axis=0) for pid in pids])
            sims = _cs(centroids)
            for i in range(len(pids)):
                for j in range(i + 1, len(pids)):
                    s = float(sims[i][j])
                    if s >= content_threshold:
                        a, b = pids[i], pids[j]
                        union(a, b)
                        key = (min(a, b), max(a, b))
                        if key not in reasons:
                            reasons[key] = f"content cosine={s:.3f}"
    except Exception as e:  # noqa: BLE001
        # Embeddings unavailable — skip the content-similarity signal.
        import logging
        logging.getLogger(__name__).info(
            "Skipping content-similarity duplicate detection: %s", e
        )

    # Group by union-find root, only emit groups with > 1 member
    groups: dict[int, list[int]] = {}
    for pid in info_by_id:
        groups.setdefault(find(pid), []).append(pid)
    out = []
    for members in groups.values():
        if len(members) < 2:
            continue
        # Collect all reasons that involve a member of this group
        group_reasons = []
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                key = (min(a, b), max(a, b))
                if key in reasons:
                    group_reasons.append(f"{a}↔{b}: {reasons[key]}")
        # Tier label = strongest signal
        if any("doi=" in r for r in group_reasons):
            tier = "doi"
        elif any("arxiv=" in r for r in group_reasons):
            tier = "arxiv"
        elif any("title+author" in r for r in group_reasons):
            tier = "title"
        elif any("content cosine" in r for r in group_reasons):
            tier = "content"
        else:
            tier = "?"
        out.append({
            "tier": tier,
            "members": sorted(
                [info_by_id[m] for m in members],
                key=lambda x: x["id"],
            ),
            "reasons": group_reasons,
        })
    # Sort: doi/arxiv first, then title, then content
    tier_rank = {"doi": 0, "arxiv": 1, "title": 2, "content": 3, "?": 4}
    out.sort(key=lambda g: (tier_rank.get(g["tier"], 9), g["members"][0]["id"]))
    return out


# ---------- Stats / search-by-name ----------

def text_search(session: Session, query: str, limit: int = 25) -> list[Paper]:
    """Cheap LIKE-based search across title / abstract / authors."""
    if not query:
        return []
    pat = f"%{query}%"
    stmt = (
        select(Paper)
        .where(
            or_(
                Paper.title.ilike(pat),
                Paper.abstract.ilike(pat),
            )
        )
        .order_by(Paper.year.desc().nullslast())
        .limit(limit)
    )
    return list(session.scalars(stmt))


def library_stats(session: Session) -> dict:
    n_papers = session.scalar(select(func.count(Paper.id))) or 0
    n_authors = session.scalar(select(func.count(Author.id))) or 0
    n_tags = session.scalar(select(func.count(Tag.id))) or 0
    n_citations = session.scalar(select(func.count(Citation.id))) or 0
    n_chunks = session.scalar(select(func.coalesce(func.sum(Paper.n_chunks), 0))) or 0
    by_year = session.execute(
        select(Paper.year, func.count(Paper.id))
        .where(Paper.year.is_not(None))
        .group_by(Paper.year)
        .order_by(Paper.year.desc())
    ).all()
    return {
        "papers": n_papers,
        "authors": n_authors,
        "tags": n_tags,
        "citations": n_citations,
        "vector_chunks": n_chunks,
        "by_year": [(y, c) for y, c in by_year],
    }
