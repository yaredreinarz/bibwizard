"""Query router for chat — sends questions to the right backend.

Qwen2.5:7b (and similar small quantized models) have poor "needle in
haystack" performance once the context grows past ~5-8k tokens of unstructured
text. We were asking it to "find paper 28 in this 20k-token dump and list
2021 papers". That doesn't work reliably — the model attends to the end of
context (the retrieved chunks) and skims the middle (the library overview).

This module classifies the question and routes structural questions
(list/count/specific-paper-lookup) to deterministic SQL queries, then asks
the LLM only to format / narrate the result. RAG-style semantic questions
still go through the embedding pipeline.

Routing keeps the LLM doing what it's good at (writing prose, summarizing,
explaining concepts) and stops it doing what it's bad at (acting as a
search engine over a giant pasted context).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from bibwizard.database.migrations import session_scope
from bibwizard.database.models import Author, Paper, PaperAuthor, Tag


QueryType = Literal[
    "list_papers",
    "count_papers",
    "specific_paper",
    "library_summary",
    "rag",
]


@dataclass
class QueryIntent:
    type: QueryType
    year: int | None = None
    year_range: tuple[int, int] | None = None
    author_surname: str | None = None
    tag: str | None = None
    paper_id: int | None = None
    title_query: str | None = None
    raw_question: str = ""
    notes: list[str] = field(default_factory=list)


# ---------- regex patterns ----------

_RE_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")
_RE_YEAR_RANGE = re.compile(r"\b((?:19|20)\d{2})\s*[-–to]+\s*((?:19|20)\d{2})\b")
_RE_PAPER_ID = re.compile(r"\bpaper\s*(?:id\s*[:=]\s*|#)?(\d{1,4})\b", re.IGNORECASE)
_RE_AUTHOR_YEAR = re.compile(
    r"\b([A-Z][a-zà-ÿ'\-]{2,})\s+(?:et\s*al\.?\s*)?\(?((?:19|20)\d{2})\)?",
)
# 'by Smith' / 'by Mawet et al' / 'authored by Smith'
_RE_BY_AUTHOR = re.compile(
    r"\b(?:by|authored\s+by|from)\s+([A-Z][a-zà-ÿ'\-]{2,})(?:\s+et\s+al\.?)?",
)

_LIST_VERBS = re.compile(
    r"\b(list|show|enumerate|what\s+(papers|are\s+the\s+papers)|which\s+papers)\b",
    re.IGNORECASE,
)
_COUNT_VERBS = re.compile(
    r"\b(how\s+many|count(\s+of)?\s+(papers|articles))\b",
    re.IGNORECASE,
)
_SUMMARY_VERBS = re.compile(
    r"\b(summari[sz]e\s+(my|the)\s+(library|database|papers)|"
    r"overview\s+of\s+(the|my)\s+(library|field|database|papers)|"
    r"what\s+(do|did)\s+(i|you)\s+have|"
    r"what.?s\s+in\s+(my|the)\s+(library|database)|"
    r"what\s+(topics|fields|areas|themes))\b",
    re.IGNORECASE,
)
_SPECIFIC_PAPER_VERBS = re.compile(
    r"\b(what\s+does|tell\s+me\s+about|describe|explain|summari[sz]e|"
    r"in\s+the\s+paper|according\s+to)\b",
    re.IGNORECASE,
)
_TAG_HINT = re.compile(
    r"\b(?:about|on|tagged|with\s+the\s+tag)\s+[\"']?([A-Za-z][A-Za-z0-9\- ]{2,40})[\"']?",
    re.IGNORECASE,
)


# ---------- classification ----------

def classify(question: str) -> QueryIntent:
    """Decide which path handles this question."""
    q = (question or "").strip()
    intent = QueryIntent(type="rag", raw_question=q)

    # Year filters
    rng = _RE_YEAR_RANGE.search(q)
    if rng:
        y1, y2 = int(rng.group(1)), int(rng.group(2))
        intent.year_range = (min(y1, y2), max(y1, y2))
    else:
        years = _RE_YEAR.findall(q)
        if len(years) == 1:
            intent.year = int(years[0])

    # Author filters: "by Smith", or "Smith et al" (without year)
    m = _RE_BY_AUTHOR.search(q)
    if m:
        intent.author_surname = m.group(1)

    # Paper id
    m = _RE_PAPER_ID.search(q)
    if m:
        intent.paper_id = int(m.group(1))

    # Author + year together — strong "specific paper" signal
    m = _RE_AUTHOR_YEAR.search(q)
    if m:
        surname, yr = m.group(1), int(m.group(2))
        # Don't get tricked by phrases like "in 2021 papers"
        if surname.lower() not in {"papers", "paper", "library", "database"}:
            intent.author_surname = intent.author_surname or surname
            if not intent.year_range:
                intent.year = intent.year or yr

    # Order matters: most-specific intents first.
    if intent.paper_id is not None:
        intent.type = "specific_paper"
        return intent

    if _RE_AUTHOR_YEAR.search(q) and _SPECIFIC_PAPER_VERBS.search(q):
        intent.type = "specific_paper"
        return intent

    if _COUNT_VERBS.search(q):
        intent.type = "count_papers"
        return intent

    if _LIST_VERBS.search(q):
        intent.type = "list_papers"
        return intent

    if _SUMMARY_VERBS.search(q):
        intent.type = "library_summary"
        return intent

    # "Author + year" without an explicit verb still strongly implies lookup
    if _RE_AUTHOR_YEAR.search(q):
        intent.type = "specific_paper"
        return intent

    return intent


# ---------- handlers (return plain text answers) ----------

def _format_paper_line(p: Paper) -> str:
    authors = [a.name for a in p.authors]
    surname = ""
    if authors:
        first = authors[0]
        surname = first.split(",", 1)[0].strip() if "," in first else (first.split()[-1] if first.split() else first)
    cite = f"{surname} et al." if len(authors) > 2 else (
        f"{surname} & {(authors[1].split(',',1)[0].strip() if ',' in authors[1] else authors[1].split()[-1])}"
        if len(authors) == 2 else surname
    )
    year = p.year or "?"
    title = (p.title or "(untitled)").strip()
    return f"- **[paper {p.id}]** {cite} ({year}) — {title}"


def handle_list(intent: QueryIntent) -> str:
    """Plain SQL list of papers matching intent's filters."""
    with session_scope() as session:
        stmt = session.query(Paper)
        if intent.year is not None:
            stmt = stmt.filter(Paper.year == intent.year)
        if intent.year_range is not None:
            lo, hi = intent.year_range
            stmt = stmt.filter(Paper.year >= lo, Paper.year <= hi)
        if intent.author_surname:
            # Match authors whose name CONTAINS the surname (case-insensitive).
            # Paper.authors is an association_proxy — join through the
            # PaperAuthor association object instead.
            pat = f"%{intent.author_surname}%"
            stmt = (
                stmt.join(PaperAuthor, PaperAuthor.paper_id == Paper.id)
                .join(Author, Author.id == PaperAuthor.author_id)
                .filter(Author.name.ilike(pat))
                .distinct()
            )
        if intent.tag:
            pat = intent.tag.lower()
            stmt = stmt.join(Paper.tags).filter(Tag.name == pat).distinct()
        papers = stmt.order_by(Paper.year.desc().nullslast(), Paper.id.asc()).all()
        # Format outside the session
        rows = [(p.id, p.year, p.title or "(untitled)",
                 [a.name for a in p.authors]) for p in papers]

    if not rows:
        filters = []
        if intent.year:
            filters.append(f"year={intent.year}")
        if intent.year_range:
            filters.append(f"year {intent.year_range[0]}-{intent.year_range[1]}")
        if intent.author_surname:
            filters.append(f"author~={intent.author_surname}")
        if intent.tag:
            filters.append(f"tag={intent.tag}")
        return f"No papers in your library match {', '.join(filters) or 'that query'}."

    # Format the results
    header = f"**{len(rows)} paper(s)"
    bits = []
    if intent.year:
        bits.append(f"from {intent.year}")
    if intent.year_range:
        bits.append(f"from {intent.year_range[0]}-{intent.year_range[1]}")
    if intent.author_surname:
        bits.append(f"with an author matching '{intent.author_surname}'")
    if intent.tag:
        bits.append(f"tagged '{intent.tag}'")
    if bits:
        header += " " + ", ".join(bits)
    header += ":**"

    lines = [header, ""]
    for pid, year, title, authors in rows:
        surname = ""
        if authors:
            first = authors[0]
            surname = first.split(",", 1)[0].strip() if "," in first else (first.split()[-1] if first.split() else first)
        cite = (
            f"{surname} et al." if len(authors) > 2
            else (f"{surname} & {(authors[1].split(',',1)[0].strip() if ',' in authors[1] else authors[1].split()[-1])}" if len(authors) == 2 else surname)
        )
        lines.append(f"- **[paper {pid}]** {cite} {year or '?'} — {title}")
    return "\n".join(lines)


def handle_count(intent: QueryIntent) -> str:
    with session_scope() as session:
        stmt = session.query(Paper)
        if intent.year is not None:
            stmt = stmt.filter(Paper.year == intent.year)
        if intent.year_range is not None:
            lo, hi = intent.year_range
            stmt = stmt.filter(Paper.year >= lo, Paper.year <= hi)
        if intent.author_surname:
            pat = f"%{intent.author_surname}%"
            stmt = (
                stmt.join(PaperAuthor, PaperAuthor.paper_id == Paper.id)
                .join(Author, Author.id == PaperAuthor.author_id)
                .filter(Author.name.ilike(pat))
                .distinct()
            )
        if intent.tag:
            stmt = stmt.join(Paper.tags).filter(Tag.name == intent.tag.lower()).distinct()
        n = stmt.count()
    bits = []
    if intent.year:
        bits.append(f"from {intent.year}")
    if intent.year_range:
        bits.append(f"from {intent.year_range[0]}-{intent.year_range[1]}")
    if intent.author_surname:
        bits.append(f"by an author matching '{intent.author_surname}'")
    if intent.tag:
        bits.append(f"tagged '{intent.tag}'")
    qualifier = " " + ", ".join(bits) if bits else ""
    return f"You have **{n} paper(s){qualifier}**."


def find_paper_by_reference(intent: QueryIntent) -> Paper | None:
    """Look up a paper by id OR by author surname + year."""
    with session_scope() as session:
        if intent.paper_id is not None:
            return session.get(Paper, intent.paper_id)
        if intent.author_surname:
            pat = f"%{intent.author_surname}%"
            stmt = (
                session.query(Paper)
                .join(PaperAuthor, PaperAuthor.paper_id == Paper.id)
                .join(Author, Author.id == PaperAuthor.author_id)
                .filter(Author.name.ilike(pat))
            )
            if intent.year is not None:
                stmt = stmt.filter(Paper.year == intent.year)
            candidates = stmt.distinct().all()
            if len(candidates) == 1:
                return candidates[0]
            # If multiple, return the most recent (the user probably named the
            # latest paper from that author)
            if candidates:
                return sorted(
                    candidates,
                    key=lambda p: -(p.year or 0),
                )[0]
    return None


def build_library_aggregates() -> dict:
    """Stats used for the library_summary path."""
    from collections import Counter

    with session_scope() as session:
        papers = session.query(Paper).all()
        years = [p.year for p in papers if p.year]
        year_counts = Counter(years)
        tag_counts: Counter = Counter()
        author_counts: Counter = Counter()
        venue_counts: Counter = Counter()
        for p in papers:
            for t in p.tags:
                tag_counts[t.name] += 1
            for a in p.authors:
                author_counts[a.name] += 1
            if p.venue:
                venue_counts[p.venue] += 1
        # Per-decade
        decade_counts: Counter = Counter()
        for y in years:
            decade_counts[(y // 10) * 10] += 1
        return {
            "n": len(papers),
            "year_range": (min(years), max(years)) if years else None,
            "year_counts": year_counts.most_common(),
            "decade_counts": decade_counts.most_common(),
            "top_authors": author_counts.most_common(15),
            "top_tags": tag_counts.most_common(15),
            "top_venues": venue_counts.most_common(8),
        }
