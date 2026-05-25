"""Chat-callable tools.

The chat REPL exposes a handful of read-only library operations as natural-
language-addressable tools. The LLM tool-router (see `chat._maybe_handle_tool`)
inspects the user's question, picks one of these tools (or "none"), extracts
arguments, and runs the corresponding handler.

Each handler:
  - prints rich output to the console as a side effect (tables, panels, etc.),
  - returns a short plain-text summary so the answer is preserved in chat history.

Every tool here MUST be read-only. Side-effecting commands (fetch-refs, add,
scan, enrich, remove, clean, …) are deliberately not exposed yet — they need a
confirmation flow before going through the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

from rich.table import Table

from bibwizard.database.migrations import session_scope
from bibwizard.database.models import Paper
from bibwizard.database.queries import (
    find_duplicate_groups,
    library_stats,
    list_recent_papers,
    text_search,
)
from bibwizard.utils.display import console, info, panel, papers_table, stats_table, warn


@dataclass(frozen=True)
class ArgSpec:
    type: str            # "int", "str", "bool"
    required: bool = False
    default: Any = None
    description: str = ""


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str          # one-line, used by the LLM router
    args: dict[str, ArgSpec]  # name -> spec
    handler: Callable[[dict], str]


# ---------- helpers shared across tools ----------

def _humanize_since(now: datetime, then: datetime) -> str:
    """Local copy of cli._humanize_since to avoid a circular import."""
    delta = now - then
    secs = int(delta.total_seconds())
    if secs < 60:
        return "just now"
    if secs < 3600:
        m = secs // 60
        return f"{m} minute{'s' if m != 1 else ''} ago"
    if secs < 86400:
        h = secs // 3600
        return f"{h} hour{'s' if h != 1 else ''} ago"
    days = secs // 86400
    if days < 14:
        return f"{days} day{'s' if days != 1 else ''} ago"
    if days < 60:
        return f"{days // 7} week{'s' if days // 7 != 1 else ''} ago"
    if days < 730:
        return f"{days // 30} month{'s' if days // 30 != 1 else ''} ago"
    return f"{days // 365} year{'s' if days // 365 != 1 else ''} ago"


def _coerce_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# ---------- tool: whats_new ----------

def _t_whats_new(args: dict) -> str:
    days = _coerce_int(args.get("days"), 7)
    since_str = args.get("since") or None
    if since_str:
        try:
            cutoff = datetime.strptime(str(since_str), "%Y-%m-%d")
            label = f"since {cutoff:%Y-%m-%d}"
        except ValueError:
            warn(f"Ignoring malformed --since {since_str!r}; falling back to --days {days}.")
            cutoff = datetime.utcnow() - timedelta(days=days)
            label = f"last {days} day(s)"
    else:
        cutoff = datetime.utcnow() - timedelta(days=days)
        label = f"last {days} day{'s' if days != 1 else ''}"

    with session_scope() as session:
        papers = list_recent_papers(session, since=cutoff)
        rows = [
            {
                "id": p.id,
                "title": p.title,
                "authors": [a.name for a in p.authors],
                "year": p.year,
                "created_at": p.created_at,
            }
            for p in papers
        ]

    if not rows:
        msg = f"No papers added in the {label}."
        panel("What's new", msg, style="cyan")
        return msg

    now = datetime.utcnow()
    table = Table(
        title=f"What's new — {label} ({len(rows)} paper{'s' if len(rows)!=1 else ''})",
        header_style="bold cyan",
    )
    table.add_column("ID", justify="right", style="dim", no_wrap=True)
    table.add_column("Added", no_wrap=True)
    table.add_column("Title", overflow="fold")
    table.add_column("Authors", overflow="fold")
    table.add_column("Year", justify="right")
    for r in rows:
        ts = r["created_at"]
        added = f"{ts:%Y-%m-%d} [dim]({_humanize_since(now, ts)})[/]"
        auth_short = ", ".join(r["authors"][:3]) + (" …" if len(r["authors"]) > 3 else "")
        table.add_row(
            str(r["id"]),
            added,
            r["title"] or "(untitled)",
            auth_short,
            str(r["year"] or ""),
        )
    console.print(table)
    return f"{len(rows)} paper(s) added in the {label}."


# ---------- tool: find (semantic search) ----------

def _t_find(args: dict) -> str:
    from bibwizard.search.paper_search import find_papers

    query = (args.get("query") or "").strip()
    if not query:
        msg = "No query provided. Try: \"find papers about Bessel beams\"."
        panel("Find", msg, style="yellow")
        return msg
    top_k = _coerce_int(args.get("top_k"), 10)

    hits = find_papers(query, top_k=top_k)
    if not hits:
        msg = f"No papers matched “{query}”."
        panel("Find (semantic)", msg, style="cyan")
        return msg

    table = Table(title=f"Find — “{query}” (top {len(hits)})", header_style="bold cyan")
    table.add_column("Score", justify="right")
    table.add_column("ID", justify="right", style="dim", no_wrap=True)
    table.add_column("Cite", no_wrap=True)
    table.add_column("Title", overflow="fold")
    table.add_column("Page", justify="right")
    table.add_column("Snippet", overflow="fold")
    for h in hits:
        table.add_row(
            f"{h.score:.3f}",
            str(h.paper_id),
            h.cite,
            h.title,
            str(h.best_page if h.best_page > 0 else ""),
            h.best_snippet,
        )
    console.print(table)
    return f"{len(hits)} paper(s) match “{query}”."


# ---------- tool: grep (substring) ----------

def _t_grep(args: dict) -> str:
    query = (args.get("query") or "").strip()
    if not query:
        msg = "No query provided. Try: \"grep for 'coronagraph'\"."
        panel("Grep", msg, style="yellow")
        return msg
    limit = _coerce_int(args.get("limit"), 20)

    with session_scope() as session:
        papers = text_search(session, query, limit=limit)
        rows = [
            {
                "id": p.id,
                "title": p.title,
                "authors": ", ".join(a.name for a in p.authors[:3])
                + (" …" if len(p.authors) > 3 else ""),
                "year": p.year,
                "tags": ", ".join(t.name for t in p.tags),
            }
            for p in papers
        ]

    if not rows:
        msg = f"No paper titles or abstracts contain “{query}”."
        panel("Grep", msg, style="cyan")
        return msg
    console.print(papers_table(rows))
    return f"{len(rows)} paper(s) contain “{query}” in title or abstract."


# ---------- tool: stats ----------

def _t_stats(args: dict) -> str:
    with session_scope() as session:
        st = library_stats(session)
    by_year = st.pop("by_year", []) or []
    console.print(stats_table(st))
    if by_year:
        sub = ", ".join(f"{y}:{c}" for y, c in by_year[:12])
        info(f"by year: {sub}")
    return (
        f"{st.get('papers', 0)} papers, {st.get('authors', 0)} authors, "
        f"{st.get('citations', 0)} parsed citations, "
        f"{st.get('vector_chunks', 0)} vector chunks."
    )


# ---------- tool: duplicates ----------

def _t_duplicates(args: dict) -> str:
    with session_scope() as session:
        groups = find_duplicate_groups(session)
    if not groups:
        msg = "No likely duplicates found."
        panel("Duplicates", msg, style="green")
        return msg
    lines: list[str] = []
    for g in groups:
        head = f"[bold]{g['tier'].upper()}[/]: " + ", ".join(
            f"paper {m['id']} ({m['year']}) — {m['title'][:60]}"
            for m in g["members"]
        )
        reasons = "; ".join(g.get("reasons", []))
        lines.append(head + ("\n  " + reasons if reasons else ""))
    panel(f"Duplicates ({len(groups)} group(s))", "\n".join(lines), style="yellow")
    return f"{len(groups)} duplicate group(s) found."


# ---------- tool: cite_finder ----------

def _t_cite_finder(args: dict) -> str:
    from bibwizard.search.cite_search import find_citations
    from bibwizard.search.reranker import CrossEncoderReranker, get_reranker
    from bibwizard.utils.wizard_spinner import WizardLive

    claim = (args.get("claim") or args.get("statement") or args.get("query") or "").strip()
    if not claim:
        msg = (
            "No claim provided. Give me a sentence to find a citation for, "
            "e.g. \"single-mode fibers improve spectrograph stability\"."
        )
        panel("Cite finder", msg, style="yellow")
        return msg
    max_results = _coerce_int(args.get("max_results"), 5)
    pool = _coerce_int(args.get("pool_size"), 20)
    want_tex = bool(args.get("tex"))
    cite_command = (args.get("cite_command") or "citep").strip().lower()

    reranker = get_reranker()

    # Animated wizard reads through the shelves while the LLM grinds.
    # Transient — clears when find_citations returns so the results
    # panel is the first persistent output the user sees.
    with WizardLive(
        console,
        status=f"Preparing search (reranker={reranker.name})...",
        total=pool,
    ) as wiz:
        if (
            isinstance(reranker, CrossEncoderReranker)
            and reranker._model is None
        ):
            wiz.update(
                status=(
                    f"Conjuring reranker model ({reranker.model_name}, "
                    "one-time)..."
                ),
            )
            reranker._ensure_model()

        wiz.update(
            status=f"Searching {pool} candidate passages...",
            done=0,
            total=pool,
        )

        def _progress(done: int, total: int) -> None:
            wiz.update(
                status=f"Reading candidate passage {done}/{total}",
                done=done,
                total=total,
            )

        hits = find_citations(
            claim,
            pool_size=pool,
            max_results=max_results,
            progress_cb=_progress,
            reranker=reranker,
        )

    if not hits:
        msg = (
            "I couldn't find a passage in your library that supports that "
            "claim. Either the evidence isn't in any indexed paper, or the "
            "claim is too general — try a more specific statement."
        )
        panel("Cite finder", msg, style="cyan")
        return msg

    table = Table(
        title=f"Citations supporting: “{claim[:120]}”",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Conf.", justify="right", no_wrap=True)
    table.add_column("ID", justify="right", style="dim", no_wrap=True)
    table.add_column("Cite", no_wrap=True)
    table.add_column("Page", justify="right", no_wrap=True)
    table.add_column("Quoted passage", overflow="fold")
    table.add_column("Why", overflow="fold")
    for h in hits:
        table.add_row(
            f"{h.confidence:.2f}",
            str(h.paper_id),
            h.paper_cite,
            str(h.page) if h.page > 0 else "?",
            f"“{h.quoted_sentence}”",
            h.rationale,
        )
    console.print(table)

    if want_tex:
        # Render the LaTeX block (\citep macro + .bib entries) for the
        # papers we just found. Mirrors the `bibwizard cite ... --tex`
        # CLI behaviour so the user gets paste-into-manuscript output
        # without leaving chat.
        from bibwizard.output.bibtex import format_citation_block

        hit_ids = [h.paper_id for h in hits]
        with session_scope() as session:
            papers = (
                session.query(Paper)
                .filter(Paper.id.in_(hit_ids))
                .all()
            )
            by_id = {p.id: p for p in papers}
            ordered = [by_id[pid] for pid in hit_ids if pid in by_id]
            # format_citation_block touches paper.authors which is lazy-
            # loaded, so render INSIDE the session.
            tex_block = format_citation_block(ordered, command=cite_command)

        panel(
            "LaTeX (\\citep + .bib entries)",
            tex_block,
            style="green",
        )

    return (
        f"Found {len(hits)} supporting passage(s). "
        f"Top: {hits[0].paper_cite} p.{hits[0].page} (confidence {hits[0].confidence:.2f})."
    )


# ---------- tool: show ----------

def _t_show(args: dict) -> str:
    pid = _coerce_int(args.get("paper_id"), -1)
    if pid < 0:
        msg = "No paper_id provided. Try: \"show paper 12\"."
        panel("Show", msg, style="yellow")
        return msg
    with session_scope() as session:
        p = session.get(Paper, pid)
        if p is None:
            msg = f"No paper with id {pid}."
            panel("Show", msg, style="red")
            return msg
        title = p.title or "(untitled)"
        authors = [a.name for a in p.authors]
        year = p.year
        doi = p.doi or "(none)"
        arxiv_id = p.arxiv_id or "(none)"
        venue = p.venue or "(none)"
        tags = ", ".join(t.name for t in p.tags) or "(none)"
        n_chunks = p.n_chunks
        file_path = p.file_path or "(no file)"
        created = p.created_at
        n_outgoing = len(p.outgoing_citations or [])

    body = (
        f"[bold]paper {pid}[/]\n"
        f"title:    {title}\n"
        f"authors:  {', '.join(authors) if authors else '(unknown)'}\n"
        f"year:     {year if year is not None else '(unknown)'}\n"
        f"venue:    {venue}\n"
        f"doi:      {doi}\n"
        f"arxiv:    {arxiv_id}\n"
        f"tags:     {tags}\n"
        f"chunks:   {n_chunks}\n"
        f"refs:     {n_outgoing} parsed\n"
        f"added:    {created:%Y-%m-%d}\n"
        f"file:     {file_path}"
    )
    panel(f"Paper {pid}", body, style="cyan")
    return f"Paper {pid}: {title} ({year})."


# ---------- registry ----------

TOOLS: dict[str, ToolDef] = {
    "whats_new": ToolDef(
        name="whats_new",
        description="Show papers that were recently added to the library.",
        args={
            "days": ArgSpec("int", default=7, description="Look back N days (default 7)."),
            "since": ArgSpec("str", description="Cutoff date YYYY-MM-DD (overrides days)."),
        },
        handler=_t_whats_new,
    ),
    "find": ToolDef(
        name="find",
        description="Semantic search for papers about a topic, concept, or method.",
        args={
            "query": ArgSpec("str", required=True, description="What to search for."),
            "top_k": ArgSpec("int", default=10, description="How many results to return."),
        },
        handler=_t_find,
    ),
    "cite_finder": ToolDef(
        name="cite_finder",
        description=(
            "Find a CITATION that supports a specific claim or sentence. "
            "Use this when the user asks 'find a citation for X', 'who showed Y?', "
            "'is there a paper that demonstrates Z?'. The tool reads candidate "
            "passages and returns verbatim quotes + page numbers — different "
            "from find(), which returns papers about a topic. Set tex=true to "
            "also output a LaTeX \\citep{} macro + .bib entries (use for "
            "'give me the bibtex for X', 'cite X as latex')."
        ),
        args={
            "claim": ArgSpec(
                "str",
                required=True,
                description="The exact statement to find evidence for. "
                "Must be the full claim, not a topic keyword.",
            ),
            "max_results": ArgSpec("int", default=5),
            "tex": ArgSpec(
                "bool",
                default=False,
                description="If true, output a LaTeX \\citep{key1, key2} "
                "macro plus the .bib entries for paste-into-manuscript use.",
            ),
            "cite_command": ArgSpec(
                "str",
                default="citep",
                description="natbib command for the --tex output (citep, "
                "citet, cite). Only matters when tex=true.",
            ),
        },
        handler=_t_cite_finder,
    ),
    "grep": ToolDef(
        name="grep",
        description="Substring search across paper titles and abstracts. Use this for exact phrases or author surnames; use find() for concepts.",
        args={
            "query": ArgSpec("str", required=True, description="Exact substring to look for."),
            "limit": ArgSpec("int", default=20),
        },
        handler=_t_grep,
    ),
    "stats": ToolDef(
        name="stats",
        description="Show overall library statistics (paper / author / citation / chunk counts, year distribution).",
        args={},
        handler=_t_stats,
    ),
    "duplicates": ToolDef(
        name="duplicates",
        description="Find papers in the library that look like duplicates of each other.",
        args={},
        handler=_t_duplicates,
    ),
    "show": ToolDef(
        name="show",
        description="Show a metadata card for a specific paper by id (file path, DOI, arXiv id, tag list, chunk count, etc.). Use this when the user names a paper id directly.",
        args={
            "paper_id": ArgSpec("int", required=True, description="The paper id to display."),
        },
        handler=_t_show,
    ),
}


def tool_catalogue_for_prompt() -> str:
    """Render the registry as a compact text catalogue the router LLM can read."""
    lines: list[str] = []
    for t in TOOLS.values():
        arg_bits: list[str] = []
        for name, spec in t.args.items():
            req = "" if spec.required else "?"
            default = f"={spec.default}" if spec.default is not None and not spec.required else ""
            arg_bits.append(f"{name}{req}: {spec.type}{default}")
        sig = f"{t.name}({', '.join(arg_bits)})" if arg_bits else f"{t.name}()"
        lines.append(f"- {sig}\n    {t.description}")
    return "\n".join(lines)


def run_tool(name: str, args: dict | None = None) -> str:
    """Dispatch a tool by name. Raises KeyError if unknown."""
    tool = TOOLS[name]
    return tool.handler(args or {})
