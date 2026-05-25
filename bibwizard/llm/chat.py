"""Interactive RAG chat loop.

Pipeline:
  1. Embed user query with nomic-embed-text via Ollama.
  2. Retrieve top-k chunks from ChromaDB.
  3. Build a [PAPER #] labelled context block and inject as system context.
  4. Stream the DeepSeek response token-by-token to the terminal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from bibwizard.database.migrations import session_scope
from bibwizard.database.models import Paper
from bibwizard.ingestion.embedder import query_chunks
from bibwizard.utils.config import settings
from bibwizard.utils.display import console, error, info, panel

from . import router
from . import tools as chat_tools
from .client import ChatMessage, OllamaUnavailable, get_client
from .prompts import (
    CHAT_SYSTEM,
    CHAT_USER,
    LIBRARY_SUMMARY_SYSTEM,
    LIBRARY_SUMMARY_USER,
    SPECIFIC_PAPER_SYSTEM,
    SPECIFIC_PAPER_USER,
    TOOL_ROUTER_SYSTEM,
    TOOL_ROUTER_USER,
)


# DeepSeek-R1 emits <think>...</think>. We hide it in chat output by default.
_THINK_OPEN = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think\s*>", re.IGNORECASE)


@dataclass
class ChatTurn:
    role: str
    content: str


def _build_context_block(chunks: list[dict]) -> tuple[str, list[dict]]:
    """Render top-k chunks as a numbered context block + return label map.

    Looks up author + year for each cited paper from SQLite so the [PAPER N]
    label carries enough context for both the LLM and the sources panel.
    """
    paper_ids = sorted({int((ch.get("metadata") or {}).get("paper_id", -1)) for ch in chunks})
    paper_ids = [pid for pid in paper_ids if pid >= 0]
    paper_info: dict[int, dict] = {}
    if paper_ids:
        with session_scope() as session:
            for p in session.query(Paper).filter(Paper.id.in_(paper_ids)).all():
                authors = [a.name for a in p.authors]
                paper_info[p.id] = {
                    "title": p.title,
                    "authors": authors,
                    "year": p.year,
                    "doi": p.doi,
                    "arxiv_id": p.arxiv_id,
                    "short_cite": _short_cite(authors, p.year),
                }

    lines: list[str] = []
    label_rows: list[dict] = []
    for ch in chunks:
        meta = ch.get("metadata", {}) or {}
        pid = int(meta.get("paper_id", -1))
        title = (paper_info.get(pid) or {}).get("title") or meta.get("title", "(unknown)")
        info_row = paper_info.get(pid, {})
        authors = info_row.get("authors") or []
        year = info_row.get("year")
        short_cite = info_row.get("short_cite") or "(unknown)"
        page = meta.get("page", -1)
        # IMPORTANT: use the database paper.id as the [PAPER N] label so it
        # MATCHES the label used in the LIBRARY OVERVIEW block. Otherwise the
        # LLM sees two label schemes and can't reconcile them — it
        # silently ignores the overview because the cite-keys don't match.
        label = pid
        lines.append(
            f"[PAPER {label}] ({short_cite}; p.{page}, "
            f"score={ch.get('score', 0):.3f})\n"
            f"Title: {title}\n"
            f"Authors: {', '.join(authors) if authors else '(unknown)'}\n"
            f"Year: {year if year is not None else '(unknown)'}\n"
            f"{ch.get('text', '').strip()}\n"
        )
        label_rows.append(
            {
                "label": label,
                "paper_id": pid,
                "title": title,
                "authors": authors,
                "year": year,
                "short_cite": short_cite,
                "page": page,
                "score": ch.get("score", 0),
            }
        )
    return "\n---\n".join(lines), label_rows


def _short_cite(authors: list[str], year: int | None) -> str:
    """Build a 'Lastname et al. YYYY' string for the sources panel + LLM context."""
    if not authors:
        return f"(unknown){f' {year}' if year else ''}".strip()
    first = authors[0]
    # Pull a likely surname: 'Reinarz, Y.' → 'Reinarz'; 'Y. Reinarz' → 'Reinarz'.
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
        sec_last = second.split(",", 1)[0].strip() if "," in second else (second.split()[-1] if second.split() else second)
        return f"{last} & {sec_last}{yr}"
    return f"{last} et al.{yr}"


# Query intent — "library-wide" questions need more papers detailed than
# narrow technical questions do. These patterns boost the detail-K.
_LIBRARY_WIDE_RE = re.compile(
    r"\b("
    r"summari[sz]e|summary|overview|overall|"
    r"what topics|what fields|what areas|"
    r"list( all)?|all papers|every paper|how many papers|"
    r"my library|the library|my database|the database|"
    r"what (?:do|did)\s+(?:i|you)\s+have|"
    r"what'?s in|the field"
    r")\b",
    re.IGNORECASE,
)


def _detect_intent(question: str) -> str:
    """Return 'library_wide' if the question looks like a meta-library
    question (summarize, list topics, overall field), else 'default'."""
    return "library_wide" if _LIBRARY_WIDE_RE.search(question or "") else "default"


def _build_stats_block(papers: list) -> str:
    """Always-shown compact statistics about the whole library."""
    from collections import Counter

    n = len(papers)
    if n == 0:
        return "LIBRARY STATS: (empty)"
    years = [p.year for p in papers if p.year]
    year_str = f"{min(years)}–{max(years)}" if years else "(no years)"
    year_counts = Counter(years)
    top_years = year_counts.most_common(8)

    tag_counts: Counter = Counter()
    for p in papers:
        for t in p.tags:
            tag_counts[t.name] += 1
    top_tags = tag_counts.most_common(12)

    author_counts: Counter = Counter()
    for p in papers:
        for a in p.authors:
            author_counts[a.name] += 1
    top_authors = author_counts.most_common(8)

    venues: Counter = Counter()
    for p in papers:
        v = (p.venue or "").strip()
        if v:
            venues[v] += 1
    top_venues = venues.most_common(5)

    parts = [
        f"LIBRARY STATS:",
        f"  total papers: {n}",
        f"  year range:   {year_str}",
    ]
    if top_years:
        parts.append(
            "  by year:      " + ", ".join(f"{y}:{c}" for y, c in top_years)
        )
    if top_tags:
        parts.append(
            "  top tags:     " + ", ".join(f"{t} ({c})" for t, c in top_tags)
        )
    if top_authors:
        parts.append(
            "  top authors:  " + ", ".join(f"{a} ({c})" for a, c in top_authors)
        )
    if top_venues:
        parts.append(
            "  top venues:   " + ", ".join(f"{v} ({c})" for v, c in top_venues)
        )
    return "\n".join(parts)


def _format_paper_block(p) -> str:
    """One compact entry for the detailed-papers section."""
    import json as _json

    authors = [a.name for a in p.authors]
    cite = _short_cite(authors, p.year)
    digest = ""
    if p.summary_json:
        try:
            s = _json.loads(p.summary_json)
            contributions = s.get("key_contributions") or []
            methodology = (s.get("methodology") or "").strip()
            if contributions:
                digest = "; ".join(str(c) for c in contributions[:2])
            elif methodology:
                digest = methodology
        except Exception:
            pass
    if not digest and p.abstract:
        digest = p.abstract.strip().replace("\n", " ")
    digest = digest[:200] + ("…" if digest and len(digest) > 200 else "")
    tag_names = ", ".join(t.name for t in p.tags)
    return (
        f"[PAPER {p.id}] {cite} — {(p.title or '(untitled)')[:140]}\n"
        f"   authors: {', '.join(authors[:6])}{' …' if len(authors) > 6 else ''}\n"
        + (f"   summary: {digest}\n" if digest else "")
        + (f"   tags: {tag_names}\n" if tag_names else "")
    )


def build_library_overview(
    *,
    relevant_paper_ids: list[int] | None = None,
    intent: str = "default",
    max_detail_papers: int | None = None,
    max_chars: int = 20000,
) -> str:
    """Tiered library overview for the chat LLM.

    Layout:
      1) LIBRARY STATS — always shown, ~300 tokens regardless of library
         size: counts, year histogram, top tags / authors / venues.
      2) DETAILED ENTRIES — ~150 tokens per paper for a sample sized to fit
         the budget. Selection rules:
           - if `relevant_paper_ids` is supplied, those come first (they're
             the RAG-relevant papers for this question);
           - then the most recent papers are appended to fill the budget.

    For ≤80 papers the detail section just shows everything. For larger
    libraries `intent='library_wide'` widens the detail set; specific
    questions stay focused on the RAG-relevant papers.
    """
    with session_scope() as session:
        all_papers = (
            session.query(Paper)
            .order_by(Paper.year.desc().nullslast(), Paper.id.asc())
            .all()
        )
        # Stats block first — uses ALL papers so it's a true global view.
        stats = _build_stats_block(all_papers)
        n = len(all_papers)
        if n == 0:
            return stats

        # Decide how many to detail.
        if max_detail_papers is None:
            if intent == "library_wide":
                k = min(n, 50)
            else:
                k = min(n, 20)
        else:
            k = min(n, max_detail_papers)

        # Selection: relevant papers first (in given order), then recent.
        by_id = {p.id: p for p in all_papers}
        selected_ids: list[int] = []
        seen: set[int] = set()
        if relevant_paper_ids:
            for pid in relevant_paper_ids:
                if pid in by_id and pid not in seen:
                    selected_ids.append(pid)
                    seen.add(pid)
                if len(selected_ids) >= k:
                    break
        # Fill with most-recent papers
        for p in all_papers:  # already sorted by year desc
            if p.id in seen:
                continue
            selected_ids.append(p.id)
            seen.add(p.id)
            if len(selected_ids) >= k:
                break

        # Build detail block in selection order
        detail_blocks = [_format_paper_block(by_id[pid]) for pid in selected_ids]
        detail = "\n".join(detail_blocks)

    n_shown = len(selected_ids)
    header = f"DETAILED ENTRIES (showing {n_shown} of {n} papers):"
    if n_shown < n:
        if relevant_paper_ids:
            header += " — most-relevant + most-recent"
        else:
            header += " — most-recent"

    full = f"{stats}\n\n{header}\n{detail}"
    if len(full) > max_chars:
        # Trim the detail tail rather than the stats
        budget_for_detail = max_chars - len(stats) - 200
        # Crude trim — preserve whole paper entries where possible
        cut_detail = detail[:budget_for_detail].rsplit("[PAPER ", 1)[0]
        full = (
            f"{stats}\n\n{header}\n{cut_detail}"
            f"\n[... detail truncated to fit context budget ...]"
        )
    return full


def build_rag_messages(
    history: list[ChatTurn],
    user_question: str,
    chunks: list[dict],
    *,
    wider_chunks: list[dict] | None = None,
) -> list[ChatMessage]:
    """Convert history + retrieved chunks into the message list for /api/chat.

    The overview block scales with library size: stats are always shipped;
    detailed paper entries are the RAG-relevant papers (derived from
    `wider_chunks` if provided, otherwise from `chunks`) plus the most-recent
    papers to fill any remaining budget.
    """
    context_block, _labels = _build_context_block(chunks)

    # Derive the set of "relevant" paper ids from the wider RAG hit
    relevant_ids: list[int] = []
    seen: set[int] = set()
    for ch in (wider_chunks or chunks):
        pid = int((ch.get("metadata") or {}).get("paper_id", -1))
        if pid >= 0 and pid not in seen:
            relevant_ids.append(pid)
            seen.add(pid)

    intent = _detect_intent(user_question)
    overview_block = build_library_overview(
        relevant_paper_ids=relevant_ids, intent=intent
    )

    # Count headers for the user prompt
    with session_scope() as session:
        n_papers = session.query(Paper).count()
    n_in_overview = overview_block.count("[PAPER ")

    user_payload = CHAT_USER.substitute(
        overview=overview_block,
        k=len(chunks),
        context=context_block,
        question=user_question,
        n_papers=n_papers,
        n_in_overview=n_in_overview,
    )
    messages: list[ChatMessage] = [ChatMessage("system", CHAT_SYSTEM)]
    for turn in history:
        messages.append(ChatMessage(turn.role, turn.content))
    messages.append(ChatMessage("user", user_payload))
    return messages


class _ThinkingFilter:
    """Strip <think>...</think> blocks from a streaming token feed."""

    def __init__(self) -> None:
        self.in_think = False
        self.buffer = ""

    def feed(self, token: str) -> str:
        self.buffer += token
        out: list[str] = []
        while self.buffer:
            if self.in_think:
                m = _THINK_CLOSE.search(self.buffer)
                if not m:
                    self.buffer = ""
                    break
                self.buffer = self.buffer[m.end() :]
                self.in_think = False
            else:
                m = _THINK_OPEN.search(self.buffer)
                if not m:
                    out.append(self.buffer)
                    self.buffer = ""
                    break
                out.append(self.buffer[: m.start()])
                self.buffer = self.buffer[m.end() :]
                self.in_think = True
        return "".join(out)


def _stream_messages_to_panel(messages: list[ChatMessage]) -> str:
    """Stream a chat completion into the bibwizard panel. Returns full text."""
    client = get_client()
    filt = _ThinkingFilter()
    pieces: list[str] = []
    text = Text()
    with Live(
        Panel(text, title="bibwizard", border_style="cyan"),
        refresh_per_second=20,
        console=console,
    ) as live:
        for token in client.chat(messages, stream=True):
            visible = filt.feed(token)
            if visible:
                pieces.append(visible)
                text.append(visible)
                live.update(Panel(text, title="bibwizard", border_style="cyan"))
    return "".join(pieces).strip()


def _print_direct_answer(answer: str) -> None:
    """Show a router-supplied (non-LLM) answer in the bibwizard panel."""
    console.print(Panel(Text(answer), title="bibwizard", border_style="cyan"))


def _format_aggregates_block(agg: dict) -> str:
    """Render router.build_library_aggregates() as a plain-text FACTS block."""
    lines: list[str] = []
    lines.append(f"total papers: {agg.get('n', 0)}")
    yr = agg.get("year_range")
    if yr:
        lines.append(f"year range:   {yr[0]}–{yr[1]}")
    decades = agg.get("decade_counts") or []
    if decades:
        decades_sorted = sorted(decades, key=lambda x: x[0])
        lines.append(
            "by decade:    "
            + ", ".join(f"{d}s:{c}" for d, c in decades_sorted)
        )
    yc = agg.get("year_counts") or []
    if yc:
        yc_sorted = sorted(yc, key=lambda x: -x[1])[:10]
        lines.append("by year:      " + ", ".join(f"{y}:{c}" for y, c in yc_sorted))
    top_tags = agg.get("top_tags") or []
    if top_tags:
        lines.append(
            "top tags:     " + ", ".join(f"{t} ({c})" for t, c in top_tags)
        )
    top_authors = agg.get("top_authors") or []
    if top_authors:
        lines.append(
            "top authors:  " + ", ".join(f"{a} ({c})" for a, c in top_authors)
        )
    top_venues = agg.get("top_venues") or []
    if top_venues:
        lines.append(
            "top venues:   " + ", ".join(f"{v} ({c})" for v, c in top_venues)
        )
    return "\n".join(lines)


def _handle_specific_paper(
    intent: "router.QueryIntent",
    history: list[ChatTurn],
    *,
    top_k: int,
    show_sources: bool,
) -> str:
    """Route: user named one paper. Look it up, scope chunks to it, stream."""
    import json as _json

    paper = router.find_paper_by_reference(intent)
    if paper is None:
        msg = (
            f"I couldn't find that paper in your library. "
            f"Try `bibwizard list` to see what's available, or rephrase the "
            f"reference (e.g. `Smith 2021` or `paper 42`)."
        )
        _print_direct_answer(msg)
        return msg

    # Pull the bits we need INSIDE a session, then format outside.
    with session_scope() as session:
        p = session.get(Paper, paper.id)
        pid = p.id
        title = p.title or "(untitled)"
        authors = [a.name for a in p.authors]
        year = p.year
        venue = p.venue or ""
        doi = p.doi or ""
        arxiv_id = p.arxiv_id or ""
        tags = ", ".join(t.name for t in p.tags) or "(none)"
        summary_json = p.summary_json or ""

    # Format the summary digest for the prompt
    summary_block = "(no summary on file)"
    if summary_json:
        try:
            s = _json.loads(summary_json)
            parts = []
            kc = s.get("key_contributions") or []
            if kc:
                parts.append("Key contributions:")
                parts.extend(f"  - {c}" for c in kc)
            meth = (s.get("methodology") or "").strip()
            if meth:
                parts.append(f"Methodology: {meth}")
            lim = (s.get("limitations") or "").strip()
            if lim:
                parts.append(f"Limitations: {lim}")
            if parts:
                summary_block = "\n".join(parts)
        except Exception:
            summary_block = summary_json[:2000]

    # Retrieve chunks SCOPED to this paper only
    chunks = query_chunks(
        intent.raw_question or "summary",
        top_k=top_k,
        paper_ids=[pid],
    )
    excerpts_block, label_rows = _build_context_block(chunks) if chunks else (
        "(no excerpts retrieved)",
        [],
    )

    if show_sources:
        cite = _short_cite(authors, year)
        src = (
            f"  [paper {pid}] {cite} — {title}\n"
            f"           {len(chunks)} excerpt(s) retrieved from this paper"
        )
        panel("Paper located", src, style="dim")

    user_payload = SPECIFIC_PAPER_USER.substitute(
        paper_id=pid,
        title=title,
        authors=", ".join(authors) if authors else "(unknown)",
        year=year if year is not None else "(unknown)",
        venue=venue or "(unknown)",
        doi=doi or "(none)",
        arxiv_id=arxiv_id or "(none)",
        tags=tags,
        summary=summary_block,
        k=len(chunks),
        excerpts=excerpts_block,
        question=intent.raw_question,
    )
    # Substitute $paper_id in the system prompt too (it references the id).
    system_payload = SPECIFIC_PAPER_SYSTEM.replace("$paper_id", str(pid))

    messages: list[ChatMessage] = [ChatMessage("system", system_payload)]
    for turn in history:
        messages.append(ChatMessage(turn.role, turn.content))
    messages.append(ChatMessage("user", user_payload))
    return _stream_messages_to_panel(messages)


def _handle_library_summary(
    intent: "router.QueryIntent",
    history: list[ChatTurn],
    *,
    show_sources: bool,
) -> str:
    """Route: user wants a narrative overview of the whole library."""
    agg = router.build_library_aggregates()
    if agg.get("n", 0) == 0:
        msg = "Your library is empty. Add a paper with `bibwizard add <pdf>`."
        _print_direct_answer(msg)
        return msg

    facts = _format_aggregates_block(agg)
    # A small detailed sample so the LLM can name specific papers
    detail_block = build_library_overview(
        relevant_paper_ids=None, intent="library_wide", max_detail_papers=30,
    )

    if show_sources:
        n = agg["n"]
        yr = agg.get("year_range")
        yr_str = f"{yr[0]}–{yr[1]}" if yr else "?"
        panel(
            "Aggregates",
            f"  {n} paper(s), years {yr_str}\n"
            f"  computed deterministically from SQLite (not from RAG)",
            style="dim",
        )

    user_payload = LIBRARY_SUMMARY_USER.substitute(
        question=intent.raw_question,
        facts=facts,
        detail=detail_block,
    )
    messages: list[ChatMessage] = [ChatMessage("system", LIBRARY_SUMMARY_SYSTEM)]
    for turn in history:
        messages.append(ChatMessage(turn.role, turn.content))
    messages.append(ChatMessage("user", user_payload))
    return _stream_messages_to_panel(messages)


def _format_history_for_router(history: list[ChatTurn], max_turns: int = 6) -> str:
    """Render the tail of the conversation for the router prompt.

    We include the LAST max_turns turns (default 6 — three back-and-forths)
    so the router can stitch together multi-turn intents: "find a citation"
    on turn 1 + the statement on turn 2 should resolve to a single tool call.
    Older context is dropped because the routing decision is local — long-
    range memory belongs to the RAG path, not classification.
    """
    if not history:
        return "(no prior turns)"
    tail = history[-max_turns:]
    lines: list[str] = []
    for turn in tail:
        role = "USER" if turn.role == "user" else "ASSISTANT"
        # Truncate each turn so a long pasted PDF excerpt doesn't dominate
        # the router prompt budget.
        content = (turn.content or "").strip().replace("\n", " ")
        if len(content) > 500:
            content = content[:500] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# Deterministic keyword routes — handled BEFORE the LLM router so a tiny
# routing model can't second-guess an obvious command-style request. Each
# pattern captures the claim/query text in group 1.
_EXPLICIT_CITE_PATTERNS = [
    # cite "claim"  /  cite 'claim'  /  cite “claim”
    # Optional `bibwizard ` prefix in case the user pastes the whole shell command.
    re.compile(
        r"""^\s*(?:bibwizard\s+)?cite\s+["'“‘](.+)["'”’]\s*$""",
        re.IGNORECASE | re.DOTALL,
    ),
    # cite: claim   /  cite_finder claim
    re.compile(
        r"""^\s*(?:bibwizard\s+)?cite(?:_finder)?\s*:\s*(.+)$""",
        re.IGNORECASE | re.DOTALL,
    ),
    # find (a/the) (cite|citation|reference) (for|to support|that supports|justifying|to justify|that justifies) <claim>
    re.compile(
        r"""^\s*(?:find|give|get|i\s+need)\s+(?:me\s+)?(?:a|an|the)?\s*"""
        r"""(?:cite|citation|reference|source)s?\s+"""
        r"""(?:for|to\s+support|that\s+supports?|justifying|to\s+justify|"""
        r"""that\s+justif(?:ies|y)|backing|backing\s+up)\s+(.+)$""",
        re.IGNORECASE | re.DOTALL,
    ),
    # "who showed that <claim>" / "who demonstrated that <claim>"
    re.compile(
        r"""^\s*who\s+(?:showed|demonstrated|proved|reported|found)\s+(?:that\s+)?(.+\??)$""",
        re.IGNORECASE | re.DOTALL,
    ),
]

# Strip trailing CLI flags so the explicit cite regex still matches when the
# user pastes a shell command into chat (`bibwizard cite "..." --tex --debug`).
# Some flags take values (--max 5, --pool 40, --cite-command citep), some don't
# (--tex, --debug). The optional-value branch covers both. Flags that should
# actually take effect in chat (currently --tex and --cite-command) are
# extracted separately by _extract_chat_cite_flags() before stripping.
_CLI_FLAG_TAIL_RE = re.compile(
    r"""\s+--(?:debug|pool|min-confidence|max|n|tex|cite-command|
             dump-passages|no-rerank)
       (?:[=\s][^\s"']*)?(?=\s|$)""",
    re.IGNORECASE | re.VERBOSE,
)

# Regexes for extracting flag VALUES (vs. just stripping). Operate on the
# ORIGINAL question text, not the stripped version.
_CHAT_TEX_FLAG_RE = re.compile(r"(?:^|\s)--tex\b", re.IGNORECASE)
_CHAT_CITE_COMMAND_RE = re.compile(
    r"--cite-command[=\s]+([A-Za-z]+)", re.IGNORECASE
)


def _extract_chat_cite_flags(question: str) -> dict:
    """Extract chat-supported cite_finder flags from a user message.

    Currently recognized:
      --tex                    → tex=True
      --cite-command <name>    → cite_command="<name>" (lowercase)

    Other CLI flags (--debug, --pool, --max, etc.) are stripped by
    _CLI_FLAG_TAIL_RE but not surfaced here because they have no effect
    inside the chat REPL — debug output and pool-size tuning belong to the
    CLI, not the conversational interface.

    Returns an args dict suitable for merging into a cite_finder tool call.
    """
    args: dict = {}
    if _CHAT_TEX_FLAG_RE.search(question):
        args["tex"] = True
    m = _CHAT_CITE_COMMAND_RE.search(question)
    if m:
        args["cite_command"] = m.group(1).lower()
    return args


def _explicit_cite_claim(question: str) -> str | None:
    """Return the extracted claim if `question` is an explicit cite request.

    Catches command-style phrasings the LLM router occasionally misclassifies
    as RAG ("cite '<claim>'"), or natural-language phrasings where the intent
    is unambiguous ("find a citation for <claim>"). Returns None when the
    question doesn't match any explicit pattern — in that case fall through
    to the LLM router as usual.
    """
    if not question:
        return None
    q = question.strip()
    # Strip CLI-only flag tails like "--debug" so a pasted shell command still
    # routes correctly (the flag has no effect in chat).
    q = _CLI_FLAG_TAIL_RE.sub("", q).strip()
    for pat in _EXPLICIT_CITE_PATTERNS:
        m = pat.match(q)
        if m:
            claim = (m.group(1) or "").strip().strip(" \"'“”‘’")
            # Trim trailing question marks so "who showed X?" gives us "X".
            claim = claim.rstrip("?").strip()
            if len(claim) >= 8:  # avoid matching trivially short tokens
                return claim
    return None


def _classify_tool_call(
    user_question: str, history: list[ChatTurn] | None = None
) -> tuple[str, dict]:
    """Ask the LLM router what to do with the question.

    Returns a (action, payload) tuple. `action` is one of:
      - "tool" : payload = {"tool": "<name>", "args": {...}}
      - "ask"  : payload = {"question": "<clarification to show the user>"}
      - "rag"  : payload = {}     (default — fall through to retrieval)
    """
    import json as _json

    try:
        client = get_client()
        sys_msg = ChatMessage("system", TOOL_ROUTER_SYSTEM)
        user_msg = ChatMessage(
            "user",
            TOOL_ROUTER_USER.substitute(
                catalogue=chat_tools.tool_catalogue_for_prompt(),
                history=_format_history_for_router(history or []),
                question=user_question,
            ),
        )
        # Non-streaming, low-temperature, JSON-format-constrained.
        raw = client.chat(
            [sys_msg, user_msg],
            stream=False,
            options={"temperature": 0.0},
            format="json",
        )
    except Exception:
        return "rag", {}

    text = _strip_think_and_fences(raw if isinstance(raw, str) else "")
    try:
        obj = _json.loads(text)
    except Exception:
        return "rag", {}

    action = (obj.get("action") or "").strip().lower()

    if action == "tool":
        name = obj.get("tool")
        if not name or name not in chat_tools.TOOLS:
            return "rag", {}
        args = obj.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        # Reject obviously empty required-arg tool calls — the router should
        # have asked instead. We treat this as an implicit "ask".
        tool_def = chat_tools.TOOLS[name]
        missing = [
            n for n, spec in tool_def.args.items()
            if spec.required and not str(args.get(n) or "").strip()
        ]
        if missing:
            return "ask", {
                "question": _default_clarification(name, missing),
            }
        return "tool", {"tool": name, "args": args}

    if action == "ask":
        question = (obj.get("question") or "").strip()
        if question:
            return "ask", {"question": question}
        return "rag", {}

    # Unknown action ("rag", "null", or anything else) → fall through.
    return "rag", {}


def _default_clarification(tool_name: str, missing_args: list[str]) -> str:
    """Friendly fallback question when the router picks a tool with empty
    required args (the LLM should have done this itself, but we backstop it)."""
    a = ", ".join(missing_args)
    nice = {
        "find": "What topic or concept should I search for?",
        "grep": "What exact phrase should I look for?",
        "show": "Which paper id should I show?",
    }
    return nice.get(tool_name, f"I need more info — could you provide: {a}?")


def _strip_think_and_fences(text: str) -> str:
    """Remove <think> blocks, ```json fences, and surrounding whitespace."""
    s = text or ""
    s = _THINK_OPEN.sub("", s)
    s = _THINK_CLOSE.sub("", s)
    # Drop any stray content before the first '{' or after the last '}'.
    open_idx = s.find("{")
    close_idx = s.rfind("}")
    if open_idx >= 0 and close_idx > open_idx:
        s = s[open_idx : close_idx + 1]
    return s.strip()


def _handle_rag(
    user_question: str,
    history: list[ChatTurn],
    *,
    top_k: int,
    show_sources: bool,
) -> str:
    """Default RAG path (unchanged from the original behaviour)."""
    chunks = query_chunks(user_question, top_k=top_k)
    if not chunks:
        info("No chunks retrieved — answering from model knowledge only.")

    intent = _detect_intent(user_question)
    wider_k = 50 if intent == "library_wide" else 25
    try:
        wider_chunks = query_chunks(user_question, top_k=wider_k)
    except Exception:
        wider_chunks = chunks

    messages = build_rag_messages(
        history, user_question, chunks, wider_chunks=wider_chunks
    )

    if show_sources and chunks:
        _, labels = _build_context_block(chunks)
        by_paper: dict[int, dict] = {}
        for row in labels:
            pid = row["paper_id"]
            cur = by_paper.get(pid)
            if cur is None or row["score"] > cur["score"]:
                by_paper[pid] = row
        ordered = sorted(by_paper.values(), key=lambda r: r["label"])
        src_lines = "\n".join(
            f"  [PAPER {row['label']}] {row['short_cite']} — {row['title']}\n"
            f"           p.{row['page']} (score={row['score']:.3f})"
            for row in ordered
        )
        panel("Sources used", src_lines, style="dim")

    return _stream_messages_to_panel(messages)


def stream_answer(
    user_question: str,
    history: list[ChatTurn],
    *,
    top_k: int | None = None,
    show_sources: bool = True,
) -> str:
    """Classify the question, then dispatch.

    The router decides whether this is a structural question (list / count /
    specific-paper lookup / library summary) or a semantic one (RAG). For
    structural questions we use deterministic SQL and either answer directly
    (list / count) or hand the LLM a focused payload (specific paper /
    library summary). For semantic questions we use the original RAG path.
    """
    client = get_client()
    client.ensure_ready(need_llm=True, need_embed=True)

    qi = router.classify(user_question)
    effective_top_k = top_k or settings.rag_top_k

    if qi.type == "count_papers":
        if show_sources:
            panel(
                "Route",
                f"  count_papers (deterministic SQL — no LLM)\n"
                f"  filters: year={qi.year}, year_range={qi.year_range}, "
                f"author~={qi.author_surname}, tag={qi.tag}",
                style="dim",
            )
        answer = router.handle_count(qi)
        _print_direct_answer(answer)
        return answer

    if qi.type == "list_papers":
        if show_sources:
            panel(
                "Route",
                f"  list_papers (deterministic SQL — no LLM)\n"
                f"  filters: year={qi.year}, year_range={qi.year_range}, "
                f"author~={qi.author_surname}, tag={qi.tag}",
                style="dim",
            )
        answer = router.handle_list(qi)
        _print_direct_answer(answer)
        return answer

    if qi.type == "specific_paper":
        if show_sources:
            panel(
                "Route",
                f"  specific_paper (SQL lookup → focused LLM)\n"
                f"  ref: paper_id={qi.paper_id}, "
                f"author~={qi.author_surname}, year={qi.year}",
                style="dim",
            )
        return _handle_specific_paper(
            qi, history, top_k=effective_top_k, show_sources=show_sources
        )

    if qi.type == "library_summary":
        if show_sources:
            panel(
                "Route",
                "  library_summary (SQL aggregates → LLM narrates the facts)",
                style="dim",
            )
        return _handle_library_summary(qi, history, show_sources=show_sources)

    # Explicit keyword route — catches command-style cite requests
    # (`cite "..."`, `find a citation for ...`) BEFORE the LLM router has a
    # chance to misclassify them. The LLM router is a small model (qwen2.5:7b)
    # and tends to fall back to RAG on anything that looks like prose; we
    # don't want that for an obvious cite request.
    explicit_claim = _explicit_cite_claim(user_question)
    if explicit_claim is not None:
        # Extract any chat-supported flags (--tex, --cite-command) from the
        # original question. These get merged into the tool args so a user
        # can type `cite "..." --tex` and get the LaTeX block.
        extra_args = _extract_chat_cite_flags(user_question)
        cite_args = {"claim": explicit_claim, **extra_args}
        if show_sources:
            # Show the full claim — Rich wraps. The previous 120-char ellipsis
            # display was purely cosmetic but looked like the claim was being
            # truncated, which it wasn't (the full claim always goes to the
            # tool). Showing it in full removes that confusion.
            flag_note = ""
            if extra_args:
                flag_note = "\n  flags: " + ", ".join(
                    f"{k}={v}" for k, v in extra_args.items()
                )
            panel(
                "Route",
                f"  cite_finder (explicit cite request)\n"
                f"  claim: “{explicit_claim}”{flag_note}",
                style="dim",
            )
        return chat_tools.run_tool("cite_finder", cite_args)

    # No regex route matched. Ask the LLM router what to do — it may
    # invoke a tool, ask the user for a clarification, or punt to RAG.
    action, payload = _classify_tool_call(user_question, history=history)

    if action == "tool":
        tool_name = payload["tool"]
        tool_args = payload.get("args", {})
        if show_sources:
            arg_repr = ", ".join(f"{k}={v}" for k, v in tool_args.items()) or "(no args)"
            panel(
                "Route",
                f"  tool_call: [bold]{tool_name}[/]({arg_repr})\n"
                f"  (picked by LLM router; output rendered as a table/panel)",
                style="dim",
            )
        try:
            return chat_tools.run_tool(tool_name, tool_args)
        except Exception as e:  # noqa: BLE001
            error(f"Tool {tool_name} failed: {e}. Falling back to RAG.")

    if action == "ask":
        question = payload.get("question") or "Could you clarify what you meant?"
        if show_sources:
            panel(
                "Route",
                "  clarify (router needs more info before running a tool)",
                style="dim",
            )
        _print_direct_answer(question)
        # Return the question as the assistant's turn so it lands in history;
        # the user's next reply will arrive with this context attached.
        return question

    # Default: RAG
    if show_sources:
        panel("Route", "  rag (semantic retrieval → LLM)", style="dim")
    return _handle_rag(
        user_question, history, top_k=effective_top_k, show_sources=show_sources
    )


def run_chat_loop() -> None:
    """REPL entry point for `bibwizard chat`."""
    info(
        f"Chat session — model={settings.ollama_llm_model}, "
        f"embed={settings.ollama_embed_model}, top_k={settings.rag_top_k}. "
        "Type /quit to exit, /help for commands."
    )

    history: list[ChatTurn] = []
    while True:
        try:
            question = console.input("[bold cyan]you ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not question:
            continue
        if question in {"/quit", "/exit", "/q"}:
            return
        if question == "/help":
            panel(
                "Commands",
                "  /quit, /exit, /q   leave the chat\n"
                "  /clear             clear conversation history\n"
                "  /tools             list tools the LLM router can call\n"
                "  /help              show this help",
            )
            continue
        if question == "/clear":
            history.clear()
            info("History cleared.")
            continue
        if question == "/tools":
            panel(
                "Tools the router can invoke",
                chat_tools.tool_catalogue_for_prompt(),
                style="cyan",
            )
            continue

        try:
            answer = stream_answer(question, history)
        except OllamaUnavailable as e:
            error(str(e))
            return
        except Exception as e:  # noqa: BLE001
            error(f"Chat failed: {e}")
            continue

        history.append(ChatTurn("user", question))
        history.append(ChatTurn("assistant", answer))
        # Render markdown version too, since DeepSeek likes lists & code blocks
        console.print(Markdown(answer))
