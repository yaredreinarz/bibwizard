"""Structured summary generation using DeepSeek via Ollama."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from bibwizard.ingestion.metadata import PaperMetadata
from bibwizard.ingestion.parser import ParsedPDF
from bibwizard.ingestion.structure import (
    FrontMatter,
    extract_front_matter,
    parse_authors_from_byline,
)

from .client import ChatMessage, OllamaClient, get_client
from .prompts import (
    AUTHORS_ONLY_SYSTEM,
    AUTHORS_ONLY_USER,
    SUMMARY_SYSTEM,
    SUMMARY_USER,
)


# DeepSeek-R1 emits <think> reasoning. We strip it before parsing JSON.
_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class StructuredSummary:
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    key_contributions: list[str] = field(default_factory=list)
    methodology: str = ""
    limitations: str = ""
    tags: list[str] = field(default_factory=list)
    raw: str = ""  # the LLM's raw response, for debugging

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "key_contributions": list(self.key_contributions),
            "methodology": self.methodology,
            "limitations": self.limitations,
            "tags": list(self.tags),
        }


def _strip_thinking(text: str) -> str:
    return _THINK_BLOCK.sub("", text).strip()


def _coerce_int(v) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _coerce_str_list(v) -> list[str]:
    """Normalize LLM output to list[str]. Unwraps dict-shaped author objects
    like {"name": "X", "affiliation": "Y"} that some models return."""
    if not v:
        return []
    if isinstance(v, str):
        return [s.strip() for s in re.split(r"[;,\n]", v) if s.strip()]
    if isinstance(v, list):
        out: list[str] = []
        for x in v:
            if isinstance(x, dict):
                for k in ("name", "Name", "full_name", "fullName", "author"):
                    if x.get(k):
                        out.append(str(x[k]).strip())
                        break
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    if isinstance(v, dict):
        return _coerce_str_list([v])
    return []


def parse_summary_response(raw: str) -> StructuredSummary:
    """Best-effort parsing of the LLM's JSON response."""
    cleaned = _strip_thinking(raw)
    # Strip ```json fences if present
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned.strip(), flags=re.MULTILINE)

    data: dict = {}
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(cleaned)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = {}

    return StructuredSummary(
        title=str(data.get("title", "") or "").strip(),
        authors=_coerce_str_list(data.get("authors")),
        year=_coerce_int(data.get("year")),
        key_contributions=_coerce_str_list(data.get("key_contributions")),
        methodology=str(data.get("methodology", "") or "").strip(),
        limitations=str(data.get("limitations", "") or "").strip(),
        tags=[t.lower().strip().replace(" ", "-") for t in _coerce_str_list(data.get("tags"))],
        raw=raw,
    )


def _truncate_for_context(text: str, max_chars: int = 24000) -> str:
    """Keep head + tail of body since key info often sits in intro & conclusion."""
    if len(text) <= max_chars:
        return text
    head = text[: max_chars * 2 // 3]
    tail = text[-max_chars // 3 :]
    return head + "\n\n[... truncated middle ...]\n\n" + tail


def _extract_authors_focused(
    front: FrontMatter, client: OllamaClient
) -> list[str]:
    """Last-resort: tiny focused LLM call with ONLY the byline as context."""
    if not front.byline_text or not front.byline_text.strip():
        return []
    user_msg = AUTHORS_ONLY_USER.substitute(
        title=front.title or "",
        byline=front.byline_text,
    )
    try:
        response = client.chat(
            [
                ChatMessage("system", AUTHORS_ONLY_SYSTEM),
                ChatMessage("user", user_msg),
            ],
            stream=False,
            options={"temperature": 0.0},
            format="json",
        )
    except Exception:
        return []
    if not isinstance(response, str):
        return []
    try:
        data = json.loads(_strip_thinking(response))
    except json.JSONDecodeError:
        return []
    return _coerce_str_list(data.get("authors"))


def summarize_paper(
    parsed: ParsedPDF,
    metadata: PaperMetadata,
    client: OllamaClient | None = None,
    *,
    front_matter: FrontMatter | None = None,
) -> StructuredSummary:
    """Run the structured summary prompt over the parsed paper.

    Pipeline:
      1. Structure-scrape front matter (title / byline / abstract) from page 1.
      2. Pass those clean regions to the main summary LLM call.
      3. If the model returns no authors, retry with a focused author-only call
         using just the byline as context (much harder to ignore).
    """
    client = client or get_client()
    client.ensure_ready(need_llm=True, need_embed=False)

    front = front_matter or extract_front_matter(parsed.path)
    body = _truncate_for_context(parsed.body_text or parsed.raw_text)

    user_msg = SUMMARY_USER.substitute(
        title=metadata.title or "",
        authors=", ".join(metadata.authors) if metadata.authors else "",
        year=str(metadata.year) if metadata.year else "",
        doi=metadata.doi or "",
        arxiv_id=metadata.arxiv_id or "",
        front_title="\n".join(front.title_lines) or front.title or "(unknown)",
        front_byline=front.byline_text or "(empty)",
        front_abstract=front.abstract or metadata.abstract or "(empty)",
        body=body,
    )

    response = client.chat(
        [
            ChatMessage("system", SUMMARY_SYSTEM),
            ChatMessage("user", user_msg),
        ],
        stream=False,
        options={"temperature": 0.2},
        # Constrain Ollama to emit valid JSON. Without this small reasoning
        # models often wrap the result in prose or truncate mid-object.
        format="json",
    )
    assert isinstance(response, str)
    summary = parse_summary_response(response)

    # Backfill with metadata if the model omitted fields
    if not summary.title and metadata.title:
        summary.title = metadata.title
    if not summary.authors and metadata.authors:
        summary.authors = list(metadata.authors)
    if summary.year is None and metadata.year is not None:
        summary.year = metadata.year

    # If we STILL don't have authors, retry with a focused author-only call
    # using just the byline text. Models can ignore a single field of a big
    # JSON ask but rarely refuse a single-field call with tight context.
    if not summary.authors:
        focused = _extract_authors_focused(front, client)
        if focused:
            summary.authors = focused

    # And as a last resort, parse the byline ourselves (regex/heuristic).
    if not summary.authors:
        heuristic = parse_authors_from_byline(front.byline_text)
        if heuristic:
            summary.authors = heuristic

    return summary
