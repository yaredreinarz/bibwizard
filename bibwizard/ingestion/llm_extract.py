"""Careful LLM-driven front-matter extraction.

When the heuristic structure scraper can't be trusted (weird layouts, multi-
column journal templates, OCR artifacts), the local LLM (qwen2.5 et al via
Ollama) does a much better job at "look at this front matter and tell me
what you actually see". This module wraps that flow with:

  - a structured extraction call,
  - an optional second-pass self-review call,
  - validation of the result (no future years, no journal-name titles, no
    obvious garbage),
  - graceful fallback to the heuristic when the LLM result is bad.

It's slow (≈30s–3min per paper on a 7B model), so it's opt-in via the
`LLM_EXTRACT_METADATA=true` env var or the `--llm-extract` CLI flag.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
from dataclasses import dataclass, field

from bibwizard.ingestion.metadata import (
    PaperMetadata,
    _DOI_RE,
    is_valid_arxiv_id,
)
from bibwizard.ingestion.parser import ParsedPDF
from bibwizard.ingestion.structure import (
    FrontMatter,
    _JOURNAL_HEADER_RE,
    _is_banner_line,
    extract_front_matter,
)
from bibwizard.llm.client import OllamaClient, get_client
from bibwizard.llm.prompts import (
    FRONT_MATTER_SYSTEM,
    FRONT_MATTER_USER,
    FRONT_MATTER_VERIFY_SYSTEM,
    FRONT_MATTER_VERIFY_USER,
)
from bibwizard.llm.client import ChatMessage

log = logging.getLogger(__name__)


# DeepSeek-R1 wraps responses in <think>...</think>. Qwen doesn't, but we
# strip them defensively so the same parser works for either model.
_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class ExtractedMetadata:
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    doi: str | None = None
    arxiv_id: str | None = None
    # Diagnostic
    raw_first_pass: str = ""
    raw_verify_pass: str = ""

    def to_paper_metadata(self) -> PaperMetadata:
        return PaperMetadata(
            title=self.title,
            authors=list(self.authors),
            year=self.year,
            doi=self.doi,
            arxiv_id=self.arxiv_id,
            abstract=self.abstract or None,
        )


def _truncate_for_front_matter(text: str, max_chars: int = 12000) -> str:
    """Front matter lives in the first ~2 pages. Cap input length so we don't
    exceed the model's context window on unusually long first pages."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 50] + "\n\n[... truncated ...]"


def _clean_response(raw: str) -> str:
    cleaned = _THINK_BLOCK.sub("", raw or "").strip()
    cleaned = _FENCE_RE.sub("", cleaned).strip()
    return cleaned


def _safe_json_parse(raw: str) -> dict:
    text = _clean_response(raw)
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _coerce_str(v) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _coerce_int(v) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _coerce_str_list(v) -> list[str]:
    """Normalize whatever the LLM returns into a list[str].

    Qwen sometimes returns `[{"name": "X", "affiliation": "Y"}, ...]` instead
    of `["X", "Y", ...]` — unwrap that. Also handles bare strings and lists
    of mixed types.
    """
    if not v:
        return []
    if isinstance(v, list):
        out: list[str] = []
        for x in v:
            if isinstance(x, dict):
                # Prefer 'name', fall back to anything name-shaped.
                for k in ("name", "Name", "full_name", "fullName", "author"):
                    if x.get(k):
                        out.append(str(x[k]).strip())
                        break
                else:
                    # Last-resort: join all string values
                    parts = [str(val).strip() for val in x.values() if isinstance(val, str) and val.strip()]
                    if parts:
                        out.append(parts[0])
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    if isinstance(v, str):
        return [s.strip() for s in re.split(r"[;|\n]", v) if s.strip()]
    if isinstance(v, dict):
        # Single dict instead of a list — unwrap once.
        return _coerce_str_list([v])
    return []


def _parse_extraction(raw: str) -> ExtractedMetadata:
    data = _safe_json_parse(raw)
    return ExtractedMetadata(
        title=_coerce_str(data.get("title")),
        authors=_coerce_str_list(data.get("authors")),
        year=_coerce_int(data.get("year")),
        abstract=_coerce_str(data.get("abstract")),
        doi=_coerce_str(data.get("doi")) or None,
        arxiv_id=_coerce_str(data.get("arxiv_id")) or None,
    )


# ---------- post-processing ----------

_PAGE_YEAR_RE = re.compile(r"\b(19[5-9]\d|20\d{2})\b")


def _ground_year(llm_year: int | None, raw_text: str) -> int | None:
    """Reject hallucinated years.

    LLMs occasionally invent a publication year out of nothing — e.g. for
    SPIE proceedings PDFs that print no date on page 1, the model may guess
    based on prior papers it has seen. We refuse any year the LLM returns
    if that exact year doesn't appear in the raw front-matter text.
    """
    if llm_year is None:
        return None
    visible = {int(y) for y in _PAGE_YEAR_RE.findall(raw_text or "")}
    if llm_year not in visible:
        log.info(
            "LLM year %d not in front-matter text %s — dropping as hallucination.",
            llm_year,
            sorted(visible)[:10] if visible else "(none)",
        )
        return None
    return llm_year




# Typical SPIE/proceedings affiliation-marker letters. Restricted to the
# range that actually appears in real bylines (rarely past 'k'), AND we
# require multiple DISTINCT letters to fire — so a single real surname
# ending in 'a' or a list of authors all coincidentally ending in 'n' won't
# trigger stripping.
_AFFIL_LETTERS = frozenset("abcdefghijklm")


def _strip_fused_affiliation_letters(authors: list[str]) -> list[str]:
    """Detect the 'Blinda/Kühnb/Chazelasa' pattern — affiliation superscript
    letters concatenated to the surname with no separator — and strip them.

    Real affiliation-marker runs are contiguous starting at 'a' (`a, b, c, d, …`).
    We compute the longest contiguous prefix from 'a' in the set of trailing
    letters observed, and only strip letters that belong to that prefix.
    This protects legitimate surnames that happen to end in single lowercase
    letters (e.g. 'Restori' → 'i' is far outside the typical run, so keep).

    Activation requires ≥3 authors with trailing letters in the contiguous
    prefix AND prefix length ≥2 (so at least 'a, b').
    """
    if len(authors) < 3:
        return authors

    def _surname(name: str) -> str:
        toks = (name or "").split()
        return toks[-1] if toks else ""

    def _looks_fused(s: str) -> bool:
        return (
            len(s) >= 5
            and s[-1] in _AFFIL_LETTERS
            and s[-2].isalpha() and s[-2].islower()
            and all(c.isalpha() for c in s[-4:-1])
        )

    surnames = [_surname(n) for n in authors]
    candidate_letters = {s[-1] for s in surnames if _looks_fused(s)}
    if not candidate_letters:
        return authors

    # Longest contiguous prefix from 'a' in the candidate set:
    # if {a, b, c, d, e, i} → strip prefix = {a, b, c, d, e}, keep 'i'.
    strip_set: set[str] = set()
    for i, ch in enumerate("abcdefghijklm"):
        if ch in candidate_letters:
            strip_set.add(ch)
        else:
            break
    if len(strip_set) < 2:
        # Not a strong enough run (need at least 'a' and 'b' both seen).
        return authors
    # Need at least 3 fused authors with trailing letters in strip_set.
    if sum(1 for s in surnames if _looks_fused(s) and s[-1] in strip_set) < 3:
        return authors

    out: list[str] = []
    for n in authors:
        toks = (n or "").split()
        if not toks:
            out.append(n)
            continue
        s = toks[-1]
        if _looks_fused(s) and s[-1] in strip_set:
            toks[-1] = s[:-1]
            out.append(" ".join(toks))
        else:
            out.append(n)
    return out


# ---------- validation ----------

def _is_plausible_title(title: str) -> bool:
    """A real paper title is non-empty, not just a journal name / banner,
    and has at least a handful of words."""
    if not title:
        return False
    s = title.strip()
    if len(s) < 6:
        return False
    if _is_banner_line(s):
        return False
    if _JOURNAL_HEADER_RE.match(s):
        return False
    if s.lower() in {"untitled", "abstract", "introduction", "research paper"}:
        return False
    return True


def _is_plausible_year(year: int | None) -> bool:
    if year is None:
        return True  # null is allowed
    upper = datetime.datetime.now().year + 1
    return 1900 <= year <= upper


def _looks_like_real_author(name: str) -> bool:
    s = name.strip()
    if not (3 <= len(s) <= 80):
        return False
    if not re.search(r"[A-Za-zÀ-ÿ]", s):
        return False
    # Anything affiliation-shaped — reject
    low = s.lower()
    if any(
        kw in low
        for kw in (
            "university",
            "institute",
            "laboratory",
            "observatory",
            "department",
            "school of",
            "center for",
            "centre for",
            "max-planck",
            "cnrs",
            "inaf",
            "esa",
            "nasa",
        )
    ):
        return False
    return True


def _validate(em: ExtractedMetadata) -> tuple[bool, list[str]]:
    """Return (ok, list_of_problems)."""
    problems: list[str] = []
    if not _is_plausible_title(em.title):
        problems.append(f"implausible title: {em.title!r}")
    if not _is_plausible_year(em.year):
        problems.append(f"implausible year: {em.year}")
    if em.authors:
        bad = [a for a in em.authors if not _looks_like_real_author(a)]
        if bad:
            problems.append(f"non-name-shaped authors: {bad[:4]}")
    if em.doi and not _DOI_RE.match(em.doi):
        problems.append(f"invalid doi: {em.doi!r}")
    if em.arxiv_id and not is_valid_arxiv_id(em.arxiv_id):
        problems.append(f"invalid arxiv_id: {em.arxiv_id!r}")
    return (len(problems) == 0), problems


# ---------- main entry point ----------

def llm_extract_metadata(
    parsed: ParsedPDF,
    *,
    heuristic: PaperMetadata,
    front_matter: FrontMatter | None = None,
    client: OllamaClient | None = None,
    verify: bool = True,
) -> ExtractedMetadata | None:
    """Run a careful LLM extraction over the paper's front matter.

    Returns the LLM's metadata if it passes validation, else None — the
    caller should fall back to the heuristic. Two-pass when `verify=True`:
    first extract, then send the extraction back through a skeptical-review
    prompt.
    """
    client = client or get_client()
    try:
        client.ensure_ready(need_llm=True, need_embed=False)
    except Exception as e:
        log.warning("LLM extraction skipped — Ollama not ready: %s", e)
        return None

    fm = front_matter or extract_front_matter(parsed.path)

    # Build the raw-text context: page 1, optionally page 2.
    page1_text = parsed.pages[0][1] if parsed.pages else ""
    page2_text = parsed.pages[1][1] if len(parsed.pages) > 1 else ""
    raw_text = _truncate_for_front_matter(page1_text + "\n\n" + page2_text)

    # First pass — structured extraction
    user_msg = FRONT_MATTER_USER.substitute(
        heuristic_title=heuristic.title or fm.title or "(none)",
        heuristic_byline=fm.byline_text or ", ".join(heuristic.authors) or "(none)",
        heuristic_abstract=(heuristic.abstract or fm.abstract or "(none)")[:600],
        raw_text=raw_text,
    )

    try:
        first_response = client.chat(
            [
                ChatMessage("system", FRONT_MATTER_SYSTEM),
                ChatMessage("user", user_msg),
            ],
            stream=False,
            options={"temperature": 0.0},
            format="json",
        )
    except Exception as e:
        log.warning("LLM extract first-pass failed: %s", e)
        return None
    if not isinstance(first_response, str):
        return None

    em = _parse_extraction(first_response)
    em.raw_first_pass = first_response
    em.authors = _strip_fused_affiliation_letters(em.authors)
    em.year = _ground_year(em.year, raw_text)

    # Validation
    ok, issues = _validate(em)
    if not ok:
        log.info("LLM extract first-pass issues: %s", issues)

    # Second pass — self-verification (only if first pass parsed something)
    if verify and (em.title or em.authors):
        prev_json = json.dumps(em.to_paper_metadata().__dict__, default=str, ensure_ascii=False)
        verify_msg = FRONT_MATTER_VERIFY_USER.substitute(
            prev_json=prev_json,
            raw_text=raw_text,
        )
        try:
            second_response = client.chat(
                [
                    ChatMessage("system", FRONT_MATTER_VERIFY_SYSTEM),
                    ChatMessage("user", verify_msg),
                ],
                stream=False,
                options={"temperature": 0.0},
                format="json",
            )
        except Exception as e:
            log.warning("LLM verify pass failed: %s — keeping first-pass result", e)
            second_response = ""
        if isinstance(second_response, str) and second_response.strip():
            em2 = _parse_extraction(second_response)
            em2.raw_first_pass = em.raw_first_pass
            em2.raw_verify_pass = second_response
            em2.authors = _strip_fused_affiliation_letters(em2.authors)
            em2.year = _ground_year(em2.year, raw_text)
            ok2, _ = _validate(em2)
            if ok2 or (em2.title and not em.title):
                em = em2

    # Final validation
    ok, issues = _validate(em)
    if not ok:
        log.warning(
            "LLM extraction rejected after validation (%s) — falling back to heuristic.",
            "; ".join(issues),
        )
        return None
    return em
