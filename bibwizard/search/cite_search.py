"""Citation finder — given a CLAIM, return papers + verbatim quotes that
support it.

Pipeline (strict mode):
  1. Embed the claim and pull a wide pool of candidate chunks from ChromaDB.
  2. For each chunk, ask the LLM whether it ENTAILS the claim and, if so,
     to quote the supporting sentence verbatim.
  3. Verify each quote actually appears in the chunk (no hallucinated quotes).
  4. Dedup by paper, keep the highest-confidence hit per paper.
  5. Return the top N citations sorted by confidence.

The entailment step is what separates this from `find` — `find` returns
papers ABOUT a topic; this returns papers that CONTAIN evidence for a claim,
with the exact passage you can paste into your manuscript.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from bibwizard.database.migrations import session_scope
from bibwizard.database.models import Paper
from bibwizard.ingestion.embedder import query_chunks
from bibwizard.llm.client import ChatMessage, OllamaClient, get_client
from bibwizard.llm.prompts import (
    CITE_ENTAILMENT_SENT_SYSTEM,
    CITE_ENTAILMENT_SENT_USER,
    CITE_ENTAILMENT_SYSTEM,
    CITE_ENTAILMENT_USER,
)
from bibwizard.search.reranker import Reranker, get_reranker
from bibwizard.utils.config import settings


@dataclass
class CitationHit:
    paper_id: int
    paper_title: str
    paper_cite: str          # "Smith et al. 2021"
    paper_authors: list[str]
    paper_year: int | None
    page: int
    quoted_sentence: str
    confidence: float
    rationale: str
    chunk_score: float       # original ChromaDB similarity score

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "paper_cite": self.paper_cite,
            "page": self.page,
            "quoted_sentence": self.quoted_sentence,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "chunk_score": self.chunk_score,
        }


def _short_cite(authors: list[str], year: int | None) -> str:
    if not authors:
        return f"(unknown){f' {year}' if year else ''}".strip()
    first = authors[0]
    last = first.split(",", 1)[0].strip() if "," in first else (
        first.split()[-1] if first.split() else first
    )
    yr = f" {year}" if year else ""
    if len(authors) == 1:
        return f"{last}{yr}"
    if len(authors) == 2:
        second = authors[1]
        sec_last = second.split(",", 1)[0].strip() if "," in second else (
            second.split()[-1] if second.split() else second
        )
        return f"{last} & {sec_last}{yr}"
    return f"{last} et al.{yr}"


def _strip_json_fences(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", s, flags=re.IGNORECASE | re.DOTALL)
    open_idx = s.find("{")
    close_idx = s.rfind("}")
    if open_idx >= 0 and close_idx > open_idx:
        s = s[open_idx : close_idx + 1]
    return s.strip()


# Common PDF ligatures (and a few other typographic artifacts) that PyMuPDF
# preserves verbatim from the source PDF but that the LLM naturally
# normalizes when quoting. Without folding these the verbatim-in-passage
# check rejects perfectly real quotes — see the Halverson 2015 case.
_LIGATURE_MAP = {
    "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬀ": "ff",
    "ﬅ": "ft", "ﬆ": "st",
    # Smart quotes → straight; LLM produces straight quotes, PDFs often
    # have smart ones (or vice versa).
    "“": '"', "”": '"', "‘": "'", "’": "'",
    # Various dashes → hyphen.
    "–": "-", "—": "-", "−": "-",
    # Non-breaking space and other oddities
    " ": " ", "​": "",
}
_LIGATURE_RE = re.compile("|".join(re.escape(k) for k in _LIGATURE_MAP))


def _normalize_for_compare(s: str) -> str:
    """Normalize a string for the 'is the quote actually in the passage?'
    check. Folds PDF ligatures, smart quotes, dashes, casing, and whitespace
    so that an LLM quote like "These fibers support…" matches a PDF chunk
    that actually contains "These ﬁbers support…" with a ligature.
    """
    s = _LIGATURE_RE.sub(lambda m: _LIGATURE_MAP[m.group(0)], s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    # Strip leading/trailing quotes the LLM sometimes adds
    return s.strip(" \"'“”‘’")


def _quote_is_in_passage(quote: str, passage: str) -> bool:
    """Forgiving substring check — the LLM sometimes drops figure refs / page
    numbers when quoting, so we require ≥ 60% of the quote's words to appear
    contiguously in the passage."""
    q = _normalize_for_compare(quote)
    p = _normalize_for_compare(passage)
    if not q or not p:
        return False
    if q in p:
        return True
    # Try: longest prefix of quote that is contained in passage
    words = q.split()
    if len(words) < 4:
        return False
    # Walk back from the full quote, shaving 10% at a time
    for cut in (1.0, 0.9, 0.8, 0.7, 0.6):
        n = max(4, int(len(words) * cut))
        candidate = " ".join(words[:n])
        if candidate in p:
            return True
    return False


# --- Sentence splitting for sentence-level entailment ---------------------
#
# A robust sentence splitter is hard, but for entailment we don't need
# perfection — over-splitting is fine (LLM sees adjacent context anyway),
# under-splitting is fine (the merged sentence still contains the evidence).
# We use a simple regex that handles common cases: punctuation followed by
# whitespace + capital letter. We protect a handful of abbreviations (e.g.,
# i.e., et al., Fig., Eq.) to avoid splitting mid-citation.

_SENT_PROTECT = re.compile(
    r"\b(?:e\.g|i\.e|et al|cf|vs|fig|figs|eq|eqs|sec|ref|refs|"
    r"vol|no|pp|p|approx|min|max|avg|mr|mrs|dr|prof|st|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
    r"\.\s+",
    re.IGNORECASE,
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

# Front-matter detection patterns. Used to skip noisy context sentences
# (byline / affiliation / submission metadata) when building the displayed
# quote — the picked sentence itself is never filtered. A chunk that
# happens to start mid-front-matter (common for page-1 chunks) would
# otherwise produce a context window like
#   "Crepp, Ryan Ketterer ... abechter@nd.edu Received 2020 April 12 ..."
# instead of useful body prose.
_FRONTMATTER_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.IGNORECASE)
_FRONTMATTER_DATE_RE = re.compile(
    r"\b(?:Received|Accepted|Published|Submitted|Revised)\s+\d{4}\b",
    re.IGNORECASE,
)
_FRONTMATTER_PROCS_RE = re.compile(
    r"\b(?:Proc\.|Proceedings)\s+(?:of\s+)?SPIE\b|"
    r"\bSPIE\s+Vol\.\b|"
    r"\bDOI:\s*10\.\d{4}/|"
    r"\b(?:ISSN|ISBN)\s*[:0-9-]",
    re.IGNORECASE,
)


def _looks_like_frontmatter(sentence: str) -> bool:
    """Heuristic: does this sentence look like paper front-matter (byline,
    affiliations, dates, journal IDs) rather than body prose?

    Used to skip noisy sentences when building the displayed context
    window around a picked sentence. Conservative — only flags clear
    metadata signals (email, "Received YYYY", "Proc. SPIE", DOI prefix).
    Body sentences that incidentally mention an institution or date are
    NOT filtered.
    """
    if not sentence:
        return True
    if _FRONTMATTER_EMAIL_RE.search(sentence):
        return True
    if _FRONTMATTER_DATE_RE.search(sentence):
        return True
    if _FRONTMATTER_PROCS_RE.search(sentence):
        return True
    return False


def _split_into_sentences(passage: str, max_sentences: int = 60) -> list[str]:
    """Split a chunk into sentences for sentence-level entailment.

    The goal isn't linguistic perfection — it's giving the LLM short,
    discrete units to evaluate against the claim. A few over-splits or
    under-splits are fine; what matters is that no single unit is so long
    the LLM loses the thread.

    Args:
      passage: the chunk text.
      max_sentences: hard cap to keep the prompt budget bounded.

    Returns:
      List of cleaned sentence strings, in original order.
    """
    if not passage:
        return []
    # Normalize whitespace
    text = re.sub(r"\s+", " ", passage).strip()
    # Protect common abbreviations from splitting by replacing the trailing
    # space with a special marker. This is a cheap hack but effective.
    placeholder = "\x00"
    text = _SENT_PROTECT.sub(lambda m: m.group(0).rstrip() + placeholder, text)
    # Split on sentence terminators
    parts = _SENT_SPLIT.split(text)
    # Undo placeholder
    parts = [p.replace(placeholder, " ").strip() for p in parts]
    # Drop empties and very short fragments (figure numbers, equation
    # labels). Keep anything ≥ 30 chars or that contains a digit (numeric
    # claims often live in short sentences).
    sentences = [
        s for s in parts
        if s and (len(s) >= 30 or any(ch.isdigit() for ch in s))
    ]
    return sentences[:max_sentences]


def _entail(
    *, claim: str, paper_id: int, page: int, passage: str, client: OllamaClient,
) -> dict | None:
    """Sentence-level entailment. Pre-splits the passage and asks the LLM
    to PICK a sentence index, not generate a verbatim quote. This is much
    more reliable on small models — the task is multiple-choice over short
    pieces, not haystack-search.

    Returns a dict normalized to the chunk-level schema for the rest of
    the pipeline:
      {"supports": bool, "quote": str, "confidence": float, "rationale": str}
    where `quote` is looked up from the splitter's sentence list (so it's
    guaranteed to be verbatim from the passage).

    Returns None on parse / network error.
    """
    sentences = _split_into_sentences(passage)
    if not sentences:
        return None

    # Render numbered list with a small budget. Truncate any single
    # sentence to ~500 chars to keep total prompt sane.
    numbered_lines = []
    for i, s in enumerate(sentences, start=1):
        snippet = s if len(s) <= 500 else s[:500] + "…"
        numbered_lines.append(f"[{i}] {snippet}")
    numbered = "\n".join(numbered_lines)

    user_msg = CITE_ENTAILMENT_SENT_USER.substitute(
        claim=claim,
        paper_id=paper_id,
        page=page,
        sentences=numbered,
    )
    try:
        raw = client.chat(
            [
                ChatMessage("system", CITE_ENTAILMENT_SENT_SYSTEM),
                ChatMessage("user", user_msg),
            ],
            stream=False,
            options={"temperature": 0.0},
            format="json",
        )
    except Exception:
        return None
    if not isinstance(raw, str):
        return None
    text = _strip_json_fences(raw)
    try:
        obj = json.loads(text)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None

    # Normalize to chunk-level schema (so the rest of find_citations
    # doesn't need to know which entailment mode was used). Accept both
    # the new list schema (sentence_indices) and the legacy single-int
    # schema (sentence_index) so a model that occasionally falls back to
    # the older form still produces a working result.
    supports = bool(obj.get("supports"))
    raw_indices = obj.get("sentence_indices")
    if raw_indices is None and "sentence_index" in obj:
        # Legacy single-int form
        raw_indices = [obj.get("sentence_index")]
    if not isinstance(raw_indices, list):
        raw_indices = []
    # Coerce to ints, drop invalid / out-of-range entries, dedupe, sort
    indices: list[int] = []
    seen_idx: set[int] = set()
    for v in raw_indices:
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if 1 <= iv <= len(sentences) and iv not in seen_idx:
            indices.append(iv)
            seen_idx.add(iv)
    indices.sort()

    try:
        confidence = float(obj.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    rationale = (obj.get("rationale") or "").strip()

    # Build the quote from the picked sentence(s).
    #
    # Single-sentence case: expand to a 3-sentence window (picked ±1) for
    # readability, skipping front-matter junk in the surrounding context.
    # This preserves the test-9 / test-11 behaviour for context-dependent
    # fragments.
    #
    # Multi-sentence case: the picked sentences themselves ARE the
    # multi-sentence span. No extra context expansion — they tell the
    # whole story.
    quote = ""
    primary_sentence = ""
    if supports and indices:
        if len(indices) == 1:
            idx = indices[0]
            primary_sentence = sentences[idx - 1]
            prior = None
            prior_pos = idx - 2
            if prior_pos >= 0 and not _looks_like_frontmatter(sentences[prior_pos]):
                prior = sentences[prior_pos]
            nxt = None
            next_pos = idx
            if next_pos < len(sentences) and not _looks_like_frontmatter(sentences[next_pos]):
                nxt = sentences[next_pos]
            parts: list[str] = []
            if prior is not None:
                parts.append(prior)
            parts.append(f"«{primary_sentence}»")
            if nxt is not None:
                parts.append(nxt)
            quote = " ".join(parts).strip()
        else:
            # Multi-sentence: render all picked sentences in order, each
            # wrapped in guillemets. If they're non-adjacent (gaps in
            # indices), join with " [...] " so the user sees the elision.
            primary_sentence = sentences[indices[0] - 1]  # first picked, for verification
            chunks_out: list[str] = []
            prev = None
            for idx in indices:
                s = sentences[idx - 1]
                if prev is not None and idx > prev + 1:
                    chunks_out.append("[...]")
                chunks_out.append(f"«{s}»")
                prev = idx
            quote = " ".join(chunks_out).strip()
    elif supports and not indices:
        # LLM said supports=true but returned no indices — invalid.
        supports = False
        rationale = (rationale + " [LLM said supports without picking sentences]").strip()

    return {
        "supports": supports,
        "quote": quote,
        "confidence": confidence,
        "rationale": rationale,
        # Debug surface — expose the full pick list so the debug table
        # can show "[3,4,5]" instead of just one index.
        "_sentence_index": indices[0] if indices else 0,
        "_sentence_indices": indices,
        "_n_sentences": len(sentences),
        "_primary_sentence": primary_sentence,
    }


def find_citations(
    claim: str,
    *,
    pool_size: int = 20,
    max_results: int = 5,
    min_confidence: float = 0.5,
    progress_cb: Callable[[int, int], None] | None = None,
    debug_cb: Callable[[dict], None] | None = None,
    reranker: Reranker | None = None,
    max_per_paper: int | None = None,
) -> list[CitationHit]:
    """Return up to `max_results` papers whose chunks entail the claim.

    Args:
      claim: the statement you want a citation for.
      pool_size: how many candidate chunks to retrieve from ChromaDB.
        20 is a good default — large enough to cover the top few papers
        in detail, small enough to keep entailment latency to ~30-60s on
        qwen2.5:7b.
      max_results: how many distinct papers to return.
      min_confidence: drop hits below this LLM-reported confidence.
      progress_cb: optional callback `(done, total)` for showing progress
        while the LLM grinds through the pool.
      debug_cb: optional callback receiving a dict per candidate chunk with
        the full entailment verdict and reason. Useful for diagnosing why
        a chunk that obviously should support a claim is being rejected.
        Fields:
          - i, total
          - paper_id, page, chunk_score
          - passage_preview                  (first 1500 chars)
          - passage_full                     (entire chunk text)
          - verdict                          : "accepted" or "rejected_*"
          - supports, confidence, quote, rationale (when LLM responded)

    Returns:
      List of CitationHit, sorted by confidence DESC. One row per paper —
      the highest-confidence chunk wins when a paper has several supportive
      passages.
    """
    if not claim or not claim.strip():
        return []
    claim = claim.strip()

    # Two-stage retrieval: pull a wider pool from ChromaDB cheaply, then
    # rerank with a cross-encoder (or lexical fallback) to surface the chunks
    # that actually support the claim. Without this, noisy/topical chunks
    # routinely outrank the chunk that contains the answer.
    if reranker is None:
        reranker = get_reranker()
    overscan = max(1, int(getattr(settings, "reranker_overscan", 5) or 5))
    if reranker.name == "passthrough":
        # Skip the wider pull when reranking is disabled — no point.
        wide_pool = max(pool_size, 1)
    else:
        wide_pool = max(pool_size * overscan, pool_size)

    wide_chunks = query_chunks(claim, top_k=wide_pool)
    if not wide_chunks:
        return []
    # Rerank the entire wide pool first so we have an accurate score
    # ordering, then apply the per-paper cap so the entailment pool stays
    # diverse. Asking the reranker for top-N first would defeat the cap.
    reranked = reranker.rerank(claim, wide_chunks, top_k=len(wide_chunks))
    if not reranked:
        return []

    # Per-paper diversification: cap any single paper at max_per_paper
    # chunks in the entailment pool. Without this, a paper with many
    # semantically-similar chunks (a topical web article, a long noise
    # doc) can flood the pool and crowd out the paper that actually
    # contains the answer.
    if max_per_paper is None:
        max_per_paper = max(
            1, int(getattr(settings, "reranker_max_per_paper", 3) or 3)
        )
    seen: dict[int, int] = {}
    chunks: list[dict] = []
    for ch in reranked:
        pid = int((ch.get("metadata") or {}).get("paper_id", -1))
        if pid < 0:
            continue
        if seen.get(pid, 0) >= max_per_paper:
            continue
        seen[pid] = seen.get(pid, 0) + 1
        chunks.append(ch)
        if len(chunks) >= pool_size:
            break
    if not chunks:
        return []

    client = get_client()
    client.ensure_ready(need_llm=True, need_embed=True)

    # Hydrate paper metadata for the chunks we got back so we don't hit SQLite
    # once per chunk.
    paper_ids = sorted({
        int((ch.get("metadata") or {}).get("paper_id", -1))
        for ch in chunks
    })
    paper_ids = [pid for pid in paper_ids if pid >= 0]
    paper_info: dict[int, dict] = {}
    with session_scope() as session:
        for p in session.query(Paper).filter(Paper.id.in_(paper_ids)).all():
            authors = [a.name for a in p.authors]
            paper_info[p.id] = {
                "title": p.title or "(untitled)",
                "authors": authors,
                "year": p.year,
                "cite": _short_cite(authors, p.year),
            }

    # Entailment pass — one LLM call per candidate chunk.
    # Best-by-paper map so the final list has one entry per paper.
    best_by_paper: dict[int, CitationHit] = {}
    total = len(chunks)

    def _emit_debug(
        *, i: int, ch: dict, pid: int, page: int, passage: str,
        verdict: str, result: dict | None = None,
    ) -> None:
        if not debug_cb:
            return
        row: dict = {
            "i": i,
            "total": total,
            "paper_id": pid,
            "page": page,
            "chunk_score": float(ch.get("score", 0.0)),
            "rerank_score": float(ch.get("rerank_score", 0.0)),
            "rerank_components": ch.get("rerank_components") or {},
            # Bumped from 200 → 1500 chars: a 200-char preview routinely cut
            # off the actual target sentence, making the debug table useless
            # for confirming whether a chunk did or didn't contain the
            # evidence. 1500 fits most chunks in full.
            "passage_preview": (passage[:1500] + ("…" if len(passage) > 1500 else "")),
            "passage_full": passage,
            "verdict": verdict,
            "supports": None,
            "confidence": None,
            "quote": None,
            "rationale": None,
        }
        if isinstance(result, dict):
            row["supports"] = bool(result.get("supports"))
            try:
                row["confidence"] = float(result.get("confidence") or 0.0)
            except (TypeError, ValueError):
                row["confidence"] = None
            row["quote"] = (result.get("quote") or "").strip() or None
            row["rationale"] = (result.get("rationale") or "").strip() or None
            # Surface the picked sentence index for sentence-level entailment
            si = result.get("_sentence_index")
            ns = result.get("_n_sentences")
            if si is not None and ns is not None:
                row["sentence_index"] = int(si)
                row["n_sentences"] = int(ns)
        debug_cb(row)

    for i, ch in enumerate(chunks, start=1):
        if progress_cb:
            progress_cb(i, total)
        meta = ch.get("metadata") or {}
        pid = int(meta.get("paper_id", -1))
        if pid < 0:
            continue
        page = int(meta.get("page", -1))
        passage = (ch.get("text") or "").strip()
        if len(passage) < 40:
            _emit_debug(
                i=i, ch=ch, pid=pid, page=page, passage=passage,
                verdict="rejected_passage_too_short",
            )
            continue

        result = _entail(
            claim=claim, paper_id=pid, page=page, passage=passage, client=client,
        )
        if not result:
            _emit_debug(
                i=i, ch=ch, pid=pid, page=page, passage=passage,
                verdict="rejected_no_response",
            )
            continue
        supports = bool(result.get("supports"))
        if not supports:
            _emit_debug(
                i=i, ch=ch, pid=pid, page=page, passage=passage,
                verdict="rejected_not_supports", result=result,
            )
            continue
        quote = (result.get("quote") or "").strip()
        try:
            confidence = float(result.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        rationale = (result.get("rationale") or "").strip()
        if confidence < min_confidence:
            _emit_debug(
                i=i, ch=ch, pid=pid, page=page, passage=passage,
                verdict="rejected_low_confidence", result=result,
            )
            continue
        if not quote:
            _emit_debug(
                i=i, ch=ch, pid=pid, page=page, passage=passage,
                verdict="rejected_empty_quote", result=result,
            )
            continue
        # Reject hallucinated quotes that don't actually appear in the chunk.
        # For sentence-level entailment, verify against the primary picked
        # sentence (which is guaranteed verbatim from the passage by
        # construction). The displayed quote includes ±1 sentence of context
        # plus guillemet decorators, so it might not pass a strict substring
        # check even when nothing's hallucinated.
        verify_text = result.get("_primary_sentence") or quote
        if not _quote_is_in_passage(verify_text, passage):
            _emit_debug(
                i=i, ch=ch, pid=pid, page=page, passage=passage,
                verdict="rejected_quote_not_in_passage", result=result,
            )
            continue

        info = paper_info.get(pid)
        if not info:
            _emit_debug(
                i=i, ch=ch, pid=pid, page=page, passage=passage,
                verdict="rejected_unknown_paper", result=result,
            )
            continue
        hit = CitationHit(
            paper_id=pid,
            paper_title=info["title"],
            paper_cite=info["cite"],
            paper_authors=info["authors"],
            paper_year=info["year"],
            page=page,
            quoted_sentence=quote,
            confidence=confidence,
            rationale=rationale,
            chunk_score=float(ch.get("score", 0.0)),
        )
        _emit_debug(
            i=i, ch=ch, pid=pid, page=page, passage=passage,
            verdict="accepted", result=result,
        )
        # Keep only the strongest hit per paper.
        existing = best_by_paper.get(pid)
        if existing is None or hit.confidence > existing.confidence:
            best_by_paper[pid] = hit

    hits = sorted(best_by_paper.values(), key=lambda h: -h.confidence)
    return hits[:max_results]
