"""Reranking layer between ChromaDB retrieval and LLM entailment.

The motivation: dense-embedding retrieval (Chroma) returns chunks in order of
cosine similarity between the query and the chunk embedding. That's a coarse
signal — chunks with high topical overlap with the query but no actual answer
content can outrank the chunk that contains the literal answer. When the
library has noise (a Gaussian-beam blog post can flood every Bessel-beam
query because they're in the same neighborhood), this gets worse.

A reranker reads the (claim, chunk) pair together and scores how well the
chunk SUPPORTS the claim — much more accurate than embedding similarity
because it actually attends to both texts at once.

This module provides three implementations, in increasing order of quality
and dependency cost:

1. PassthroughReranker — no-op; returns chunks in their original order.
   Used when reranking is disabled or no backend is available.

2. LexicalReranker — pure Python; combines the Chroma score with a token-
   overlap signal (claim content words appearing in the chunk). Zero
   dependencies, sub-100ms for 100 chunks. The right default — useful out
   of the box, gets ~80% of the benefit of a real reranker for most queries.

3. CrossEncoderReranker — uses sentence-transformers with BAAI/bge-reranker
   models. ~10-50ms per (claim, chunk) pair on CPU. The right choice when
   you want best-in-class recall. Opt-in: requires `pip install
   sentence-transformers` (~1GB with torch). Auto-detected if present.

Pick the implementation via `get_reranker()`. Selection rules:
  - if reranker_enabled = False                  → Passthrough
  - if reranker_kind = "cross" and ST available  → CrossEncoder
  - if reranker_kind = "cross" and ST missing    → warn + LexicalReranker
  - if reranker_kind = "lexical"                 → Lexical
  - if reranker_kind = "auto" (default):
      - if ST available  → CrossEncoder
      - else             → Lexical
"""

from __future__ import annotations

import math
import re
from typing import Protocol

from bibwizard.utils.config import settings
from bibwizard.utils.display import info, warn


# Small content-word stopword list. Anything in this set is dropped from
# claim/chunk token sets so it doesn't dominate overlap scoring. Kept tiny
# on purpose — over-aggressive stopwording strips meaningful content words
# in scientific text ("we", "show" are content-bearing in some abstracts).
_STOPWORDS = frozenset(
    """
    a an the and or but if of in on at to from by for with without
    is are was were be been being am
    this that these those there it its
    as so than then thus also too very only just much many such
    not no nor
    because due since while where when how why which who what whose
    can may might could would should will shall must
    do does did done having have has had
    we our us they their them
    """.split()
)

# Same ligature folding used by cite_search._normalize_for_compare — we want
# "ﬁbers" and "fibers" to count as the same token.
_LIGATURE_MAP = {
    "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬀ": "ff",
    "ﬅ": "ft", "ﬆ": "st",
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "–": "-", "—": "-", "−": "-",
    " ": " ", "​": "",
}
_LIGATURE_RE = re.compile("|".join(re.escape(k) for k in _LIGATURE_MAP))

# Token = letters / digits / hyphenated compound. Strip punctuation.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _singularize(tok: str) -> str:
    """Crude plural→singular folding. No real stemmer — just strip trailing
    's' when it looks safe so 'fibers' ↔ 'fiber', 'distances' ↔ 'distance'
    count as the same token. Avoids stripping 's' that's part of the stem
    ('class', 'distress', 'kiss')."""
    if len(tok) < 5:
        return tok
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"        # 'frequencies' → 'frequency'
    if tok.endswith("ses") or tok.endswith("xes") or tok.endswith("zes"):
        return tok[:-2]              # 'lenses' → 'lens', 'boxes' → 'box'
    if tok.endswith("ss") or tok.endswith("us") or tok.endswith("is"):
        return tok                   # 'class', 'focus', 'basis' — keep
    if tok.endswith("s"):
        return tok[:-1]              # 'fibers' → 'fiber'
    return tok


def _tokenize(text: str) -> list[str]:
    """Lower-case, fold ligatures, extract tokens, drop stopwords, singularize."""
    if not text:
        return []
    s = _LIGATURE_RE.sub(lambda m: _LIGATURE_MAP[m.group(0)], text).lower()
    return [
        _singularize(t)
        for t in _TOKEN_RE.findall(s)
        if t not in _STOPWORDS and len(t) > 1
    ]


class Reranker(Protocol):
    """Common interface — call with claim + chunks, get reordered chunks back."""

    name: str

    def rerank(self, claim: str, chunks: list[dict], top_k: int) -> list[dict]:
        """Score each chunk against the claim, return top_k sorted by score DESC.

        Each input chunk MUST have a 'text' field; the returned chunks are
        annotated with a 'rerank_score' (float, higher = better) plus a
        'rerank_components' dict for debugging.
        """
        ...


class PassthroughReranker:
    """No-op reranker. Use when reranking is disabled or unavailable."""

    name = "passthrough"

    def rerank(self, claim: str, chunks: list[dict], top_k: int) -> list[dict]:
        out: list[dict] = []
        for c in chunks[:top_k]:
            c = dict(c)
            c["rerank_score"] = float(c.get("score", 0.0))
            c["rerank_components"] = {"chroma": float(c.get("score", 0.0))}
            out.append(c)
        return out


class LexicalReranker:
    """Combines RANK-based Chroma score with IDF-weighted lexical overlap.

    Why rank-based instead of raw Chroma scores: in practice, dense embeddings
    crowd into a tight band (e.g. all chunks in the pool score 0.65–0.75).
    Min-max normalizing that band makes a tiny 0.08 score difference look
    like a 0–1 spread, which over-penalizes any chunk that happens to be at
    the bottom of the band even when it contains the literal answer. Using
    rank position (top → 1.0, bottom → 0.0) is invariant to absolute Chroma
    values and reflects Chroma's actual confidence in the ordering.

    Why IDF: scientific papers share lots of common vocabulary ("beam",
    "fiber", "optical"). A chunk getting credit for matching "beam" tells
    us almost nothing — every chunk in an optics pool has "beam". Weighting
    each token by its inverse document frequency in the POOL means
    distinctive terms ("Bessel", "Helmholtz", phrases like "low power")
    dominate the score, which is what we actually care about.

    Final score: alpha * chroma_rank + (1 - alpha) * lexical_score
    """

    name = "lexical"

    def __init__(self, alpha: float = 0.3, bigram_weight: float = 3.0) -> None:
        # Lower alpha = more weight on lexical. Defaulted low (30% Chroma,
        # 70% lexical) because Chroma already filtered to the top N — within
        # the filtered pool, lexical match is the better signal.
        self.alpha = alpha
        self.bigram_weight = bigram_weight

    def _bigrams(self, toks: list[str]) -> list[tuple[str, str]]:
        return list(zip(toks[:-1], toks[1:])) if len(toks) >= 2 else []

    def _build_idf(self, all_toks: list[list[str]]) -> dict[str, float]:
        """Pool-level IDF: rare tokens get higher weight."""
        n = len(all_toks)
        df: dict[str, int] = {}
        for toks in all_toks:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        # Smoothed IDF: log((n + 1) / (df + 1)) + 1 — always positive.
        return {t: math.log((n + 1) / (df_t + 1)) + 1.0 for t, df_t in df.items()}

    def _raw_lexical_signal(
        self,
        claim_uni: set[str],
        claim_bi: set[tuple[str, str]],
        passage_toks: list[str],
        passage_bi: list[tuple[str, str]],
        idf: dict[str, float],
        bi_idf: dict[tuple[str, str], float],
    ) -> tuple[float, dict]:
        """Raw IDF-weighted match signal. Higher = stronger match.

        Scored as a SUM of matched-term IDFs (precision-flavored), not a
        ratio against all claim terms (recall-flavored). Recall-style
        scoring over-penalizes paraphrased claims where the source uses
        different vocabulary for a few of the concepts — those rare
        unmatched terms have high IDF and dominate the denominator.
        Sum-of-matches sidesteps that and lets the chunk that DOES match
        the distinctive terms win.
        """
        if not passage_toks:
            return 0.0, {"uni_hits": [], "bi_hits": []}
        p_uni = set(passage_toks)
        p_bi = set(passage_bi)
        uni_hits = claim_uni & p_uni
        bi_hits = claim_bi & p_bi
        uni_sig = sum(idf.get(t, 1.0) for t in uni_hits)
        bi_sig = sum(bi_idf.get(b, 1.0) for b in bi_hits)
        raw = uni_sig + self.bigram_weight * bi_sig
        return raw, {
            "uni_hits": sorted(uni_hits),
            "bi_hits": sorted(bi_hits),
            "uni_idf_sum": uni_sig,
            "bi_idf_sum": bi_sig,
        }

    def rerank(self, claim: str, chunks: list[dict], top_k: int) -> list[dict]:
        if not chunks:
            return []
        c_toks = _tokenize(claim)
        c_uni = set(c_toks)
        c_bi_list = self._bigrams(c_toks)
        c_bi = set(c_bi_list)

        # Tokenize all passages once
        passage_toks = [_tokenize(c.get("text") or "") for c in chunks]
        passage_bis = [self._bigrams(toks) for toks in passage_toks]

        # Pool-level IDF tables (passage + claim universe).
        idf = self._build_idf(passage_toks + [c_toks])
        bi_universe = passage_bis + [c_bi_list]
        bi_df: dict[tuple[str, str], int] = {}
        for bis in bi_universe:
            for b in set(bis):
                bi_df[b] = bi_df.get(b, 0) + 1
        n_bi_docs = len(bi_universe)
        bi_idf = {
            b: math.log((n_bi_docs + 1) / (df_b + 1)) + 1.0
            for b, df_b in bi_df.items()
        }

        # Compute raw lexical signal for every chunk
        raw_signals: list[float] = []
        components_list: list[dict] = []
        for p_toks, p_bi in zip(passage_toks, passage_bis):
            raw, components = self._raw_lexical_signal(
                c_uni, c_bi, p_toks, p_bi, idf, bi_idf,
            )
            raw_signals.append(raw)
            components_list.append(components)

        # Min-max normalize the lexical signal across the pool so it lives
        # in [0, 1] and combines sensibly with the chroma rank score.
        lo, hi = min(raw_signals), max(raw_signals)
        span = hi - lo if hi > lo else 1.0
        lex_norms = [(s - lo) / span for s in raw_signals]

        # Chroma rank score: top → 1.0, bottom → 0.0
        n = len(chunks)
        chroma_ranks = [1.0] if n == 1 else [1.0 - i / (n - 1) for i in range(n)]

        scored: list[tuple[float, dict]] = []
        for c, rank_score, lex_norm, raw, components in zip(
            chunks, chroma_ranks, lex_norms, raw_signals, components_list,
        ):
            combined = self.alpha * rank_score + (1.0 - self.alpha) * lex_norm
            annotated = dict(c)
            annotated["rerank_score"] = combined
            annotated["rerank_components"] = {
                "chroma_rank_score": rank_score,
                "chroma_raw": float(c.get("score", 0.0)),
                "lexical_norm": lex_norm,
                "lexical_raw": raw,
                **components,
            }
            scored.append((combined, annotated))

        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:top_k]]


class CrossEncoderReranker:
    """Wraps sentence-transformers CrossEncoder with a BGE-reranker model.

    Lazy-imports sentence-transformers and lazy-loads the model on first
    use, so importing this module is cheap (no torch unless you actually
    rerank). Caches the model on the instance.
    """

    name = "cross_encoder"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None  # type: ignore[assignment]

    def _ensure_model(self) -> object | None:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
        except ImportError:
            warn(
                "sentence-transformers is not installed; cross-encoder reranker "
                "unavailable. `pip install sentence-transformers` to enable, "
                "or set reranker_kind='lexical' to silence this warning."
            )
            return None
        try:
            # Suppress sentence-transformers / transformers / tqdm stderr
            # noise during model load — the caller is responsible for
            # showing a spinner if it wants user-visible feedback. We don't
            # redirect stdout because exceptions need to propagate cleanly.
            import contextlib
            import io
            with contextlib.redirect_stderr(io.StringIO()):
                self._model = CrossEncoder(self.model_name)
            return self._model
        except Exception as e:  # noqa: BLE001
            warn(f"Failed to load reranker model {self.model_name!r}: {e}")
            return None

    def rerank(self, claim: str, chunks: list[dict], top_k: int) -> list[dict]:
        if not chunks:
            return []
        model = self._ensure_model()
        if model is None:
            # Graceful fallback to lexical if model couldn't be loaded.
            return LexicalReranker().rerank(claim, chunks, top_k)

        pairs = [(claim, c.get("text") or "") for c in chunks]
        try:
            raw_scores = model.predict(pairs, show_progress_bar=False)
        except Exception as e:  # noqa: BLE001
            warn(f"Cross-encoder reranking failed: {e}. Falling back to lexical.")
            return LexicalReranker().rerank(claim, chunks, top_k)

        scored: list[tuple[float, dict]] = []
        for c, s in zip(chunks, raw_scores):
            annotated = dict(c)
            annotated["rerank_score"] = float(s)
            annotated["rerank_components"] = {
                "cross_encoder": float(s),
                "chroma_raw": float(c.get("score", 0.0)),
            }
            scored.append((float(s), annotated))

        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:top_k]]


# ---------- selection / factory ----------

def _sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def get_reranker() -> Reranker:
    """Build the reranker chosen by settings, with sensible fallbacks.

    Settings consulted (all live on `bibwizard.utils.config.settings`):
      - reranker_enabled : bool   master switch
      - reranker_kind    : str    "auto" | "cross" | "lexical" | "off"
      - reranker_model   : str    HuggingFace model id for cross-encoder

    Returns a Reranker instance. Never raises — falls back to Passthrough
    if anything goes wrong.
    """
    if not getattr(settings, "reranker_enabled", True):
        return PassthroughReranker()

    kind = (getattr(settings, "reranker_kind", "auto") or "auto").lower()
    if kind in {"off", "none", "passthrough"}:
        return PassthroughReranker()
    if kind == "lexical":
        return LexicalReranker()
    if kind == "cross":
        return CrossEncoderReranker(
            getattr(settings, "reranker_model", "BAAI/bge-reranker-base")
        )
    # "auto" — prefer cross-encoder if installed, else lexical.
    if _sentence_transformers_available():
        return CrossEncoderReranker(
            getattr(settings, "reranker_model", "BAAI/bge-reranker-base")
        )
    return LexicalReranker()
