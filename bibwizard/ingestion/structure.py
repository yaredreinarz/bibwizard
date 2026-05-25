"""Dedicated PDF structure scraper.

PyMuPDF's `dict` mode exposes per-span font size and bounding boxes. We use
that to identify three regions on page 1 of an academic paper:

  - title block: the lines sharing the largest font size in the upper half
  - byline block: the cluster of lines immediately below the title with a
    smaller but still prominent font (authors + affiliations)
  - abstract block: text following an "Abstract" / "ABSTRACT" header, ending
    at "Introduction", "Keywords", "1." or end of page

Compared to plain-text extraction, this gives the downstream LLM a clean,
targeted region for each field instead of a soup of merged columns. The
output is consumed by `extract_metadata` (heuristic) and `summarize_paper`
(LLM, focused prompts).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz


_ABSTRACT_RE = re.compile(r"^\s*abstract[\s\.:—–-]*", re.IGNORECASE)
_ABSTRACT_END_RE = re.compile(
    r"^\s*(keywords?|key words|introduction|1\.?\s+introduction|i\.\s*introduction|"
    r"contents|table of contents)\b",
    re.IGNORECASE,
)

# Lines that look like the arXiv banner, journal stamps, "Submitted to" / "Draft
# version" / preprint headers — they often sit at the top of page 1 and get
# mistaken for the title.
_BANNER_RE = re.compile(
    r"arxiv\s*:\s*\d{4}\.\d{4,5}"
    r"|^\s*(preprint|draft|submitted|accepted|received|published|to appear|"
    r"astro-?ph|astro-?ph\.|copyright|©|\\textcopyright|"
    r"manuscript|in press|under review)\b",
    re.IGNORECASE,
)

# Journals + proceedings frequently typeset their name at the top of page 1
# (often with trailing "manuscript no. ..." or volume/page numbers) in a
# large font that beats the actual title. Detect any line that *starts* with
# a known journal/proceedings header.
_JOURNAL_HEADER_RE = re.compile(
    r"^\s*(?:"
    r"astronomy\s*(?:&|and|amp;?)?\s*astrophysics?"
    r"|a\s*&\s*a\b"
    r"|the\s+astrophysical\s+journal(?:\s+(?:letters|supplement(?:\s+series)?))?"
    r"|astrophys(?:ical)?\.?\s+j(?:ournal)?\.?"
    r"|the\s+astronomical\s+journal"
    r"|astron(?:omical)?\.?\s+j(?:ournal)?\.?"
    r"|monthly\s+notices\s+of\s+the\s+royal\s+astronomical\s+society"
    r"|mnras\b"
    r"|publications\s+of\s+the\s+astronomical\s+society\s+of\s+the\s+pacific"
    r"|pasp\b"
    r"|optics\s+communications?\b"
    r"|optics\s+(?:letters|express)\b"
    r"|applied\s+optics\b"
    r"|j(?:ournal)?\.?\s+opt(?:ics|ical)?\.?\s+soc(?:iety)?\.?\s+(?:of\s+)?am(?:erica)?\b"
    r"|josa\s*[ab]?\b"
    r"|nature(?:\s+(?:astronomy|communications|methods|physics))?\b"
    r"|science(?:\s+advances)?\b"
    r"|physical\s+review\s+(?:letters|[a-e])\b"
    r"|phys(?:ical)?\.?\s+rev(?:iew)?\.?\s+[a-e]?\b"
    r"|proc(?:eedings|\.)\s+(?:of\s+)?(?:the\s+)?spie\b"
    r"|spie\s+proceedings\b"
    r"|icarus\b"
    r")",
    re.IGNORECASE,
)

# Affiliation indicators — if a candidate "byline" cluster is dominated by
# these, it's actually an affiliation block.
_AFFIL_KEYWORDS = (
    "university",
    "institute",
    "laboratory",
    "observatory",
    "department",
    "school of",
    "centre for",
    "center for",
    "faculty of",
    "academy",
    "agency",
    "max-planck",
    "max planck",
    "cnrs",
    "inaf",
    "esa",
    "nasa",
    "jpl",
    "harvard",
    "stanford",
    "mit ",
    "caltech",
    "consejo",
    "universidad",
    "università",
    "université",
    "instituto",
)
# US zip / state, postal codes
_POSTAL_RE = re.compile(r"\b[A-Z]{2}\s*\d{4,5}\b|\b[A-Z]{2}\s+\d{5}\b")


def _is_banner_line(text: str) -> bool:
    if _BANNER_RE.search(text):
        return True
    # Journal headers — match at line start, no length cap. Real paper titles
    # don't open with a journal name like "Astronomy & Astrophysics ...".
    if _JOURNAL_HEADER_RE.match(text):
        return True
    return False


# Patterns that PyMuPDF tends to fuse into the byline cluster: date stamps
# ("Received: 19 October 2015"), digit/letter affiliation superscripts
# (" 1,2,3" / " a,b"), and dropcap-style first letters ("r eceived").
_DATESTAMP_RE = re.compile(
    r"\b[rRaApP]?\s*"
    r"(?:r\s*eceived|received|a\s*ccepted|accepted|p\s*ublished|published|"
    r"revised|in\s+press|submitted)"
    r"\s*:?\s*"
    r"(?:\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s*\d{4}|\d{4}\s+\w+\s+\d{1,2})",
    re.IGNORECASE,
)
# Affiliation marker sequences attached to a name: "Smith 1", "Smith 1,2,3",
# "Smith a", "Smith a,b". We strip the WHOLE sequence (digits/letters and
# their internal commas) when it sits between a word and a comma/end-of-text.
# Negative lookahead `(?!\s*\d{4})` keeps us from eating a year that happens
# to follow a comma (e.g. ", 2015, ..." in a date stamp).
_AFFIL_MARK_RE = re.compile(
    r"\s+(?:\d{1,3}|[a-z])(?:\s*,\s*(?:\d{1,3}|[a-z]))*"
    r"(?=\s*(?:,(?!\s*\d{4})|;|$|\s+and\s|\s+&\s|\sand\s|\s&\s))"
)


def _strip_byline_noise(text: str) -> str:
    """Remove date stamps and digit/letter affiliation superscripts that
    PyMuPDF tends to fuse into the byline cluster.

    Three passes:
      1. Date stamps ("Received: 19 October 2015" etc.) — wiped out.
      2. Affiliation markers followed by a real separator — wiped.
      3. ORPHAN markers between two names with only whitespace around them
         (no comma) — replaced by ", " so the two authors get split, not
         fused. This is the "Pin Lü 3 Wenke Xie" case where PyMuPDF kept the
         digit but lost the surrounding visual layout.
    """
    if not text:
        return text
    text = _DATESTAMP_RE.sub(" ", text)
    text = _AFFIL_MARK_RE.sub("", text)
    # Orphan-after-name: "...Smith 3 Bob..." → "...Smith, Bob..."
    text = re.sub(
        r"(?<=[a-zà-ÿ])\s+(?:\d{1,3}|[a-z])\b(?=\s+[A-ZÀ-Ÿ])",
        ",",
        text,
    )
    # Orphan-after-comma: "..., a Bob ..." → "..., Bob ..."
    text = re.sub(
        r"(?<=[,;])\s+(?:\d{1,3}|[a-z])\b(?=\s+[A-ZÀ-Ÿ])",
        "",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[,\s]+|[,\s]+$", "", text)
    # Collapse accidental ",," that two passes can leave behind.
    text = re.sub(r",\s*,+", ",", text)
    return text


@dataclass
class Line:
    text: str
    size: float  # average font size of spans on this line
    y: float     # top-of-bbox y coordinate
    x: float     # left edge x
    page: int    # 1-indexed page number


@dataclass
class FrontMatter:
    """Structured first-page extraction. Fields may be empty if heuristics
    couldn't find them — callers should fall back gracefully."""

    title: str = ""
    title_size: float = 0.0
    title_lines: list[str] = field(default_factory=list)

    byline_text: str = ""
    byline_size: float = 0.0
    byline_lines: list[str] = field(default_factory=list)

    abstract: str = ""

    # Debug payload — list of (page, y, size, text) tuples for the first two
    # pages, useful for the `bibwizard inspect` command.
    page_lines: list[Line] = field(default_factory=list)


def _page_lines(page: fitz.Page, page_no: int) -> list[Line]:
    out: list[Line] = []
    try:
        blocks = page.get_text("dict").get("blocks", [])
    except Exception:
        return out
    for b in blocks:
        for ln in b.get("lines", []):
            spans = ln.get("spans", [])
            if not spans:
                continue
            sizes = [s.get("size", 0.0) for s in spans]
            avg = sum(sizes) / len(sizes) if sizes else 0.0
            text = " ".join((s.get("text") or "") for s in spans).strip()
            if not text:
                continue
            bbox = ln.get("bbox") or (0, 0, 0, 0)
            out.append(Line(text=text, size=avg, y=bbox[1], x=bbox[0], page=page_no))
    return out


def _norm_ws(s: str) -> str:
    return " ".join(s.split())


def _is_marker_line(text: str) -> bool:
    """Lines that mark section boundaries we don't want inside the byline."""
    low = text.strip().lower()
    # Single-word document-structure headers ("Contents", "References", etc.).
    # White papers / technical reports put these right under the title in the
    # same font as the body, which fools the byline scorer.
    if low.strip(".:") in {
        "contents",
        "table of contents",
        "references",
        "bibliography",
        "appendix",
        "acknowledgements",
        "acknowledgments",
        "summary",
        "conclusion",
        "conclusions",
        "discussion",
        "methods",
        "results",
        "background",
    }:
        return True
    return any(
        low.startswith(m)
        for m in (
            "abstract",
            "introduction",
            "keywords",
            "key words",
            "i.",
            "1 ",
            "1.",
            "received",
            "accepted",
            "submitted",
            "draft",
            "preprint",
        )
    )


def _find_title(lines: list[Line], page_height: float) -> tuple[list[Line], float]:
    """Return (title_lines, title_size). Picks the largest-font cluster in the
    top half of page 1, EXCLUDING arXiv banners / preprint stamps / section
    markers.

    Crucially, after picking a cluster, we *also* check whether its joined
    text matches a journal-header pattern — A&A in particular splits its
    masthead across separate lines ("Astronomy" / "&" / "Astrophysics") that
    individually don't match the banner regex but together do. If so, we
    drop that font size and try the next-largest cluster.
    """
    upper = [
        ln for ln in lines
        if ln.y < page_height * 0.55
        and len(ln.text) >= 1
        and not _is_banner_line(ln.text)
        and not _is_marker_line(ln.text)
    ]
    if not upper:
        return [], 0.0

    # Walk fonts from biggest to smallest, picking the first cluster whose
    # joined text isn't itself a journal banner.
    candidate_pool = list(upper)
    while candidate_pool:
        biggest = max(ln.size for ln in candidate_pool)
        at_biggest = [ln for ln in candidate_pool if abs(ln.size - biggest) < 0.5]
        if not at_biggest:
            return [], 0.0
        at_biggest.sort(key=lambda ln: ln.y)
        kept: list[Line] = [at_biggest[0]]
        for ln in at_biggest[1:]:
            if ln.y - kept[-1].y < max(kept[-1].size, 8) * 2.5:
                kept.append(ln)
            else:
                break
        # Validate: joined-line text must not be a journal banner / not too short.
        joined = _norm_ws(" ".join(ln.text for ln in kept))
        if (
            len(joined) >= 8
            and not _is_banner_line(joined)
            and not _JOURNAL_HEADER_RE.match(joined)
        ):
            return kept, biggest
        # This font cluster IS a journal banner (or junk) — drop ALL lines at
        # this size and retry with the next-biggest.
        candidate_pool = [ln for ln in candidate_pool if abs(ln.size - biggest) >= 0.5]
    return [], 0.0


# Common-word stoplist for name-likeness detection. Body text contains plenty
# of these; real author names never do.
_BODY_STOPWORDS = frozenset(
    {
        "the", "of", "in", "is", "are", "be", "we", "our", "this", "that",
        "these", "those", "with", "which", "where", "however", "such",
        "based", "on", "at", "to", "for", "from", "by", "as", "and", "or",
        "but", "if", "then", "than", "may", "can", "will", "would", "should",
        "could", "must", "have", "has", "had", "been", "being", "do", "does",
        "did", "use", "used", "using", "show", "shows", "showed", "find",
        "finds", "found", "see", "into", "out", "over", "under", "an", "a",
        "also", "very", "still", "yet", "between", "among", "across",
        "while", "when", "what", "how", "many", "more", "most", "less",
        "least", "some", "any", "all", "no", "not", "only", "first", "next",
        "previous", "results", "result", "model", "models", "system",
        "systems", "data", "method", "methods", "paper", "papers", "present",
        "presented", "study", "studies", "studied", "report", "reports",
        "due", "fields", "areas", "evolution", "intensity", "telescope",
        "telescopes", "ground", "limited",
    }
)


def _is_namelike(piece: str) -> bool:
    """Return True if `piece` (one comma-separated chunk) looks like a single
    author name. A name is:
      - 1-5 word tokens
      - all word tokens are Title-case OR initials ('Y.' / 'Y.-M.')
      - no body-text stopwords ('the', 'we', 'based', 'is', ...)
      - reasonable length (3-60 chars)
    """
    s = piece.strip()
    if not (3 <= len(s) <= 60):
        return False
    # Allow trailing affiliation superscript like 'Last a' / 'Last b' for now
    s = re.sub(r"(\s+[a-z](?:,[a-z])*)+\s*$", "", s).strip()
    tokens = s.split()
    if not (1 <= len(tokens) <= 6):
        return False
    has_alpha = False
    for tok in tokens:
        # Strip punctuation for the check
        bare = re.sub(r"[^\w-]", "", tok)
        if not bare:
            return False
        low = bare.lower()
        if low in _BODY_STOPWORDS:
            return False
        # Token must be either:
        #  * an INITIAL — single capital, or capitals separated by `.` or `-`
        #    (e.g. "Y", "Y.", "J.K.", "J.-M."). Acronyms like "VLT" / "PIAAN"
        #    must NOT match — that's the bug that let abstract text score
        #    higher than the actual byline.
        #  * a TITLE-CASE word ("Reinarz", "McGill", "de'Luca").
        is_initial = bool(
            re.fullmatch(
                r"[A-ZÀ-Ÿ]\.?(?:[.\-]+[A-ZÀ-Ÿ]\.?)*",
                tok.replace(",", ""),
            )
        )
        is_titlecase = bool(re.match(r"^[A-ZÀ-Ÿ][a-zà-ÿ'\-]+$", bare)) or bool(
            re.match(r"^[A-ZÀ-Ÿ][a-zà-ÿ'\-]*[A-ZÀ-Ÿ][a-zà-ÿ'\-]+$", bare)
        )
        if not (is_initial or is_titlecase):
            return False
        if any(ch.isalpha() for ch in bare):
            has_alpha = True
    return has_alpha


def _name_ratio(text: str) -> tuple[float, int]:
    """Of the comma- and semicolon-separated pieces in `text`, what fraction
    look like author names? Returns (ratio, n_pieces)."""
    if not text:
        return 0.0, 0
    # Also split on ' and ' and ' & '
    parts = re.split(r"\s+and\s+|\s+&\s+|[,;]", text)
    pieces = [p.strip() for p in parts if p.strip()]
    if not pieces:
        return 0.0, 0
    n_name = sum(1 for p in pieces if _is_namelike(p))
    return n_name / len(pieces), len(pieces)


def _byline_score(text: str) -> float:
    """Heuristic 'how byline-like is this text?' score.

    The single strongest signal is the **name-likeness ratio**: real bylines
    have most/all comma-separated pieces parse as proper names; body text
    does not. We multiply that ratio in heavily so a sentence with a few
    commas can't beat a real author list.
    """
    if not text:
        return -100.0
    # Normalize away PyMuPDF noise (date stamps, digit/letter superscripts)
    # before scoring — otherwise the byline ratio looks much worse than it is.
    text = _strip_byline_noise(text)
    if not text:
        return -100.0
    low = text.lower()
    score = 0.0

    # Name-likeness — the dominant signal
    ratio, n_pieces = _name_ratio(text)
    if n_pieces >= 2:
        # A real byline averages ~0.9 name-ratio; body text averages ~0.2.
        # Be generous in the middle (mixed byline+affiliation block) so a
        # cluster that has at least some real names still wins over a pure
        # body-text cluster with 0 name-likeness.
        score += (ratio - 0.3) * 10.0
    elif n_pieces == 1:
        score += 5.0 if ratio >= 0.5 else -5.0

    # Initial-style name tokens add a small extra push (covers single-name
    # cases like 'M. Jones' that already pass the ratio check).
    initials = re.findall(r"\b[A-ZÄÖÜÅÆ]\.\s*(?:-?[A-ZÄÖÜÅÆ]\.\s*)?[A-ZÄÖÜÅÆ][a-zà-ÿ]+", text)
    score += min(len(initials), 6) * 0.5
    surname_init = re.findall(r"\b[A-ZÄÖÜÅÆ][a-zà-ÿ]+,\s*[A-ZÄÖÜÅÆ]\.", text)
    score += min(len(surname_init), 6) * 0.5

    # 'and' between authors
    if re.search(r"\band\b|\s&\s", low):
        score += 0.5

    # Affiliation keywords → strongly negative
    for kw in _AFFIL_KEYWORDS:
        if kw in low:
            score -= 3.0

    # Postal / ZIP codes → likely an address
    if _POSTAL_RE.search(text):
        score -= 3.0

    # Emails → present in some bylines but more often in contact block.
    if "@" in text:
        score -= 1.0

    # Affiliation cross-refs like "[1] Dept. of..." or "(2)"
    if re.search(r"\[\d", text):
        score -= 2.0

    return score


def _byline_clusters(
    lines: list[Line], title_lines: list[Line], title_size: float
) -> list[list[Line]]:
    """Yield all reasonable below-title font-size clusters as candidates."""
    if not title_lines or title_size <= 0:
        return []
    title_bottom = max(ln.y for ln in title_lines)
    below = [
        ln for ln in lines
        if ln.y > title_bottom
        and ln.y < title_bottom + 400
        and ln.size < title_size * 0.97
        and not _is_marker_line(ln.text)
        and not _is_banner_line(ln.text)
    ]
    below.sort(key=lambda ln: ln.y)
    clusters: list[list[Line]] = []
    cur: list[Line] = []
    for ln in below:
        if not cur:
            cur = [ln]
            continue
        # Compare to the FIRST line of the current cluster, not the previous
        # line — otherwise font-size drift accumulates and a byline (11.3pt)
        # silently merges with affiliations (11.7pt) merges with the next
        # block (11.96pt), each within 0.4pt of the prior.
        first = cur[0]
        prev = cur[-1]
        if (
            abs(ln.size - first.size) < 0.35
            and ln.y - prev.y < max(prev.size, 8) * 2.5
        ):
            cur.append(ln)
        else:
            clusters.append(cur)
            cur = [ln]
    if cur:
        clusters.append(cur)
    return clusters


# Affiliation lines often START with a lowercase letter (superscript marker
# matching the one attached to each author in the byline, like "a Observatoire"
# / "b Space Sciences Institute"). Catches multilingual affiliations the
# keyword list misses (Observatoire, Universidad, Institut, Università, ...).
_AFFIL_LEADING_LETTER_RE = re.compile(r"^\s*[a-z]\s+[A-ZÀ-Ÿ]")


def _trim_trailing_affiliations(cluster: list[Line]) -> list[Line]:
    """Within a cluster, drop trailing lines that look like affiliations.

    Three signals trigger the cut: an English affiliation keyword, a postal
    code, or a single-letter affiliation-marker prefix ("a Observatoire").
    """
    kept: list[Line] = []
    for ln in cluster:
        text_low = ln.text.lower()
        if (
            any(kw in text_low for kw in _AFFIL_KEYWORDS)
            or _POSTAL_RE.search(ln.text)
            or _AFFIL_LEADING_LETTER_RE.match(ln.text)
        ):
            break
        kept.append(ln)
    return kept if kept else cluster


def _find_byline(
    lines: list[Line], title_lines: list[Line], title_size: float
) -> list[Line]:
    """Pick the byline cluster: most author-shaped chunk below the title.

    We look at every contiguous font-size cluster under the title and score it
    by author-likeness. Highest non-negative score wins. Falls back to the
    first cluster if everything scores negative (rare).
    """
    clusters = _byline_clusters(lines, title_lines, title_size)
    if not clusters:
        return []
    scored: list[tuple[float, list[Line]]] = []
    for cluster in clusters:
        text = " ".join(ln.text for ln in cluster)
        scored.append((_byline_score(text), cluster))
    scored.sort(key=lambda t: -t[0])
    best_score, best_cluster = scored[0]
    if best_score <= -2:
        # Strongly negative → clearly an affiliation/body block; refuse it.
        return []
    return _trim_trailing_affiliations(best_cluster)


def _find_abstract(lines: list[Line]) -> str:
    """Scan in reading order for an `Abstract` header and capture text up to
    `Introduction` / `Keywords` / a numbered section header."""
    # Sort lines by (page, y) — reading order.
    ordered = sorted(lines, key=lambda ln: (ln.page, ln.y))
    in_abstract = False
    collected: list[str] = []
    for ln in ordered:
        if not in_abstract:
            m = _ABSTRACT_RE.match(ln.text)
            if m:
                in_abstract = True
                rest = ln.text[m.end():].strip()
                if rest:
                    collected.append(rest)
            continue
        if _ABSTRACT_END_RE.match(ln.text):
            break
        # Stop on a very-large-font line (likely the next section title).
        collected.append(ln.text)
        if len(" ".join(collected)) > 4000:
            break
    text = _norm_ws(" ".join(collected))
    # Sanity guard — too short means no real abstract, drop it.
    if len(text) < 60:
        return ""
    return text


def extract_front_matter(pdf_path: Path) -> FrontMatter:
    """Open `pdf_path` and run the structure heuristics on pages 1-2."""
    fm = FrontMatter()
    try:
        with fitz.open(pdf_path) as doc:
            if doc.page_count == 0:
                return fm
            page1 = doc[0]
            page_height = page1.rect.height
            page1_lines = _page_lines(page1, 1)
            page2_lines: list[Line] = []
            if doc.page_count >= 2:
                page2_lines = _page_lines(doc[1], 2)
    except Exception:
        return fm

    fm.page_lines = page1_lines + page2_lines

    title_lines, title_size = _find_title(page1_lines, page_height)
    if title_lines:
        fm.title = _norm_ws(" ".join(ln.text for ln in title_lines))
        fm.title_size = title_size
        fm.title_lines = [ln.text for ln in title_lines]

    byline_lines = _find_byline(page1_lines, title_lines, title_size)
    if byline_lines:
        # Join byline lines with a comma between each pair so a multi-line
        # byline that wraps mid-author-list (e.g. "...Delorme b" newline
        # "Jason J. Wang b, ...") doesn't fuse two authors into one. If the
        # previous line already ends in "," or ";" we don't add another.
        # Join byline lines with a plain space — NOT a forced comma. Authors
        # whose surname wraps across two lines (e.g. "M. Shin\nMartinez") must
        # remain a single name; a forced comma would split them. Orphan
        # affiliation markers that previously separated two names get
        # restored as commas by _strip_byline_noise() below.
        fm.byline_text = _strip_byline_noise(_norm_ws(" ".join(ln.text for ln in byline_lines)))
        fm.byline_size = sum(ln.size for ln in byline_lines) / len(byline_lines)
        fm.byline_lines = [ln.text for ln in byline_lines]

    fm.abstract = _find_abstract(page1_lines + page2_lines)
    return fm


# ---------- Author parsing from a byline string ----------

# Affiliation markers we want to strip: superscripts, parenthesised numbers,
# email addresses, ORCID URLs, asterisks, daggers.
_AFFIL_NOISE_RE = re.compile(
    r"\b\S+@\S+\.\S+\b"
    r"|https?://\S+"
    r"|\([^()]*\)"
    r"|\[[^\[\]]*\]"
    # Superscript & footnote markers. Includes Unicode asterisk-operator (∗
    # U+2217), bullet operator (∙), reference mark (※), and the more common
    # dagger/double-dagger glyphs LaTeX uses for corresponding-author flags.
    r"|[¹²³⁴⁵⁶⁷⁸⁹⁰\*†‡§¶★∗∙※♯⋆]+"
    r"|\b\d{1,2}\b"
)
_AND_SPLIT_RE = re.compile(r"\s+(?:and|&)\s+", re.IGNORECASE)


# Trailing single-letter affiliation superscript: " Jovanovic a" / " Jovanovic a,b"
_TRAILING_SUPER_RE = re.compile(r"\s+[a-z](?:,\s*[a-z])*\s*$")
# Leading single-letter affiliation superscript: ", a Daniel" → "Daniel"
_LEADING_SUPER_RE = re.compile(r"^\s*[a-z](?:,\s*[a-z])*\s+(?=[A-ZÀ-Ÿ])")


def _strip_affiliation_superscripts(name: str) -> str:
    s = name.strip()
    # Several passes — superscript markers can stack ("a,b" / "1,2")
    for _ in range(3):
        new = _TRAILING_SUPER_RE.sub("", s)
        new = _LEADING_SUPER_RE.sub("", new)
        if new == s:
            break
        s = new
    return s.strip(" ,;")


def parse_authors_from_byline(byline: str) -> list[str]:
    """Heuristic byline → author-list parser.

    Strips superscripts (Unicode + ASCII single-letter), emails, parenthesised
    affiliations, then splits on commas / semicolons / 'and'. Filters obvious
    non-name tokens.
    """
    if not byline:
        return []

    # Strip date stamps + digit/letter superscripts first.
    text = _strip_byline_noise(byline)
    text = _AFFIL_NOISE_RE.sub(" ", text)
    text = _norm_ws(text)

    # Replace " and " / " & " with comma so a single split handles both.
    text = _AND_SPLIT_RE.sub(", ", text)

    raw = re.split(r"\s*[,;]\s*", text)
    out: list[str] = []
    seen: set[str] = set()
    for tok in raw:
        tok = tok.strip().strip(",").strip()
        if not tok:
            continue
        # Strip ASCII single-letter superscript affiliation markers:
        #   'Nemanja Jovanovic a' -> 'Nemanja Jovanovic'
        #   'a Daniel Echeverri'  -> 'Daniel Echeverri'
        tok = _strip_affiliation_superscripts(tok)
        if not tok:
            continue
        # Drop affiliation-only fragments
        low = tok.lower()
        if any(
            kw in low
            for kw in (
                "university",
                "institute",
                "department",
                "laboratory",
                "observatory",
                "school of",
                "centre for",
                "center for",
                "faculty of",
                "academy",
                "agency",
                "max-planck",
                "cnrs",
                "inaf",
                "esa",
                "nasa",
            )
        ):
            continue
        if not re.search(r"[A-Za-zÀ-ÿ]", tok):
            continue
        if len(tok) < 3 or len(tok) > 80:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out
