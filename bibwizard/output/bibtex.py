"""BibTeX formatter for bibwizard papers.

Produces deterministic citation keys and clean .bib entries that paste
directly into a natbib/biblatex bibliography. Goals:
  - Keys are stable across runs (same paper → same key) so the macro
    bibwizard prints today still works in your .tex file tomorrow.
  - Entries include every field bibwizard can reliably fill from the
    Paper model (author, title, year, journal/venue, doi, arxiv_id).
  - LaTeX-special characters in titles / author names are escaped so the
    output compiles without manual cleanup.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from bibwizard.database.models import Paper

# Characters that need escaping in LaTeX. The dollar sign, ampersand, etc.
# trip up TeX without a backslash; tildes and carets need special handling
# because backslash-tilde IS a TeX command.
_LATEX_SPECIAL = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}

# Stopwords stripped from titles when building the key — same intuition as
# the lexical reranker stopwords. We keep them tiny because over-stopping
# produces weird keys ("the" → ø → next word).
_KEY_TITLE_STOPWORDS = frozenset(
    """
    a an the and or of in on at to from by for with without
    is are was were be been
    this that these those it its as
    """.split()
)


def _slugify_token(s: str) -> str:
    """Strip accents and non-letter characters from a single name/token
    for use in a BibTeX key. 'Eyyuboğlu' → 'eyyuboglu'."""
    # NFKD splits accented chars into base + combining; encode/decode to
    # drop the combining marks.
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _first_author_surname(authors: list[str]) -> str:
    """Pull the surname of the first author for the key.

    Handles both "Lastname, F." and "F. Lastname" forms — and a few uglier
    cases ("Lastname F", "First Middle Lastname"). Falls back to the first
    token if nothing clearly works.
    """
    if not authors:
        return "unknown"
    name = authors[0].strip()
    if "," in name:
        # "Halverson, Samuel" → "Halverson"
        return _slugify_token(name.split(",", 1)[0]) or "unknown"
    tokens = name.split()
    if not tokens:
        return "unknown"
    # Heuristic: surname is usually the last multi-letter token, ignoring
    # trailing initials like "Jr." or "II".
    candidates = [t for t in tokens if len(t) >= 2 and not t.endswith(".")]
    if candidates:
        return _slugify_token(candidates[-1]) or _slugify_token(tokens[-1]) or "unknown"
    return _slugify_token(tokens[-1]) or "unknown"


def _first_significant_title_word(title: str) -> str:
    """Pull a single distinctive lowercase word from the title for the key.

    'Modal Noise in Single-Mode Fibers...' → 'modal'
    'A Cautionary Note for High Precision...' → 'cautionary' (skip 'a')
    """
    if not title:
        return ""
    for raw in title.split():
        slug = _slugify_token(raw)
        if slug and slug not in _KEY_TITLE_STOPWORDS and len(slug) >= 3:
            return slug
    # Fall back: any non-stopword, any length
    for raw in title.split():
        slug = _slugify_token(raw)
        if slug and slug not in _KEY_TITLE_STOPWORDS:
            return slug
    return ""


def bibtex_key(paper: Paper) -> str:
    """Build a deterministic BibTeX key from a Paper.

    Format: <surname><year><title-word>, e.g. 'halverson2015modal'. If the
    paper has no year, omit it. If it has no usable title word, fall back
    to just <surname><year>. Always lowercase, ASCII-safe.
    """
    authors = [a.name for a in paper.authors] if paper.authors else []
    surname = _first_author_surname(authors)
    year = str(paper.year) if paper.year else ""
    title_word = _first_significant_title_word(paper.title or "")
    key = f"{surname}{year}{title_word}"
    # If we got back a totally uninformative key ("unknown" surname only,
    # no year, no title) fall back to paper{id} so keys are at least unique.
    if not key or key in {"unknown", "unknownunknown"}:
        key = f"paper{paper.id}"
    return key


def _escape_latex(s: str) -> str:
    """Escape TeX-special chars so a title/author string compiles cleanly."""
    if not s:
        return ""
    out: list[str] = []
    for ch in s:
        out.append(_LATEX_SPECIAL.get(ch, ch))
    return "".join(out)


def _format_authors_for_bib(authors: list[str]) -> str:
    """BibTeX author field: 'Last, First and Last, First and ...'.

    We try to preserve whatever form the input is already in — if it's
    already 'Lastname, Firstname' we leave it; if it's 'Firstname Lastname'
    we flip it. This isn't perfect (middle initials, suffixes) but covers
    the common cases.
    """
    formatted: list[str] = []
    for raw in authors:
        name = raw.strip()
        if not name:
            continue
        if "," in name:
            # Already in 'Last, First' form
            formatted.append(_escape_latex(name))
        else:
            tokens = name.split()
            if len(tokens) >= 2:
                last = tokens[-1]
                rest = " ".join(tokens[:-1])
                formatted.append(_escape_latex(f"{last}, {rest}"))
            else:
                formatted.append(_escape_latex(name))
    return " and ".join(formatted)


def _detect_entry_type(venue: str | None, doi: str | None = None) -> str:
    """Heuristic: @inproceedings for conference proceedings, @article otherwise.

    First check the venue string (most reliable). If venue is missing or
    inconclusive, fall back to known proceedings DOI prefixes:
      - 10.1117/...  → SPIE proceedings
      - 10.1063/...  → AIP conference proceedings (most are, some aren't —
                       we err on the side of journal here)

    SPIE proceedings, AAS Meeting abstracts, and similar venues are
    conferences. Journal abbreviations (A&A, ApJ, MNRAS, Optics Express,
    Scientific Reports) are journals.
    """
    if venue:
        v = venue.lower()
        if any(
            marker in v
            for marker in (
                "proc.", "proceedings", "spie", "meeting", "conference",
                "workshop", "symposium",
            )
        ):
            return "inproceedings"
    # Fall back to DOI prefix — useful when venue wasn't extracted at
    # ingestion time (common for SPIE papers where the title page doesn't
    # have a "Proc. SPIE" header).
    if doi:
        d = doi.strip().lower()
        if d.startswith("10.1117/"):
            return "inproceedings"
    return "article"


def bibtex_entry(paper: Paper, *, key: str | None = None) -> str:
    """Format a Paper as a BibTeX entry.

    Includes every available field. The exact entry type (@article vs
    @inproceedings) is inferred from the venue.

    Args:
      paper: a Paper instance with hydrated authors/tags.
      key: optional override for the citation key; auto-generated if None.

    Returns:
      A multi-line string. No trailing newline.
    """
    k = key or bibtex_key(paper)
    entry_type = _detect_entry_type(paper.venue, paper.doi)
    authors_str = _format_authors_for_bib([a.name for a in paper.authors])
    title_str = _escape_latex(paper.title or "")
    venue_str = _escape_latex(paper.venue or "")

    fields: list[tuple[str, str]] = []
    if authors_str:
        fields.append(("author", f"{{{authors_str}}}"))
    if title_str:
        fields.append(("title", f"{{{title_str}}}"))
    if paper.year:
        fields.append(("year", f"{{{paper.year}}}"))
    if venue_str:
        # @article uses 'journal', @inproceedings uses 'booktitle'
        venue_field = "booktitle" if entry_type == "inproceedings" else "journal"
        fields.append((venue_field, f"{{{venue_str}}}"))
    if paper.doi:
        fields.append(("doi", f"{{{paper.doi}}}"))
    if paper.arxiv_id:
        # Standard biblatex convention
        fields.append(("eprint", f"{{{paper.arxiv_id}}}"))
        fields.append(("eprinttype", "{arxiv}"))

    # Render
    lines = [f"@{entry_type}{{{k},"]
    for name, value in fields:
        lines.append(f"  {name} = {value},")
    lines.append("}")
    return "\n".join(lines)


def citep_macro(keys: Iterable[str], *, command: str = "citep") -> str:
    r"""Build a \citep{key1, key2, ...} macro string.

    `command` lets you swap natbib variants:
      - "citep"  → parenthetical (Smith et al. 2020)        [default]
      - "citet"  → textual       Smith et al. (2020)
      - "cite"   → plain biblatex
      - "citeauthor", "citeyear", etc. — any command name
    """
    keys = [k for k in keys if k]
    if not keys:
        return ""
    return f"\\{command}{{{', '.join(keys)}}}"


def format_citation_block(
    papers: list[Paper], *, command: str = "citep",
) -> str:
    r"""Render a complete \cite + .bib block for a set of papers.

    Output layout:

      \citep{key1, key2}

      @article{key1,
        ...
      }

      @article{key2,
        ...
      }

    Ready to paste straight into a manuscript draft.
    """
    if not papers:
        return ""
    # Build keys once so the macro and entries match.
    keys = [bibtex_key(p) for p in papers]
    macro = citep_macro(keys, command=command)
    entries = [bibtex_entry(p, key=k) for p, k in zip(papers, keys)]
    return macro + "\n\n" + "\n\n".join(entries)
