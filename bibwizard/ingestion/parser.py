"""PDF parsing with PyMuPDF (fitz).

Splits a paper into:
  - body_text  : main content up to (but excluding) the bibliography
  - references : raw text of the References / Bibliography section
  - pages      : list of (page_number, text) tuples
  - raw_text   : full document text (joined pages)

Citations are then parsed out of `references` separately.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


# Patterns that mark the start of the bibliography section. We scan from the
# bottom of the document up so we catch the LAST occurrence (the real one).
_REF_HEADERS = re.compile(
    r"^\s*(references|bibliography|works cited|literature cited)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class ParsedPDF:
    path: Path
    sha256: str
    pages: list[tuple[int, str]] = field(default_factory=list)
    raw_text: str = ""
    body_text: str = ""
    references: str = ""

    @property
    def n_pages(self) -> int:
        return len(self.pages)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_at_references(full_text: str) -> tuple[str, str]:
    """Return (body, references) by finding the last References-like header."""
    matches = list(_REF_HEADERS.finditer(full_text))
    if not matches:
        return full_text, ""
    last = matches[-1]
    return full_text[: last.start()], full_text[last.end():]


def parse_pdf(path: str | Path) -> ParsedPDF:
    """Parse a PDF page-by-page and split out the bibliography."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {p}")
    if not p.is_file():
        raise ValueError(f"Not a file: {p}")

    sha = _hash_file(p)
    pages: list[tuple[int, str]] = []
    with fitz.open(p) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            pages.append((i, text))

    raw = "\n".join(t for _, t in pages)
    body, refs = _split_at_references(raw)

    return ParsedPDF(
        path=p,
        sha256=sha,
        pages=pages,
        raw_text=raw,
        body_text=body,
        references=refs,
    )


# ---------- Reference parsing ----------

# Numbered reference: "[12] Author..." or "12. Author..."
_NUMBERED_REF = re.compile(r"^\s*(?:\[\d+\]|\d+\.)\s+", re.MULTILINE)
# Year-anchored fallback: "Author A. 2019. Title..."
_YEAR_REF = re.compile(r"\b(19|20)\d{2}\b")


def split_references(refs_text: str) -> list[str]:
    """Best-effort split of a bibliography blob into individual entries."""
    if not refs_text or not refs_text.strip():
        return []

    text = refs_text.strip()

    # Strategy 1: numbered references — split on the marker, drop the leader.
    matches = list(_NUMBERED_REF.finditer(text))
    if len(matches) >= 3:
        entries: list[str] = []
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            entries.append(_normalize_whitespace(text[start:end]))
        return [e for e in entries if e]

    # Strategy 2: blank-line separation
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) >= 3:
        return [_normalize_whitespace(b) for b in blocks]

    # Strategy 3: split on every line that looks like the start of a new
    # reference (typically begins with an author surname + initial OR a year).
    lines = text.split("\n")
    entries = []
    buf: list[str] = []
    for line in lines:
        starts_new = bool(re.match(r"^[A-Z][A-Za-z'\-]+,\s*[A-Z]\.", line.strip()))
        if starts_new and buf:
            entries.append(_normalize_whitespace(" ".join(buf)))
            buf = [line]
        else:
            buf.append(line)
    if buf:
        entries.append(_normalize_whitespace(" ".join(buf)))
    entries = [e for e in entries if e and (_YEAR_REF.search(e) or len(e) > 30)]
    return entries


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()
