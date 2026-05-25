"""Extract paper metadata (title, authors, DOI, year, arXiv id) from a parsed PDF.

We use a few simple heuristics in priority order:
  1. PDF embedded metadata (Title / Author).
  2. DOI / arXiv id regex on the first 2 pages.
  3. Year regex on the first page.
  4. Title fallback: largest-font line on page 1.

The `fetch_arxiv_metadata()` helper uses the public arXiv API for `--arxiv <id>`.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import fitz
import httpx

from bibwizard.utils.config import settings

from .parser import ParsedPDF


_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s,;\"<>(){}\[\]]+", re.IGNORECASE)
_ARXIV_RE = re.compile(
    r"\b(?:arXiv:)?(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+/\d{7}(?:v\d+)?)\b",
    re.IGNORECASE,
)
# Accept any plausible publication year (1900–2099). The earlier 1950-floor
# was too aggressive — historically important refs (Lyot 1933, Maxwell 1873,
# etc.) silently fell through and got mis-classified as untitled junk.
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

# Sanity-checks for arXiv IDs. The new (post-April-2007) format encodes year
# and month in the leading 4 digits as YYMM, where MM must be 01..12 and YY
# is in arXiv's actual lifetime. Without this check, bibliography text like
# "MNRAS, 2966, 2012" (PyMuPDF can flatten table cells with periods) gets
# misread as the arXiv id "2966.2012".
def is_valid_arxiv_id(s: str) -> bool:
    if not s:
        return False
    s = s.strip().replace("arXiv:", "").replace("arxiv:", "")
    # Modern format: YYMM.NNNN[N][vN]
    m = re.fullmatch(r"(\d{2})(\d{2})\.(\d{4,5})(v\d+)?", s)
    if m:
        yy, mm = int(m.group(1)), int(m.group(2))
        if not (1 <= mm <= 12):
            return False
        # arXiv switched to this format in April 2007. Allow some headroom
        # past the current year so future submissions still parse.
        import datetime

        current_yy = datetime.datetime.now().year - 2000
        if yy < 7 or yy > current_yy + 1:
            return False
        if yy == 7 and mm < 4:
            return False
        return True
    # Legacy format: subject-class/YYMMNNN[vN]
    return bool(re.fullmatch(r"[a-z\-]+/\d{7}(v\d+)?", s))


def find_valid_arxiv_id(text: str) -> str | None:
    """Return the first regex-matched arXiv id in `text` that passes validation."""
    for m in _ARXIV_RE.finditer(text or ""):
        cand = m.group(1)
        if is_valid_arxiv_id(cand):
            return cand
    return None


@dataclass
class PaperMetadata:
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    abstract: str | None = None
    venue: str | None = None


# ---------- helpers ----------

def _split_authors(raw: str) -> list[str]:
    if not raw:
        return []
    # Common separators: ',', ';', ' and '
    parts = re.split(r",| and |;|/", raw)
    out = []
    for p in parts:
        p = p.strip().strip(".")
        # Filter junk like emails and affiliations
        if not p or "@" in p or len(p) < 3:
            continue
        if any(ch.isdigit() for ch in p):
            continue
        out.append(p)
    return out


def _largest_font_line(pdf_path: Path) -> str:
    """Heuristic: pick the line with the largest average font size on page 1."""
    try:
        with fitz.open(pdf_path) as doc:
            if doc.page_count == 0:
                return ""
            page = doc[0]
            blocks = page.get_text("dict")["blocks"]
    except Exception:
        return ""

    best_size = 0.0
    best_text = ""
    for block in blocks:
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            avg = sum(s["size"] for s in spans) / len(spans)
            text = " ".join(s["text"] for s in spans).strip()
            if not text or len(text) < 4:
                continue
            if avg > best_size:
                best_size = avg
                best_text = text
    return best_text


def extract_metadata(parsed: ParsedPDF) -> PaperMetadata:
    """Build a PaperMetadata from a ParsedPDF using heuristics + structure scraper."""
    from .structure import extract_front_matter, parse_authors_from_byline

    meta = PaperMetadata()

    # 0. Structured scrape of page-1 font / layout — best source of truth for
    #    title + authors + abstract on most academic PDFs.
    fm = extract_front_matter(parsed.path)

    # 1. PDF embedded metadata (sometimes correct)
    try:
        with fitz.open(parsed.path) as doc:
            info = doc.metadata or {}
    except Exception:
        info = {}

    if info.get("title"):
        meta.title = info["title"].strip()
    if info.get("author"):
        meta.authors = _split_authors(info["author"])

    head = "\n".join(t for i, t in parsed.pages[:2])

    # 2. DOI
    m = _DOI_RE.search(head)
    if m:
        meta.doi = m.group(0).rstrip(".,;)]}>\\\"'")

    # 3. arXiv id (validated — month must be 01..12, year ≥ 2007)
    aid = find_valid_arxiv_id(head)
    if aid:
        meta.arxiv_id = aid

    # 4. Year — pick the most plausible 4-digit year on page 1.
    # Bound to a sensible window: published papers can be from ~1900 up to
    # next calendar year (preprints often dated for the year they'll appear).
    # Without this bound, figure labels like "Fig. 2041" or table cell values
    # win the `max()` and we end up with absurd dates.
    import datetime

    upper_bound = datetime.datetime.now().year + 1
    page1 = parsed.pages[0][1] if parsed.pages else ""
    candidate_years = [
        int(y) for y in _YEAR_RE.findall(page1)
        if 1900 <= int(y) <= upper_bound
    ]
    if candidate_years:
        # Prefer the most recent plausible year (publication > old references).
        meta.year = max(candidate_years)

    # 5. Title — prefer structured scrape; fall back to PDF metadata; then
    #    the old largest-font heuristic.
    if fm.title and len(fm.title) >= 4:
        meta.title = fm.title
    if not meta.title or meta.title.lower() in {"untitled", "title"}:
        meta.title = _largest_font_line(parsed.path) or "(untitled)"

    # 6. Authors — parse the byline block. If embedded PDF metadata gave us
    #    something different, keep that only when the byline scrape failed.
    byline_authors = parse_authors_from_byline(fm.byline_text)
    if byline_authors:
        meta.authors = byline_authors

    # 7. Abstract — prefer structured scrape; fall back to body-text regex.
    if fm.abstract and 60 < len(fm.abstract) < 4000:
        meta.abstract = fm.abstract
    else:
        abstract_match = re.search(
            r"abstract[\s\.:]*\n?(.+?)\n\s*(?:1\s+introduction|introduction|keywords|i\.\s+introduction)",
            head,
            re.IGNORECASE | re.DOTALL,
        )
        if abstract_match:
            abs_text = re.sub(r"\s+", " ", abstract_match.group(1)).strip()
            if 60 < len(abs_text) < 4000:
                meta.abstract = abs_text

    return meta


# ---------- arXiv fetcher ----------

_ARXIV_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivRateLimited(RuntimeError):
    """Raised when arXiv replies with HTTP 429."""


def fetch_arxiv_metadata(arxiv_id: str) -> tuple[PaperMetadata, str]:
    """Query arXiv API and return (metadata, pdf_url)."""
    aid = arxiv_id.strip().replace("arXiv:", "").replace("arxiv:", "")
    url = settings.arxiv_api
    params = {"id_list": aid, "max_results": 1}
    headers = {"User-Agent": settings.user_agent}
    with httpx.Client(timeout=30.0, headers=headers, follow_redirects=True) as client:
        resp = client.get(url, params=params)
        if resp.status_code == 429:
            raise ArxivRateLimited(
                "arXiv returned 429 — rate-limited. Wait a few minutes before retrying."
            )
        resp.raise_for_status()
        body = resp.text

    root = ET.fromstring(body)
    entry = root.find("a:entry", _ARXIV_NS)
    if entry is None:
        raise LookupError(f"No arXiv entry for id {aid}")

    title_el = entry.find("a:title", _ARXIV_NS)
    summary_el = entry.find("a:summary", _ARXIV_NS)
    pub_el = entry.find("a:published", _ARXIV_NS)
    authors = [
        (a.findtext("a:name", default="", namespaces=_ARXIV_NS) or "").strip()
        for a in entry.findall("a:author", _ARXIV_NS)
    ]
    authors = [a for a in authors if a]

    title = (title_el.text or "").strip() if title_el is not None else ""
    summary = (summary_el.text or "").strip() if summary_el is not None else ""
    year: int | None = None
    if pub_el is not None and pub_el.text:
        try:
            year = int(pub_el.text[:4])
        except ValueError:
            year = None

    pdf_url = ""
    for link in entry.findall("a:link", _ARXIV_NS):
        if link.get("title") == "pdf" or link.get("type") == "application/pdf":
            pdf_url = link.get("href", "")
            break
    if not pdf_url:
        # Construct it manually
        pdf_url = f"https://arxiv.org/pdf/{aid}.pdf"

    md = PaperMetadata(
        title=title,
        authors=authors,
        year=year,
        arxiv_id=aid,
        abstract=re.sub(r"\s+", " ", summary).strip() or None,
    )
    return md, pdf_url


def download_pdf(url: str, target: Path) -> Path:
    """Stream-download a PDF to `target`. Verifies content-type looks like a PDF."""
    target.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": settings.user_agent, "Accept": "application/pdf,*/*"}
    with httpx.stream(
        "GET", url, follow_redirects=True, headers=headers, timeout=120.0
    ) as resp:
        resp.raise_for_status()
        ctype = (resp.headers.get("content-type") or "").lower()
        final_url = str(resp.url).lower()
        if "pdf" not in ctype and not final_url.endswith(".pdf"):
            raise ValueError(
                f"Expected a PDF but got content-type={ctype!r} from {resp.url}"
            )
        with target.open("wb") as f:
            for chunk in resp.iter_bytes(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
    return target


# ---------- URL helpers ----------

# arXiv URL → id. Handles both `abs/` and `pdf/` paths, modern + legacy formats.
_ARXIV_URL_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf)/(?P<id>(?:[a-z\-]+/\d{7}|\d{4}\.\d{4,5}))(?:v\d+)?(?:\.pdf)?",
    re.IGNORECASE,
)
# NASA ADS bibcode in URL.
_ADS_URL_RE = re.compile(
    r"adsabs\.harvard\.edu/abs/(?P<bibcode>[^/?#]+)", re.IGNORECASE
)


def is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def extract_arxiv_id_from_url(url: str) -> str | None:
    m = _ARXIV_URL_RE.search(url)
    return m.group("id") if m else None


def extract_ads_bibcode_from_url(url: str) -> str | None:
    m = _ADS_URL_RE.search(url)
    return m.group("bibcode") if m else None


def resolve_ads_to_pdf_url(bibcode: str) -> str:
    """ADS exposes a stable redirect at `/abs/<bibcode>/EPRINT_PDF` that lands
    on the arXiv preprint when one exists. We follow it and return the final
    URL — usually an `arxiv.org/pdf/...` link.
    """
    headers = {"User-Agent": settings.user_agent}
    resolver = f"https://ui.adsabs.harvard.edu/abs/{bibcode}/EPRINT_PDF"
    with httpx.Client(timeout=30.0, headers=headers, follow_redirects=True) as client:
        resp = client.get(resolver)
        resp.raise_for_status()
        return str(resp.url)


def search_arxiv_by_doi(doi: str) -> str | None:
    """Try to find an arXiv id for a DOI via the arXiv API search."""
    if not doi:
        return None
    headers = {"User-Agent": settings.user_agent}
    params = {"search_query": f"doi:{doi}", "max_results": 1}
    try:
        with httpx.Client(timeout=20.0, headers=headers, follow_redirects=True) as client:
            resp = client.get(settings.arxiv_api, params=params)
            resp.raise_for_status()
            body = resp.text
    except httpx.HTTPError:
        return None
    root = ET.fromstring(body)
    entry = root.find("a:entry", _ARXIV_NS)
    if entry is None:
        return None
    id_el = entry.find("a:id", _ARXIV_NS)
    if id_el is None or not id_el.text:
        return None
    m = _ARXIV_URL_RE.search(id_el.text)
    return m.group("id") if m else None


@dataclass
class ArxivCandidate:
    arxiv_id: str
    title: str
    authors: list[str]
    year: int | None
    doi: str | None
    summary: str | None
    pdf_url: str


def search_arxiv(
    *,
    title: str | None = None,
    author: str | None = None,
    year: int | None = None,
    max_results: int = 5,
) -> list[ArxivCandidate]:
    """Multi-criteria arXiv search. Combines ti:, au:, submittedDate: with AND."""
    parts: list[str] = []
    if title:
        # Trim to keep the URL sane; arXiv tolerates partial titles fine.
        parts.append(f'ti:"{title[:200]}"')
    if author:
        parts.append(f'au:"{author[:120]}"')
    if year:
        parts.append(f"submittedDate:[{year}01010000 TO {year}12312359]")
    if not parts:
        return []
    query = " AND ".join(parts)

    headers = {"User-Agent": settings.user_agent}
    params = {"search_query": query, "max_results": max(1, min(max_results, 25))}
    try:
        with httpx.Client(
            timeout=30.0, headers=headers, follow_redirects=True
        ) as client:
            resp = client.get(settings.arxiv_api, params=params)
            resp.raise_for_status()
            body = resp.text
    except httpx.HTTPError:
        return []

    root = ET.fromstring(body)
    out: list[ArxivCandidate] = []
    for entry in root.findall("a:entry", _ARXIV_NS):
        id_text = entry.findtext("a:id", default="", namespaces=_ARXIV_NS) or ""
        m = _ARXIV_URL_RE.search(id_text)
        if not m:
            continue
        aid = m.group("id")
        title_el = entry.findtext("a:title", default="", namespaces=_ARXIV_NS) or ""
        summary_el = entry.findtext("a:summary", default="", namespaces=_ARXIV_NS) or ""
        pub_el = entry.findtext("a:published", default="", namespaces=_ARXIV_NS) or ""
        authors = [
            (a.findtext("a:name", default="", namespaces=_ARXIV_NS) or "").strip()
            for a in entry.findall("a:author", _ARXIV_NS)
        ]
        authors = [a for a in authors if a]
        # arxiv embeds DOI in a child element when known
        doi = None
        for el in entry.findall("{http://arxiv.org/schemas/atom}doi"):
            if el.text and el.text.strip():
                doi = el.text.strip()
                break
        yr: int | None = None
        if pub_el:
            try:
                yr = int(pub_el[:4])
            except ValueError:
                yr = None
        pdf_url = ""
        for link in entry.findall("a:link", _ARXIV_NS):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href", "")
                break
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{aid}.pdf"
        out.append(
            ArxivCandidate(
                arxiv_id=aid,
                title=re.sub(r"\s+", " ", title_el.strip()),
                authors=authors,
                year=yr,
                doi=doi,
                summary=re.sub(r"\s+", " ", summary_el.strip()) or None,
                pdf_url=pdf_url,
            )
        )
    return out


def title_similarity(a: str, b: str) -> float:
    """Cheap normalized-token Jaccard similarity for two titles."""
    if not a or not b:
        return 0.0

    def _toks(s: str) -> set[str]:
        s = re.sub(r"[^a-z0-9]+", " ", s.lower())
        return {t for t in s.split() if len(t) > 2}

    ta, tb = _toks(a), _toks(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def search_arxiv_by_title(title: str) -> str | None:
    """Last-resort title search; only trust an exact-ish match (80-char prefix)."""
    if not title or len(title) < 10:
        return None
    headers = {"User-Agent": settings.user_agent}
    params = {"search_query": f'ti:"{title[:200]}"', "max_results": 1}
    try:
        with httpx.Client(timeout=20.0, headers=headers, follow_redirects=True) as client:
            resp = client.get(settings.arxiv_api, params=params)
            resp.raise_for_status()
            body = resp.text
    except httpx.HTTPError:
        return None
    root = ET.fromstring(body)
    entry = root.find("a:entry", _ARXIV_NS)
    if entry is None:
        return None
    cand_title_el = entry.find("a:title", _ARXIV_NS)
    if cand_title_el is None or not cand_title_el.text:
        return None
    cand_title = re.sub(r"\s+", " ", cand_title_el.text.strip())
    target_title = re.sub(r"\s+", " ", title.strip())
    if cand_title.lower()[:80] != target_title.lower()[:80]:
        return None
    id_el = entry.find("a:id", _ARXIV_NS)
    if id_el is None or not id_el.text:
        return None
    m = _ARXIV_URL_RE.search(id_el.text)
    return m.group("id") if m else None


# ---------- Crossref + Unpaywall ----------

def _crossref_user_agent() -> str:
    ua = settings.user_agent
    if settings.unpaywall_email:
        # Polite-pool: Crossref ranks UAs with a mailto higher and gives better SLA.
        ua = f"{ua} (mailto:{settings.unpaywall_email})"
    return ua


def _norm_journal(s: str) -> str:
    """Normalize a journal abbreviation to a comparable form."""
    s = (s or "").lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^\w\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_page(s: str) -> str:
    """Page strings often differ only in leading zeros / `e`-prefix /
    page-range. Normalize for a forgiving compare."""
    s = (s or "").lower()
    s = re.sub(r"^[a-z]+", "", s)              # drop leading letter (A155 -> 155)
    s = s.split("-", 1)[0].split("–", 1)[0]  # first page of a range
    s = s.lstrip("0") or s
    return s


def crossref_lookup_by_locator(
    *,
    author_surname: str,
    year: int,
    journal: str,
    volume: str,
    page: str,
    timeout: float = 20.0,
) -> str | None:
    """Look up a DOI by (first-author, year, journal, volume, page).

    Crossref's `/works` endpoint takes free-text queries; we filter to the
    publication year, ask for up to 20 candidates, then verify each by checking
    `volume` and `page` match what we expect.
    """
    if not (author_surname and year and journal and volume):
        return None
    headers = {"User-Agent": _crossref_user_agent(), "Accept": "application/json"}
    params = {
        "query.author": author_surname,
        "query.container-title": journal,
        "filter": f"from-pub-date:{year},until-pub-date:{year}",
        "rows": 20,
        "select": "DOI,volume,page,container-title,issued,author,title",
    }
    try:
        with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
            resp = client.get(settings.crossref_api, params=params)
            if resp.status_code != 200:
                return None
            data = resp.json()
    except Exception:  # noqa: BLE001
        return None

    items = (data.get("message") or {}).get("items") or []
    want_journal = _norm_journal(journal)
    want_page = _norm_page(page)
    want_vol = str(volume).strip()

    # Crossref has already pre-filtered by author + journal + year.
    # An exact (volume, page) match on top of that is high-confidence; we
    # use the normalized journal string only as a soft tie-breaker when
    # several items match.
    exact: list[dict] = []
    for it in items:
        if str(it.get("volume", "")).strip() != want_vol:
            continue
        cand_page = it.get("page") or ""
        if want_page and _norm_page(cand_page) != want_page:
            continue
        exact.append(it)

    if not exact:
        return None
    if len(exact) == 1:
        doi = (exact[0].get("DOI") or "").strip()
        return doi.lower() if doi else None

    # Multiple exact (vol, page) hits — pick the one whose container-title
    # overlaps best with the requested journal.
    def _journal_overlap(it: dict) -> int:
        cts = it.get("container-title") or []
        best = 0
        for ct in cts:
            nct = _norm_journal(ct)
            common = set(want_journal.split()) & set(nct.split())
            best = max(best, len(common))
        return best

    exact.sort(key=_journal_overlap, reverse=True)
    doi = (exact[0].get("DOI") or "").strip()
    return doi.lower() if doi else None


def crossref_lookup_by_title(
    *,
    title: str,
    author_surname: str | None = None,
    year: int | None = None,
    timeout: float = 20.0,
    min_score: float = 0.7,
) -> str | None:
    """Look up a DOI by title (+ optional author/year). Verifies match before
    returning to avoid false positives — requires Jaccard ≥ `min_score`
    between Crossref candidate title and the queried title."""
    if not title or len(title) < 10:
        return None
    headers = {"User-Agent": _crossref_user_agent(), "Accept": "application/json"}
    params: dict = {
        "query.bibliographic": title[:300],
        "rows": 5,
        "select": "DOI,title,author,issued",
    }
    if author_surname:
        params["query.author"] = author_surname
    if year:
        params["filter"] = f"from-pub-date:{year},until-pub-date:{year}"
    try:
        with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
            resp = client.get(settings.crossref_api, params=params)
            if resp.status_code != 200:
                return None
            data = resp.json()
    except Exception:  # noqa: BLE001
        return None
    items = (data.get("message") or {}).get("items") or []
    for it in items:
        cand_titles = it.get("title") or []
        if not cand_titles:
            continue
        cand = cand_titles[0]
        if title_similarity(title, cand) < min_score:
            continue
        doi = (it.get("DOI") or "").strip()
        if doi:
            return doi.lower()
    return None


def unpaywall_pdf_url(doi: str, timeout: float = 20.0) -> str | None:
    """Return the best open-access PDF URL for a DOI, via Unpaywall.

    Returns None if no OA copy is available or UNPAYWALL_EMAIL is unset.
    """
    if not doi or not settings.unpaywall_email:
        return None
    headers = {"User-Agent": settings.user_agent}
    url = f"{settings.unpaywall_api.rstrip('/')}/{doi}"
    params = {"email": settings.unpaywall_email}
    try:
        with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
            resp = client.get(url, params=params)
            if resp.status_code != 200:
                return None
            data = resp.json()
    except Exception:  # noqa: BLE001
        return None
    loc = data.get("best_oa_location") or {}
    return (loc.get("url_for_pdf") or loc.get("url") or "").strip() or None


# ---------- Reference-line metadata extraction ----------

def parse_reference_line(line: str) -> dict:
    """Extract DOI / arXiv id / year / title / journal locator from a ref line.

    Output keys (when present):
      doi, arxiv_id, year, title, journal, volume, page, raw_text

    For ADS-style entries (`Authors. YEAR, Journal, Vol, Page`) there is no
    title in the citation at all — `title` is left unset and we populate
    `journal`/`volume`/`page` instead so the resolver has something to query
    Crossref with.
    """
    out: dict = {"raw_text": line}
    m = _DOI_RE.search(line)
    if m:
        out["doi"] = m.group(0).rstrip(".,;)")
    aid = find_valid_arxiv_id(line)
    if aid:
        out["arxiv_id"] = aid
    m = _YEAR_RE.search(line)
    if m:
        try:
            out["year"] = int(m.group(0))
        except ValueError:
            pass

    # ADS-style: "Authors et al. YEAR, Journal, Vol, Page" — has no title.
    # Try this first so we don't mistake the journal name for the title.
    locator = _parse_ads_locator(line, out.get("year"))
    if locator:
        out.update(locator)
        return out

    title = _guess_reference_title(line, out.get("year"))
    if title:
        out["title"] = title
    return out


# ADS/A&A no-title pattern. The bit AFTER the year is `Journal, Vol, Page`.
# Journal can include `&`, `.`, spaces ("J. Astron. Telescopes Instrum. Syst.",
# "SPIE Conf. Ser.", "A&A", "ApJ"). Volume is a positive integer. Page can be
# an integer ("92"), a letter+integer ("A155"), or a SPIE-style identifier
# ("99080I").
_ADS_LOCATOR_RE = re.compile(
    r"""
    ,\s*                                # comma after the year
    (?P<journal>[A-Za-z&][A-Za-z0-9&\.\-\s]+?)  # journal name (non-greedy)
    ,\s*
    (?P<volume>\d{1,5})                 # volume
    ,\s*
    (?P<page>[A-Z]?\d{1,7}[A-Z]?|\d+[\-–]\d+)   # page or article id
    \s*(?:\.|$)                          # end with period or EOL
    """,
    re.VERBOSE,
)


def _parse_ads_locator(line: str, year: int | None) -> dict | None:
    """If `line` looks like a no-title ADS-style ref, return the journal/vol/page.

    Returns None if it doesn't match the pattern (e.g. APA / Nature style with
    a real title).
    """
    if year is None:
        return None
    # Find the year's position; the locator must come immediately after it.
    yr = str(year)
    pos = line.find(yr)
    if pos < 0:
        return None
    tail = line[pos + len(yr):]
    m = _ADS_LOCATOR_RE.match(tail)
    if not m:
        return None
    journal = re.sub(r"\s+", " ", m.group("journal")).strip(" .,")
    if not journal or len(journal) < 2:
        return None
    # Reject if the "journal" looks like an English sentence (i.e. a real
    # title got matched). Real journal abbreviations don't contain common
    # English connective words.
    low = journal.lower()
    if any(f" {w} " in f" {low} " for w in ("and", "the", "of", "for", "with", "in", "by")):
        return None
    return {
        "journal": journal,
        "volume": m.group("volume"),
        "page": m.group("page"),
    }


def looks_like_ads_locator_title(s: str | None) -> bool:
    """True if `s` is something the OLD title extractor mis-stored — i.e. it
    looks like a bare `Journal, Vol, Page` locator rather than a real title."""
    if not s:
        return False
    # Try to parse it as `Journal, vol, page` with no leading year present.
    # We accept matches that have no obvious English sentence structure.
    bits = [b.strip() for b in s.split(",")]
    if len(bits) < 2 or len(bits) > 4:
        return False
    # Reject if any chunk looks like a sentence (multiple lower-case words).
    for b in bits:
        words = [w for w in b.split() if w]
        n_lower = sum(1 for w in words if w[0].islower())
        if n_lower >= 3:
            return False
    # Require at least one chunk to be a small number (volume).
    if not any(re.fullmatch(r"\d{1,5}", b) for b in bits):
        return False
    return True


# Author-list-end anchors: tokens that almost always sit between the byline
# and the title in academic references. Listed in approximate priority order.
# - "et al." (with optional comma/space variants)
# - "<Surname>, <Initial(s)>." possibly followed by " &" or ","
# - APA-style "(year)." right after the byline
_ETAL_RE = re.compile(r"\bet al\.[,\s]+", re.IGNORECASE)
# "<Initial>. " followed by a capital letter — last author's initial in
# Vancouver/Nature style ("Segev, M. Nondiffracting ...").
_INITIAL_BOUNDARY_RE = re.compile(r"\b[A-Z]\.\s+(?=[A-Z])")
# Title-end heuristic: the title ends at the first ". " that's followed by
# either a capital letter (start of journal name) OR a digit (start of vol).
# Demand the char before the period be a letter/closing-paren so initials
# like "G." inside the title don't split it (rare in practice).
_TITLE_END_RE = re.compile(r"(.+?[a-z0-9\)\]])\.\s+(?=[A-Z0-9])")


def _guess_reference_title(line: str, year: int | None) -> str | None:
    """Extract a title from a bibliography entry.

    Handles both common styles:
      Vancouver / Nature : "Authors. Title. Journal vol, pages (year)."
      APA                : "Authors (year). Title. Journal vol, pages."

    Strategy: locate where the author list ends, then take the next
    sentence-shaped chunk (everything up to the next ". <Cap>" boundary).
    Iterate the candidate anchors from latest to earliest so we prefer the
    anchor that sits just before the title rather than one mid-byline.
    """
    if not line:
        return None

    anchors: list[int] = []
    # "et al." style boundary
    for m in _ETAL_RE.finditer(line):
        anchors.append(m.end())
    # "<Initial>. <Cap>" style boundary
    for m in _INITIAL_BOUNDARY_RE.finditer(line):
        anchors.append(m.end())
    # APA "(year). " style boundary
    if year is not None:
        m = re.search(r"\(\s*" + str(year) + r"\s*\)\.\s+", line)
        if m:
            anchors.append(m.end())

    if not anchors:
        return None

    # Try anchors from latest (closest to title) to earliest; for each, pull
    # the first sentence-shaped chunk and accept it if its length looks like
    # a real title.
    for start in sorted(set(anchors), reverse=True):
        rest = line[start:].lstrip()
        if not rest:
            continue
        m = _TITLE_END_RE.match(rest)
        if m:
            title = m.group(1).strip().rstrip(",;:")
            if _looks_like_title(title):
                return title
        # If the title runs to the end of the line (no ". <Cap>" follows),
        # accept up to the first period.
        head = rest.split(".", 1)[0].strip()
        if _looks_like_title(head):
            return head
    return None


# Words that strongly suggest the candidate text is NOT a title (it's
# splitter noise like a figure caption that got fused into the entry).
_TITLE_NOISE_PREFIXES = (
    "figure ",
    "fig. ",
    "fig ",
    "table ",
    "supplementary ",
    "see also ",
    "comparison of ",
    "note that ",
)


def _looks_like_title(s: str) -> bool:
    if not s:
        return False
    if not (8 <= len(s) <= 300):
        return False
    low = s.lower()
    if any(low.startswith(p) for p in _TITLE_NOISE_PREFIXES):
        return False
    # A real title should contain at least one space (i.e. more than one word).
    if " " not in s:
        return False
    # Reject fragments that are just abbreviations and initials — common for
    # journal-name fragments mis-extracted as titles. Heuristic: count
    # genuine word tokens (≥3 letters, all lowercase or Title-cased). A real
    # title has at least 2 such tokens; "R. Astron" has 0 (R is 1 char,
    # Astron is 6 chars but is the only word).
    word_tokens = [
        t for t in re.split(r"[\s.,;:]+", s)
        if len(t) >= 4 and t.isalpha()
    ]
    if len(word_tokens) < 2:
        return False
    return True
