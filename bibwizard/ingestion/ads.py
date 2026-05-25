"""NASA ADS search (optional, requires ADS_API_TOKEN).

ADS has authoritative metadata for astronomy/astrophysics papers (final journal
info, full author lists, official DOIs, arXiv ↔ bibcode mapping). This module
is opt-in: if ADS_API_TOKEN isn't set, `search_ads()` returns an empty list.

Get a free token at https://ui.adsabs.harvard.edu/user/settings/token
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from bibwizard.utils.config import settings


ADS_BASE_URL = "https://api.adsabs.harvard.edu/v1/search/query"


@dataclass
class ADSPaper:
    bibcode: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    venue: str | None = None  # journal / publication
    pubdate: str | None = None  # "YYYY-MM"

    @property
    def adsabs_url(self) -> str:
        return f"https://ui.adsabs.harvard.edu/abs/{self.bibcode}"

    @property
    def eprint_pdf_url(self) -> str:
        return f"https://ui.adsabs.harvard.edu/abs/{self.bibcode}/EPRINT_PDF"


def is_configured() -> bool:
    return bool(settings.ads_api_token)


def _extract_arxiv_from_identifiers(identifiers: list[str] | None) -> str | None:
    """Pick the arXiv id out of ADS's `identifier` list (mixed bag of ids)."""
    if not identifiers:
        return None
    for raw in identifiers:
        s = str(raw).strip()
        low = s.lower()
        if low.startswith("arxiv:"):
            return s.split(":", 1)[1]
        # Modern bare IDs: 1706.03762
        if len(s) >= 9 and s[4] == "." and s[:4].isdigit() and s[5:].split("v")[0].isdigit():
            return s
    return None


def search_ads(
    *,
    title: str | None = None,
    author: str | None = None,
    year: int | None = None,
    bibcode: str | None = None,
    arxiv_id: str | None = None,
    doi: str | None = None,
    max_results: int = 5,
) -> list[ADSPaper]:
    """Query ADS by any combination of fields. Returns [] when no token is set."""
    if not is_configured():
        return []

    parts: list[str] = []
    if bibcode:
        parts.append(f'bibcode:"{bibcode}"')
    if title:
        parts.append(f'title:"{title}"')
    if author:
        parts.append(f'author:"{author}"')
    if year:
        parts.append(f"year:{year}")
    if arxiv_id:
        parts.append(f'identifier:"arXiv:{arxiv_id}"')
    if doi:
        parts.append(f'doi:"{doi}"')
    if not parts:
        return []

    headers = {
        "Authorization": f"Bearer {settings.ads_api_token}",
        "User-Agent": settings.user_agent,
        "Accept": "application/json",
    }
    params = {
        "q": " ".join(parts),
        "fl": "bibcode,title,author,year,doi,pub,pubdate,identifier",
        "rows": max(1, min(max_results, 25)),
        "sort": "score desc",
    }

    try:
        with httpx.Client(timeout=20.0, headers=headers, follow_redirects=True) as client:
            resp = client.get(ADS_BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError:
        return []

    docs = (data.get("response") or {}).get("docs") or []
    out: list[ADSPaper] = []
    for d in docs:
        title_list = d.get("title") or []
        doi_list = d.get("doi") or []
        out.append(
            ADSPaper(
                bibcode=str(d.get("bibcode", "")),
                title=str(title_list[0]) if title_list else "",
                authors=[str(a) for a in (d.get("author") or [])],
                year=int(d["year"]) if d.get("year") else None,
                doi=str(doi_list[0]) if doi_list else None,
                arxiv_id=_extract_arxiv_from_identifiers(d.get("identifier")),
                venue=d.get("pub"),
                pubdate=d.get("pubdate"),
            )
        )
    return out
