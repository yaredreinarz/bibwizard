"""Typer entry point for the `bibwizard` CLI."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

import typer
from rich.markdown import Markdown
from rich.panel import Panel

from bibwizard import __version__
from bibwizard.context.content_map import export_clusters
from bibwizard.context.reference_map import export_dot, export_json
from bibwizard.database.migrations import init_db, session_scope
from bibwizard.database.models import Paper
from bibwizard.database.queries import (  # noqa: E402  -- grouped import
    backfill_paper_from_summary,
    delete_all_papers,
    delete_orphan_authors,
    delete_orphan_tags,
    delete_paper,
    get_or_create_author,
    reset_chunk_counts,
    reset_summary_columns,
)
from bibwizard.database.queries import (
    add_citation,
    add_tags,
    create_paper,
    find_paper_by_identity,
    library_stats,
    list_papers,
    list_recent_papers,
    set_chunk_count,
    text_search,
    update_summary,
)
from bibwizard.ingestion.embedder import (
    delete_paper_chunks,
    ingest_paper_chunks,
    query_chunks,
    reset_collection,
    save_pdf_to_library,
)
from bibwizard.ingestion import ads as ads_mod
from bibwizard.ingestion.metadata import (
    ArxivRateLimited,
    crossref_lookup_by_locator,
    crossref_lookup_by_title,
    download_pdf,
    extract_arxiv_id_from_url,
    extract_ads_bibcode_from_url,
    extract_metadata,
    fetch_arxiv_metadata,
    is_url,
    looks_like_ads_locator_title,
    parse_reference_line,
    resolve_ads_to_pdf_url,
    search_arxiv,
    search_arxiv_by_doi,
    search_arxiv_by_title,
    title_similarity,
    unpaywall_pdf_url,
)
from bibwizard.ingestion.parser import parse_pdf, split_references
from bibwizard.ingestion.structure import (
    extract_front_matter,
    parse_authors_from_byline,
)
from bibwizard.llm.chat import run_chat_loop
from bibwizard.llm.client import OllamaModelMissing, OllamaUnavailable, get_client
from bibwizard.llm.summarizer import summarize_paper
from bibwizard.utils.config import ensure_dirs, settings
from bibwizard.utils.display import (
    banner,
    console,
    error,
    info,
    panel,
    papers_table,
    search_results_table,
    stats_table,
    success,
    warn,
)

app = typer.Typer(
    name="bibwizard",
    help="bibwizard — local-first, LLM-powered research paper manager.",
    rich_markup_mode="rich",
)

map_app = typer.Typer(help="Export reference / content maps.", no_args_is_help=True)
app.add_typer(map_app, name="map")


# ---------- helpers ----------

def _safe_run(fn, *, ollama_check: bool = False, **kwargs) -> None:
    """Wrapper that turns Ollama errors into clean CLI messages."""
    try:
        if ollama_check:
            get_client().ensure_ready(**kwargs)
        fn()
    except (OllamaUnavailable, OllamaModelMissing) as e:
        error(str(e))
        raise typer.Exit(code=2) from None


def _persist_summary(paper_id: int, summary) -> Path:
    summary_path = settings.summaries_dir / f"paper_{paper_id}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary_path


def _ingest_pdf_into_library(
    pdf_path: Path,
    *,
    arxiv_id: str | None = None,
    skip_summary: bool = False,
    skip_embed: bool = False,
    external_metadata=None,
    llm_extract: bool | None = None,
) -> int:
    """Shared ingest pipeline for both `add <path>` and `add --arxiv`.

    `external_metadata` (a PaperMetadata) overrides PDF-scraped fields when
    provided. We use this for arxiv downloads — the arxiv API's title/authors/
    year/DOI are authoritative and almost always better than what we can
    recover from the PDF.
    """
    info(f"Parsing {pdf_path.name}...")
    parsed = parse_pdf(pdf_path)

    with session_scope() as session:
        existing = find_paper_by_identity(session, sha256=parsed.sha256)
        if existing is not None:
            warn(f"Paper already in library as id={existing.id} ({existing.title}).")
            return existing.id

    metadata = extract_metadata(parsed)
    if arxiv_id and not metadata.arxiv_id:
        metadata.arxiv_id = arxiv_id

    # Secondary dedup: sha256 didn't catch this PDF (different bytes — e.g.
    # re-downloaded under a different filename) but the metadata DOI / arXiv
    # id may already be in the library. Skip cleanly instead of crashing on
    # the UNIQUE constraint.
    if metadata.doi or metadata.arxiv_id:
        with session_scope() as session:
            existing = find_paper_by_identity(
                session, doi=metadata.doi, arxiv_id=metadata.arxiv_id
            )
            if existing is not None:
                ident = metadata.doi or metadata.arxiv_id
                warn(
                    f"Paper already in library as id={existing.id} "
                    f"(matched by {ident}). Skipping {pdf_path.name}."
                )
                return existing.id

    # Careful LLM-driven metadata extraction (slow, opt-in). Overrides the
    # heuristic when it returns a result that passes validation. Falls back
    # to the heuristic silently on failure.
    use_llm = llm_extract if llm_extract is not None else settings.llm_extract
    if use_llm:
        from bibwizard.ingestion.llm_extract import llm_extract_metadata

        info("Running careful LLM front-matter extraction (this can take a few minutes)...")
        try:
            em = llm_extract_metadata(
                parsed, heuristic=metadata, verify=settings.llm_extract_verify
            )
        except Exception as e:  # noqa: BLE001
            warn(f"LLM extraction errored, falling back: {e}")
            em = None
        if em is not None:
            success(
                f"LLM extracted: title={em.title[:60]!r}, "
                f"{len(em.authors)} authors, year={em.year}, "
                f"abstract={len(em.abstract)} chars"
            )
            # LLM result wins over the heuristic for these fields.
            if em.title:
                metadata.title = em.title
            if em.authors:
                metadata.authors = list(em.authors)
            if em.year is not None:
                metadata.year = em.year
            if em.abstract and not metadata.abstract:
                metadata.abstract = em.abstract
            if em.doi and not metadata.doi:
                metadata.doi = em.doi
            if em.arxiv_id and not metadata.arxiv_id:
                metadata.arxiv_id = em.arxiv_id
        else:
            warn("LLM extraction returned no usable result; keeping heuristic.")

    # If we have authoritative metadata from arxiv (or any future source),
    # let it win wholesale. PDF-scrape values are only kept for fields the
    # external source doesn't provide.
    if external_metadata is not None:
        if external_metadata.title:
            metadata.title = external_metadata.title
        if external_metadata.authors:
            metadata.authors = list(external_metadata.authors)
        if external_metadata.year is not None:
            metadata.year = external_metadata.year
        if external_metadata.doi:
            metadata.doi = external_metadata.doi
        if external_metadata.arxiv_id:
            metadata.arxiv_id = external_metadata.arxiv_id
        if external_metadata.abstract and not metadata.abstract:
            metadata.abstract = external_metadata.abstract

    # Copy into library if not already there
    target_path = save_pdf_to_library(pdf_path)

    # Persist row first so we have an id, then update with summary
    with session_scope() as session:
        paper = create_paper(
            session,
            title=metadata.title or pdf_path.stem,
            authors=metadata.authors,
            year=metadata.year,
            doi=metadata.doi,
            arxiv_id=metadata.arxiv_id,
            abstract=metadata.abstract,
            file_path=str(target_path),
            sha256=parsed.sha256,
        )
        paper_id = paper.id

    # References → citations
    refs = split_references(parsed.references)
    if refs:
        info(f"Parsed {len(refs)} references from bibliography.")
        with session_scope() as session:
            for r in refs:
                fields = parse_reference_line(r)
                add_citation(
                    session,
                    source_paper_id=paper_id,
                    raw_text=r,
                    target_title=fields.get("title"),
                    target_doi=fields.get("doi"),
                    target_arxiv_id=fields.get("arxiv_id"),
                    target_year=fields.get("year"),
                )

    # Summary
    if not skip_summary:
        info("Generating structured summary via DeepSeek...")
        try:
            summary = summarize_paper(parsed, metadata)
            sp = _persist_summary(paper_id, summary)
            with session_scope() as session:
                update_summary(
                    session,
                    paper_id,
                    json.dumps(summary.to_dict(), ensure_ascii=False),
                    str(sp),
                )
                # Backfill any paper fields the heuristic extractor missed
                # (e.g. authors when the PDF has no embedded metadata).
                _, filled = backfill_paper_from_summary(
                    session,
                    paper_id,
                    title=summary.title or None,
                    authors=summary.authors or None,
                    year=summary.year,
                    abstract=metadata.abstract,
                    doi=metadata.doi,
                    arxiv_id=metadata.arxiv_id,
                )
                tags_filled = []
                if settings.auto_tag and summary.tags:
                    add_tags(session, paper_id, summary.tags)
                    tags_filled = [f"tags[{len(summary.tags)}]"]
            filled_msg = ", ".join(filled + tags_filled) or "nothing new"
            success(f"Summary saved to {sp}  (filled: {filled_msg})")
        except (OllamaUnavailable, OllamaModelMissing) as e:
            warn(f"Skipping summary: {e}")
        except Exception as e:  # noqa: BLE001
            warn(f"Summary failed: {e}")

    # Embedding
    if not skip_embed:
        info("Embedding chunks into ChromaDB...")
        try:
            n = ingest_paper_chunks(
                paper_id, metadata.title or pdf_path.stem, parsed.pages
            )
            with session_scope() as session:
                set_chunk_count(session, paper_id, n)
            success(f"Indexed {n} chunks for paper id={paper_id}.")
        except (OllamaUnavailable, OllamaModelMissing) as e:
            warn(f"Skipping embedding: {e}")
        except Exception as e:  # noqa: BLE001
            warn(f"Embedding failed: {e}")

    return paper_id


# ---------- commands ----------

@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-V", help="Show version and exit.", is_eager=True
    ),
) -> None:
    if version:
        console.print(f"bibwizard {__version__}")
        raise typer.Exit()
    # Mimic `no_args_is_help=True` but only after eager flags have run.
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


@app.command()
def init() -> None:
    """Create the database, vector store, and literature/ folder."""
    banner("bibwizard init", f"home={settings.home}")
    ensure_dirs(settings)
    init_db()
    success(f"Initialized database at {settings.sqlite_path}")
    success(f"Vector store at {settings.vectors_dir}")
    success(f"Literature drop folder at {settings.literature_dir}")
    info("Drop PDFs into the literature/ folder and run `bibwizard scan`.")


def _resolve_source_to_pdf(target: str):
    """Turn an `add` argument (path | arxiv id | URL) into a local PDF path.

    Returns (pdf_path, arxiv_id, external_metadata).
      - arxiv_id is filled in when we can determine it from the source
      - external_metadata is the authoritative arxiv-API metadata when the
        source resolved through arxiv (None otherwise)
    """
    # 1. Plain local path
    if not is_url(target) and not _looks_like_arxiv_id(target):
        p = Path(target).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        return p, None, None

    # 2. Bare arxiv id (e.g. "2106.04561")
    if _looks_like_arxiv_id(target):
        return _download_arxiv(target)

    # 3. URL — sniff for arxiv / ADS / direct PDF
    url = target
    aid = extract_arxiv_id_from_url(url)
    if aid:
        return _download_arxiv(aid)

    bibcode = extract_ads_bibcode_from_url(url)
    if bibcode:
        info(f"Resolving ADS bibcode {bibcode} via EPRINT_PDF redirect...")
        try:
            resolved = resolve_ads_to_pdf_url(bibcode)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"ADS resolution failed: {e}") from e
        aid = extract_arxiv_id_from_url(resolved)
        if aid:
            return _download_arxiv(aid)
        # ADS pointed to a non-arxiv PDF — download it directly
        target_path = settings.literature_dir / f"ads_{bibcode}.pdf"
        info(f"Downloading {resolved} → {target_path}")
        download_pdf(resolved, target_path)
        return target_path, None, None

    # 4. Generic URL (often a direct PDF link)
    fname = url.rsplit("/", 1)[-1].split("?")[0] or "download.pdf"
    if not fname.lower().endswith(".pdf"):
        fname += ".pdf"
    target_path = settings.literature_dir / fname
    info(f"Downloading {url} → {target_path}")
    download_pdf(url, target_path)
    return target_path, None, None


def _looks_like_arxiv_id(s: str) -> bool:
    import re

    return bool(re.fullmatch(r"(?:arXiv:)?\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+/\d{7}(?:v\d+)?", s))


def _download_arxiv(arxiv_id: str):
    """Fetch arxiv metadata + download the PDF.

    Returns (pdf_path, arxiv_id, paper_metadata). The metadata is authoritative
    (title + author list + year + DOI come from the arxiv API), so callers
    should prefer it over PDF-scrape heuristics.
    """
    info(f"Looking up arXiv id {arxiv_id}...")
    md, pdf_url = fetch_arxiv_metadata(arxiv_id)
    aid = md.arxiv_id or arxiv_id
    target = settings.literature_dir / f"arxiv_{aid.replace('/', '_')}.pdf"
    info(f"Downloading {pdf_url} → {target}")
    download_pdf(pdf_url, target)
    return target, aid, md


@app.command("add")
def add_paper(
    target: Optional[str] = typer.Argument(
        None,
        help="Local PDF path, arXiv id, or URL (arXiv abs/pdf, NASA ADS, or any direct PDF).",
        metavar="TARGET",
    ),
    arxiv: Optional[str] = typer.Option(
        None, "--arxiv", help="Explicit arXiv id (alias for passing it as TARGET)."
    ),
    no_summary: bool = typer.Option(False, "--no-summary", help="Skip LLM summary."),
    no_embed: bool = typer.Option(False, "--no-embed", help="Skip vector indexing."),
    llm_extract: bool = typer.Option(
        False,
        "--llm-extract",
        help="Use the LLM (slow but careful) to extract title/authors/year/abstract. "
        "Adds ~30s-3min per paper depending on model + hardware.",
    ),
) -> None:
    """Ingest a paper.

    Examples:
      bibwizard add ./paper.pdf
      bibwizard add 2106.04561
      bibwizard add https://arxiv.org/abs/1706.03762
      bibwizard add https://ui.adsabs.harvard.edu/abs/2017Natur.551..547L/abstract
      bibwizard add https://example.com/some/paper.pdf
    """
    init_db()
    ensure_dirs(settings)

    src = arxiv or target
    if not src:
        error("Provide a path, an arXiv id, or a URL.")
        raise typer.Exit(code=2)

    try:
        pdf_path, arxiv_id, external_md = _resolve_source_to_pdf(src)
    except FileNotFoundError as e:
        error(str(e))
        raise typer.Exit(code=2) from None
    except Exception as e:  # noqa: BLE001
        error(f"Could not fetch source: {e}")
        raise typer.Exit(code=1) from None

    if pdf_path.suffix.lower() != ".pdf":
        warn(f"{pdf_path.name} doesn't end in .pdf — proceeding anyway.")

    paper_id = _ingest_pdf_into_library(
        pdf_path, arxiv_id=arxiv_id, external_metadata=external_md,
        skip_summary=no_summary, skip_embed=no_embed,
        llm_extract=llm_extract or None,
    )
    success(f"Done. paper id = {paper_id}")


@app.command("scan")
def scan_cmd(
    folder: Optional[Path] = typer.Option(
        None,
        "--folder",
        "-d",
        help="Folder to scan (defaults to ./literature/).",
    ),
    no_summary: bool = typer.Option(False, "--no-summary", help="Skip LLM summary."),
    no_embed: bool = typer.Option(False, "--no-embed", help="Skip vector indexing."),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Recurse into subfolders."),
    llm_extract: bool = typer.Option(
        False,
        "--llm-extract",
        help="Use the LLM to carefully extract metadata from each PDF (slow). "
        "Recommended for overnight runs over a large literature folder.",
    ),
) -> None:
    """Ingest every PDF in the literature folder that isn't already in the library."""
    init_db()
    ensure_dirs(settings)
    target_dir = (folder or settings.literature_dir).expanduser().resolve()
    if not target_dir.exists():
        error(f"Folder not found: {target_dir}")
        raise typer.Exit(code=2)

    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdfs = sorted(p for p in target_dir.glob(pattern) if p.is_file())
    if not pdfs:
        warn(f"No PDFs found under {target_dir}.")
        return

    total = len(pdfs)
    info(f"Scanning {total} PDF(s) under {target_dir}{' (LLM extract ON)' if llm_extract else ''}...")
    n_new = n_skipped = n_failed = 0
    for idx, p in enumerate(pdfs, start=1):
        info(f"[{idx}/{total}] {p.name}")
        try:
            from bibwizard.ingestion.parser import parse_pdf as _peek_parse

            parsed = _peek_parse(p)  # quick hash + parse
            with session_scope() as s:
                if find_paper_by_identity(s, sha256=parsed.sha256) is not None:
                    n_skipped += 1
                    info(f"  skip (already in library)")
                    continue
            _ingest_pdf_into_library(
                p, skip_summary=no_summary, skip_embed=no_embed,
                llm_extract=llm_extract or None,
            )
            n_new += 1
        except KeyboardInterrupt:
            warn(f"Interrupted at paper {idx}/{total}. Progress so far: {n_new} new, {n_skipped} skipped, {n_failed} failed.")
            raise
        except Exception as e:  # noqa: BLE001
            n_failed += 1
            warn(f"  failed {p.name}: {e}")

    success(f"Scan complete: {n_new} added, {n_skipped} skipped, {n_failed} failed.")


def _first_author_surname(raw: str) -> str:
    """Best-effort first-author surname from a raw reference line."""
    if not raw:
        return ""
    head = raw.split(",", 1)[0].strip()
    # Strip common abbreviations on what should be a surname
    head = head.rstrip(".")
    if " " in head:
        head = head.split()[-1]
    return head


@app.command("fetch-refs")
def fetch_refs_cmd(
    paper_id: int = typer.Argument(..., help="Paper id whose references to fetch."),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="Stop after this many successful fetches."
    ),
    no_summary: bool = typer.Option(False, "--no-summary", help="Skip LLM summary on each."),
    no_embed: bool = typer.Option(False, "--no-embed", help="Skip embedding on each."),
    use_title_search: bool = typer.Option(
        False,
        "--title-search",
        help="Last-resort: search arXiv by title for refs lacking a DOI/arXiv id.",
    ),
    refresh_fields: bool = typer.Option(
        False,
        "--refresh-fields",
        help="Re-run the per-line field extractor on each stored raw_text "
        "before resolving. Backfills titles/years/DOIs/arXiv ids and clears "
        "old junk titles that turn out to be journal locators.",
    ),
    no_crossref: bool = typer.Option(
        False,
        "--no-crossref",
        help="Disable the Crossref DOI lookup for references without a DOI.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Resolve references but don't download anything. Useful for "
        "seeing what fetch-refs WOULD do.",
    ),
) -> None:
    """Resolve every reference of a paper and download the ones we can.

    Resolution chain per reference:
      1. arXiv id already in the citation row     → download from arXiv
      2. DOI already in the citation row          → arXiv-by-DOI, else Unpaywall
      3. ADS-style (journal/vol/page in raw text) → Crossref-by-locator → DOI
      4. Title in citation row                    → Crossref-by-title → DOI
      5. (opt-in) --title-search                  → arXiv-by-title

    Resolved DOIs are persisted back to the citation row, so a second run
    short-circuits to the dedup-by-DOI check immediately.
    """
    init_db()
    ensure_dirs(settings)

    with session_scope() as session:
        paper = session.get(Paper, paper_id)
        if paper is None:
            error(f"No paper with id {paper_id}")
            raise typer.Exit(code=2)
        citations = list(paper.outgoing_citations)
        source_title = paper.title

        if refresh_fields and citations:
            n_updated = 0
            n_cleared = 0
            for cit in citations:
                fields = parse_reference_line(cit.raw_text)
                changed = False
                # Fill missing identifying fields
                if not cit.target_doi and fields.get("doi"):
                    cit.target_doi = fields["doi"]
                    changed = True
                if not cit.target_arxiv_id and fields.get("arxiv_id"):
                    cit.target_arxiv_id = fields["arxiv_id"]
                    changed = True
                if not cit.target_year and fields.get("year"):
                    cit.target_year = fields["year"]
                    changed = True
                # Title: new parser may have decided the entry has no title
                # (ADS style). If we currently store a journal-locator-shaped
                # string, clear it — it was junk.
                new_title = fields.get("title")
                if new_title and not cit.target_title:
                    cit.target_title = new_title
                    changed = True
                elif (
                    not new_title
                    and cit.target_title
                    and looks_like_ads_locator_title(cit.target_title)
                ):
                    cit.target_title = None
                    changed = True
                    n_cleared += 1
                if changed:
                    n_updated += 1
            session.flush()
            info(
                f"  refreshed fields on {n_updated}/{len(citations)} citations "
                f"(cleared {n_cleared} junk titles)."
            )
            citations = list(paper.outgoing_citations)

    if not citations:
        warn(f"Paper {paper_id} ({source_title}) has no parsed references.")
        return

    info(f"Paper {paper_id}: {len(citations)} parsed references. Resolving...")
    n_added = n_have = n_unresolved = n_failed = n_xref = 0
    for cit in citations:
        if limit is not None and n_added >= limit:
            break

        # 1. Already in library by arxiv id or DOI?
        with session_scope() as s:
            if cit.target_arxiv_id and find_paper_by_identity(s, arxiv_id=cit.target_arxiv_id):
                n_have += 1
                continue
            if cit.target_doi and find_paper_by_identity(s, doi=cit.target_doi):
                n_have += 1
                continue

        # 2. No DOI? Try Crossref — by locator first, then by title.
        if not cit.target_doi and not no_crossref:
            fields = parse_reference_line(cit.raw_text)
            surname = _first_author_surname(cit.raw_text)
            year = cit.target_year or fields.get("year")
            doi_found: str | None = None
            if year and fields.get("journal") and fields.get("volume"):
                doi_found = crossref_lookup_by_locator(
                    author_surname=surname,
                    year=year,
                    journal=fields["journal"],
                    volume=fields["volume"],
                    page=fields.get("page", ""),
                )
                if doi_found:
                    info(f"  crossref(locator) → {doi_found}")
            if not doi_found and cit.target_title:
                doi_found = crossref_lookup_by_title(
                    title=cit.target_title,
                    author_surname=surname or None,
                    year=year,
                )
                if doi_found:
                    info(f"  crossref(title)   → {doi_found}")
            if doi_found:
                # Persist the freshly-resolved DOI to the citation row.
                with session_scope() as s:
                    row = s.get(type(cit), cit.id)
                    if row is not None:
                        row.target_doi = doi_found
                        # Link to a library paper if we now match one.
                        existing = find_paper_by_identity(s, doi=doi_found)
                        if existing is not None and existing.id != paper_id:
                            row.target_paper_id = existing.id
                cit.target_doi = doi_found
                n_xref += 1
                # Recheck dedup with the freshly-resolved DOI.
                with session_scope() as s:
                    if find_paper_by_identity(s, doi=doi_found):
                        n_have += 1
                        continue

        # 3. Resolve to a downloadable PDF: arXiv first (preferred), then Unpaywall.
        aid = cit.target_arxiv_id
        if not aid and cit.target_doi:
            aid = search_arxiv_by_doi(cit.target_doi)
        if not aid and use_title_search and cit.target_title:
            aid = search_arxiv_by_title(cit.target_title)

        snippet = (cit.target_title or cit.raw_text)[:80]
        if dry_run:
            if aid:
                info(f"  [dry-run] would download arXiv:{aid} — {snippet}")
                n_added += 1
            elif cit.target_doi and settings.unpaywall_email:
                info(f"  [dry-run] would try Unpaywall for {cit.target_doi}")
                n_added += 1
            elif cit.target_doi:
                # DOI resolved but no download path (no arXiv preprint and
                # UNPAYWALL_EMAIL not set). Count separately so the summary
                # doesn't over-promise.
                info(f"  [dry-run] DOI {cit.target_doi} — set UNPAYWALL_EMAIL to download")
                n_unresolved += 1
            else:
                n_unresolved += 1
            continue

        try:
            if aid:
                info(f"  arxiv:{aid} — {snippet}")
                pdf_path, aid_resolved, ext_md = _download_arxiv(aid)
                _ingest_pdf_into_library(
                    pdf_path,
                    arxiv_id=aid_resolved,
                    external_metadata=ext_md,
                    skip_summary=no_summary,
                    skip_embed=no_embed,
                )
                n_added += 1
                time.sleep(settings.arxiv_min_delay)
            elif cit.target_doi:
                # No arXiv preprint — try Unpaywall for an OA copy.
                if not settings.unpaywall_email:
                    n_unresolved += 1
                    continue
                pdf_url = unpaywall_pdf_url(cit.target_doi)
                if not pdf_url:
                    n_unresolved += 1
                    continue
                info(f"  unpaywall: {cit.target_doi} → {pdf_url}")
                safe_doi = re.sub(r"[^A-Za-z0-9._-]+", "_", cit.target_doi)
                target = settings.literature_dir / f"doi_{safe_doi}.pdf"
                download_pdf(pdf_url, target)
                _ingest_pdf_into_library(
                    target,
                    skip_summary=no_summary,
                    skip_embed=no_embed,
                )
                n_added += 1
            else:
                n_unresolved += 1
        except ArxivRateLimited as e:
            warn(f"    {e}")
            warn("Stopping early to back off arXiv. Re-run later.")
            break
        except Exception as e:  # noqa: BLE001
            warn(f"    failed: {e}")
            n_failed += 1

    extra = f", {n_xref} new DOIs via Crossref" if n_xref else ""
    success(
        f"fetch-refs done: {n_added} added, {n_have} already had, "
        f"{n_unresolved} unresolved, {n_failed} failed{extra}."
    )


@app.command()
def chat() -> None:
    """Open a RAG chat session over your library."""
    init_db()

    def _go() -> None:
        run_chat_loop()

    _safe_run(_go, ollama_check=True, need_llm=True, need_embed=True)


@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="Natural-language search query."),
    top_k: int = typer.Option(8, "--top-k", "-k", help="Max results."),
    paper: Optional[int] = typer.Option(
        None, "--paper", help="Restrict search to a single paper id."
    ),
) -> None:
    """Semantic search across all chunks in your library."""
    init_db()
    try:
        get_client().ensure_ready(need_embed=True, need_llm=False)
    except (OllamaUnavailable, OllamaModelMissing) as e:
        error(str(e))
        raise typer.Exit(code=2) from None

    chunks = query_chunks(
        query, top_k=top_k, paper_ids=[paper] if paper is not None else None
    )
    if not chunks:
        warn("No matches.")
        return

    rows = []
    for ch in chunks:
        m = ch.get("metadata", {}) or {}
        rows.append(
            {
                "score": ch.get("score", 0),
                "paper": f"id={m.get('paper_id', '?')} p.{m.get('page', '?')} — {m.get('title', '')}",
                "snippet": ch.get("text", ""),
            }
        )
    console.print(search_results_table(rows))


@app.command("show")
def show_cmd(paper_id: int = typer.Argument(..., help="Paper id from `bibwizard list`.")) -> None:
    """Display a paper's metadata and structured summary."""
    init_db()
    with session_scope() as session:
        paper = session.get(Paper, paper_id)
        if paper is None:
            error(f"No paper with id {paper_id}")
            raise typer.Exit(code=2)

        header = (
            f"[bold]{paper.title}[/]\n"
            f"[dim]id={paper.id}  year={paper.year or '—'}  "
            f"doi={paper.doi or '—'}  arxiv={paper.arxiv_id or '—'}[/]\n"
            f"Authors: {', '.join(a.name for a in paper.authors) or '—'}\n"
            f"Tags:    {', '.join(t.name for t in paper.tags) or '—'}\n"
            f"File:    {paper.file_path or '—'}\n"
            f"Chunks:  {paper.n_chunks}"
        )
        console.print(Panel(header, title="Paper", border_style="cyan"))

        if paper.abstract:
            console.print(Panel(paper.abstract, title="Abstract", border_style="dim"))

        if paper.summary_json:
            try:
                data = json.loads(paper.summary_json)
            except json.JSONDecodeError:
                data = None
            if data:
                md = "## Key contributions\n" + "\n".join(
                    f"- {k}" for k in data.get("key_contributions", [])
                )
                md += f"\n\n## Methodology\n{data.get('methodology', '—')}"
                md += f"\n\n## Limitations\n{data.get('limitations', '—')}"
                console.print(Panel(Markdown(md), title="Summary", border_style="green"))


@app.command("resummarize")
def resummarize_cmd(
    paper_id: int = typer.Argument(..., help="Paper id to re-summarize."),
    llm_extract: bool = typer.Option(
        False,
        "--llm-extract",
        help="Re-extract metadata with the careful LLM pass before summarizing.",
    ),
) -> None:
    """Retry the LLM summary on an existing paper and backfill missing fields."""
    init_db()
    with session_scope() as session:
        paper = session.get(Paper, paper_id)
        if paper is None:
            error(f"No paper with id {paper_id}")
            raise typer.Exit(code=2)
        file_path = paper.file_path

    if not file_path or not Path(file_path).exists():
        error(f"PDF file missing on disk: {file_path}")
        raise typer.Exit(code=2)

    info(f"Re-parsing {file_path}...")
    parsed = parse_pdf(Path(file_path))
    metadata = extract_metadata(parsed)

    # Optional careful LLM front-matter pass
    if llm_extract or settings.llm_extract:
        from bibwizard.ingestion.llm_extract import llm_extract_metadata

        info("Running careful LLM front-matter extraction...")
        try:
            em = llm_extract_metadata(
                parsed, heuristic=metadata, verify=settings.llm_extract_verify
            )
        except Exception as e:  # noqa: BLE001
            warn(f"LLM extraction errored: {e}")
            em = None
        if em is not None:
            success(
                f"LLM extracted: title={em.title[:60]!r}, "
                f"{len(em.authors)} authors, year={em.year}"
            )
            if em.title:
                metadata.title = em.title
            if em.authors:
                metadata.authors = list(em.authors)
            if em.year is not None:
                metadata.year = em.year
            if em.abstract:
                metadata.abstract = em.abstract
            if em.doi and not metadata.doi:
                metadata.doi = em.doi
            if em.arxiv_id and not metadata.arxiv_id:
                metadata.arxiv_id = em.arxiv_id

    try:
        summary = summarize_paper(parsed, metadata)
    except (OllamaUnavailable, OllamaModelMissing) as e:
        error(str(e))
        raise typer.Exit(code=2) from None

    # Show what the LLM actually returned so you can see why something might
    # still be empty — empty here = DeepSeek didn't extract that field.
    info(
        f"LLM returned: title={summary.title!r}, "
        f"authors={summary.authors}, year={summary.year}, "
        f"tags={summary.tags}, "
        f"contributions={len(summary.key_contributions)}, "
        f"methodology={len(summary.methodology)} chars"
    )

    sp = _persist_summary(paper_id, summary)
    with session_scope() as session:
        update_summary(
            session,
            paper_id,
            json.dumps(summary.to_dict(), ensure_ascii=False),
            str(sp),
        )
        # When --llm-extract was used we OVERWRITE the paper row with the
        # LLM-extracted values (user explicitly asked to re-extract).
        # Otherwise the backfill is conservative (only empty fields).
        if llm_extract or settings.llm_extract:
            paper = session.get(Paper, paper_id)
            if paper is not None:
                if metadata.title:
                    paper.title = metadata.title
                if metadata.year is not None:
                    paper.year = metadata.year
                if metadata.doi:
                    paper.doi = metadata.doi
                if metadata.arxiv_id:
                    paper.arxiv_id = metadata.arxiv_id
                if metadata.abstract:
                    paper.abstract = metadata.abstract
                if metadata.authors:
                    paper.authors.clear()
                    for n in metadata.authors:
                        n = " ".join((n or "").split())
                        if n:
                            paper.authors.append(get_or_create_author(session, n))
        _, filled = backfill_paper_from_summary(
            session,
            paper_id,
            title=summary.title or None,
            authors=summary.authors or None,
            year=summary.year,
            abstract=metadata.abstract,
            doi=metadata.doi,
            arxiv_id=metadata.arxiv_id,
        )
        tags_filled = []
        if settings.auto_tag and summary.tags:
            add_tags(session, paper_id, summary.tags)
            tags_filled = [f"tags[{len(summary.tags)}]"]
    filled_msg = ", ".join(filled + tags_filled) or "nothing"
    success(f"Resummarized paper {paper_id}. Backfilled: {filled_msg}")
    info(f"Raw summary JSON: {sp}")


@app.command("list")
def list_cmd(
    year: Optional[int] = typer.Option(None, "--year", help="Filter by year."),
    tag: Optional[str] = typer.Option(None, "--tag", help="Filter by tag."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Max rows."),
) -> None:
    """List papers in the library."""
    init_db()
    with session_scope() as session:
        papers = list_papers(session, year=year, tag=tag, limit=limit)
        rows = [
            {
                "id": p.id,
                "title": p.title,
                "authors": ", ".join(a.name for a in p.authors[:3])
                + (" …" if len(p.authors) > 3 else ""),
                "year": p.year,
                "tags": ", ".join(t.name for t in p.tags),
            }
            for p in papers
        ]
    if not rows:
        warn("Library is empty.")
        return
    console.print(papers_table(rows))


def _humanize_since(now, then) -> str:
    """Return a human-readable 'N days ago' style relative timestamp."""
    delta = now - then
    secs = int(delta.total_seconds())
    if secs < 60:
        return "just now"
    if secs < 3600:
        m = secs // 60
        return f"{m} minute{'s' if m != 1 else ''} ago"
    if secs < 86400:
        h = secs // 3600
        return f"{h} hour{'s' if h != 1 else ''} ago"
    days = secs // 86400
    if days < 14:
        return f"{days} day{'s' if days != 1 else ''} ago"
    if days < 60:
        w = days // 7
        return f"{w} week{'s' if w != 1 else ''} ago"
    if days < 730:
        mo = days // 30
        return f"{mo} month{'s' if mo != 1 else ''} ago"
    y = days // 365
    return f"{y} year{'s' if y != 1 else ''} ago"


@app.command("whats-new")
def whats_new_cmd(
    days: int = typer.Option(
        7, "--days", help="Show papers added in the last N days (default 7)."
    ),
    since: Optional[str] = typer.Option(
        None,
        "--since",
        help="Show papers added on or after this date (YYYY-MM-DD). "
        "Overrides --days.",
    ),
    limit: Optional[int] = typer.Option(None, "--limit", help="Max rows."),
) -> None:
    """Show papers recently added to your library.

    Examples:
      bibwizard whats-new                  # last 7 days
      bibwizard whats-new --days 30        # last 30 days
      bibwizard whats-new --since 2026-01-01
    """
    from datetime import datetime, timedelta
    from rich.table import Table

    init_db()
    if since:
        try:
            cutoff = datetime.strptime(since, "%Y-%m-%d")
        except ValueError:
            error(f"--since must be YYYY-MM-DD, got {since!r}.")
            raise typer.Exit(code=2)
        label = f"since {cutoff:%Y-%m-%d}"
    else:
        if days < 1:
            error("--days must be ≥ 1.")
            raise typer.Exit(code=2)
        cutoff = datetime.utcnow() - timedelta(days=days)
        label = f"last {days} day{'s' if days != 1 else ''}"

    with session_scope() as session:
        papers = list_recent_papers(session, since=cutoff, limit=limit)
        rows = [
            {
                "id": p.id,
                "title": p.title,
                "authors": ", ".join(a.name for a in p.authors[:3])
                + (" …" if len(p.authors) > 3 else ""),
                "year": p.year,
                "created_at": p.created_at,
            }
            for p in papers
        ]

    if not rows:
        warn(f"No papers added in the {label}.")
        return

    now = datetime.utcnow()
    table = Table(
        title=f"What's new — {label} ({len(rows)} paper{'s' if len(rows)!=1 else ''})",
        show_lines=False,
        header_style="bold cyan",
    )
    table.add_column("ID", justify="right", style="dim", no_wrap=True)
    table.add_column("Added", no_wrap=True)
    table.add_column("Title", overflow="fold")
    table.add_column("Authors", overflow="fold")
    table.add_column("Year", justify="right")
    for r in rows:
        ts = r["created_at"]
        added = f"{ts:%Y-%m-%d} [dim]({_humanize_since(now, ts)})[/]"
        table.add_row(
            str(r["id"]),
            added,
            r["title"] or "(untitled)",
            r["authors"] or "",
            str(r["year"] or ""),
        )
    console.print(table)


@app.command("tag")
def tag_cmd(
    paper_id: int = typer.Argument(..., help="Paper id."),
    tags: list[str] = typer.Argument(..., help="One or more tags to add."),
) -> None:
    """Tag a paper."""
    init_db()
    with session_scope() as session:
        try:
            paper = add_tags(session, paper_id, tags)
        except LookupError as e:
            error(str(e))
            raise typer.Exit(code=2) from None
        success(f"Tags on paper {paper.id}: {', '.join(t.name for t in paper.tags)}")


# ---------- PDF inspection ----------

@app.command("inspect")
def inspect_cmd(
    paper_id: int = typer.Argument(..., help="Paper id from `bibwizard list`."),
    show_lines: bool = typer.Option(
        False, "--lines", help="Also dump every detected page-1/page-2 line with font size."
    ),
    refs: bool = typer.Option(
        False,
        "--refs",
        help="Dump the bibliography pipeline (raw section text, splitter output, "
        "stored Citation rows) instead of front matter.",
    ),
) -> None:
    """Show what the PDF structure scraper extracted from a paper.

    Useful for debugging when authors / abstract come back empty after ingest,
    or — with `--refs` — when reference parsing looks off.
    """
    init_db()
    with session_scope() as session:
        paper = session.get(Paper, paper_id)
        if paper is None:
            error(f"No paper with id {paper_id}")
            raise typer.Exit(code=2)
        file_path = paper.file_path

    if not file_path or not Path(file_path).exists():
        error(f"PDF file missing on disk: {file_path}")
        raise typer.Exit(code=2)

    if refs:
        _inspect_refs(paper_id, Path(file_path))
        return

    info(f"Scraping {file_path}...")
    fm = extract_front_matter(Path(file_path))
    heuristic_authors = parse_authors_from_byline(fm.byline_text)

    panel(
        "Front matter",
        f"  title:        {fm.title or '(empty)'}\n"
        f"  title font:   {fm.title_size:.1f}pt\n"
        f"  title lines:  {len(fm.title_lines)}\n\n"
        f"  byline text:  {(fm.byline_text or '(empty)')[:300]}\n"
        f"  byline font:  {fm.byline_size:.1f}pt\n\n"
        f"  abstract:     {(fm.abstract or '(empty)')[:300]}"
        + ("..." if len(fm.abstract) > 300 else ""),
    )

    panel(
        "Heuristic byline parser",
        "  " + ("\n  ".join(heuristic_authors) if heuristic_authors else "(no authors parsed)"),
        style="cyan",
    )

    # Show every cluster the byline picker considered + its score, so we can
    # see why a particular cluster was chosen or rejected.
    from bibwizard.ingestion.structure import (
        _byline_clusters,
        _byline_score,
        _name_ratio,
    )

    title_line_objs = [ln for ln in fm.page_lines if ln.text in fm.title_lines]
    clusters = _byline_clusters(fm.page_lines, title_line_objs, fm.title_size)
    if clusters:
        rows = []
        for i, c in enumerate(clusters):
            text = " ".join(ln.text for ln in c)
            score = _byline_score(text)
            ratio, n = _name_ratio(text)
            rows.append(
                f"  [{i}] size~{c[0].size:.1f}pt  score={score:+.2f}  "
                f"name_ratio={ratio:.2f}({n} pieces)\n"
                f"      {text[:180]}"
            )
        panel(
            f"Byline cluster scoring ({len(clusters)} candidate(s))",
            "\n\n".join(rows),
            style="magenta",
        )

    if show_lines:
        rows = "\n".join(
            f"  p{ln.page} y={ln.y:6.1f} size={ln.size:5.2f}  {ln.text[:90]}"
            for ln in fm.page_lines[:80]
        )
        panel(
            f"First 80 page-1/2 lines (of {len(fm.page_lines)})",
            rows or "(none)",
            style="dim",
        )


def _inspect_refs(paper_id: int, pdf_path: Path) -> None:
    """Dump everything about how this paper's bibliography was processed.

    Shows three layers of the pipeline so we can see where it broke:
      1. Did we even FIND the References section? (raw section text length)
      2. Did the SPLITTER produce sensible entries? (count + samples)
      3. What did the per-line FIELD EXTRACTOR pull from each entry, and
         what's actually stored in the citations table?
    """
    info(f"Re-parsing {pdf_path} to inspect bibliography...")
    parsed = parse_pdf(pdf_path)

    # 1) Bibliography section split
    refs_blob = parsed.references or ""
    panel(
        "Bibliography section split (from parser._split_at_references)",
        f"  raw section length: {len(refs_blob)} chars\n"
        f"  first 800 chars:\n"
        f"{(refs_blob[:800] or '(no References header found in PDF)')}",
        style="dim",
    )

    # 2) Re-run the splitter on this raw blob
    entries = split_references(refs_blob)
    sample = "\n\n".join(
        f"  [{i}] ({len(e)} chars)  {e[:200]}{'…' if len(e) > 200 else ''}"
        for i, e in enumerate(entries[:6])
    )
    panel(
        f"split_references() output — {len(entries)} entries",
        sample or "  (splitter produced 0 entries)",
        style="cyan",
    )

    # 3) Re-run the per-line extractor on the splitter output (live, with no
    # DB involvement) so we can compare it to what's in the DB.
    fresh = [parse_reference_line(e) for e in entries]
    n = len(fresh)
    n_doi = sum(1 for r in fresh if r.get("doi"))
    n_arx = sum(1 for r in fresh if r.get("arxiv_id"))
    n_yr = sum(1 for r in fresh if r.get("year"))
    n_ti = sum(1 for r in fresh if r.get("title"))
    panel(
        "parse_reference_line() field extraction — coverage on the SPLIT output",
        f"  total entries:       {n}\n"
        f"  with DOI extracted:  {n_doi} ({100*n_doi//n if n else 0}%)\n"
        f"  with arXiv extracted:{n_arx} ({100*n_arx//n if n else 0}%)\n"
        f"  with year extracted: {n_yr} ({100*n_yr//n if n else 0}%)\n"
        f"  with title guess:    {n_ti} ({100*n_ti//n if n else 0}%)",
        style="cyan",
    )
    sample_rows = []
    for i, r in enumerate(fresh[:5]):
        sample_rows.append(
            f"  [{i}] doi={r.get('doi') or '-'}\n"
            f"      arxiv={r.get('arxiv_id') or '-'}  year={r.get('year') or '-'}\n"
            f"      title={(r.get('title') or '-')[:120]}\n"
            f"      raw={r['raw_text'][:160]}"
        )
    panel("First 5 fresh-extractor rows", "\n\n".join(sample_rows) or "(none)", style="dim")

    # 4) What's actually stored in the DB for this paper
    with session_scope() as s:
        paper = s.get(Paper, paper_id)
        rows = list(paper.outgoing_citations)
        db_total = len(rows)
        db_doi = sum(1 for r in rows if r.target_doi)
        db_arx = sum(1 for r in rows if r.target_arxiv_id)
        db_ti = sum(1 for r in rows if r.target_title)
        db_yr = sum(1 for r in rows if r.target_year)
        db_linked = sum(1 for r in rows if r.target_paper_id is not None)
        # Snapshot first 5 raw_text values + their stored fields
        db_sample = []
        for i, r in enumerate(rows[:5]):
            db_sample.append(
                f"  [{i}] doi={r.target_doi or '-'}  arxiv={r.target_arxiv_id or '-'}  "
                f"year={r.target_year or '-'}  linked_to_paper_id={r.target_paper_id or '-'}\n"
                f"      title={(r.target_title or '-')[:120]}\n"
                f"      raw={r.raw_text[:160]}"
            )
    panel(
        "Stored Citation rows (DB)",
        f"  total rows:          {db_total}\n"
        f"  with DOI:            {db_doi} ({100*db_doi//db_total if db_total else 0}%)\n"
        f"  with arXiv id:       {db_arx} ({100*db_arx//db_total if db_total else 0}%)\n"
        f"  with title:          {db_ti} ({100*db_ti//db_total if db_total else 0}%)\n"
        f"  with year:           {db_yr} ({100*db_yr//db_total if db_total else 0}%)\n"
        f"  linked to a paper:   {db_linked}",
        style="magenta",
    )
    panel("First 5 DB Citation rows", "\n\n".join(db_sample) or "(none)", style="dim")

    # 5) Sanity: do the splitter and the DB agree on count?
    if n != db_total:
        warn(
            f"  split_references() says {n} entries but DB stores {db_total}. "
            f"Re-ingest may have used a different splitter, or the parser "
            f"didn't capture the same References section."
        )


# ---------- manual editing & API enrichment ----------

@app.command("edit")
def edit_cmd(
    paper_id: int = typer.Argument(..., help="Paper id from `bibwizard list`."),
    title: Optional[str] = typer.Option(None, "--title", help="Set title."),
    authors: Optional[str] = typer.Option(
        None,
        "--authors",
        help="Set authors. Pass a string like 'Last, F.; Other, A.' (separator: ; or |).",
    ),
    year: Optional[int] = typer.Option(None, "--year", help="Set publication year."),
    doi: Optional[str] = typer.Option(None, "--doi", help="Set DOI."),
    arxiv_id: Optional[str] = typer.Option(None, "--arxiv", help="Set arXiv id."),
    venue: Optional[str] = typer.Option(None, "--venue", help="Set publication venue / journal."),
    abstract: Optional[str] = typer.Option(None, "--abstract", help="Set abstract."),
    replace_authors: bool = typer.Option(
        True,
        "--replace-authors/--append-authors",
        help="Replace existing authors (default) vs. append to them.",
    ),
) -> None:
    """Manually edit a paper's metadata. Useful when LLM extraction misses fields."""
    init_db()
    if all(
        v is None
        for v in (title, authors, year, doi, arxiv_id, venue, abstract)
    ):
        error("Pass at least one of --title / --authors / --year / --doi / --arxiv / --venue / --abstract.")
        raise typer.Exit(code=2)

    changed: list[str] = []
    with session_scope() as session:
        paper = session.get(Paper, paper_id)
        if paper is None:
            error(f"No paper with id {paper_id}")
            raise typer.Exit(code=2)

        if title is not None:
            paper.title = " ".join(title.split())
            changed.append("title")
        if year is not None:
            paper.year = year
            changed.append("year")
        if doi is not None:
            paper.doi = doi.strip() or None
            changed.append("doi")
        if arxiv_id is not None:
            paper.arxiv_id = arxiv_id.strip() or None
            changed.append("arxiv_id")
        if venue is not None:
            paper.venue = venue.strip() or None
            changed.append("venue")
        if abstract is not None:
            paper.abstract = abstract.strip() or None
            changed.append("abstract")

        if authors is not None:
            # Accept ; or | as separators (commas are inside names like "Last, F.")
            names = [n.strip() for n in re.split(r"[;|]", authors) if n.strip()]
            if replace_authors:
                # SQLAlchemy: clearing the list emits the right delete on the join table.
                paper.authors.clear()
            existing = {a.name for a in paper.authors}
            n_added = 0
            for n in names:
                if n in existing:
                    continue
                paper.authors.append(get_or_create_author(session, n))
                existing.add(n)
                n_added += 1
            changed.append(f"authors[{n_added}]")
        session.flush()
    success(f"Updated paper {paper_id}: {', '.join(changed)}")


def _confirm_or_pick(candidates: list[dict], original_title: str, *, yes: bool) -> dict | None:
    """Show ranked candidates and let the user pick. Auto-accept if --yes and
    there's a single high-confidence match.

    Each candidate is a dict with keys: source ('arxiv'/'ads'), title, authors,
    year, doi, arxiv_id, venue, score.
    """
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)
    best = candidates[0]

    # Auto-accept the top hit if confidence is reasonable.
    if yes and best.get("score", 0) >= 0.70:
        return best

    panel("Candidates", "\n".join(
        f"  [{i}] ({c['source']:>5}, score={c.get('score', 0):.2f}) "
        f"{c.get('title', '')[:90]}\n"
        f"        {', '.join(c.get('authors', [])[:4])}"
        f"{' …' if len(c.get('authors', [])) > 4 else ''}"
        f"  | {c.get('year')}"
        + (f" | doi:{c.get('doi')}" if c.get('doi') else "")
        + (f" | arxiv:{c.get('arxiv_id')}" if c.get('arxiv_id') else "")
        for i, c in enumerate(candidates)
    ), style="cyan")

    if yes:
        info(f"--yes: top match score={best.get('score', 0):.2f} below 0.70, refusing to auto-pick.")
        return None
    raw = typer.prompt(
        "Apply which candidate? [0-N or 'n' to cancel]", default="0"
    ).strip().lower()
    if raw in {"n", "no", "cancel", ""}:
        return None
    try:
        idx = int(raw)
        return candidates[idx]
    except (ValueError, IndexError):
        warn("Invalid selection.")
        return None


@app.command("enrich")
def enrich_cmd(
    paper_id: int = typer.Argument(..., help="Paper id whose metadata to enrich."),
    title: Optional[str] = typer.Option(
        None, "--title", help="Override title used for the search (default: paper's current title)."
    ),
    author: Optional[str] = typer.Option(
        None,
        "--author",
        help="Filter the search by an author's surname (improves disambiguation).",
    ),
    year: Optional[int] = typer.Option(
        None, "--year", help="Filter the search by year (defaults to paper's year)."
    ),
    source: str = typer.Option(
        "auto", "--source", help="Where to look: 'arxiv', 'ads', or 'auto' (both, ADS only if token set)."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Auto-apply when confidence is high (>=0.85)."
    ),
) -> None:
    """Search arXiv (and NASA ADS) by title/author/year and backfill metadata."""
    init_db()
    with session_scope() as session:
        paper = session.get(Paper, paper_id)
        if paper is None:
            error(f"No paper with id {paper_id}")
            raise typer.Exit(code=2)
        cur_title = title or paper.title or ""
        cur_year = year if year is not None else paper.year
        cur_authors = [a.name for a in paper.authors]

    if not cur_title:
        error("No title to search by. Pass --title explicitly.")
        raise typer.Exit(code=2)

    src = source.lower()
    if src not in {"arxiv", "ads", "auto"}:
        error("--source must be one of: arxiv | ads | auto")
        raise typer.Exit(code=2)

    candidates: list[dict] = []

    # arXiv search (always free + unauthenticated)
    if src in {"arxiv", "auto"}:
        info(f"Searching arXiv for: title~={cur_title!r} year={cur_year} author~={author!r}")
        try:
            arx = search_arxiv(title=cur_title, author=author, year=cur_year, max_results=5)
        except Exception as e:  # noqa: BLE001
            warn(f"arXiv search failed: {e}")
            arx = []
        for c in arx:
            candidates.append({
                "source": "arxiv",
                "title": c.title,
                "authors": list(c.authors),
                "year": c.year,
                "doi": c.doi,
                "arxiv_id": c.arxiv_id,
                "venue": None,
                "score": title_similarity(cur_title, c.title),
            })

    # NASA ADS search (optional, needs token)
    if src in {"ads", "auto"}:
        if not ads_mod.is_configured():
            if src == "ads":
                error("ADS_API_TOKEN is not set. Get a free token at https://ui.adsabs.harvard.edu/user/settings/token")
                raise typer.Exit(code=2)
            info("ADS not configured (set ADS_API_TOKEN to also query NASA ADS).")
        else:
            info(f"Searching NASA ADS for: title~={cur_title!r} year={cur_year} author~={author!r}")
            try:
                ads_hits = ads_mod.search_ads(
                    title=cur_title, author=author, year=cur_year, max_results=5
                )
            except Exception as e:  # noqa: BLE001
                warn(f"ADS search failed: {e}")
                ads_hits = []
            for c in ads_hits:
                candidates.append({
                    "source": "ads",
                    "title": c.title,
                    "authors": list(c.authors),
                    "year": c.year,
                    "doi": c.doi,
                    "arxiv_id": c.arxiv_id,
                    "venue": c.venue,
                    "score": title_similarity(cur_title, c.title),
                })

    if not candidates:
        warn("No matches.")
        raise typer.Exit(code=1)

    chosen = _confirm_or_pick(candidates, cur_title, yes=yes)
    if chosen is None:
        info("No match applied.")
        return

    panel(
        "Applying",
        f"  source:  {chosen['source']}\n"
        f"  title:   {chosen.get('title', '')}\n"
        f"  authors: {', '.join(chosen.get('authors', []))}\n"
        f"  year:    {chosen.get('year')}\n"
        f"  doi:     {chosen.get('doi')}\n"
        f"  arxiv:   {chosen.get('arxiv_id')}\n"
        f"  venue:   {chosen.get('venue')}",
        style="green",
    )

    with session_scope() as session:
        paper = session.get(Paper, paper_id)
        if paper is None:
            error("Paper disappeared mid-flight?")
            raise typer.Exit(code=2)
        # Title: replace if existing was weak; otherwise leave alone.
        from bibwizard.database.queries import _looks_weak_title

        if chosen.get("title") and _looks_weak_title(paper.title):
            paper.title = " ".join(chosen["title"].split())
        if chosen.get("year") is not None and paper.year is None:
            paper.year = chosen["year"]
        if chosen.get("doi") and not paper.doi:
            paper.doi = chosen["doi"]
        if chosen.get("arxiv_id") and not paper.arxiv_id:
            paper.arxiv_id = chosen["arxiv_id"]
        if chosen.get("venue") and not paper.venue:
            paper.venue = chosen["venue"]
        # Authors: only attach if we have nothing locally.
        if chosen.get("authors") and not paper.authors:
            for n in chosen["authors"]:
                n = " ".join((n or "").split())
                if not n:
                    continue
                paper.authors.append(get_or_create_author(session, n))
        session.flush()

    if not cur_authors and chosen.get("authors"):
        success(f"Backfilled {len(chosen['authors'])} author(s) and other fields.")
    else:
        success("Enrichment applied.")


# ---------- destructive commands ----------

def _is_under(path: Path, parent: Path) -> bool:
    """True if `path` is the same as or lives inside `parent` (resolved)."""
    try:
        path = path.resolve()
        parent = parent.resolve()
    except OSError:
        return False
    return path == parent or parent in path.parents


@app.command("remove")
def remove_cmd(
    paper_id: int = typer.Argument(..., help="Paper id from `bibwizard list`."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
    keep_pdf: bool = typer.Option(
        False, "--keep-pdf", help="Keep the PDF file in ~/.bibwizard/papers/."
    ),
) -> None:
    """Remove a single paper (DB row, vector chunks, summary file, and library PDF)."""
    init_db()
    with session_scope() as session:
        paper = session.get(Paper, paper_id)
        if paper is None:
            error(f"No paper with id {paper_id}")
            raise typer.Exit(code=2)
        title = paper.title
        file_path = paper.file_path
        n_chunks = paper.n_chunks
        summary_path = paper.summary_path

    panel(
        f"About to remove paper id={paper_id}",
        f"  title:   {title}\n"
        f"  pdf:     {file_path or '—'}\n"
        f"  chunks:  {n_chunks}\n"
        f"  summary: {'yes' if summary_path else 'no'}\n"
        f"  (your literature/ folder is NOT touched)",
        style="yellow",
    )
    if not yes and not typer.confirm("Continue?", default=False):
        info("Cancelled.")
        raise typer.Exit()

    n_deleted = delete_paper_chunks(paper_id)
    info(f"Removed {n_deleted} vector chunk(s) from ChromaDB")

    # Try the DB-tracked path first, then fall back to our naming convention.
    sp_candidates = []
    if summary_path:
        sp_candidates.append(Path(summary_path))
    sp_candidates.append(settings.summaries_dir / f"paper_{paper_id}.json")
    for sp in sp_candidates:
        if sp.is_file():
            try:
                sp.unlink()
                info(f"Removed summary file {sp.name}")
            except OSError as e:
                warn(f"Could not remove summary file: {e}")
            break

    if not keep_pdf and file_path:
        fp = Path(file_path)
        if fp.is_file() and _is_under(fp, settings.papers_dir):
            try:
                fp.unlink()
                info(f"Removed library PDF {fp.name}")
            except OSError as e:
                warn(f"Could not remove PDF: {e}")
        elif fp.is_file():
            info(f"Left PDF in place (outside library): {fp}")

    with session_scope() as session:
        delete_paper(session, paper_id)
        delete_orphan_authors(session)
        delete_orphan_tags(session)
    success(f"Paper id={paper_id} removed.")


@app.command("duplicates")
def duplicates_cmd(
    threshold: float = typer.Option(
        0.97,
        "--threshold", "-t",
        help="Cosine similarity threshold for content-based dupe detection.",
    ),
    remove: bool = typer.Option(
        False,
        "--remove",
        help="Interactively keep one paper per group and remove the others.",
    ),
    auto_remove: bool = typer.Option(
        False,
        "--auto-remove",
        help="Keep the LOWEST id in each strong-tier (doi/arxiv) group and remove the rest. Implies --yes.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip per-paper confirmation."),
) -> None:
    """Find (and optionally remove) duplicate papers.

    Detection runs through four signal tiers, ordered strongest to weakest:
      1. DOI match  — same DOI = same paper
      2. arXiv id match (version-stripped)
      3. Same normalized title + first-author surname
      4. Content similarity (mean chunk-embedding cosine ≥ threshold)

    Without flags it just LISTS duplicate groups. Add `--remove` to dedupe
    interactively, or `--auto-remove` to wipe everything but the lowest-id
    paper in each strong-tier (doi/arxiv) group automatically.
    """
    from bibwizard.database.queries import find_duplicate_groups
    from bibwizard.ingestion.embedder import delete_paper_chunks

    init_db()
    with session_scope() as session:
        groups = find_duplicate_groups(session, content_threshold=threshold)

    if not groups:
        success("No duplicates found.")
        return

    tier_color = {"doi": "red", "arxiv": "red", "title": "yellow", "content": "cyan"}
    info(f"Found {len(groups)} duplicate group(s).")
    print()
    for i, group in enumerate(groups, 1):
        c = tier_color.get(group["tier"], "white")
        console.print(
            f"[bold {c}]Group {i}  — tier: {group['tier']}  — {len(group['members'])} papers[/]"
        )
        for m in group["members"]:
            fn = (m["file_path"] or "").rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
            console.print(
                f"  id={m['id']:>3}  year={m['year'] or '?'}  "
                f"{m['title'][:75]!r}"
            )
            console.print(f"        file: {fn}")
            console.print(f"        doi:  {m['doi']}    arxiv: {m['arxiv_id']}")
        for r in group["reasons"][:3]:
            console.print(f"  [dim]  reason: {r}[/]")
        print()

    if not (remove or auto_remove):
        info("Pass --remove for interactive dedup, or --auto-remove for strong-tier auto-keep-lowest-id.")
        return

    # ---------- removal phase ----------
    n_removed = 0
    for group in groups:
        members = group["members"]
        keep_id: int | None = None
        if auto_remove:
            if group["tier"] not in {"doi", "arxiv"}:
                info(f"Skipping group (tier={group['tier']}) — auto-remove only acts on doi/arxiv.")
                continue
            keep_id = members[0]["id"]   # lowest id
            console.print(f"\n[bold green]auto: keep id={keep_id}, remove rest[/]")
        else:
            console.print()
            choices = ", ".join(str(m["id"]) for m in members)
            raw = typer.prompt(
                f"Which paper id to KEEP from this group? ({choices}, 's' to skip)",
                default=str(members[0]["id"]),
            ).strip()
            if raw.lower() in {"s", "skip", "no", "n"}:
                continue
            try:
                keep_id = int(raw)
            except ValueError:
                warn(f"Bad input {raw!r}; skipping group.")
                continue
            if keep_id not in {m["id"] for m in members}:
                warn(f"{keep_id} isn't in the group; skipping.")
                continue

        for m in members:
            if m["id"] == keep_id:
                continue
            if not (yes or auto_remove):
                if not typer.confirm(f"  Remove paper id={m['id']} ({m['title'][:50]!r})?", default=True):
                    continue
            # Wipe chunks + summary file + PDF (if in library) + DB row
            try:
                n = delete_paper_chunks(m["id"])
                if n:
                    info(f"  removed {n} chunks for id={m['id']}")
            except Exception as e:  # noqa: BLE001
                warn(f"  chunk cleanup failed: {e}")
            # Summary file
            sp = Path(settings.summaries_dir) / f"paper_{m['id']}.json"
            if sp.is_file():
                try:
                    sp.unlink()
                except OSError as e:
                    warn(f"  could not remove summary file: {e}")
            # Library PDF (only if under papers_dir, never literature/)
            if m["file_path"]:
                fp = Path(m["file_path"])
                if fp.is_file() and _is_under(fp, settings.papers_dir):
                    try:
                        fp.unlink()
                    except OSError as e:
                        warn(f"  could not remove pdf: {e}")
            # DB row
            with session_scope() as session:
                delete_paper(session, m["id"])
                delete_orphan_authors(session)
                delete_orphan_tags(session)
            n_removed += 1
            success(f"  ✓ removed id={m['id']}")

    success(f"Done. Removed {n_removed} paper(s) total.")


@app.command("clean")
def clean_cmd(
    all_data: bool = typer.Option(
        False, "--all", help="Wipe EVERYTHING: papers, vectors, summaries, library PDFs."
    ),
    vectors: bool = typer.Option(
        False, "--vectors", help="Wipe just the ChromaDB vector store (DB rows kept)."
    ),
    summaries: bool = typer.Option(
        False, "--summaries", help="Wipe just summary JSON files (DB rows kept)."
    ),
    orphans: bool = typer.Option(
        False, "--orphans", help="Drop authors/tags that have no remaining papers."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Clean library state. Pass exactly one of --all / --vectors / --summaries / --orphans.

    Your literature/ drop folder is NEVER touched — only `~/.bibwizard/`.
    """
    init_db()
    chosen = sum([all_data, vectors, summaries, orphans])
    if chosen != 1:
        error("Pass exactly one of --all, --vectors, --summaries, --orphans.")
        raise typer.Exit(code=2)

    if all_data:
        with session_scope() as session:
            stats = library_stats(session)
        n_summary_files = (
            len(list(settings.summaries_dir.glob("*.json")))
            if settings.summaries_dir.exists()
            else 0
        )
        n_pdfs = (
            len([p for p in settings.papers_dir.glob("*") if p.is_file()])
            if settings.papers_dir.exists()
            else 0
        )
        panel(
            "DESTRUCTIVE — about to delete",
            f"  • {stats['papers']} papers (with citations, tags, authors)\n"
            f"  • {stats['vector_chunks']} vector chunks (entire ChromaDB collection)\n"
            f"  • {n_summary_files} summary JSON files\n"
            f"  • {n_pdfs} files in {settings.papers_dir}\n\n"
            f"NOT touched: {settings.literature_dir}",
            style="red",
        )
        if not yes:
            answer = typer.prompt('Type "yes" to confirm', default="")
            if answer.strip().lower() != "yes":
                info("Cancelled.")
                raise typer.Exit()

        reset_collection()
        success("Cleared ChromaDB collection")

        n = 0
        if settings.summaries_dir.exists():
            for f in settings.summaries_dir.glob("*.json"):
                try:
                    f.unlink()
                    n += 1
                except OSError:
                    pass
        success(f"Removed {n} summary file(s)")

        n = 0
        if settings.papers_dir.exists():
            for f in settings.papers_dir.glob("*"):
                if f.is_file():
                    try:
                        f.unlink()
                        n += 1
                    except OSError:
                        pass
        success(f"Removed {n} file(s) from {settings.papers_dir}")

        with session_scope() as session:
            deleted = delete_all_papers(session)
            delete_orphan_authors(session)
            delete_orphan_tags(session)
        success(f"Removed {deleted} paper row(s) from database. Library is empty.")
        return

    if vectors:
        if not yes and not typer.confirm(
            "Wipe ChromaDB vector store? (DB rows kept; you'll need to re-embed.)",
            default=False,
        ):
            info("Cancelled.")
            raise typer.Exit()
        reset_collection()
        with session_scope() as session:
            n = reset_chunk_counts(session)
        success(
            f"Cleared ChromaDB. Reset n_chunks=0 on {n} paper(s). "
            f"Re-embed via `bibwizard scan` or by re-adding."
        )
        return

    if summaries:
        if not yes and not typer.confirm(
            "Wipe summary JSON files and clear summary columns?", default=False
        ):
            info("Cancelled.")
            raise typer.Exit()
        n = 0
        if settings.summaries_dir.exists():
            for f in settings.summaries_dir.glob("*.json"):
                try:
                    f.unlink()
                    n += 1
                except OSError:
                    pass
        with session_scope() as session:
            reset_summary_columns(session)
        success(
            f"Removed {n} summary file(s). "
            f"Re-run via `bibwizard resummarize <id>`."
        )
        return

    if orphans:
        if not yes and not typer.confirm(
            "Drop authors/tags that have no remaining papers?", default=False
        ):
            info("Cancelled.")
            raise typer.Exit()
        with session_scope() as session:
            na = delete_orphan_authors(session)
            nt = delete_orphan_tags(session)
        success(f"Dropped {na} orphan author(s), {nt} orphan tag(s).")


@map_app.command("ref")
def map_ref(
    fmt: str = typer.Option("dot", "--format", "-f", help="dot | json"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output path."),
) -> None:
    """Export the citation graph."""
    init_db()
    fmt = fmt.lower()
    default_name = "references.dot" if fmt == "dot" else "references.json"
    target = (out or (settings.home / default_name)).expanduser().resolve()
    if fmt == "dot":
        path = export_dot(target)
    elif fmt == "json":
        path = export_json(target)
    else:
        error(f"Unknown format: {fmt}")
        raise typer.Exit(code=2)
    success(f"Wrote {path}")


@map_app.command("content")
def map_content(
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output path."),
    n_clusters: Optional[int] = typer.Option(
        None, "--clusters", "-n", help="Force a specific cluster count (JSON only)."
    ),
    fmt: str = typer.Option(
        "json", "--format", "-f",
        help="json (cluster summary) or html (interactive force graph).",
    ),
    threshold: float = typer.Option(
        0.65, "--threshold", "-t",
        help="Cosine-similarity threshold for edges in the HTML graph.",
    ),
) -> None:
    """Cluster papers semantically — JSON or interactive HTML graph."""
    init_db()
    fmt = fmt.lower()
    if fmt == "json":
        target = (out or (settings.home / "content_map.json")).expanduser().resolve()
        path = export_clusters(target, n_clusters=n_clusters)
        success(f"Wrote {path}")
        return
    if fmt == "html":
        from bibwizard.context.visualize import (
            build_content_graph,
            render_content_graph_html,
        )

        target = (out or (settings.home / "content_map.html")).expanduser().resolve()
        info(f"Building content-similarity graph (threshold={threshold})...")
        g = build_content_graph(threshold=threshold)
        path = render_content_graph_html(g, target, threshold=threshold)
        success(
            f"Wrote {path}  ({g.number_of_nodes()} papers, {g.number_of_edges()} edges)"
        )
        return
    error(f"Unknown format: {fmt}. Use 'json' or 'html'.")
    raise typer.Exit(code=2)


@map_app.command("authors")
def map_authors(
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Output path (default: ~/.bibwizard/author_map.html)."
    ),
    fmt: str = typer.Option(
        "html", "--format", "-f", help="html (interactive) or json (raw graph)."
    ),
) -> None:
    """Build the co-authorship graph: papers connected by shared authors."""
    init_db()
    from bibwizard.context.visualize import (
        build_author_graph,
        render_author_graph_html,
    )

    info("Building co-authorship graph...")
    g = build_author_graph()
    fmt = fmt.lower()
    if fmt == "html":
        target = (out or (settings.home / "author_map.html")).expanduser().resolve()
        path = render_author_graph_html(g, target)
        success(
            f"Wrote {path}  ({g.number_of_nodes()} papers, {g.number_of_edges()} co-authorship edges)"
        )
    elif fmt == "json":
        import json as _json

        target = (out or (settings.home / "author_map.json")).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": [{"id": n, **{k: v for k, v in d.items() if k != "id"}} for n, d in g.nodes(data=True)],
            "edges": [{"source": u, "target": v, **d} for u, v, d in g.edges(data=True)],
        }
        target.write_text(_json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        success(f"Wrote {target}")
    else:
        error(f"Unknown format: {fmt}. Use 'html' or 'json'.")
        raise typer.Exit(code=2)


@app.command("export")
def export_cmd(
    fmt: str = typer.Option("bibtex", "--format", "-f", help="Currently: bibtex"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output file path."),
) -> None:
    """Export the library to a citation format (BibTeX)."""
    init_db()
    if fmt.lower() != "bibtex":
        error("Only --format bibtex is supported right now.")
        raise typer.Exit(code=2)

    target = (out or (settings.home / "library.bib")).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    with session_scope() as session:
        papers = list_papers(session)
        entries = []
        for p in papers:
            key = _bibtex_key(p)
            entry = [f"@article{{{key},"]
            entry.append(f"  title = {{{_escape_bib(p.title)}}},")
            if p.authors:
                authors = " and ".join(a.name for a in p.authors)
                entry.append(f"  author = {{{_escape_bib(authors)}}},")
            if p.year:
                entry.append(f"  year = {{{p.year}}},")
            if p.doi:
                entry.append(f"  doi = {{{p.doi}}},")
            if p.arxiv_id:
                entry.append(f"  eprint = {{{p.arxiv_id}}},")
                entry.append("  archivePrefix = {arXiv},")
            if p.venue:
                entry.append(f"  journal = {{{_escape_bib(p.venue)}}},")
            entry.append("}")
            entries.append("\n".join(entry))

    target.write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    success(f"Wrote {len(entries)} entries → {target}")


def _bibtex_key(paper) -> str:
    last = "anon"
    if paper.authors:
        last = paper.authors[0].name.split()[-1]
    year = paper.year or "nd"
    word = ""
    for tok in (paper.title or "").split():
        clean = "".join(c for c in tok if c.isalnum())
        if clean and clean.lower() not in {"a", "an", "the", "on", "of", "in"}:
            word = clean.lower()
            break
    return f"{last.lower()}{year}{word}".replace(" ", "")


def _escape_bib(s: str) -> str:
    return (s or "").replace("{", "(").replace("}", ")")


@app.command("stats")
def stats_cmd() -> None:
    """Show library stats."""
    init_db()
    with session_scope() as session:
        s = library_stats(session)
    flat = {
        "papers": s["papers"],
        "authors": s["authors"],
        "tags": s["tags"],
        "citations": s["citations"],
        "vector_chunks": s["vector_chunks"],
    }
    console.print(stats_table(flat))
    if s["by_year"]:
        rows = "\n".join(f"  {y}: {c}" for y, c in s["by_year"])
        panel("Papers by year", rows, style="dim")


@app.command("grep")
def grep_cmd(query: str = typer.Argument(..., help="Substring search across title/abstract.")) -> None:
    """Literal substring search across title/abstract (no embedding required).

    Use this when you know an exact phrase from a paper's title or abstract.
    For semantic / topical search ("papers about PIAA nullers"), use
    `bibwizard find` instead.
    """
    init_db()
    with session_scope() as session:
        papers = text_search(session, query)
        rows = [
            {
                "id": p.id,
                "title": p.title,
                "authors": ", ".join(a.name for a in p.authors[:3]),
                "year": p.year,
                "tags": ", ".join(t.name for t in p.tags),
            }
            for p in papers
        ]
    if not rows:
        warn("No matches.")
        return
    console.print(papers_table(rows))


@app.command("find")
def find_cmd(
    query: str = typer.Argument(..., help="What kind of paper to find (free-text topic / description)."),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of papers to return."),
    chunks_per_paper: int = typer.Option(
        3,
        "--chunks-per-paper",
        help="How many of each paper's best chunks contribute to its score. "
        "Higher = rewards papers with many medium-relevance passages.",
    ),
    pool_size: int = typer.Option(
        50,
        "--pool",
        help="Width of the chunk-retrieval pool before paper-level aggregation. "
        "Raise for very large libraries.",
    ),
    why: bool = typer.Option(
        False,
        "--why",
        help="Add a one-line LLM-generated 'why this matched' blurb per result.",
    ),
) -> None:
    """Find papers semantically related to a topic, ranked at the paper level.

    Unlike `bibwizard search`, which returns individual chunks, this returns
    one row per PAPER, ordered by how well the paper as a whole matches.
    """
    from bibwizard.search.paper_search import find_papers
    from rich.table import Table

    init_db()
    try:
        get_client().ensure_ready(need_embed=True, need_llm=why)
    except (OllamaUnavailable, OllamaModelMissing) as e:
        error(str(e))
        raise typer.Exit(code=2) from None

    hits = find_papers(
        query,
        top_k=top_k,
        chunks_per_paper=chunks_per_paper,
        pool_size=pool_size,
    )
    if not hits:
        warn("No matches.")
        return

    # Optional LLM 'why' blurbs — one cheap call per row.
    reasons: dict[int, str] = {}
    if why:
        from bibwizard.llm.client import ChatMessage
        client = get_client()
        sys_msg = (
            "You are a concise research assistant. Given a search query and a "
            "single excerpt from a paper, explain in ONE short sentence why "
            "this paper matches the query. No preamble, no 'this paper...', "
            "just the reason. Maximum 20 words."
        )
        for h in hits:
            user_msg = (
                f"Query: {query}\n\n"
                f"Paper: {h.cite} — {h.title}\n"
                f"Excerpt: {h.best_snippet}"
            )
            try:
                parts = []
                for token in client.chat(
                    [ChatMessage("system", sys_msg), ChatMessage("user", user_msg)],
                    stream=True,
                ):
                    parts.append(token)
                reasons[h.paper_id] = "".join(parts).strip().splitlines()[0][:140]
            except Exception:  # noqa: BLE001
                reasons[h.paper_id] = ""

    table = Table(title=f"Papers matching: {query!r}", show_lines=True)
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("cite", style="cyan", no_wrap=True)
    table.add_column("title", style="bold")
    table.add_column("score", justify="right", style="green", width=6)
    table.add_column("p.", justify="right", style="dim", width=4)
    table.add_column("snippet" if not why else "why / snippet")

    for i, h in enumerate(hits, start=1):
        snippet = h.best_snippet
        if why:
            reason = reasons.get(h.paper_id, "")
            snippet = f"[italic green]{reason}[/]\n[dim]{h.best_snippet}[/]"
        cite_cell = f"[paper {h.paper_id}]\n{h.cite}"
        table.add_row(
            str(i),
            cite_cell,
            h.title[:90],
            f"{h.score:.3f}",
            str(h.best_page) if h.best_page > 0 else "?",
            snippet,
        )
    console.print(table)


@app.command("cite")
def cite_cmd(
    claim: str = typer.Argument(..., help="The exact statement to find a citation for."),
    max_results: int = typer.Option(5, "--max", "-n", help="Max papers to return."),
    pool_size: int = typer.Option(
        20,
        "--pool",
        help="Candidate chunks to check via LLM entailment. Higher = more "
        "thorough, slower. Each chunk = one LLM call.",
    ),
    min_confidence: float = typer.Option(
        0.5,
        "--min-confidence",
        help="Reject hits below this LLM-reported confidence.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Print the entailment verdict for EVERY candidate chunk (kept or "
        "rejected). Use when a claim isn't matching and you want to see why.",
    ),
    dump_passages: bool = typer.Option(
        False,
        "--dump-passages",
        help="In addition to --debug, dump the FULL passage text of every "
        "candidate chunk (not just the preview shown in the table). Use when "
        "the table preview is cut off mid-sentence and you need to see what "
        "the entailment LLM actually had access to.",
    ),
    no_rerank: bool = typer.Option(
        False,
        "--no-rerank",
        help="Disable the reranker (skip the cross-encoder / lexical reordering "
        "between ChromaDB and entailment). Use to A/B-test whether rerank is "
        "helping for a given claim.",
    ),
    max_per_paper: int = typer.Option(
        0,
        "--max-per-paper",
        help="Cap how many chunks from a single paper can sit in the "
        "entailment pool. Default 0 = use the setting (3). Use 0 or a "
        "high number to disable the cap; lower it (e.g. 1) to force max "
        "diversity across papers.",
    ),
    tex: bool = typer.Option(
        False,
        "--tex",
        help="After the results table, print a LaTeX block with a "
        r"\citep{...} macro and the BibTeX entries for every paper found. "
        "Designed for paste-into-manuscript natbib/biblatex workflows.",
    ),
    cite_command: str = typer.Option(
        "citep",
        "--cite-command",
        help=r"natbib command name for --tex output. 'citep' → (Smith 2020), "
        r"'citet' → Smith (2020), 'cite' → plain biblatex. Default: citep.",
    ),
) -> None:
    """Find a citation that supports a specific statement.

    Unlike `find` (which returns papers ABOUT a topic), this looks for
    papers whose passages CONTAIN evidence for the exact claim, and
    returns the verbatim supporting sentence + page number.

    Example:
      bibwizard cite "single-mode fibers reduce modal noise in spectrographs"
    """
    from bibwizard.search.cite_search import find_citations
    from bibwizard.search.reranker import (
        CrossEncoderReranker, PassthroughReranker, get_reranker,
    )
    from bibwizard.utils.wizard_spinner import WizardLive
    from rich.table import Table

    init_db()
    try:
        get_client().ensure_ready(need_llm=True, need_embed=True)
    except (OllamaUnavailable, OllamaModelMissing) as e:
        error(str(e))
        raise typer.Exit(code=2) from None

    reranker = PassthroughReranker() if no_rerank else get_reranker()

    debug_rows: list[dict] = []

    def _debug(row: dict) -> None:
        debug_rows.append(row)

    # Animated ASCII wizard reads through the library while the LLM grinds.
    # Transient — auto-clears so the results table is the first persistent
    # output once find_citations returns.
    with WizardLive(
        console,
        status=f"Preparing search (reranker={reranker.name})...",
        total=pool_size,
    ) as wiz:
        # Pre-load cross-encoder model under the wizard so the ~30s wait
        # has user-visible feedback. We do this BEFORE setting the
        # entailment total so the progress bar tracks the LLM loop, not
        # the model download.
        if (
            isinstance(reranker, CrossEncoderReranker)
            and reranker._model is None
        ):
            wiz.update(
                status=(
                    f"Conjuring reranker model ({reranker.model_name}, "
                    "one-time, ~30s)..."
                ),
            )
            reranker._ensure_model()

        wiz.update(
            status=f"Searching {pool_size} candidate passages...",
            done=0,
            total=pool_size,
        )

        def _progress(done: int, total: int) -> None:
            wiz.update(
                status=f"Reading candidate passage {done}/{total}",
                done=done,
                total=total,
            )

        hits = find_citations(
            claim,
            pool_size=pool_size,
            max_results=max_results,
            min_confidence=min_confidence,
            progress_cb=_progress,
            debug_cb=_debug if debug else None,
            reranker=reranker,
            max_per_paper=(max_per_paper if max_per_paper > 0 else None),
        )

    if debug and debug_rows:
        dbg_table = Table(
            title=f"DEBUG — entailment verdicts for {len(debug_rows)} candidate chunk(s)",
            header_style="bold magenta",
            show_lines=True,
        )
        dbg_table.add_column("#", justify="right", no_wrap=True)
        dbg_table.add_column("Paper", justify="right", no_wrap=True)
        dbg_table.add_column("Pg", justify="right", no_wrap=True)
        dbg_table.add_column("RAG", justify="right", no_wrap=True)
        dbg_table.add_column("Rerank", justify="right", no_wrap=True)
        dbg_table.add_column("Verdict", no_wrap=True)
        dbg_table.add_column("Sup", no_wrap=True)
        dbg_table.add_column("Conf", justify="right", no_wrap=True)
        dbg_table.add_column("Rationale", overflow="fold")
        dbg_table.add_column("Quote / passage preview", overflow="fold")
        for r in debug_rows:
            conf = r.get("confidence")
            sup = r.get("supports")
            verdict = r["verdict"]
            # Color the verdict
            if verdict == "accepted":
                v_str = f"[green]{verdict}[/]"
            else:
                v_str = f"[red]{verdict}[/]"
            quote = r.get("quote") or ""
            preview = r.get("passage_preview") or ""
            quote_or_preview = (
                f"[yellow]quote:[/] “{quote}”\n[dim]passage:[/] {preview}"
                if quote
                else f"[dim]passage:[/] {preview}"
            )
            rr = r.get("rerank_score")
            rr_str = f"{rr:.3f}" if rr is not None else "—"
            dbg_table.add_row(
                str(r["i"]),
                str(r["paper_id"]),
                str(r["page"]) if r["page"] > 0 else "?",
                f"{r['chunk_score']:.3f}",
                rr_str,
                v_str,
                ("Y" if sup else "N") if sup is not None else "—",
                (f"{conf:.2f}" if conf is not None else "—"),
                r.get("rationale") or "",
                quote_or_preview,
            )
        console.print(dbg_table)

        if dump_passages:
            console.print()
            console.print(
                Panel(
                    f"Full passage text for all {len(debug_rows)} candidate "
                    "chunks (in entailment order). Use this when the table "
                    "preview is cut off and you need to confirm whether a "
                    "specific sentence is actually in the chunk.",
                    title="DEBUG — full passages",
                    border_style="magenta",
                )
            )
            for r in debug_rows:
                full = r.get("passage_full") or r.get("passage_preview") or ""
                console.print(
                    Panel(
                        full,
                        title=(
                            f"#{r['i']}  paper {r['paper_id']}  p.{r['page']}  "
                            f"RAG={r['chunk_score']:.3f}  verdict={r['verdict']}"
                        ),
                        border_style="dim",
                    )
                )

    if not hits:
        warn(
            "No supporting passage found. Try a more specific / concrete "
            "version of the claim, or raise --pool to widen the search."
        )
        return

    table = Table(
        title=f"Citations supporting: {claim!r}",
        show_lines=True,
        header_style="bold cyan",
    )
    table.add_column("Conf.", justify="right", no_wrap=True)
    table.add_column("ID", justify="right", style="dim", no_wrap=True)
    table.add_column("Cite", style="cyan", no_wrap=True)
    table.add_column("Page", justify="right", no_wrap=True)
    table.add_column("Quoted passage", overflow="fold")
    table.add_column("Why", overflow="fold")
    for h in hits:
        table.add_row(
            f"{h.confidence:.2f}",
            str(h.paper_id),
            h.paper_cite,
            str(h.page) if h.page > 0 else "?",
            f"“{h.quoted_sentence}”",
            h.rationale,
        )
    console.print(table)

    if tex:
        # Resolve each hit's full Paper record from the DB so the BibTeX
        # entries include venue / DOI / arxiv_id (CitationHit doesn't carry
        # those — it's optimized for display, not export). Render INSIDE
        # the session because format_citation_block iterates paper.authors
        # which is lazy-loaded.
        from bibwizard.output.bibtex import format_citation_block

        hit_ids = [h.paper_id for h in hits]
        with session_scope() as session:
            papers = (
                session.query(Paper)
                .filter(Paper.id.in_(hit_ids))
                .all()
            )
            by_id = {p.id: p for p in papers}
            # Preserve hits order (highest confidence first)
            ordered_papers = [by_id[pid] for pid in hit_ids if pid in by_id]
            tex_block = format_citation_block(
                ordered_papers, command=cite_command,
            )

        console.print()
        console.print(
            Panel(
                tex_block,
                title="LaTeX (\\citep + .bib entries)",
                border_style="green",
            )
        )


if __name__ == "__main__":  # pragma: no cover
    app()
