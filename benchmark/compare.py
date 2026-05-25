#!/usr/bin/env python3
"""Compare bibwizard's database against the curated ground-truth dataset.

Usage:
    python compare.py                       # uses defaults from bibwizard config
    python compare.py --ground-truth path/to/ground_truth.json
    python compare.py --report report.md    # write markdown report
    python compare.py --strict              # also flag minor whitespace/case diffs

Outputs:
    - stdout: per-paper summary table + aggregate scores
    - optional markdown report file with the full diff

The comparison keys papers by SHA-256, so the ordering / counts in either side
don't have to match. Aggregate metrics reported:
    - matched / unmatched / extra
    - title exact-match rate
    - title token-Jaccard mean
    - author exact-match rate
    - author set-Jaccard mean
    - year correct rate
    - doi correct rate
    - arxiv_id correct rate
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# bibwizard must be importable
try:
    from bibwizard.database.migrations import session_scope, init_db
    from bibwizard.database.models import Paper
    from bibwizard.utils.config import settings as bw_settings
except ImportError as e:
    print(
        f"Could not import bibwizard ({e}).\n"
        "Make sure you've run `pip install -e .` from the bibwizard project root.",
        file=sys.stderr,
    )
    sys.exit(2)


# ---------- text normalization ----------

def normalize_title(s: str) -> str:
    s = (s or "").strip().lower()
    # Collapse PyMuPDF whitespace artifacts
    s = re.sub(r"\s+", " ", s)
    # Strip surrounding punctuation
    s = s.strip(" .,;:\"'“”‘’")
    # Remove ligature-like quirks
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff")
    return s


def normalize_name(s: str) -> str:
    s = (s or "").strip()
    # NFKD-normalize so 'Eyyuboğlu' (NFC) and 'Eyyubog˘lu' (with U+02D8 spacing
    # breve) collapse to the same string. The breve's compatibility
    # decomposition is `space + combining breve` so after we strip combining
    # marks we ALSO need to drop the leftover modifier-category characters
    # (Lm = modifier letter, Sk = modifier symbol) — otherwise the residual
    # space splits the surname.
    s = unicodedata.normalize("NFKD", s)
    s = "".join(
        ch for ch in s
        if unicodedata.category(ch) not in ("Mn", "Lm", "Sk")
    )
    # NFKD decomposes some modifier letters into `space + combining mark`
    # (e.g. ˘ → ' ' + ̆). After we strip the combining mark, a stray space
    # remains *inside* the surname (e.g. 'Eyyubog lu'). Glue letter runs
    # back together when the space is sandwiched between two lowercase
    # ASCII letters — real word boundaries always involve a capital letter
    # or punctuation, so this is safe for name comparison.
    s = re.sub(r"(?<=[a-z])\s+(?=[a-z])", "", s)
    s = re.sub(r"\s+", " ", s)
    # 'Last, F.' → 'F. Last' so set comparison is order-agnostic per-name
    if "," in s and re.search(r"^[A-Z][a-z]", s):
        try:
            last, rest = s.split(",", 1)
            s = f"{rest.strip()} {last.strip()}"
        except ValueError:
            pass
    # Strip trailing affiliation-letter superscripts that may have leaked
    s = re.sub(r"\s+[a-z](?:\s*,\s*[a-z])*\s*$", "", s)
    return s.strip()


def normalize_doi(s: str | None) -> str:
    """Strip trailing punctuation noise that often follows DOIs in PDFs."""
    if not s:
        return ""
    return s.strip().rstrip(".,;)]}>\\\"'").lower()


def normalize_arxiv(s: str | None) -> str:
    """Strip arXiv version suffixes — '0903.5001v2' ≡ '0903.5001'."""
    if not s:
        return ""
    return re.sub(r"v\d+$", "", s.strip(), flags=re.IGNORECASE)


def title_tokens(s: str) -> set[str]:
    s = normalize_title(s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return {t for t in s.split() if len(t) > 2}


def title_similarity(a: str, b: str) -> float:
    ta, tb = title_tokens(a), title_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def author_set(authors: list[str]) -> set[str]:
    """Compare authors as a set of normalized surnames so first-name-vs-initial
    differences ('Y. Reinarz' vs 'Yared Reinarz') don't punish us.

    Surname extraction: take the rightmost contiguous run of ≥3 ASCII letters
    (after NFKD decomposition + combining-mark strip + non-letter strip).
    This handles modifier-letter decomposition residues that would otherwise
    split a surname like 'Eyyuboğlu' → 'Eyyubog lu' → surname='lu'."""
    out = set()
    for a in authors:
        n = normalize_name(a)
        if not n:
            continue
        # Re-decompose to ASCII letters + hyphens + whitespace; then find the
        # last run of ≥3 ASCII letters (possibly hyphenated for "Jensen-Clem").
        ascii_only = re.sub(r"[^A-Za-z\- ]", "", n).lower()
        # Reject trailing whitespace and find rightmost surname run.
        m_iter = list(re.finditer(r"[a-z]+(?:-[a-z]+)*", ascii_only))
        if not m_iter:
            continue
        # Walk from the right; pick the last "real" surname run (≥3 letters).
        # If the last run is < 3 chars (it's an initial like 'a' or 'jr'),
        # fall through to the previous one.
        for m in reversed(m_iter):
            if len(m.group(0).replace("-", "")) >= 3:
                out.add(m.group(0))
                break
    return out


def author_jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = author_set(a), author_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------- comparison ----------

@dataclass
class DiffRecord:
    file: str
    sha: str
    truth_title: str
    bw_title: str
    title_exact: bool
    title_jaccard: float
    truth_authors: list[str]
    bw_authors: list[str]
    author_set_match: bool
    author_jaccard: float
    n_truth_authors: int
    n_bw_authors: int
    year_match: bool | None
    truth_year: int | None
    bw_year: int | None
    doi_match: bool | None
    arxiv_match: bool | None
    issues: list[str] = field(default_factory=list)


def compare_one(truth: dict, bw: Paper, strict: bool) -> DiffRecord:
    rec = DiffRecord(
        file=truth["file"],
        sha=truth["sha256"],
        truth_title=truth.get("title") or "",
        bw_title=bw.title or "",
        title_exact=normalize_title(truth.get("title")) == normalize_title(bw.title),
        title_jaccard=title_similarity(truth.get("title", ""), bw.title or ""),
        truth_authors=list(truth.get("authors", [])),
        bw_authors=[a.name for a in bw.authors],
        author_set_match=author_set(truth.get("authors", [])) == author_set([a.name for a in bw.authors]),
        author_jaccard=author_jaccard(truth.get("authors", []), [a.name for a in bw.authors]),
        n_truth_authors=len(truth.get("authors", [])),
        n_bw_authors=len(bw.authors),
        year_match=(truth.get("year") == bw.year) if truth.get("year") is not None else None,
        truth_year=truth.get("year"),
        bw_year=bw.year,
        doi_match=(truth.get("doi") is not None and normalize_doi(truth.get("doi")) == normalize_doi(bw.doi)) if truth.get("doi") else None,
        arxiv_match=(truth.get("arxiv_id") is not None and normalize_arxiv(truth.get("arxiv_id")) == normalize_arxiv(bw.arxiv_id)) if truth.get("arxiv_id") else None,
    )

    # Auto-issue flags
    if not rec.title_exact:
        if rec.title_jaccard < 0.6:
            rec.issues.append(f"title mismatch (jaccard={rec.title_jaccard:.2f})")
        elif strict:
            rec.issues.append(f"title nearly-but-not-exact ({rec.title_jaccard:.2f})")

    if rec.author_jaccard < 0.5 and rec.n_truth_authors > 0:
        rec.issues.append(f"author set mismatch (jaccard={rec.author_jaccard:.2f})")
    if rec.n_bw_authors == 0 and rec.n_truth_authors > 0:
        rec.issues.append("no authors extracted")
    if rec.n_bw_authors > 0 and rec.n_bw_authors < rec.n_truth_authors * 0.5:
        rec.issues.append(f"only {rec.n_bw_authors}/{rec.n_truth_authors} authors extracted")

    if rec.year_match is False:
        rec.issues.append(f"year wrong (got {rec.bw_year}, expected {rec.truth_year})")

    if truth.get("doi") and rec.doi_match is False:
        rec.issues.append(f"doi wrong (got {bw.doi!r}, expected {truth.get('doi')!r})")
    if truth.get("arxiv_id") and rec.arxiv_match is False:
        rec.issues.append(f"arxiv_id wrong (got {bw.arxiv_id!r}, expected {truth.get('arxiv_id')!r})")

    return rec


# ---------- reporting ----------

def summarize(records: list[DiffRecord], unmatched_truth: list[dict], extra_bw: list[Paper]) -> dict:
    if not records:
        return {"matched": 0, "missing": len(unmatched_truth), "extra": len(extra_bw)}
    n = len(records)
    return {
        "matched": n,
        "missing_from_bibwizard": len(unmatched_truth),
        "extra_in_bibwizard": len(extra_bw),
        "title_exact_match_rate": sum(1 for r in records if r.title_exact) / n,
        "title_jaccard_mean": sum(r.title_jaccard for r in records) / n,
        "author_set_match_rate": sum(1 for r in records if r.author_set_match) / n,
        "author_jaccard_mean": sum(r.author_jaccard for r in records) / n,
        "year_correct_rate": (
            sum(1 for r in records if r.year_match is True)
            / max(1, sum(1 for r in records if r.year_match is not None))
        ),
        "doi_correct_rate": (
            sum(1 for r in records if r.doi_match is True)
            / max(1, sum(1 for r in records if r.doi_match is not None))
        ),
        "n_with_any_issue": sum(1 for r in records if r.issues),
    }


def render_markdown(records: list[DiffRecord], summary: dict, unmatched: list[dict], extras: list[Paper]) -> str:
    lines = ["# bibwizard benchmark report\n"]
    lines.append("## Summary\n")
    for k, v in summary.items():
        if isinstance(v, float):
            lines.append(f"- **{k}**: {v:.2%}" if "rate" in k or "mean" in k else f"- **{k}**: {v:.2f}")
        else:
            lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines.append("## Per-paper diffs\n")
    lines.append("| File | Title✓ | Auth Jaccard | Yr✓ | DOI✓ | Issues |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in sorted(records, key=lambda r: (-len(r.issues), r.file)):
        ti = "✓" if r.title_exact else f"{r.title_jaccard:.2f}"
        ai = f"{r.author_jaccard:.2f}"
        yi = "—" if r.year_match is None else ("✓" if r.year_match else "✗")
        di = "—" if r.doi_match is None else ("✓" if r.doi_match else "✗")
        issues = "; ".join(r.issues) or "—"
        # Truncate filename for the table
        short = r.file if len(r.file) <= 60 else r.file[:57] + "..."
        lines.append(f"| {short} | {ti} | {ai} | {yi} | {di} | {issues} |")

    if unmatched:
        lines.append("\n## Missing from bibwizard (in ground truth but not in DB)\n")
        for u in unmatched:
            lines.append(f"- {u['file']} (sha={u['sha256'][:12]}…)")
    if extras:
        lines.append("\n## Extra in bibwizard (in DB but not in ground truth)\n")
        for p in extras:
            lines.append(f"- id={p.id} title={p.title!r} (sha={(p.sha256 or '')[:12]}…)")

    # Detail section for high-issue papers
    high_issue = [r for r in records if r.issues]
    if high_issue:
        lines.append("\n## Detail for papers with issues\n")
        for r in high_issue:
            lines.append(f"### {r.file}\n")
            lines.append(f"- **truth title**: `{r.truth_title}`")
            lines.append(f"- **bibwizard title**: `{r.bw_title}`")
            lines.append(f"- **truth authors** ({r.n_truth_authors}): {', '.join(r.truth_authors)}")
            lines.append(f"- **bibwizard authors** ({r.n_bw_authors}): {', '.join(r.bw_authors)}")
            lines.append(f"- **truth year**: {r.truth_year}  **bw year**: {r.bw_year}")
            lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path(__file__).parent / "ground_truth.json",
        help="Path to ground_truth.json (default: alongside this script).",
    )
    parser.add_argument("--report", type=Path, help="Write a markdown report to this path.")
    parser.add_argument("--strict", action="store_true", help="Flag near-but-not-exact title matches.")
    args = parser.parse_args()

    if not args.ground_truth.exists():
        print(f"Ground truth file not found: {args.ground_truth}", file=sys.stderr)
        return 2

    truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    truth_papers = truth["papers"]
    truth_by_sha = {p["sha256"]: p for p in truth_papers}

    init_db()
    with session_scope() as session:
        all_papers = session.query(Paper).all()
        bw_by_sha = {(p.sha256 or ""): p for p in all_papers if p.sha256}
        # Eager-load authors before session closes
        records: list[DiffRecord] = []
        extras: list[Paper] = []
        for p in all_papers:
            if not p.sha256 or p.sha256 not in truth_by_sha:
                extras.append(p)
        unmatched_truth: list[dict] = []
        for sha, t in truth_by_sha.items():
            bw = bw_by_sha.get(sha)
            if bw is None:
                unmatched_truth.append(t)
                continue
            # Read authors while session is open
            _ = [a.name for a in bw.authors]
            rec = compare_one(t, bw, args.strict)
            records.append(rec)

    summary = summarize(records, unmatched_truth, extras)

    # Pretty-print to stdout
    print(f"bibwizard benchmark — comparing {bw_settings.sqlite_path}")
    print()
    print("=== SUMMARY ===")
    for k, v in summary.items():
        if isinstance(v, float):
            label = f"{v:.1%}" if ("rate" in k or "mean" in k) else f"{v:.2f}"
        else:
            label = str(v)
        print(f"  {k:.<35} {label}")
    print()

    # Issue table
    if records:
        print("=== PER-PAPER (sorted by # of issues) ===")
        for r in sorted(records, key=lambda r: (-len(r.issues), r.file)):
            tag = "PASS" if not r.issues else f"ISSUES: {', '.join(r.issues)}"
            short = r.file if len(r.file) <= 70 else r.file[:67] + "..."
            print(f"  [{tag[:60]}]")
            print(f"    {short}")
    if unmatched_truth:
        print()
        print(f"=== {len(unmatched_truth)} GROUND-TRUTH PAPER(S) NOT FOUND IN BIBWIZARD ===")
        for t in unmatched_truth:
            print(f"  - {t['file']}")
    if extras:
        print()
        print(f"=== {len(extras)} BIBWIZARD PAPER(S) NOT IN GROUND TRUTH ===")
        for p in extras:
            print(f"  - id={p.id} title={p.title!r}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_markdown(records, summary, unmatched_truth, extras), encoding="utf-8")
        print(f"\nMarkdown report written to: {args.report}")

    # Exit non-zero if anything's wrong, so it's CI-friendly
    if summary.get("n_with_any_issue", 0) or unmatched_truth:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
