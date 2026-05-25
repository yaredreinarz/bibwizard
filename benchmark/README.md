# bibwizard benchmark

A curated ground-truth dataset for the 36 PDFs in `D:\bibwizard\literature\`,
plus a script that diffs it against whatever metadata bibwizard ended up
extracting. Useful for measuring extraction quality after a scan.

## Files

- **`ground_truth.json`** — one entry per PDF in `literature/`, with:
  - `file` and `sha256` (the key)
  - `title` — clean, properly-cased title
  - `authors` — full list in byline order
  - `year`, `venue`, `doi`, `arxiv_id` — when known
  - `summary` — 2–3 sentence description from the abstract
  - `confidence` — `high` / `medium` / `low` reflecting how authoritative the source was (PDF embedded metadata = high, page-1 text only = medium, edge cases like the HTML scrape = low)
- **`compare.py`** — diff script.

## Running the benchmark

After bibwizard finishes its `bibwizard scan --llm-extract` run:

```powershell
cd D:\bibwizard
.\.venv\Scripts\Activate.ps1
python benchmark\compare.py                            # stdout report
python benchmark\compare.py --report benchmark\report.md   # also write markdown
python benchmark\compare.py --strict                   # flag near-misses
```

## What it reports

Aggregate metrics across all matched papers:

- **title_exact_match_rate** — fraction with character-identical (case-insensitive, whitespace-normalised, ligature-fixed) title.
- **title_jaccard_mean** — mean token-Jaccard similarity. Catches "almost correct" titles.
- **author_set_match_rate** — fraction where the set of *surnames* matches exactly. Order-insensitive.
- **author_jaccard_mean** — mean surname-set Jaccard.
- **year_correct_rate** — fraction of papers with the correct year (only counts papers where ground truth has a year).
- **doi_correct_rate** — same, for DOIs.
- **n_with_any_issue** — count of papers that triggered any auto-issue flag.

Per-paper diff table lists each paper with `title✓`, `author Jaccard`, `year✓`, `doi✓`, and a comma-separated `issues` column. Sorted by issues-first so the broken ones are at the top.

Missing-from-bibwizard and extra-in-bibwizard sections list any papers that didn't match by SHA-256 in either direction.

## Auto-issue flags

The script flags a paper as "has issues" when any of these are true:

- title Jaccard < 0.6 → `title mismatch`
- author Jaccard < 0.5 (with non-empty truth authors) → `author set mismatch`
- bibwizard has zero authors but truth has some → `no authors extracted`
- bibwizard found fewer than half the truth authors → `only N/M authors extracted`
- year doesn't match → `year wrong (got X, expected Y)`
- DOI doesn't match (when truth has one) → `doi wrong (got X, expected Y)`
- arXiv ID doesn't match (when truth has one) → `arxiv_id wrong`

With `--strict`, near-but-not-exact title matches (Jaccard between 0.6 and 1.0) also trip a flag.

## Notes on the ground-truth dataset

- **`http_Gaussian_Beams_Optical_design_essentials.pdf`** isn't a real paper (it's a print-to-PDF of a web article). Marked `confidence: low`. Recommend `bibwizard remove`ing it.
- **`KPIC_Keck_SPIE_2021_*.pdf`** appears to be an exact duplicate of **`2021_JATIS_Delorme_*.pdf`** — both will hit the same SHA-256 and bibwizard's `find_paper_by_identity` dedup should reject the second one at ingest.
- **`KPIC_Keck_2025_SPIE_Jovanovic_*.pdf`** is the SPIE-proceedings companion to **`2025_JATIS_Jovanovic_*.pdf`** (same instrument, different venue, different SHA-256).
- For a handful of entries (Pueyo's white paper, Crass/Smous iLocater, Sirbu PIAACMC, Haffert IFS) the author list was inferred from the heuristic byline parser plus page-1 text rather than authoritative PDF embedded metadata — these are marked `confidence: medium`.

## Re-generating

If you change the literature folder (add or remove papers), regenerate the raw front-matter dump and rebuild the ground truth:

```powershell
python -c "from bibwizard.ingestion.parser import parse_pdf; from bibwizard.ingestion.structure import extract_front_matter; ..."
```

(or just delete entries that no longer apply — the comparison keys by SHA so leftover entries don't break anything).

## Author normalization

The comparison is intentionally generous on authors:

- `Smith, J.` ↔ `J. Smith` ↔ `John Smith` all hash to the surname `smith` for set comparison.
- Order within the author list is **not** compared by the aggregate metrics — only the *set* of surnames. So a paper that picks up all authors but in the wrong order still scores 1.0 on `author_set_match`. The per-paper diff prints both lists side-by-side so you can spot order issues visually.

This is deliberate: byline-order extraction is a separate (harder) problem from "did you find the right people."
