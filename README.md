# bibwizard

A local-first research paper manager for your terminal. Drop in PDFs, ask
questions of your library in plain English, and pull verbatim quotes you
can cite — all running on your own machine.

Built for astronomers and other researchers who want to search their own
literature collection more carefully than a folder of PDFs allows, without
sending anything to a cloud service.

## What it does

- **Ingests PDFs.** Parses metadata, extracts references, generates a
  structured summary, and indexes the text for search.
- **Answers questions about your library** in a chat session, citing the
  specific papers that informed the answer.
- **Finds a citation for a specific claim.** Paste a sentence; bibwizard
  returns the papers in your library that support it, with verbatim
  quotes and page numbers. Outputs LaTeX `\citep{...}` macros and BibTeX
  entries ready to paste into a manuscript.
- **Resolves references.** Given a paper, looks up its bibliography on
  Crossref / arXiv / Unpaywall and (optionally) downloads the cited PDFs
  into your library.
- **Stays local.** Everything lives under `~/.bibwizard/` on your disk.
  No API calls leave your machine except optional public-bibliography
  lookups (Crossref, arXiv, Unpaywall, NASA ADS).

## Glossary (for the field)

A handful of computing terms used below, in plain language:

| Term | What it means |
|---|---|
| **LLM** | Large language model — an AI model that reads and writes natural-language text. bibwizard uses one running locally (no cloud) for summaries, chat, and citation entailment. |
| **RAG** | Retrieval-augmented generation. Standard pattern: the LLM doesn't try to remember your library; instead, it searches your library for relevant passages each time you ask a question, then writes its answer using those passages as context. |
| **Embedding** | A numerical fingerprint of a piece of text. Two passages with similar meaning have similar fingerprints. We use embeddings to do "find me passages about X" without needing exact keyword matches. |
| **Chunk** | A small, fixed-length slice of a paper (~500 words). Search operates on chunks rather than whole papers so the LLM only sees the relevant excerpts. |
| **Reranker** | A second-pass scorer that re-orders search results by reading the query and each candidate passage together. More accurate than embedding similarity alone. |
| **Entailment** | The narrow question "does this passage actually support this specific claim?" — asked per-passage by the LLM, separate from the broader retrieval step. |
| **Ollama** | A local LLM server (free, runs on your machine). Pulls models from a registry; serves them via a local HTTP port that bibwizard talks to. |
| **ChromaDB** | A small vector database — stores embeddings and lets us look up "passages similar to this one" quickly. |

## Stack

- Python 3.11+, [Typer](https://typer.tiangolo.com/) + [Rich](https://rich.readthedocs.io/) for the terminal interface
- [Ollama](https://ollama.com/) — runs the LLM and the embedding model locally
- SQLite + SQLAlchemy 2.x — paper / author / tag / citation tables
- ChromaDB — local vector store for semantic search
- PyMuPDF — page-by-page PDF parsing
- NetworkX — citation graph
- Optional: `sentence-transformers` — for the cross-encoder reranker (≈1 GB
  install including PyTorch). Without it bibwizard falls back to a built-in
  lexical reranker.

## Install

```bash
git clone https://github.com/yaredreinarz/bibwizard.git
cd bibwizard
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .

# Optional but recommended — enables the cross-encoder reranker:
pip install sentence-transformers
```

You'll also need [Ollama](https://ollama.com/) running with the default
models pulled:

```bash
ollama serve &
ollama pull qwen2.5:7b-instruct-q4_K_M   # LLM
ollama pull nomic-embed-text             # embedding model
```

`qwen2.5:7b` is the smallest model that handles the citation-finder's
defensive prompt reliably. For more careful behaviour at the cost of
speed, swap to `qwen2.5:14b-instruct-q4_K_M`.

## First run

```bash
bibwizard init                       # creates ~/.bibwizard/{db,vectors,papers,summaries}
bibwizard add path/to/paper.pdf      # parse, summarize, index
bibwizard add --arxiv 2106.04561     # fetch from arXiv first, then ingest
bibwizard list                       # table of everything you've added
bibwizard show 1                     # full metadata + structured summary
bibwizard chat                       # ask questions about your library
```

## Finding citations for a specific claim

The `cite` command answers the question: *which paper in my library
supports this exact sentence, and what does it actually say?* This is
different from `find`, which returns papers that are *about* a topic.

```bash
bibwizard cite "Single-mode fibres reduce modal noise in spectrographs."
```

Behind the scenes it:

1. Pulls a wide pool of candidate passages from ChromaDB.
2. Re-ranks them with a cross-encoder so noisy or merely-topical chunks
   get demoted.
3. Caps each paper at three chunks in the pool so one noisy paper can't
   dominate.
4. Asks the LLM, per passage, whether the passage *actually supports*
   the claim — with explicit rules to reject inline-citation forwarding,
   reference-list entries, matching-number-but-different-quantity cases,
   and entity confusions (e.g. "lenslets" ≠ "photonic lanterns").
5. Returns the surviving papers with the verbatim sentence the LLM
   identified as evidence, plus one sentence of surrounding context.

Useful flags:

- `--tex` — also print a LaTeX `\citep{key1, key2, ...}` macro and the
  `.bib` entries for the matched papers, ready to paste into a manuscript.
- `--cite-command citet` — use `\citet{...}` instead of `\citep{...}`.
- `--max N` — limit results to top N papers.
- `--debug` — print the per-candidate entailment verdict (useful when
  something unexpected shows up).
- `--pool N` — widen the candidate pool (default 20; raise to 40 for
  harder paraphrases).

The same syntax works inside `bibwizard chat`:

```
> cite "Diffraction-limited spectrographs decouple instrument size from telescope aperture." --tex
```

## All commands

| Command | What it does |
|---|---|
| `bibwizard init` | Create the SQLite database and ChromaDB store |
| `bibwizard add <path>` | Ingest a local PDF: parse, summarize, embed |
| `bibwizard add --arxiv <id>` | Download from arXiv, then ingest |
| `bibwizard scan` | Ingest every new PDF in your `literature/` folder |
| `bibwizard list [--year Y] [--tag T]` | Tabular library listing |
| `bibwizard show <id>` | Full metadata + structured summary |
| `bibwizard tag <id> <tags...>` | Add tags to a paper |
| `bibwizard find "<query>"` | Semantic search for papers about a topic |
| `bibwizard grep "<phrase>"` | Substring search in titles/abstracts |
| `bibwizard cite "<claim>" [--tex]` | Find papers that support a specific claim, with verbatim quotes; optional LaTeX/BibTeX output |
| `bibwizard chat` | Conversational interface over your library |
| `bibwizard fetch-refs <id>` | Resolve a paper's bibliography via Crossref / arXiv / Unpaywall |
| `bibwizard whats-new [--days N]` | Show papers added recently |
| `bibwizard duplicates` | Surface likely-duplicate papers in the library |
| `bibwizard stats` | Counts and per-year breakdown |
| `bibwizard map ref [-f dot\|json]` | Export the citation graph |
| `bibwizard map content [-n K]` | Export semantic clusters of papers |
| `bibwizard export -f bibtex` | BibTeX export of the whole library |

Add `--help` to any command for full options.

## How retrieval works

1. Each PDF is split into pages by PyMuPDF. The bibliography section is
   detected (regex on `^References$` / `^Bibliography$` etc.) and parsed
   separately into citation rows.
2. The body is chunked into ~512-token sliding windows with 64-token
   overlap.
3. Each chunk is embedded with `nomic-embed-text` and stored in ChromaDB
   along with paper / page / chunk metadata.
4. On `chat` or `find`, your question is embedded with the same model.
   The top-K chunks by cosine similarity are pulled, labelled
   `[PAPER 1]`, `[PAPER 2]`, …, and injected as context. The LLM is
   instructed to cite those labels in its answer.
5. For `cite`, an additional re-ranking step and a per-passage
   entailment check tighten the result to verifiable evidence rather
   than topically-related text. See the "Finding citations" section above.

## Structured summaries

For every ingested paper the LLM is asked to return a JSON object with:

- `title`, `authors`, `year`
- `key_contributions` — 3–6 bullet-style strings
- `methodology` — 2–4 sentence description
- `limitations` — what the paper acknowledges (or `""` if none stated)
- `tags` — 3–8 lowercase, hyphenated topical tags

Tags are auto-applied to the paper if `AUTO_TAG=true` (default). The full
JSON is also written to `~/.bibwizard/summaries/paper_<id>.json` for
later use.

## Citation graphs

`bibwizard map ref --format dot` writes a Graphviz DOT file of who cites
whom. References resolved to in-library papers become real edges; the
rest are kept as `ext:` nodes so external citations aren't lost.

Render it with:

```bash
dot -Tpng ~/.bibwizard/references.dot -o references.png
```

Graphviz `dot` is a small standalone tool — install it via `brew install
graphviz` on macOS, `apt install graphviz` on Debian/Ubuntu, or
[download for Windows](https://graphviz.org/download/).

## Content map

`bibwizard map content` clusters papers by their mean chunk embedding
(KMeans clustering, ⌈√N⌉ clusters by default) and emits a JSON file with
`{cluster, papers, label}` — useful for spotting topical groupings in a
large library.

## Configuration

All settings live in `.env` at the project root (loaded via
`python-dotenv`). See `.env.example` for the full list. Important ones:

- `OLLAMA_HOST` — defaults to `http://localhost:11434`
- `OLLAMA_LLM_MODEL` — default `qwen2.5:7b-instruct-q4_K_M`
- `OLLAMA_EMBED_MODEL` — default `nomic-embed-text`
- `BIBWIZARD_HOME` — override `~/.bibwizard` if you want a custom data dir
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RAG_TOP_K` — retrieval tuning
- `RERANKER_ENABLED`, `RERANKER_KIND`, `RERANKER_MAX_PER_PAPER` — cite-finder tuning
- `AUTO_TAG` — apply LLM-suggested tags automatically
- `ADS_API_TOKEN` — optional NASA ADS access for richer reference lookup
- `UNPAYWALL_EMAIL` — optional, enables open-access PDF auto-download

## Layout on disk

```
~/.bibwizard/
├── db/bibwizard.sqlite         # SQLAlchemy metadata DB
├── vectors/                    # ChromaDB persistent client
├── papers/                     # Original PDFs (copied or downloaded)
├── summaries/                  # JSON summaries per paper
├── references.dot              # Citation graph (when exported)
└── content_map.json            # Semantic clusters (when exported)
```

## Troubleshooting

- **"Ollama doesn't appear to be running"** — start it with `ollama
  serve` or set `OLLAMA_HOST` to wherever it's actually listening.
- **"Required Ollama model(s) not installed"** — the error message tells
  you exactly which `ollama pull` to run.
- **`cite` returns "No supporting passage found"** — the candidate pool
  may not contain the right passage. Re-run with `--pool 40` (wider) or
  with `--debug` to see what got rejected and why.
- **`cite` returns a result you don't trust** — re-run with `--debug`.
  The verdict table shows every candidate the LLM saw, the verbatim
  quote it picked, and the LLM's stated rationale; that's usually enough
  to spot whether the answer is a real match or a stretch.

## A note on the authorship

bibwizard was developed in collaboration with **Claude (Anthropic),
inside the Cowork desktop application**. The architecture, design
decisions, prompt rules, and feature priorities all came from
conversational iteration. A meaningful share of the code, prompts, and
documentation were drafted by Claude under direct human supervision.

Mentioning this here so anyone who finds the repo has an accurate
picture of how it was built.

## License

MIT.
