# bibwizard / bibwizard

A local-first, LLM-powered research paper manager. Ingest PDFs (or arXiv ids), get
structured summaries via DeepSeek-R1 running in Ollama, search semantically across
your library, and chat with your papers — all from your terminal, on Windows, macOS,
or Linux.

Everything stays on your machine: SQLite for metadata, ChromaDB for vectors, raw
PDFs and JSON summaries on disk under `~/.bibwizard/`.

## Stack

- Python 3.11+, [Typer](https://typer.tiangolo.com/) + [Rich](https://rich.readthedocs.io/) for the CLI
- [Ollama](https://ollama.com/) — runs `deepseek-r1:8b` (LLM) and `nomic-embed-text` (embeddings) locally
- SQLite + SQLAlchemy 2.x — paper / author / tag / citation tables
- ChromaDB — persistent vector store (cosine distance)
- PyMuPDF (`fitz`) — page-by-page PDF parsing with bibliography detection
- NetworkX — citation graph

## Install

```bash
git clone https://github.com/yaredreinarz/bibwizard.git
cd bibwizard
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .
```

You'll also need Ollama running with both models pulled:

```bash
ollama serve &
ollama pull deepseek-r1:8b
ollama pull nomic-embed-text
```

## First run

```bash
bibwizard init                       # creates ~/.bibwizard/{db,vectors,papers,summaries}
bibwizard add path/to/paper.pdf      # parse + summarize + index
bibwizard add --arxiv 2106.04561     # fetch from arXiv first, then ingest
bibwizard list                       # table of everything you've added
bibwizard show 1                     # full metadata + structured summary
bibwizard chat                       # RAG chat over your library
```

## All commands

| Command | What it does |
| --- | --- |
| `bibwizard init` | Create the SQLite DB and ChromaDB store |
| `bibwizard add <path>` | Ingest a local PDF: parse, summarize (LLM), embed (RAG) |
| `bibwizard add --arxiv <id>` | Download from arXiv then ingest |
| `bibwizard list [--year Y] [--tag T]` | Tabular library listing |
| `bibwizard show <id>` | Full metadata + structured summary |
| `bibwizard tag <id> <tags...>` | Add tags to a paper |
| `bibwizard search "<query>"` | Semantic search over chunks |
| `bibwizard find "<query>"` | LIKE-based title/abstract search |
| `bibwizard chat` | Streaming RAG chat over your library |
| `bibwizard map ref [-f dot|json]` | Export citation graph |
| `bibwizard map content [-n K]` | Export semantic clusters |
| `bibwizard export -f bibtex` | BibTeX export of the library |
| `bibwizard stats` | Counts and per-year breakdown |

Add `--help` to any command for full options.

## How RAG works here

1. Each PDF is split into pages by PyMuPDF. The bibliography section is detected
   (regex on `^References$` / `^Bibliography$` etc.) and parsed separately into
   `Citation` rows.
2. The body is chunked into 512-token sliding windows with 64-token overlap
   (configurable via `CHUNK_SIZE` / `CHUNK_OVERLAP`).
3. Each chunk is embedded with `nomic-embed-text` and pushed to ChromaDB with
   `{paper_id, title, page, chunk_index}` metadata.
4. On `chat`, your question is embedded with the same model. The top-5 chunks
   (by cosine similarity) are pulled, labelled `[PAPER 1]`, `[PAPER 2]`, …, and
   injected as context. The LLM is instructed to cite those labels.
5. The DeepSeek-R1 response streams back token-by-token; its `<think>...</think>`
   reasoning blocks are stripped from the visible output.

## Structured summaries

For every ingested paper the LLM is asked to return a single JSON object with:

- `title`, `authors`, `year`
- `key_contributions` — 3–6 bullet-style strings
- `methodology` — 2–4 sentence description
- `limitations` — what the paper acknowledges (or `""`)
- `tags` — 3–8 lowercase, hyphenated topical tags

Tags are auto-applied to the paper if `AUTO_TAG=true` (default). The full JSON
is also written to `~/.bibwizard/summaries/paper_<id>.json` for later use.

## Citation graphs

`bibwizard map ref --format dot` writes a Graphviz DOT file of who cites whom.
Citations resolved to in-library papers become real edges (blue boxes); the rest
are kept as `ext:` nodes (grey boxes) so external references aren't lost.

Render it with:

```bash
dot -Tpng ~/.bibwizard/references.dot -o references.png
```

## Content map

`bibwizard map content` clusters papers by their mean chunk embedding (KMeans,
$\sqrt{N}$ clusters by default) and emits a JSON file with `{cluster, papers,
label}` — useful for spotting topical groupings in a large library.

## Configuration

All settings live in `.env` (loaded via `python-dotenv`). See `.env.example`.
Important ones:

- `OLLAMA_HOST` — defaults to `http://localhost:11434`
- `OLLAMA_LLM_MODEL` — default `deepseek-r1:8b`
- `OLLAMA_EMBED_MODEL` — default `nomic-embed-text`
- `BIBWIZARD_HOME` — override `~/.bibwizard` if you want a custom data dir
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RAG_TOP_K` — RAG tuning
- `AUTO_TAG` — apply LLM-suggested tags automatically

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

- **"Ollama doesn't appear to be running"** — start it with `ollama serve` or
  set `OLLAMA_HOST` to wherever it's actually listening.
- **"Required Ollama model(s) not installed"** — the error message tells you
  exactly which `ollama pull` to run.
- **PDF parses but summary is empty** — DeepSeek-R1 sometimes wraps its JSON in
  prose; the parser strips `<think>` blocks and ```json fences automatically,
  but extremely long papers may exceed the context window. Re-run with a
  shorter PDF or increase the model's context.

## License

MIT.
