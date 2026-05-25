"""Central configuration and path management for bibwizard.

All settings come from environment variables (loaded via python-dotenv) with
sensible defaults. All paths are pathlib.Path objects rooted at BIBWIZARD_HOME
(default: ~/.bibwizard) so the tool works the same on Windows / macOS / Linux.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _env_path(var: str, default: Path) -> Path:
    raw = os.getenv(var)
    if not raw:
        return default
    return Path(raw).expanduser().resolve()


def _env_str(var: str, default: str) -> str:
    return os.getenv(var, default)


def _env_int(var: str, default: int) -> int:
    try:
        return int(os.getenv(var, default))
    except (TypeError, ValueError):
        return default


def _env_bool(var: str, default: bool) -> bool:
    raw = os.getenv(var)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(var: str, default: float) -> float:
    raw = os.getenv(var)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable settings snapshot built from env at load time."""

    # Storage roots
    home: Path
    db_dir: Path
    vectors_dir: Path
    papers_dir: Path
    summaries_dir: Path
    literature_dir: Path

    # Database
    sqlite_path: Path

    # Ollama
    ollama_host: str
    ollama_llm_model: str
    ollama_embed_model: str
    ollama_timeout: int

    # RAG / chunking
    chunk_size: int
    chunk_overlap: int
    rag_top_k: int

    # Behaviour
    log_level: str
    auto_tag: bool

    # External
    arxiv_api: str
    ads_api_token: str | None = None
    arxiv_min_delay: float = 5.0
    crossref_api: str = "https://api.crossref.org/works"
    unpaywall_api: str = "https://api.unpaywall.org/v2"
    # Unpaywall requires a contact email per request. If unset, Unpaywall
    # lookups are skipped. Set via env var UNPAYWALL_EMAIL.
    unpaywall_email: str = ""

    # Careful LLM-driven metadata extraction. Slow (30s-3min per paper on
    # qwen2.5:7b) but much more robust against weird PDF layouts than the
    # heuristic scraper. Disabled by default for backward compatibility.
    llm_extract: bool = False
    llm_extract_verify: bool = True  # second-pass self-review when llm_extract is on

    # Reranker — sits between ChromaDB retrieval and LLM entailment in
    # cite_finder. Improves recall when the right chunk is in the top-100
    # by semantic similarity but not the top-20. See bibwizard/search/reranker.py.
    reranker_enabled: bool = True
    # "auto" prefers cross-encoder if sentence-transformers is installed,
    # else falls back to the zero-dep lexical reranker. Override to "cross"
    # to require sentence-transformers, "lexical" to skip it entirely, or
    # "off" to disable reranking.
    reranker_kind: str = "auto"
    reranker_model: str = "BAAI/bge-reranker-base"
    # Pull `reranker_overscan × pool_size` chunks from Chroma so the
    # reranker has enough candidates to actually do work. 5x is a good
    # default; if pool_size = 20 we retrieve 100, rerank, take top 20.
    reranker_overscan: int = 5
    # Per-paper cap in the entailment pool. After reranking, no single
    # paper contributes more than this many chunks. Without it, a paper
    # with many semantically-similar chunks (e.g. a topical web article
    # or a noise document) can crowd out other papers from the search.
    # 3 is generous enough that a paper with multiple supporting passages
    # still has its evidence checked, strict enough that no single paper
    # dominates a 20-chunk pool.
    reranker_max_per_paper: int = 3

    # Which wizard animation to show during long-running operations:
    #   "walking" — ASCII wizard walks between desk and shelf (green)
    #   "reading" — ASCII wizard parked at desk reading a book (green)
    #   "pixel"   — higher-resolution colored pixel-art wizard
    # Override via WIZARD_SCENE env var.
    wizard_scene: str = "walking"

    # Misc derived
    chroma_collection: str = "bibwizard"
    user_agent: str = field(default="bibwizard/0.1 (+https://github.com/yaredreinarz/bibwizard)")

    @property
    def db_url(self) -> str:
        # SQLAlchemy URL — pathlib's as_posix() avoids Windows backslash issues
        return f"sqlite:///{self.sqlite_path.as_posix()}"


def _resolve_home() -> Path:
    raw = os.getenv("BIBWIZARD_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".bibwizard").resolve()


def load_settings() -> Settings:
    """Load .env (if present) and build a Settings instance."""
    # Load from CWD .env, then user's BIBWIZARD_HOME/.env if present.
    load_dotenv(override=False)
    home = _resolve_home()
    env_in_home = home / ".env"
    if env_in_home.exists():
        load_dotenv(dotenv_path=env_in_home, override=False)

    db_dir = _env_path("BIBWIZARD_DB_DIR", home / "db")
    vectors_dir = _env_path("BIBWIZARD_VECTORS_DIR", home / "vectors")
    papers_dir = _env_path("BIBWIZARD_PAPERS_DIR", home / "papers")
    summaries_dir = _env_path("BIBWIZARD_SUMMARIES_DIR", home / "summaries")
    # `literature/` defaults to a folder in the current working directory so
    # users can drop PDFs next to their project. Override via LITERATURE_DIR.
    literature_dir = _env_path("LITERATURE_DIR", (Path.cwd() / "literature").resolve())

    return Settings(
        home=home,
        db_dir=db_dir,
        vectors_dir=vectors_dir,
        papers_dir=papers_dir,
        summaries_dir=summaries_dir,
        literature_dir=literature_dir,
        sqlite_path=db_dir / "bibwizard.sqlite",
        ollama_host=_env_str("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
        ollama_llm_model=_env_str("OLLAMA_LLM_MODEL", "qwen2.5:7b-instruct-q4_K_M"),
        ollama_embed_model=_env_str("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        ollama_timeout=_env_int("OLLAMA_TIMEOUT", 120),
        chunk_size=_env_int("CHUNK_SIZE", 512),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 64),
        rag_top_k=_env_int("RAG_TOP_K", 5),
        log_level=_env_str("LOG_LEVEL", "INFO").upper(),
        auto_tag=_env_bool("AUTO_TAG", True),
        arxiv_api=_env_str("ARXIV_API", "https://export.arxiv.org/api/query"),
        ads_api_token=(os.getenv("ADS_API_TOKEN") or "").strip() or None,
        arxiv_min_delay=_env_float("ARXIV_MIN_DELAY", 5.0),
        unpaywall_email=_env_str("UNPAYWALL_EMAIL", "").strip(),
        llm_extract=_env_bool("LLM_EXTRACT_METADATA", False),
        llm_extract_verify=_env_bool("LLM_EXTRACT_VERIFY", True),
        reranker_enabled=_env_bool("RERANKER_ENABLED", True),
        reranker_kind=_env_str("RERANKER_KIND", "auto"),
        reranker_model=_env_str("RERANKER_MODEL", "BAAI/bge-reranker-base"),
        reranker_overscan=_env_int("RERANKER_OVERSCAN", 5),
        reranker_max_per_paper=_env_int("RERANKER_MAX_PER_PAPER", 3),
        wizard_scene=_env_str("WIZARD_SCENE", "walking"),
    )


def ensure_dirs(settings: Settings) -> None:
    """Create all required directories under BIBWIZARD_HOME (+ literature)."""
    for p in (
        settings.home,
        settings.db_dir,
        settings.vectors_dir,
        settings.papers_dir,
        settings.summaries_dir,
        settings.literature_dir,
    ):
        p.mkdir(parents=True, exist_ok=True)


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Silence noisy third-party libraries unless the user explicitly wants
    # debug output. httpx logs every Ollama call at INFO; chromadb / urllib3
    # are similarly chatty. sentence-transformers / transformers / huggingface
    # libraries log INFO during cross-encoder reranker loading — fine for
    # debugging, distracting during the normal cite spinner.
    if settings.log_level not in {"DEBUG"}:
        for noisy in (
            "httpx", "httpcore", "urllib3", "chromadb",
            "sentence_transformers", "transformers", "huggingface_hub",
        ):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    # Env-var-level silencing for libraries that print directly rather than
    # using the logging framework. Setting before they import means the
    # initial config picks these up. setdefault so user overrides still work.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    # tokenizers fork-warning during model load (cosmetic; doesn't affect us)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# Lazy singleton — most callers do `from bibwizard.utils.config import settings`
settings: Settings = load_settings()
configure_logging(settings)
