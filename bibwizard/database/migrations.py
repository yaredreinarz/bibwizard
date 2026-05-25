"""Database engine + schema initialization.

We rely on SQLAlchemy's `create_all()` for now (no Alembic). On `bibwizard init`
this creates the SQLite file and all tables; on subsequent runs it's a no-op.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from bibwizard.utils.config import ensure_dirs, settings

from .models import Base

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        ensure_dirs(settings)
        _engine = create_engine(
            settings.db_url,
            echo=False,
            future=True,
            connect_args={"check_same_thread": False},
        )
        _SessionLocal = sessionmaker(
            bind=_engine, autoflush=False, autocommit=False, expire_on_commit=False
        )
    return _engine


def init_db() -> None:
    """Create all tables (idempotent) and apply in-place migrations."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    _apply_migrations(engine)


def _apply_migrations(engine: Engine) -> None:
    """Lightweight migrations for older SQLite databases.

    SQLite supports `ALTER TABLE ... ADD COLUMN`. We only need additive
    changes for now, so this is enough — no Alembic dance required.
    """
    with engine.begin() as conn:
        # paper_authors needs a `position` column to preserve byline order
        # per-paper. Older DBs only have (paper_id, author_id).
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(paper_authors)"))}
        if cols and "position" not in cols:
            conn.execute(
                text(
                    "ALTER TABLE paper_authors "
                    "ADD COLUMN position INTEGER NOT NULL DEFAULT 0"
                )
            )
            # Backfill: within each paper, assign positions 0..N-1 by
            # author_id. That matches the OLD implicit ordering, so legacy
            # rows behave the same as before. Re-running `bibwizard
            # resummarize <id>` or `bibwizard edit <id> --authors ...` will
            # then refresh the order to match the real byline.
            conn.execute(
                text(
                    "UPDATE paper_authors SET position = ("
                    "SELECT COUNT(*) FROM paper_authors AS pa2 "
                    "WHERE pa2.paper_id = paper_authors.paper_id "
                    "AND pa2.author_id < paper_authors.author_id)"
                )
            )


@contextmanager
def session_scope() -> Iterator[Session]:
    """Yield a session that commits on exit, rolls back on error."""
    if _SessionLocal is None:
        get_engine()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
