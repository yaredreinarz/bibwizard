"""SQLAlchemy ORM models: Paper, Author, Tag, Citation."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.orderinglist import ordering_list
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ---------- association tables / objects ----------

class PaperAuthor(Base):
    """Association object: link between a Paper and an Author with an explicit
    position so a paper's byline order is preserved independently of any other
    paper that happens to share the same authors."""

    __tablename__ = "paper_authors"

    paper_id: Mapped[int] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True
    )
    author_id: Mapped[int] = mapped_column(
        ForeignKey("authors.id", ondelete="CASCADE"), primary_key=True
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    paper: Mapped["Paper"] = relationship(back_populates="paper_authors")
    author: Mapped["Author"] = relationship(back_populates="paper_authors", lazy="joined")

    def __init__(
        self,
        author: "Author | None" = None,
        paper: "Paper | None" = None,
        position: int | None = None,
    ) -> None:
        if author is not None:
            self.author = author
        if paper is not None:
            self.paper = paper
        if position is not None:
            self.position = position


paper_tags = Table(
    "paper_tags",
    Base.metadata,
    Column("paper_id", ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True),
)


# ---------- entities ----------

class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(1024), nullable=False)
    doi: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(64), unique=True, nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    venue: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Local file path (kept as string but always built with pathlib)
    file_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    # Hash for de-dup
    sha256: Mapped[str | None] = mapped_column(String(64), unique=True, nullable=True)

    # Stored summary as JSON string
    summary_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # Number of chunks pushed into the vector store
    n_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Authors are linked via the PaperAuthor association object so we can
    # store an explicit `position` per (paper, author). `ordering_list`
    # auto-maintains position on append. The `authors` proxy keeps the old
    # `paper.authors.append(author)` API working.
    paper_authors: Mapped[list["PaperAuthor"]] = relationship(
        back_populates="paper",
        cascade="all, delete-orphan",
        order_by="PaperAuthor.position",
        collection_class=ordering_list("position"),
        lazy="selectin",
    )
    authors = association_proxy(
        "paper_authors",
        "author",
        creator=lambda author: PaperAuthor(author=author),
    )
    tags: Mapped[list["Tag"]] = relationship(
        "Tag", secondary=paper_tags, back_populates="papers", lazy="selectin"
    )
    outgoing_citations: Mapped[list["Citation"]] = relationship(
        "Citation",
        foreign_keys="Citation.source_paper_id",
        back_populates="source_paper",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Paper id={self.id} title={self.title[:40]!r}>"


class Author(Base):
    __tablename__ = "authors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)

    paper_authors: Mapped[list["PaperAuthor"]] = relationship(
        back_populates="author",
        cascade="all, delete-orphan",
    )
    papers = association_proxy(
        "paper_authors",
        "paper",
        creator=lambda paper: PaperAuthor(paper=paper),  # type: ignore[arg-type]
    )


class Tag(Base):
    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)

    papers: Mapped[list[Paper]] = relationship(
        "Paper", secondary=paper_tags, back_populates="tags"
    )


class Citation(Base):
    """A reference parsed out of a paper's bibliography section.

    `target_paper_id` is set when we can resolve the citation to a paper that
    is also in the local library (via DOI / arXiv id / fuzzy title match).
    """

    __tablename__ = "citations"
    __table_args__ = (
        UniqueConstraint("source_paper_id", "raw_text", name="uq_citation_source_raw"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_paper_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    target_paper_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("papers.id", ondelete="SET NULL"), nullable=True
    )

    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    target_title: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    target_doi: Mapped[str | None] = mapped_column(String(255), nullable=True)
    target_arxiv_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    target_year: Mapped[int | None] = mapped_column(Integer, nullable=True)

    source_paper: Mapped[Paper] = relationship(
        "Paper", foreign_keys=[source_paper_id], back_populates="outgoing_citations"
    )
    target_paper: Mapped[Paper | None] = relationship(
        "Paper", foreign_keys=[target_paper_id]
    )
