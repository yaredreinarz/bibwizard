"""Rich-based display helpers: tables, panels, progress, error formatting."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

console = Console()


# ---------- panels & messages ----------

def info(msg: str) -> None:
    console.print(f"[cyan]ℹ[/] {msg}")


def success(msg: str) -> None:
    console.print(f"[bold green]✓[/] {msg}")


def warn(msg: str) -> None:
    console.print(f"[yellow]⚠[/] {msg}")


def error(msg: str) -> None:
    console.print(f"[bold red]✗[/] {msg}")


def banner(title: str, subtitle: str | None = None) -> None:
    body = f"[bold]{title}[/]"
    if subtitle:
        body += f"\n[dim]{subtitle}[/]"
    console.print(Panel.fit(body, border_style="magenta"))


def panel(title: str, body: str, style: str = "cyan") -> None:
    console.print(Panel(body, title=title, border_style=style))


# ---------- tables ----------

def papers_table(rows: Sequence[dict]) -> Table:
    table = Table(title="Library", show_lines=False, header_style="bold cyan")
    table.add_column("ID", justify="right", style="dim", no_wrap=True)
    table.add_column("Title", overflow="fold")
    table.add_column("Authors", overflow="fold")
    table.add_column("Year", justify="right")
    table.add_column("Tags")
    for r in rows:
        table.add_row(
            str(r.get("id", "")),
            r.get("title", "(untitled)") or "(untitled)",
            r.get("authors", "") or "",
            str(r.get("year", "") or ""),
            r.get("tags", "") or "",
        )
    return table


def search_results_table(rows: Iterable[dict]) -> Table:
    table = Table(title="Semantic search", show_lines=False, header_style="bold cyan")
    table.add_column("Score", justify="right")
    table.add_column("Paper")
    table.add_column("Snippet", overflow="fold")
    for r in rows:
        table.add_row(
            f"{r.get('score', 0.0):.3f}",
            r.get("paper", ""),
            (r.get("snippet", "") or "").strip()[:240],
        )
    return table


def stats_table(stats: dict) -> Table:
    table = Table(title="Library stats", header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for k, v in stats.items():
        table.add_row(str(k), str(v))
    return table


# ---------- progress ----------

@contextmanager
def progress_bar(description: str = "Working") -> Iterator[Progress]:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        yield progress
