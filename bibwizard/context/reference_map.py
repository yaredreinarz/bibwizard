"""Citation graph builder + DOT/JSON exporter."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from bibwizard.database.migrations import session_scope
from bibwizard.database.models import Citation, Paper


def build_citation_graph() -> nx.DiGraph:
    """Build a directed graph: source paper → target paper (when resolvable).

    Unresolved citations are represented as 'external' nodes prefixed with
    `ext:` so they still show up in the export.
    """
    g = nx.DiGraph()
    with session_scope() as session:
        papers = session.query(Paper).all()
        for p in papers:
            g.add_node(
                f"p:{p.id}",
                kind="paper",
                title=p.title,
                year=p.year,
                authors=", ".join(a.name for a in p.authors),
                doi=p.doi or "",
                arxiv_id=p.arxiv_id or "",
            )
        citations = session.query(Citation).all()
        for c in citations:
            src = f"p:{c.source_paper_id}"
            if c.target_paper_id is not None:
                tgt = f"p:{c.target_paper_id}"
                if tgt not in g:
                    g.add_node(tgt, kind="paper", title="(unknown)", year=None)
            else:
                # External / unresolved
                key_parts = []
                if c.target_doi:
                    key_parts.append(f"doi:{c.target_doi}")
                elif c.target_arxiv_id:
                    key_parts.append(f"arxiv:{c.target_arxiv_id}")
                else:
                    key_parts.append((c.target_title or c.raw_text)[:80])
                tgt = f"ext:{'|'.join(key_parts)}"
                if tgt not in g:
                    g.add_node(
                        tgt,
                        kind="external",
                        title=c.target_title or c.raw_text[:120],
                        year=c.target_year,
                        doi=c.target_doi or "",
                        arxiv_id=c.target_arxiv_id or "",
                    )
            g.add_edge(src, tgt)
    return g


def export_dot(path: Path) -> Path:
    g = build_citation_graph()
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # nx.nx_pydot/agraph need graphviz. Use a hand-rolled DOT writer to avoid
    # the system-graphviz dependency.
    lines = ["digraph references {", "  rankdir=LR;", "  node [shape=box, style=rounded];"]
    for node, data in g.nodes(data=True):
        title = (data.get("title") or "").replace('"', "'")
        year = data.get("year") or ""
        kind = data.get("kind", "paper")
        label = f"{title}\\n{year}".strip()
        color = "lightblue" if kind == "paper" else "lightgrey"
        lines.append(f'  "{node}" [label="{label}", fillcolor="{color}", style="filled,rounded"];')
    for src, tgt in g.edges():
        lines.append(f'  "{src}" -> "{tgt}";')
    lines.append("}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def export_json(path: Path) -> Path:
    g = build_citation_graph()
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "nodes": [
            {"id": n, **{k: v for k, v in d.items()}} for n, d in g.nodes(data=True)
        ],
        "edges": [{"source": s, "target": t} for s, t in g.edges()],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
