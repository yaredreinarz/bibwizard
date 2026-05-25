"""Interactive HTML graph visualization for the bibwizard library.

Two graph types:

  * **Author co-authorship** — nodes are papers, edges connect papers that
    share at least one author. Edge weight = number of shared authors.
  * **Content similarity** — nodes are papers, edges connect papers whose
    mean ChromaDB chunk-embeddings exceed a cosine-similarity threshold.

Each graph is rendered as a self-contained HTML page using vis-network
(loaded from CDN), so the user just double-clicks the file to interact.
Drag nodes, scroll to zoom, hover to see title + authors.
"""

from __future__ import annotations

import html as _html
import json
from collections import defaultdict
from pathlib import Path

import networkx as nx

from bibwizard.database.migrations import session_scope
from bibwizard.database.models import Paper


# ---------- graph builders ----------

def build_author_graph() -> nx.Graph:
    """Build the paper-paper co-authorship graph.

    Nodes carry: id, title, year, authors, first_author, n_chunks.
    Edges carry: weight (# shared authors), shared (list of shared names).
    """
    g = nx.Graph()
    with session_scope() as session:
        papers = session.query(Paper).order_by(Paper.id).all()
        # Snapshot the data so the session can close before we add edges.
        rows = []
        for p in papers:
            authors = [a.name for a in p.authors]
            rows.append(
                {
                    "id": p.id,
                    "title": p.title or "(untitled)",
                    "year": p.year,
                    "authors": authors,
                    "first_author": authors[0] if authors else "",
                    "n_chunks": p.n_chunks,
                    "doi": p.doi,
                    "arxiv_id": p.arxiv_id,
                }
            )

    for r in rows:
        g.add_node(r["id"], **r)

    # Edge for every pair of papers sharing at least one author.
    for i, r1 in enumerate(rows):
        s1 = set(r1["authors"])
        for r2 in rows[i + 1 :]:
            shared = s1 & set(r2["authors"])
            if shared:
                g.add_edge(r1["id"], r2["id"], weight=len(shared), shared=sorted(shared))
    return g


def build_content_graph(threshold: float = 0.65, top_k_per_node: int = 4) -> nx.Graph:
    """Paper-paper graph: edge when mean-embedding cosine ≥ `threshold`.

    Also caps edges per node to `top_k_per_node` strongest neighbors so very
    similar clusters don't drown out weaker but informative connections.
    """
    # Imported here so the visualize module doesn't pull numpy/sklearn at
    # CLI startup time for users who never call map content.
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    from bibwizard.ingestion.embedder import all_chunk_embeddings

    g = nx.Graph()
    paper_ids, embs, _ = all_chunk_embeddings()
    if not embs:
        return g

    bucket: dict[int, list[list[float]]] = defaultdict(list)
    for pid, e in zip(paper_ids, embs):
        if pid is not None and pid >= 0:
            bucket[pid].append(e)
    if not bucket:
        return g

    pids = sorted(bucket.keys())
    centroids = np.array([np.mean(np.array(bucket[pid]), axis=0) for pid in pids])
    sims = cosine_similarity(centroids)

    # Pull paper info while session is open
    with session_scope() as session:
        papers_by_id = {
            p.id: {
                "id": p.id,
                "title": p.title or "(untitled)",
                "year": p.year,
                "authors": [a.name for a in p.authors],
                "first_author": p.authors[0].name if p.authors else "",
                "n_chunks": p.n_chunks,
                "doi": p.doi,
                "arxiv_id": p.arxiv_id,
            }
            for p in session.query(Paper).filter(Paper.id.in_(pids)).all()
        }

    for pid in pids:
        info = papers_by_id.get(pid, {"id": pid, "title": "(unknown)"})
        g.add_node(pid, **info)

    n = len(pids)
    for i in range(n):
        # Rank neighbours by similarity desc, keep top_k above threshold
        order = np.argsort(-sims[i])
        kept = 0
        for j in order:
            if i == j:
                continue
            s = float(sims[i][j])
            if s < threshold or kept >= top_k_per_node:
                break
            pi, pj = pids[i], pids[int(j)]
            if g.has_edge(pi, pj):
                kept += 1
                continue
            g.add_edge(pi, pj, weight=s)
            kept += 1
    return g


# ---------- HTML rendering ----------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  html, body { margin: 0; padding: 0; height: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  #wrap { display: flex; height: 100vh; }
  #side { width: 320px; padding: 14px; box-sizing: border-box; border-right: 1px solid #ccc; overflow-y: auto; background: #fafafa; }
  #network { flex: 1; }
  h1 { margin: 0 0 6px 0; font-size: 16px; }
  h2 { margin: 14px 0 4px 0; font-size: 13px; color: #444; }
  .hint { color: #666; font-size: 12px; }
  .selected { font-size: 12px; }
  .selected b { color: #222; }
  .selected .meta { color: #555; }
  .legend { font-size: 11px; }
  .legend .sw { display: inline-block; width: 10px; height: 10px; vertical-align: middle; margin-right: 6px; border-radius: 50%; }
</style>
</head>
<body>
<div id="wrap">
  <div id="side">
    <h1>__TITLE__</h1>
    <p class="hint">__HINT__</p>
    <h2>Selected paper</h2>
    <div class="selected" id="sel"><span class="meta">Click a node to inspect.</span></div>
    <h2>Legend</h2>
    <div class="legend" id="legend"></div>
    <h2>Stats</h2>
    <div class="legend" id="stats">__STATS__</div>
  </div>
  <div id="network"></div>
</div>

<script>
const NODES_DATA = __NODES__;
const EDGES_DATA = __EDGES__;
const COLOR_FIELD = "__COLOR_FIELD__";

// Color palette (Tableau 10)
const PALETTE = ["#4e79a7","#f28e2c","#e15759","#76b7b2","#59a14f","#edc949","#af7aa1","#ff9da7","#9c755f","#bab0ac","#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"];
const groupColors = {};
let colorIdx = 0;
function colorFor(group) {
  if (group === null || group === undefined || group === "") group = "?";
  if (!(group in groupColors)) {
    groupColors[group] = PALETTE[colorIdx % PALETTE.length];
    colorIdx++;
  }
  return groupColors[group];
}

const nodes = new vis.DataSet(NODES_DATA.map(n => ({
  id: n.id,
  label: n.short_label,
  title: n.tooltip,   // hover tooltip (HTML)
  value: n.size,
  color: { background: colorFor(n[COLOR_FIELD]), border: "#333" },
  font: { size: 13, face: "Segoe UI, sans-serif" },
  fullData: n,
})));
const edges = new vis.DataSet(EDGES_DATA.map(e => ({
  from: e.from, to: e.to,
  value: e.weight,
  title: e.tooltip,
  color: { color: "#aaa", highlight: "#222", opacity: 0.5 },
  smooth: false,
})));

const options = {
  nodes: {
    shape: "dot",
    scaling: { min: 12, max: 38, label: { enabled: true, min: 10, max: 18 } },
    borderWidth: 1,
  },
  edges: {
    scaling: { min: 1, max: 6 },
    smooth: { type: "continuous" },
  },
  physics: {
    solver: "barnesHut",
    barnesHut: { gravitationalConstant: -8000, springLength: 140, avoidOverlap: 0.3 },
    stabilization: { iterations: 250 },
  },
  interaction: { hover: true, tooltipDelay: 200 },
};

const net = new vis.Network(document.getElementById("network"), { nodes, edges }, options);

net.on("selectNode", (params) => {
  if (!params.nodes.length) return;
  const nid = params.nodes[0];
  const n = NODES_DATA.find(x => x.id === nid);
  if (!n) return;
  const author_lines = (n.authors && n.authors.length)
    ? n.authors.slice(0, 12).join(", ") + (n.authors.length > 12 ? ", …" : "")
    : "(none)";
  document.getElementById("sel").innerHTML = `
    <b>${escapeHtml(n.title)}</b><br>
    <span class="meta">Year: ${n.year ?? "?"} &nbsp;·&nbsp; id: ${n.id} &nbsp;·&nbsp; chunks: ${n.n_chunks ?? "?"}</span><br>
    <span class="meta">Authors: ${escapeHtml(author_lines)}</span><br>
    ${n.doi ? `<span class="meta">DOI: ${escapeHtml(n.doi)}</span><br>` : ""}
    ${n.arxiv_id ? `<span class="meta">arXiv: ${escapeHtml(n.arxiv_id)}</span><br>` : ""}
  `;
});
net.on("deselectNode", () => {
  document.getElementById("sel").innerHTML = '<span class="meta">Click a node to inspect.</span>';
});

// Render the legend
function renderLegend() {
  const el = document.getElementById("legend");
  const entries = Object.entries(groupColors).sort();
  el.innerHTML = entries.map(([g, c]) =>
    `<div><span class="sw" style="background:${c}"></span>${escapeHtml(String(g))}</div>`
  ).join("");
}
net.once("afterDrawing", renderLegend);

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}[c]));
}
</script>
</body>
</html>
"""


def _short_label(info: dict) -> str:
    """Single-line node label: 'Surname YYYY' or just the surname if no year."""
    first = info.get("first_author") or ""
    surname = ""
    if first:
        if "," in first:
            surname = first.split(",", 1)[0].strip()
        else:
            tokens = first.split()
            surname = tokens[-1] if tokens else first
    year = info.get("year")
    if surname and year:
        return f"{surname} {year}"
    if surname:
        return surname
    return f"#{info['id']}"


def _node_payload(info: dict, cluster: int | None = None) -> dict:
    return {
        "id": info["id"],
        "title": info.get("title") or "(untitled)",
        "year": info.get("year"),
        "authors": list(info.get("authors") or []),
        "first_author": info.get("first_author"),
        "n_chunks": info.get("n_chunks") or 1,
        "doi": info.get("doi"),
        "arxiv_id": info.get("arxiv_id"),
        "short_label": _short_label(info),
        "tooltip": _make_tooltip(info),
        "size": max(8, min(40, int((info.get("n_chunks") or 1) ** 0.5 * 4))),
        "year_str": str(info.get("year") or "?"),
        "first_author_surname": _short_label(info).split()[0] if _short_label(info) else "?",
        "cluster": cluster,
    }


def _make_tooltip(info: dict) -> str:
    """Plain-text tooltip — vis-network HTML-escapes by default."""
    authors = info.get("authors") or []
    a = ", ".join(authors[:6]) + (" …" if len(authors) > 6 else "")
    bits = [
        info.get("title") or "(untitled)",
        f"{a}" if a else "",
        f"Year: {info.get('year') or '?'}",
        f"id: {info['id']}",
    ]
    return "\n".join(b for b in bits if b)


def render_author_graph_html(g: nx.Graph, out_path: Path) -> Path:
    """Render the co-authorship graph; color nodes by year."""
    nodes_payload = []
    for nid, data in g.nodes(data=True):
        nodes_payload.append(_node_payload(data))
    edges_payload = []
    for u, v, data in g.edges(data=True):
        shared = data.get("shared") or []
        edges_payload.append(
            {
                "from": u,
                "to": v,
                "weight": data.get("weight", 1),
                "tooltip": f"shared: {', '.join(shared[:6])}"
                + (" …" if len(shared) > 6 else ""),
            }
        )

    stats = (
        f"papers: {g.number_of_nodes()}<br>"
        f"edges (≥1 shared author): {g.number_of_edges()}<br>"
        f"isolated papers: {len([n for n in g.nodes() if g.degree(n) == 0])}"
    )

    html = (
        _HTML_TEMPLATE
        .replace("__TITLE__", "bibwizard — author co-authorship")
        .replace(
            "__HINT__",
            "Each node is a paper; an edge means the two papers share at least one author. "
            "Node size scales with vector-chunk count. Node color = publication year.",
        )
        .replace("__NODES__", json.dumps(nodes_payload, ensure_ascii=False))
        .replace("__EDGES__", json.dumps(edges_payload, ensure_ascii=False))
        .replace("__COLOR_FIELD__", "year_str")
        .replace("__STATS__", stats)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def render_content_graph_html(g: nx.Graph, out_path: Path, threshold: float = 0.65) -> Path:
    """Render the content-similarity graph; color nodes by k-means cluster."""
    # Cluster the nodes for color assignment.
    cluster_for = _assign_clusters(g)

    nodes_payload = []
    for nid, data in g.nodes(data=True):
        nodes_payload.append(_node_payload(data, cluster=cluster_for.get(nid)))
    edges_payload = []
    for u, v, data in g.edges(data=True):
        edges_payload.append(
            {
                "from": u,
                "to": v,
                "weight": data.get("weight", 0),
                "tooltip": f"cosine similarity: {data.get('weight', 0):.3f}",
            }
        )

    stats = (
        f"papers (with embeddings): {g.number_of_nodes()}<br>"
        f"edges (cosine ≥ {threshold}): {g.number_of_edges()}<br>"
        f"clusters: {len(set(cluster_for.values()))}"
    )
    html = (
        _HTML_TEMPLATE
        .replace("__TITLE__", "bibwizard — content similarity")
        .replace(
            "__HINT__",
            f"Each node is a paper; an edge means cosine similarity of mean chunk "
            f"embeddings ≥ {threshold}. Color = k-means cluster on those embeddings.",
        )
        .replace("__NODES__", json.dumps(nodes_payload, ensure_ascii=False))
        .replace("__EDGES__", json.dumps(edges_payload, ensure_ascii=False))
        .replace("__COLOR_FIELD__", "cluster")
        .replace("__STATS__", stats)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _assign_clusters(g: nx.Graph) -> dict[int, int]:
    """Heuristic clustering for coloring: use connected components if dense
    enough, else delegate to the existing content_map clusterer."""
    if g.number_of_edges() == 0:
        return {nid: 0 for nid in g.nodes()}

    components = list(nx.connected_components(g))
    if len(components) >= 2:
        out: dict[int, int] = {}
        for ci, comp in enumerate(components):
            for nid in comp:
                out[nid] = ci
        return out

    # Single big component — fall back to k-means via the existing routine
    try:
        from bibwizard.context.content_map import cluster_papers

        cp = cluster_papers()
        return {p["id"]: int(p["cluster"]) for p in cp.get("papers", [])}
    except Exception:
        return {nid: 0 for nid in g.nodes()}
