"""Semantic content map: cluster papers by mean chunk embedding."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from bibwizard.database.migrations import session_scope
from bibwizard.database.models import Paper
from bibwizard.ingestion.embedder import all_chunk_embeddings


def _paper_centroids() -> tuple[list[int], np.ndarray]:
    paper_ids, embs, _docs = all_chunk_embeddings()
    if not embs:
        return [], np.zeros((0, 0))
    bucket: dict[int, list[list[float]]] = defaultdict(list)
    for pid, e in zip(paper_ids, embs):
        if pid < 0:
            continue
        bucket[pid].append(e)
    if not bucket:
        return [], np.zeros((0, 0))
    ids = sorted(bucket.keys())
    arr = np.array([np.mean(np.array(bucket[pid]), axis=0) for pid in ids])
    return ids, arr


def cluster_papers(n_clusters: int | None = None) -> dict:
    ids, centroids = _paper_centroids()
    if len(ids) == 0:
        return {"clusters": [], "papers": []}

    # Heuristic: ~sqrt(n_papers), clamped to [2, 12], unless overridden.
    if n_clusters is None:
        n_clusters = max(2, min(12, int(round(len(ids) ** 0.5))))
    n_clusters = min(n_clusters, len(ids))

    if n_clusters == 1:
        labels = np.zeros(len(ids), dtype=int)
    else:
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(centroids)

    # Pull paper info for output
    info: dict[int, dict] = {}
    with session_scope() as session:
        for p in session.query(Paper).filter(Paper.id.in_(ids)).all():
            info[p.id] = {
                "id": p.id,
                "title": p.title,
                "year": p.year,
                "authors": [a.name for a in p.authors],
                "tags": [t.name for t in p.tags],
            }

    clusters: dict[int, list[dict]] = defaultdict(list)
    for pid, lab in zip(ids, labels):
        clusters[int(lab)].append(info.get(pid, {"id": pid, "title": "(unknown)"}))

    cluster_payload = []
    for cid, papers in sorted(clusters.items()):
        # Cheap label = top tag across the cluster, else most-common year.
        tag_count: dict[str, int] = defaultdict(int)
        for p in papers:
            for t in p.get("tags", []):
                tag_count[t] += 1
        if tag_count:
            label = max(tag_count.items(), key=lambda x: x[1])[0]
        else:
            label = f"cluster-{cid}"
        cluster_payload.append({"id": cid, "label": label, "size": len(papers)})

    return {
        "n_clusters": int(max(labels) + 1) if len(labels) else 0,
        "clusters": cluster_payload,
        "papers": [
            {**info.get(pid, {"id": pid}), "cluster": int(lab)}
            for pid, lab in zip(ids, labels)
        ],
    }


def export_clusters(path: Path, n_clusters: int | None = None) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = cluster_papers(n_clusters=n_clusters)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
