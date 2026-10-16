from __future__ import annotations

import numpy as np

RECALL_KS = (100, 200, 500)
MAP_KS = (100,)
MRR_KS = (100,)

METRIC_ORDER = ["R@100", "R@200", "R@500", "MAP@100", "MRR@100", "MAP", "MRP"]


def rank_documents(scores: np.ndarray, top_k: int | None = None) -> np.ndarray:
    q, n = scores.shape
    top_k = n if top_k is None else min(top_k, n)
    if top_k < n:
        part = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1, kind="stable")
        return np.take_along_axis(part, order, axis=1)
    return np.argsort(-scores, axis=1, kind="stable")


def per_query_metrics(
    ranking: np.ndarray,
    qrels: list[set[int]],
    recall_ks: tuple[int, ...] = RECALL_KS,
    map_ks: tuple[int, ...] = MAP_KS,
    mrr_ks: tuple[int, ...] = MRR_KS,
    full_map: bool = True,
    mrp: bool = True,
) -> dict[str, np.ndarray]:
    n_q, n_ret = ranking.shape
    assert len(qrels) == n_q, "ranking and qrels are misaligned"

    out: dict[str, np.ndarray] = {}
    for k in recall_ks:
        out[f"R@{k}"] = np.zeros(n_q)
    for k in map_ks:
        out[f"MAP@{k}"] = np.zeros(n_q)
    for k in mrr_ks:
        out[f"MRR@{k}"] = np.zeros(n_q)
    if full_map:
        out["MAP"] = np.zeros(n_q)
    if mrp:
        out["MRP"] = np.zeros(n_q)

    for i in range(n_q):
        gold = qrels[i]
        n_gold = len(gold)
        if n_gold == 0:
            for key in out:
                out[key][i] = np.nan
            continue

        rel = np.fromiter((int(d in gold) for d in ranking[i]),
                          dtype=np.int8, count=n_ret)
        cum = np.cumsum(rel)
        hit_pos = np.flatnonzero(rel)

        for k in recall_ks:
            kk = min(k, n_ret)
            out[f"R@{k}"][i] = cum[kk - 1] / n_gold

        for k in map_ks:
            kk = min(k, n_ret)
            hits = hit_pos[hit_pos < kk]
            out[f"MAP@{k}"][i] = (
                (cum[hits] / (hits + 1)).sum() / n_gold if hits.size else 0.0)

        for k in mrr_ks:
            kk = min(k, n_ret)
            hits = hit_pos[hit_pos < kk]
            out[f"MRR@{k}"][i] = 1.0 / (hits[0] + 1) if hits.size else 0.0

        if full_map:
            out["MAP"][i] = ((cum[hit_pos] / (hit_pos + 1)).sum() / n_gold
                             if hit_pos.size else 0.0)
        if mrp:
            r = min(n_gold, n_ret)
            out["MRP"][i] = cum[r - 1] / n_gold

    return out


def aggregate(per_query: dict[str, np.ndarray]) -> dict[str, float]:
    return {m: float(np.nanmean(v) * 100.0) for m, v in per_query.items()}


def evaluate(
    scores: np.ndarray,
    qrels: list[set[int]],
    top_k: int | None = None,
    **kwargs,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    ranking = rank_documents(scores, top_k=top_k)
    pq = per_query_metrics(ranking, qrels, **kwargs)
    return aggregate(pq), pq


def format_table(rows: list[dict], metrics: list[str] | None = None) -> str:
    if not rows:
        return "(no results)"
    metrics = metrics or [m for m in METRIC_ORDER if m in rows[0]]
    header = "| System | " + " | ".join(metrics) + " |"
    sep = "|" + "---|" * (len(metrics) + 1)
    lines = [header, sep]
    for r in rows:
        cells = [f"{r[m]:.1f}" if isinstance(r.get(m), (int, float)) else "-"
                 for m in metrics]
        lines.append(f"| {r['system']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
