from __future__ import annotations

import itertools

import numpy as np
import scipy.sparse as sp

from .data import Corpus, QuerySet
from .structure import LegislativeStructure


def gold_size_stats(qs: QuerySet) -> dict:
    sizes = np.array([len(s) for s in qs.qrels])
    return {
        "n_questions": int(sizes.size),
        "mean_gold": float(sizes.mean()),
        "median_gold": float(np.median(sizes)),
        "max_gold": int(sizes.max()),
        "pct_multi_gold": float((sizes > 1).mean() * 100),
    }


def colocation_rates(
    qs: QuerySet,
    structure: LegislativeStructure,
    rng_seed: int = 0,
    n_random_pairs: int = 200_000,
) -> dict[str, dict]:
    rng = np.random.default_rng(rng_seed)
    n = structure.n_articles
    rand_a = rng.integers(0, n, n_random_pairs)
    rand_b = rng.integers(0, n, n_random_pairs)

    out = {}
    for lvl, g in structure.groups.items():
        same_obs, total_obs = 0, 0
        for gold in qs.qrels:
            if len(gold) < 2:
                continue
            for a, b in itertools.combinations(sorted(gold), 2):
                total_obs += 1
                same_obs += int(g[a] == g[b])
        base = float((g[rand_a] == g[rand_b]).mean())
        obs = same_obs / total_obs if total_obs else float("nan")
        out[lvl] = {
            "observed": obs * 100,
            "random_baseline": base * 100,
            "lift": (obs / base) if base > 0 else float("inf"),
            "n_pairs": total_obs,
        }
    return out


def _pair_distances(adj: sp.csr_matrix, pairs: list[tuple[int, int]],
                    max_hops: int = 2) -> np.ndarray:
    und = (adj + adj.T).tocsr()
    und.data[:] = 1.0
    two = (und @ und).tocsr()
    d = np.full(len(pairs), np.inf)
    for i, (a, b) in enumerate(pairs):
        if und[a, b] != 0:
            d[i] = 1
        elif max_hops >= 2 and two[a, b] != 0:
            d[i] = 2
    return d


def citation_connectivity(
    qs: QuerySet,
    structure: LegislativeStructure,
    rng_seed: int = 0,
    max_pairs: int = 4000,
) -> dict:
    rng = np.random.default_rng(rng_seed)
    pairs: list[tuple[int, int]] = []
    for gold in qs.qrels:
        if len(gold) < 2:
            continue
        pairs.extend(itertools.combinations(sorted(gold), 2))
    if len(pairs) > max_pairs:
        sel = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[i] for i in sel]

    n = structure.n_articles
    rand_pairs = [(int(a), int(b)) for a, b in
                  zip(rng.integers(0, n, len(pairs)),
                      rng.integers(0, n, len(pairs)))]

    d_obs = _pair_distances(structure.citations, pairs)
    d_rand = _pair_distances(structure.citations, rand_pairs)
    return {
        "n_pairs": len(pairs),
        "pct_1hop_observed": float((d_obs == 1).mean() * 100),
        "pct_le2hop_observed": float((d_obs <= 2).mean() * 100),
        "pct_1hop_random": float((d_rand == 1).mean() * 100),
        "pct_le2hop_random": float((d_rand <= 2).mean() * 100),
    }


def run_full_analysis(
    corpus: Corpus,
    qs: QuerySet,
    structure: LegislativeStructure,
    split_name: str = "train",
) -> str:
    lines = [f"## Structural locality of BSARD labels ({split_name})", ""]

    s = gold_size_stats(qs)
    lines.append(
        f"{s['n_questions']} questions; {s['mean_gold']:.2f} relevant articles "
        f"on average (median {s['median_gold']:.0f}, max {s['max_gold']}); "
        f"{s['pct_multi_gold']:.1f}% of questions have at least two."
    )
    lines.append("")
    lines.append("| Level | Co-located relevant pairs (%) | Random (%) | Lift |")
    lines.append("|---|---|---|---|")
    for lvl, r in colocation_rates(qs, structure).items():
        lines.append(f"| {lvl} | {r['observed']:.1f} | "
                     f"{r['random_baseline']:.3f} | x{r['lift']:.0f} |")
    lines.append("")

    if structure.citations.nnz:
        c = citation_connectivity(qs, structure)
        lines.append(
            f"Cross-references: {c['pct_1hop_observed']:.1f}% of relevant "
            f"article pairs are linked by a direct reference "
            f"({c['pct_1hop_random']:.2f}% at random); "
            f"{c['pct_le2hop_observed']:.1f}% within two hops "
            f"({c['pct_le2hop_random']:.2f}% at random)."
        )
    return "\n".join(lines)
