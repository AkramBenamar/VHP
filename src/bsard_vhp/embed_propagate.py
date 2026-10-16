from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import scipy.sparse as sp

from .metrics import evaluate
from .structure import LegislativeStructure, normalize_adjacency


@dataclass
class VHPEParams:
    beta: float = 0.3
    gamma: float = 0.2
    level_weights: dict[str, float] = field(default_factory=dict)
    direction: str = "out"
    scheme: str = "row"
    normalize: bool = False

    def as_dict(self) -> dict:
        return {"beta": self.beta, "gamma": self.gamma,
                "level_weights": dict(self.level_weights),
                "direction": self.direction, "scheme": self.scheme,
                "normalize": self.normalize}


def _group_indicator(group_ids: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray]:
    n = group_ids.size
    n_g = int(group_ids.max()) + 1
    ind = sp.csr_matrix(
        (np.ones(n, dtype=np.float32), (np.arange(n), group_ids)),
        shape=(n, n_g), dtype=np.float32)
    sizes = np.asarray(ind.sum(axis=0)).ravel()
    return ind, sizes


class EmbeddingSmoother:

    def __init__(self, structure: LegislativeStructure):
        self.structure = structure
        self._ind: dict[str, tuple[sp.csr_matrix, np.ndarray]] = {}
        for lvl, g in structure.groups.items():
            self._ind[lvl] = _group_indicator(g)
        self._adj: dict[tuple[str, str], sp.csr_matrix] = {}

    def level_signal(self, emb: np.ndarray, level: str) -> np.ndarray:
        ind, sizes = self._ind[level]
        g = self.structure.groups[level]
        sums = ind.T @ emb
        n_i = sizes[g][:, None]
        out = np.zeros_like(emb)
        multi = (n_i > 1).ravel()
        out[multi] = (sums[g[multi]] - emb[multi]) / (n_i[multi] - 1.0)
        return out

    def citation_signal(self, emb: np.ndarray, direction: str = "out",
                        scheme: str = "row") -> np.ndarray:
        if not self.structure.citations.nnz:
            return np.zeros_like(emb)
        key = (direction, scheme)
        if key not in self._adj:
            self._adj[key] = normalize_adjacency(
                self.structure.citations, direction=direction, scheme=scheme)
        return np.asarray(self._adj[key] @ emb, dtype=emb.dtype)

    def smooth(self, emb: np.ndarray, p: VHPEParams) -> np.ndarray:
        emb = np.asarray(emb, dtype=np.float32)
        out = emb.copy()
        weights = {l: w for l, w in (p.level_weights or {}).items() if w > 0}
        total = sum(weights.values())
        if p.beta and total > 0:
            acc = np.zeros_like(emb)
            for lvl, w in weights.items():
                acc += (w / total) * self.level_signal(emb, lvl)
            out += p.beta * acc
        if p.gamma:
            out += p.gamma * self.citation_signal(emb, p.direction, p.scheme)
        if p.normalize:
            n = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.maximum(n, 1e-9)
        return out


class VHPEScorer:

    def __init__(self, smoother: EmbeddingSmoother, doc_emb: np.ndarray,
                 query_emb: np.ndarray, direction: str = "out",
                 scheme: str = "row"):
        self.smoother = smoother
        q = np.asarray(query_emb, dtype=np.float32)
        self.base = q @ np.asarray(doc_emb, dtype=np.float32).T
        self.level: dict[str, np.ndarray] = {}
        for lvl in smoother.structure.levels:
            self.level[lvl] = q @ smoother.level_signal(doc_emb, lvl).T
        self.cite = q @ smoother.citation_signal(doc_emb, direction, scheme).T

    def score(self, p: VHPEParams) -> np.ndarray:
        out = self.base.copy()
        weights = {l: w for l, w in (p.level_weights or {}).items() if w > 0}
        total = sum(weights.values())
        if p.beta and total > 0:
            for lvl, w in weights.items():
                out += p.beta * (w / total) * self.level[lvl]
        if p.gamma:
            out += p.gamma * self.cite
        return out


def tune_vhpe(
    smoother: EmbeddingSmoother,
    doc_emb: np.ndarray,
    query_emb_train: np.ndarray,
    qrels: list[set[int]],
    objective: str = "R@100",
    betas: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8),
    gammas: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.3),
    directions: tuple[str, ...] = ("out", "sym"),
    top_k: int = 2000,
    verbose: bool = True,
) -> tuple[VHPEParams, dict]:
    from .tune import level_weight_schemes

    levels = smoother.structure.levels
    schemes = level_weight_schemes(levels)
    trace: dict = {"vertical": [], "horizontal": []}

    def ev(scorer: VHPEScorer, p: VHPEParams) -> float:
        agg, _ = evaluate(scorer.score(p), qrels, top_k=top_k)
        return agg[objective]

    scorers = {d: VHPEScorer(smoother, doc_emb, query_emb_train, direction=d)
               for d in directions}
    ref = scorers[directions[0]]
    baseline = ev(ref, VHPEParams(beta=0.0, gamma=0.0))
    trace["baseline"] = baseline
    if verbose:
        print(f"  [vhpe] base {objective} = {baseline:.2f}")

    best = VHPEParams(beta=0.0, gamma=0.0, level_weights={})
    best_score = baseline
    for sname, w in schemes.items():
        for beta in betas:
            p = VHPEParams(beta=beta, gamma=0.0, level_weights=dict(w))
            s = ev(ref, p)
            trace["vertical"].append({"scheme": sname, "beta": beta, objective: s})
            if s > best_score:
                best_score, best = s, p
                if verbose:
                    print(f"  [vhpe-V] {objective}={s:.2f}  beta={beta} "
                          f"levels={sname}")

    best_v, best_v_score = best, best_score
    for d in directions:
        for gamma in gammas:
            if gamma == 0:
                continue
            p = replace(best_v, gamma=gamma, direction=d)
            s = ev(scorers[d], p)
            trace["horizontal"].append({"direction": d, "gamma": gamma,
                                        objective: s})
            if s > best_score:
                best_score, best = s, p
                if verbose:
                    print(f"  [vhpe-H] {objective}={s:.2f}  gamma={gamma} "
                          f"dir={d}")

    trace["selected"] = best.as_dict()
    trace["selected_score"] = best_score
    trace["gain_train"] = best_score - baseline
    if verbose:
        print(f"  [vhpe] selected: {objective}={best_score:.2f} "
              f"(+{best_score - baseline:.2f})")
    return best, trace


def apply_vhpe(smoother: EmbeddingSmoother, doc_emb: np.ndarray,
               query_emb: np.ndarray, p: VHPEParams) -> np.ndarray:
    smoothed = smoother.smooth(doc_emb, p)
    q = np.asarray(query_emb, dtype=np.float32)
    if p.normalize:
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
    return q @ smoothed.T
