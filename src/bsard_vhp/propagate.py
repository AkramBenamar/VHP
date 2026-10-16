from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .structure import LegislativeStructure, normalize_adjacency


@dataclass
class VHPParams:
    beta: float = 0.3
    gamma: float = 0.2
    top_m: int = 3
    level_weights: dict[str, float] = field(default_factory=dict)
    n_hops: int = 1
    hop_decay: float = 0.5
    direction: str = "sym"
    scheme: str = "row"
    exclude_self: bool = True

    def as_dict(self) -> dict:
        return {
            "beta": self.beta, "gamma": self.gamma, "top_m": self.top_m,
            "level_weights": dict(self.level_weights), "n_hops": self.n_hops,
            "hop_decay": self.hop_decay, "direction": self.direction,
            "scheme": self.scheme, "exclude_self": self.exclude_self,
        }


def zscore_rows(scores: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    mu = scores.mean(axis=1, keepdims=True)
    sd = scores.std(axis=1, keepdims=True)
    return (scores - mu) / (sd + eps)


class VerticalAggregator:

    def __init__(self, group_ids: np.ndarray):
        self.group_ids = group_ids
        order = np.argsort(group_ids, kind="stable")
        sorted_g = group_ids[order]
        boundaries = np.flatnonzero(np.diff(sorted_g)) + 1
        self.members = np.split(order, boundaries)
        self.n_groups = len(self.members)

    def __call__(
        self,
        z: np.ndarray,
        top_m: int = 3,
        exclude_self: bool = True,
    ) -> np.ndarray:
        q, n = z.shape
        out = np.zeros((q, n), dtype=z.dtype)

        for idx in self.members:
            if idx.size == 0:
                continue
            sub = z[:, idx]
            if idx.size == 1:
                continue

            m = min(top_m, idx.size - 1) if exclude_self else min(top_m, idx.size)
            m = max(m, 1)
            take = min(m + 1, idx.size) if exclude_self else m

            part = np.partition(sub, -take, axis=1)[:, -take:]
            part = -np.sort(-part, axis=1)

            if not exclude_self:
                out[:, idx] = part[:, :m].mean(axis=1, keepdims=True)
                continue

            top_sum = part[:, :m].sum(axis=1, keepdims=True)
            threshold = part[:, m - 1:m]
            filler = (part[:, m:m + 1] if part.shape[1] > m
                      else np.full((q, 1), -np.inf, dtype=z.dtype))

            in_top = sub >= threshold
            loo = np.where(in_top, top_sum - sub + filler, top_sum) / m
            loo = np.where(np.isfinite(loo), loo, top_sum / m)
            out[:, idx] = loo

        return out


def vertical_signal(
    z: np.ndarray,
    structure: LegislativeStructure,
    params: VHPParams,
    aggregators: dict[str, VerticalAggregator] | None = None,
) -> np.ndarray:
    weights = params.level_weights or default_level_weights(structure.levels)
    total = sum(w for w in weights.values() if w > 0)
    if total <= 0:
        return np.zeros_like(z)

    acc = np.zeros_like(z)
    for lvl, w in weights.items():
        if w <= 0 or lvl not in structure.groups:
            continue
        agg = (aggregators or {}).get(lvl) or VerticalAggregator(structure.groups[lvl])
        acc += (w / total) * agg(z, top_m=params.top_m,
                                 exclude_self=params.exclude_self)
    return acc


def default_level_weights(levels: list[str]) -> dict[str, float]:
    if not levels:
        return {}
    return {lvl: float(i + 1) for i, lvl in enumerate(levels)}


def horizontal_signal(
    z: np.ndarray,
    adj: sp.csr_matrix,
    n_hops: int = 1,
    hop_decay: float = 0.5,
) -> np.ndarray:
    acc = np.zeros_like(z)
    cur = z
    w = 1.0
    for _ in range(max(n_hops, 0)):
        cur = cur @ adj
        acc += w * np.asarray(cur)
        w *= hop_decay
    return acc


class VHP:

    def __init__(self, structure: LegislativeStructure, params: VHPParams | None = None):
        self.structure = structure
        self.params = params or VHPParams()
        self._aggregators = {
            lvl: VerticalAggregator(g) for lvl, g in structure.groups.items()
        }
        self._adj_cache: dict[tuple[str, str], sp.csr_matrix] = {}

    def adjacency(self, direction: str, scheme: str) -> sp.csr_matrix:
        key = (direction, scheme)
        if key not in self._adj_cache:
            self._adj_cache[key] = normalize_adjacency(
                self.structure.citations, direction=direction, scheme=scheme
            )
        return self._adj_cache[key]

    def __call__(
        self,
        base_scores: np.ndarray,
        params: VHPParams | None = None,
        return_parts: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        p = params or self.params
        z = zscore_rows(np.asarray(base_scores, dtype=np.float32))

        v = (vertical_signal(z, self.structure, p, self._aggregators)
             if p.beta != 0 else np.zeros_like(z))
        h = (horizontal_signal(z, self.adjacency(p.direction, p.scheme),
                               n_hops=p.n_hops, hop_decay=p.hop_decay)
             if p.gamma != 0 and self.structure.citations.nnz else np.zeros_like(z))

        final = z + p.beta * v + p.gamma * h
        if return_parts:
            return final, {"base": z, "vertical": v, "horizontal": h}
        return final

    def scorer(self, base_scores: np.ndarray) -> "VHPScorer":
        return VHPScorer(self, base_scores)

    def explain(
        self,
        base_scores: np.ndarray,
        query_idx: int,
        article_idx: int,
        params: VHPParams | None = None,
    ) -> dict:
        p = params or self.params
        _, parts = self(base_scores, params=p, return_parts=True)
        contrib = {
            "base": float(parts["base"][query_idx, article_idx]),
            "vertical": float(p.beta * parts["vertical"][query_idx, article_idx]),
            "horizontal": float(p.gamma * parts["horizontal"][query_idx, article_idx]),
        }
        neighbours = self.structure.citations[article_idx].indices.tolist()
        contrib["citation_neighbours"] = neighbours[:10]
        for lvl, g in self.structure.groups.items():
            same = np.flatnonzero(g == g[article_idx])
            contrib[f"siblings_{lvl}"] = int(same.size)
        return contrib


class VHPScorer:

    def __init__(self, vhp: "VHP", base_scores: np.ndarray):
        self.vhp = vhp
        self.z = zscore_rows(np.asarray(base_scores, dtype=np.float32))
        self._v: dict[str, np.ndarray] = {}
        self._v_top_m: int | None = None
        self._v_exclude: bool | None = None
        self._h: dict[tuple, np.ndarray] = {}

    def vertical(self, level_weights: dict[str, float], top_m: int,
                 exclude_self: bool = True) -> np.ndarray:
        if top_m != self._v_top_m or exclude_self != self._v_exclude:
            self._v.clear()
            self._v_top_m, self._v_exclude = top_m, exclude_self

        weights = {l: w for l, w in (level_weights or
                                     default_level_weights(self.vhp.structure.levels)
                                     ).items() if w > 0}
        total = sum(weights.values())
        if total <= 0:
            return np.zeros_like(self.z)

        acc = np.zeros_like(self.z)
        for lvl, w in weights.items():
            if lvl not in self._v:
                agg = self.vhp._aggregators[lvl]
                self._v[lvl] = agg(self.z, top_m=top_m, exclude_self=exclude_self)
            acc += (w / total) * self._v[lvl]
        return acc

    def horizontal(self, direction: str, scheme: str, n_hops: int,
                   hop_decay: float) -> np.ndarray:
        key = (direction, scheme, n_hops, hop_decay)
        if key not in self._h:
            if not self.vhp.structure.citations.nnz:
                self._h[key] = np.zeros_like(self.z)
            else:
                adj = self.vhp.adjacency(direction, scheme)
                self._h[key] = horizontal_signal(self.z, adj, n_hops, hop_decay)
            if len(self._h) > 8:
                self._h.pop(next(iter(self._h)))
        return self._h[key]

    def score(self, p: VHPParams) -> np.ndarray:
        out = self.z.copy()
        if p.beta:
            out += p.beta * self.vertical(p.level_weights, p.top_m, p.exclude_self)
        if p.gamma:
            out += p.gamma * self.horizontal(p.direction, p.scheme,
                                             p.n_hops, p.hop_decay)
        return out
