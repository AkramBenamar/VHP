from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .data import Corpus, norm_key, normalize_article_number

_NUM = (r"(?:[A-Za-z]{1,2}\.?)?\d+(?:\.\d+)*(?:\s*er\b)?"
        r"(?:\s*(?:bis|ter|quater|quinquies|sexies|septies"
        r"|octies|novies|decies))?(?:\s*/\s*\d+)?")

_CITATION_RE = re.compile(
    rf"\bart(?:icle)?s?\.?\s*"
    rf"(?P<nums>{_NUM}(?:\s*(?:,|;|\bet\b|\bou\b|\bà\b|\bau\b|-)\s*{_NUM})*)"
    rf"(?P<tail>[^.;:()]{{0,80}})?",
    flags=re.IGNORECASE,
)

_RANGE_RE = re.compile(rf"({_NUM})\s*(?:à|au|-)\s*({_NUM})", re.IGNORECASE)
_SPLIT_RE = re.compile(r"\s*(?:,|;|\bet\b|\bou\b)\s*", re.IGNORECASE)

_CODE_RE = re.compile(
    r"\bdu\s+(?P<code>Code\s+[^,.;:()]{2,60}"
    r"|présent\s+code|même\s+code|présente\s+loi|même\s+loi)",
    flags=re.IGNORECASE,
)

MAX_RANGE_EXPANSION = 40


@dataclass
class LegislativeStructure:
    n_articles: int
    levels: list[str]
    groups: dict[str, np.ndarray]
    citations: sp.csr_matrix
    n_citation_edges: int = 0
    resolution_rate: float = 0.0
    citation_stats: dict = field(default_factory=dict)


def _self_code_column(corpus: Corpus) -> list[str]:
    df = corpus.raw
    for cand in ("code", "law_type", "reference"):
        if df is not None and cand in {c.lower() for c in df.columns}:
            col = [c for c in df.columns if c.lower() == cand][0]
            return [norm_key(v) for v in df[col].fillna("").astype(str)]
    return [""] * len(corpus)


def _expand_numbers(nums_blob: str) -> list[str]:
    out: list[str] = []
    for chunk in _SPLIT_RE.split(nums_blob):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = _RANGE_RE.fullmatch(chunk)
        if m:
            a, b = m.group(1), m.group(2)
            if a.strip().isdigit() and b.strip().isdigit():
                lo, hi = int(a), int(b)
                if 0 < hi - lo <= MAX_RANGE_EXPANSION:
                    out.extend(str(x) for x in range(lo, hi + 1))
                    continue
            out.extend([normalize_article_number(a), normalize_article_number(b)])
        else:
            n = normalize_article_number(chunk)
            if n:
                out.append(n)
    return out


def build_citation_graph(
    corpus: Corpus,
    ambiguity: str = "nearest",
    verbose: bool = True,
) -> tuple[sp.csr_matrix, dict]:
    n = len(corpus)
    self_codes = _self_code_column(corpus)

    index: dict[tuple[str, str], list[int]] = {}
    by_number: dict[str, list[int]] = {}
    for i, num in enumerate(corpus.numbers):
        if not num:
            continue
        index.setdefault((self_codes[i], num), []).append(i)
        by_number.setdefault(num, []).append(i)

    levels = list(corpus.hierarchy.values())
    level_mat = (np.stack(levels, axis=1) if levels
                 else np.zeros((n, 0), dtype=np.int64))

    def shared_depth(a: int, b: int) -> int:
        d = 0
        for k in range(level_mat.shape[1]):
            if level_mat[a, k] != level_mat[b, k]:
                break
            d += 1
        return d

    rows, cols = [], []
    attempted = resolved = ambiguous = 0

    for i, text in enumerate(corpus.texts):
        if not text:
            continue
        seen: set[int] = set()
        for m in _CITATION_RE.finditer(text):
            tail = m.group("tail") or ""
            code_m = _CODE_RE.search(tail)
            target_code = self_codes[i]
            if code_m:
                raw_code = norm_key(code_m.group("code"))
                if raw_code not in ("present code", "meme code",
                                    "presente loi", "meme loi"):
                    target_code = raw_code

            for num in _expand_numbers(m.group("nums")):
                attempted += 1
                cands = index.get((target_code, num))
                if not cands:
                    cands = index.get((self_codes[i], num))
                if not cands:
                    alt = by_number.get(num, [])
                    cands = alt if len(alt) == 1 else None
                if not cands:
                    continue

                if len(cands) > 1:
                    ambiguous += 1
                    if ambiguity == "skip":
                        continue
                    if ambiguity == "nearest":
                        cands = [max(cands, key=lambda j: (shared_depth(i, j),
                                                           -abs(j - i)))]
                resolved += 1
                for j in cands:
                    if j != i and j not in seen:
                        seen.add(j)
                        rows.append(i)
                        cols.append(j)

    data = np.ones(len(rows), dtype=np.float32)
    c = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    c.sum_duplicates()
    c.data[:] = 1.0

    stats = {
        "attempted": attempted,
        "resolved": resolved,
        "ambiguous": ambiguous,
        "resolution_rate": resolved / attempted if attempted else 0.0,
        "ambiguity_rate": ambiguous / attempted if attempted else 0.0,
        "edges": int(c.nnz),
        "edges_per_resolved": c.nnz / resolved if resolved else 0.0,
        "policy": ambiguity,
    }
    if verbose:
        print(f"[structure] references: {attempted} detected, {resolved} resolved "
              f"({stats['resolution_rate']:.1%}), {ambiguous} ambiguous "
              f"({stats['ambiguity_rate']:.1%}, policy '{ambiguity}')")
        print(f"[structure] graph: {c.nnz} edges "
              f"({stats['edges_per_resolved']:.2f} per resolved reference), "
              f"{(c.getnnz(axis=1) > 0).mean():.1%} of articles cite")
    return c, stats


def normalize_adjacency(
    a: sp.csr_matrix,
    direction: str = "sym",
    scheme: str = "row",
) -> sp.csr_matrix:
    if direction == "out":
        m = a
    elif direction == "in":
        m = a.T.tocsr()
    elif direction == "sym":
        m = (a + a.T).tocsr()
        m.data[:] = np.minimum(m.data, 1.0)
    else:
        raise ValueError(f"unknown direction: {direction}")

    m = m.astype(np.float32)
    if scheme == "row":
        deg = np.asarray(m.sum(axis=1)).ravel()
        inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
        return sp.diags(inv.astype(np.float32)) @ m
    if scheme == "gcn":
        deg = np.asarray(m.sum(axis=1)).ravel()
        inv = np.divide(1.0, np.sqrt(deg), out=np.zeros_like(deg), where=deg > 0)
        d = sp.diags(inv.astype(np.float32))
        return (d @ m @ d).tocsr()
    raise ValueError(f"unknown scheme: {scheme}")


def build_structure(corpus: Corpus, ambiguity: str = "nearest",
                    verbose: bool = True) -> LegislativeStructure:
    citations, cstats = build_citation_graph(corpus, ambiguity=ambiguity,
                                             verbose=verbose)
    levels = list(corpus.hierarchy.keys())
    if verbose:
        for lvl in levels:
            g = corpus.hierarchy[lvl]
            sizes = np.bincount(g)
            print(f"[structure] level '{lvl}': {len(sizes)} groups, "
                  f"median size {np.median(sizes):.0f}, max {sizes.max()}")
    return LegislativeStructure(
        n_articles=len(corpus),
        levels=levels,
        groups=dict(corpus.hierarchy),
        citations=citations,
        n_citation_edges=int(citations.nnz),
        resolution_rate=cstats["resolution_rate"],
        citation_stats=cstats,
    )
