from __future__ import annotations

import numpy as np

from .data import Corpus, QuerySet
from .structure import LegislativeStructure

OK, WARN, FAIL = "[OK]     ", "[WARNING] ", "[FAIL]   "


def _fmt_sizes(g: np.ndarray) -> tuple[int, float, int, float]:
    sizes = np.bincount(g)
    sizes = sizes[sizes > 0]
    singletons = float((sizes == 1).mean() * 100)
    return len(sizes), float(np.median(sizes)), int(sizes.max()), singletons


def diagnose_corpus(corpus: Corpus) -> list[str]:
    out = [f"CORPUS: {len(corpus)} articles"]
    ids = np.asarray(corpus.ids)
    contiguous = len(ids) == int(ids.max()) - int(ids.min()) + 1
    out.append(f"  identifiers {ids.min()}..{ids.max()}, "
               f"contiguous: {'yes' if contiguous else 'no'} "
               "(id/index offsets are handled by the lookup table)")

    n_num = sum(1 for x in corpus.numbers if x)
    pct = n_num / max(len(corpus), 1) * 100
    tag = OK if pct > 90 else (WARN if pct > 60 else FAIL)
    out.append(f"{tag}article number extracted for {pct:.1f}% of articles")
    if pct <= 90:
        missing = [i for i, x in enumerate(corpus.numbers) if not x][:3]
        for i in missing:
            out.append(f"          failed on: {corpus.references[i][:80]!r}")
        out.append("          -> adjust _ART_NUM_RE in data.py")

    empty = sum(1 for t in corpus.texts if not t.strip())
    if empty:
        out.append(f"{WARN}{empty} articles with empty text")
    return out


def diagnose_hierarchy(corpus: Corpus, structure: LegislativeStructure) -> list[str]:
    out = ["", "VERTICAL AXIS (legislative hierarchy)"]
    if not structure.levels:
        out.append(f"{FAIL}no level detected: the vertical axis has no effect.")
        return out

    out.append(f"  {len(structure.levels)} cumulative levels: "
               f"{', '.join(structure.levels)}")
    out.append("")
    out.append("  | level | groups | median size | max | singletons |")
    out.append("  |---|---|---|---|---|")
    usable = 0
    for lvl in structure.levels:
        n_g, med, mx, sing = _fmt_sizes(structure.groups[lvl])
        out.append(f"  | {lvl} | {n_g} | {med:.0f} | {mx} | {sing:.0f}% |")
        if n_g > 1 and sing < 80 and mx < 0.5 * len(corpus):
            usable += 1
    out.append("")
    if usable >= 2:
        out.append(f"{OK}{usable} usable levels -> the vertical axis carries signal.")
    elif usable == 1:
        out.append(f"{WARN}only one usable level: the vertical gain will be limited.")
    else:
        out.append(f"{FAIL}no usable level (singletons only, or a single group). "
                   "Check the hierarchy columns.")

    if corpus.raw is not None:
        redundant = []
        prev = None
        for lvl in structure.levels:
            g = structure.groups[lvl]
            if prev is not None and len(np.unique(g)) == len(np.unique(prev)):
                redundant.append(lvl)
            prev = g
        if redundant:
            out.append(f"  note: {', '.join(redundant)} do not subdivide the "
                       "parent level (column often empty); with no effect, "
                       "calibration will give them zero weight.")
    return out


def diagnose_citations(corpus: Corpus, structure: LegislativeStructure) -> list[str]:
    out = ["", "HORIZONTAL AXIS (cross-references between articles)"]
    c = structure.citations
    if c.nnz == 0:
        out.append(f"{FAIL}no reference resolved: the horizontal axis has no effect.")
        out.append("          -> the vertical axis alone remains usable,")
        out.append("             but check _CITATION_RE in structure.py.")
        return out

    citing = float((c.getnnz(axis=1) > 0).mean() * 100)
    cited = float((c.getnnz(axis=0) > 0).mean() * 100)
    out.append(f"  {c.nnz} edges | {citing:.1f}% citing articles | "
               f"{cited:.1f}% cited articles")
    st = structure.citation_stats
    out.append(f"  resolution rate of detected references: "
               f"{structure.resolution_rate:.1%}")
    if st:
        epr = st.get("edges_per_resolved", 0.0)
        out.append(f"  ambiguous references: {st.get('ambiguity_rate', 0):.1%} "
                   f"(policy '{st.get('policy')}') | "
                   f"{epr:.2f} edges per resolved reference")
        if epr > 1.5:
            out.append(f"{WARN}more than one edge per reference: "
                       "disambiguation is letting homonyms through. "
                       "Try --citation-ambiguity skip as an ablation.")
    if structure.resolution_rate > 0.4 and citing > 5:
        out.append(f"{OK}usable graph.")
    else:
        out.append(f"{WARN}sparse graph: calibration will probably set gamma "
                   "close to 0.")
        out.append("          -> inspect unresolved references with "
                   "`run citations --data-dir ...`")
    return out


def diagnose_questions(qs: QuerySet, name: str,
                       structure: LegislativeStructure) -> list[str]:
    sizes = np.array([len(s) for s in qs.qrels])
    out = ["", f"QUESTIONS ({name}): {len(qs)}"]
    out.append(f"  relevant articles: mean {sizes.mean():.2f}, "
               f"median {np.median(sizes):.0f}, max {sizes.max()}")
    multi = float((sizes > 1).mean() * 100)
    tag = OK if multi > 30 else WARN
    out.append(f"{tag}{multi:.1f}% of questions have >= 2 relevant articles "
               "(the population VHP can help)")
    return out


def run_diagnosis(corpus: Corpus, train: QuerySet, test: QuerySet,
                  structure: LegislativeStructure) -> str:
    lines = ["=" * 68, "BSARD / VHP DIAGNOSIS", "=" * 68, ""]
    lines += diagnose_corpus(corpus)
    lines += diagnose_hierarchy(corpus, structure)
    lines += diagnose_citations(corpus, structure)
    lines += diagnose_questions(train, "train", structure)
    lines += diagnose_questions(test, "test", structure)
    lines += ["", "=" * 68,
              "If the vertical axis is OK, continue with:  run analyze",
              "=" * 68]
    return "\n".join(lines)


def sample_unresolved_citations(corpus: Corpus, n: int = 20) -> str:
    import re

    from .structure import _CITATION_RE, _expand_numbers, _self_code_column

    self_codes = _self_code_column(corpus)
    index: dict[tuple[str, str], list[int]] = {}
    by_number: dict[str, list[int]] = {}
    for i, num in enumerate(corpus.numbers):
        if num:
            index.setdefault((self_codes[i], num), []).append(i)
            by_number.setdefault(num, []).append(i)

    shown, lines = 0, ["References detected but NOT resolved (sample):", ""]
    for i, text in enumerate(corpus.texts):
        if shown >= n:
            break
        for m in _CITATION_RE.finditer(text or ""):
            for num in _expand_numbers(m.group("nums")):
                if (self_codes[i], num) in index or len(by_number.get(num, [])) == 1:
                    continue
                ctx = re.sub(r"\s+", " ", m.group(0))[:90]
                lines.append(f"  art. {corpus.numbers[i] or '?':<10} cites "
                             f"{num!r:<12} | {ctx}")
                shown += 1
                break
            if shown >= n:
                break
    if shown == 0:
        lines.append("  (none: every detected reference was resolved)")
    return "\n".join(lines)
