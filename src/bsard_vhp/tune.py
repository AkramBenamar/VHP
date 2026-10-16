from __future__ import annotations

from dataclasses import replace

import numpy as np

from .metrics import evaluate
from .propagate import VHP, VHPParams, VHPScorer

TUNING_TOP_K = 2000


def _eval(scorer: VHPScorer, p: VHPParams, qrels, objective: str,
          top_k: int = TUNING_TOP_K) -> float:
    agg, _ = evaluate(scorer.score(p), qrels, top_k=top_k)
    return agg[objective]


def _eval_constrained(
    scorer: VHPScorer,
    p: VHPParams,
    qrels,
    objective: str,
    constraints: dict[str, float],
    top_k: int = TUNING_TOP_K,
) -> float:
    agg, _ = evaluate(scorer.score(p), qrels, top_k=top_k)
    for metric, floor in constraints.items():
        if agg.get(metric, -np.inf) < floor:
            return -np.inf
    return agg[objective]


def level_weight_schemes(levels: list[str]) -> dict[str, dict[str, float]]:
    n = len(levels)
    schemes = {
        "uniform": {l: 1.0 for l in levels},
        "increasing": {l: float(i + 1) for i, l in enumerate(levels)},
        "exponential": {l: float(2 ** i) for i, l in enumerate(levels)},
        "two_finest": {l: float(i >= n - 2) for i, l in enumerate(levels)},
        "two_middle": {l: float(n // 2 - 1 <= i <= n // 2) for i, l in enumerate(levels)},
    }
    for i, lvl in enumerate(levels):
        schemes[f"only_{lvl}"] = {l: float(j == i) for j, l in enumerate(levels)}
    return schemes


def staged_search(
    vhp: VHP,
    base_scores: np.ndarray,
    qrels: list[set[int]],
    objective: str = "R@100",
    betas: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.8, 1.2),
    gammas: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5),
    top_ms: tuple[int, ...] = (1, 3, 5),
    directions: tuple[str, ...] = ("sym", "in", "out"),
    hops: tuple[int, ...] = (1, 2),
    constrain_metrics: tuple[str, ...] = (),
    constraint_slack: float = 0.0,
    verbose: bool = True,
) -> tuple[VHPParams, dict]:
    scorer = vhp.scorer(base_scores)
    levels = vhp.structure.levels
    schemes = level_weight_schemes(levels)

    ref_agg, _ = evaluate(scorer.score(VHPParams(beta=0.0, gamma=0.0)),
                          qrels, top_k=TUNING_TOP_K)
    baseline = ref_agg[objective]
    constraints = {m: ref_agg[m] - constraint_slack
                   for m in constrain_metrics if m in ref_agg}

    def score_of(p: VHPParams) -> float:
        return (_eval_constrained(scorer, p, qrels, objective, constraints)
                if constraints else _eval(scorer, p, qrels, objective))

    trace: dict = {"baseline": baseline, "vertical": [], "horizontal": [],
                   "joint": [], "constraints": constraints}
    if verbose:
        print(f"  [tune] base {objective} = {baseline:.2f}"
              + (f" | constraints: {list(constraints)}" if constraints else ""))

    best_v = VHPParams(beta=0.0, gamma=0.0)
    best_v_score = baseline
    for top_m in top_ms:
        for sname, w in schemes.items():
            for beta in betas:
                p = VHPParams(beta=beta, gamma=0.0, top_m=top_m,
                              level_weights=dict(w))
                s = score_of(p)
                trace["vertical"].append(
                    {"top_m": top_m, "scheme": sname, "beta": beta, objective: s})
                if s > best_v_score:
                    best_v_score, best_v = s, p
                    if verbose:
                        print(f"  [tune-V] {objective}={s:.2f}  "
                              f"beta={beta} top_m={top_m} levels={sname}")

    best_h, best_h_score = best_v, best_v_score
    if vhp.structure.citations.nnz:
        for direction in directions:
            for n_hops in hops:
                for gamma in gammas:
                    p = replace(best_v, gamma=gamma, direction=direction,
                                n_hops=n_hops)
                    s = score_of(p)
                    trace["horizontal"].append(
                        {"direction": direction, "n_hops": n_hops,
                         "gamma": gamma, objective: s})
                    if s > best_h_score:
                        best_h_score, best_h = s, p
                        if verbose:
                            print(f"  [tune-H] {objective}={s:.2f}  "
                                  f"gamma={gamma} dir={direction} hops={n_hops}")

    best, best_score = best_h, best_h_score
    if best_h.gamma > 0:
        grid_b = sorted({round(max(best_h.beta * f, 0.0), 4)
                         for f in (0.5, 0.75, 1.0, 1.5, 2.0)})
        grid_g = sorted({round(max(best_h.gamma * f, 0.0), 4)
                         for f in (0.5, 0.75, 1.0, 1.5, 2.0)})
        for b in grid_b:
            for g in grid_g:
                p = replace(best_h, beta=b, gamma=g)
                s = score_of(p)
                trace["joint"].append({"beta": b, "gamma": g, objective: s})
                if s > best_score:
                    best_score, best = s, p

    if best_score <= best_v_score + 1e-9:
        best, best_score = best_v, best_v_score

    if not np.isfinite(best_score):
        best, best_score = VHPParams(beta=0.0, gamma=0.0), baseline
        if verbose:
            print("  [tune] no configuration satisfies the constraints "
                  "-> VHP disabled (beta = gamma = 0)")

    trace["selected"] = best.as_dict()
    trace["selected_score"] = best_score
    trace["gain_train"] = best_score - baseline
    if verbose:
        print(f"  [tune] selected: {objective}={best_score:.2f} "
              f"(+{best_score - baseline:.2f} vs base)")
    return best, trace


def cross_validate(
    vhp: VHP,
    base_scores: np.ndarray,
    qrels: list[set[int]],
    n_folds: int = 5,
    objective: str = "R@100",
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(qrels))
    folds = np.array_split(idx, n_folds)
    gains, picks = [], []
    for f in range(n_folds):
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != f])
        te = folds[f]
        p, _ = staged_search(vhp, base_scores[tr], [qrels[i] for i in tr],
                             objective=objective, verbose=verbose)
        held = [qrels[i] for i in te]
        agg_b, _ = evaluate(base_scores[te], held)
        agg_v, _ = evaluate(vhp(base_scores[te], params=p), held)
        gains.append(agg_v[objective] - agg_b[objective])
        picks.append(p.as_dict())
    return {
        "fold_gains": gains,
        "mean_gain": float(np.mean(gains)),
        "std_gain": float(np.std(gains)),
        "selected_params": picks,
    }


def granularity_curve(
    vhp: VHP,
    base_scores: np.ndarray,
    qrels: list[set[int]],
    params: VHPParams,
    objective: str = "R@100",
) -> list[dict]:
    scorer = vhp.scorer(base_scores)
    levels = vhp.structure.levels
    base = _eval(scorer, replace(params, beta=0.0, gamma=0.0), qrels, objective)
    out = [{"level": "(none)", "n_groups": 0, objective: base}]
    for lvl in levels:
        w = {l: float(l == lvl) for l in levels}
        p = replace(params, gamma=0.0, level_weights=w)
        n_groups = int(len(np.unique(vhp.structure.groups[lvl])))
        out.append({"level": lvl, "n_groups": n_groups,
                    objective: _eval(scorer, p, qrels, objective)})
    return out


def tradeoff_curve(
    vhp: VHP,
    base_scores: np.ndarray,
    qrels: list[set[int]],
    params: VHPParams,
    betas: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.6, 2.0),
    metrics: tuple[str, ...] = ("R@100", "R@500", "MAP@100", "MRR@100", "MRP"),
) -> list[dict]:
    scorer = vhp.scorer(base_scores)
    rows = []
    for beta in betas:
        p = replace(params, beta=beta)
        agg, _ = evaluate(scorer.score(p), qrels, top_k=TUNING_TOP_K)
        rows.append({"beta": beta, **{m: agg[m] for m in metrics if m in agg}})
    return rows


def sensitivity_curve(
    vhp: VHP,
    base_scores: np.ndarray,
    qrels: list[set[int]],
    params: VHPParams,
    axis: str = "beta",
    values: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0),
    objective: str = "R@100",
) -> list[dict]:
    scorer = vhp.scorer(base_scores)
    out = []
    for v in values:
        p = replace(params, **{axis: v})
        out.append({axis: v, objective: _eval(scorer, p, qrels, objective)})
    return out
