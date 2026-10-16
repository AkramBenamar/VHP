from __future__ import annotations

import numpy as np


def paired_bootstrap(
    baseline: np.ndarray,
    system: np.ndarray,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    diff = np.asarray(system, float) - np.asarray(baseline, float)
    diff = diff[~np.isnan(diff)]
    n = diff.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {
        "delta": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_boot": float(2 * min((boots <= 0).mean(), (boots >= 0).mean())),
        "n": int(n),
        "n_improved": int((diff > 0).sum()),
        "n_degraded": int((diff < 0).sum()),
    }


def paired_permutation(
    baseline: np.ndarray,
    system: np.ndarray,
    n_perm: int = 10_000,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    diff = np.asarray(system, float) - np.asarray(baseline, float)
    diff = diff[~np.isnan(diff)]
    obs = abs(diff.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diff.size))
    null = np.abs((signs * diff).mean(axis=1))
    return float(((null >= obs).sum() + 1) / (n_perm + 1))


def holm_bonferroni(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, rejected_all = {}, True
    for i, (k, p) in enumerate(items):
        thresh = alpha / (m - i)
        rejected_all = rejected_all and (p <= thresh)
        out[k] = rejected_all
    return out


def compare(
    baseline_pq: dict[str, np.ndarray],
    system_pq: dict[str, np.ndarray],
    metrics: list[str] | None = None,
    seed: int = 0,
) -> dict[str, dict]:
    metrics = metrics or list(baseline_pq.keys())
    res = {}
    for m in metrics:
        r = paired_bootstrap(baseline_pq[m], system_pq[m], seed=seed)
        r["p_perm"] = paired_permutation(baseline_pq[m], system_pq[m], seed=seed)
        r["delta_pct_points"] = r["delta"] * 100
        res[m] = r
    return res


def format_comparison(name: str, res: dict[str, dict]) -> str:
    lines = [f"### {name}"]
    for m, r in res.items():
        star = ("***" if r["p_perm"] < 0.001 else
                "**" if r["p_perm"] < 0.01 else
                "*" if r["p_perm"] < 0.05 else "")
        lines.append(
            f"  {m:<8} delta = {r['delta_pct_points']:+.2f} pts  "
            f"95% CI [{r['ci_low']*100:+.2f}, {r['ci_high']*100:+.2f}]  "
            f"p = {r['p_perm']:.4f}{star}  "
            f"(+{r['n_improved']}/-{r['n_degraded']} queries)"
        )
    return "\n".join(lines)


def stratify_by_gold_size(
    baseline_pq: dict[str, np.ndarray],
    system_pq: dict[str, np.ndarray],
    qrels: list[set[int]],
    metric: str = "R@100",
    buckets: tuple[tuple[int, int], ...] = ((1, 1), (2, 5), (6, 20), (21, 10_000)),
) -> list[dict]:
    sizes = np.array([len(g) for g in qrels])
    rows = []
    for lo, hi in buckets:
        mask = (sizes >= lo) & (sizes <= hi)
        if not mask.any():
            continue
        b, s = baseline_pq[metric][mask], system_pq[metric][mask]
        diff = s - b
        rows.append({
            "bucket": f"{lo}" if lo == hi else (
                f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"),
            "n_queries": int(mask.sum()),
            "baseline": float(np.nanmean(b) * 100),
            "system": float(np.nanmean(s) * 100),
            "delta": float(np.nanmean(diff) * 100),
            "n_improved": int((diff > 0).sum()),
            "n_degraded": int((diff < 0).sum()),
        })
    return rows


def format_stratified(rows: list[dict], metric: str) -> str:
    lines = [f"| Relevant articles | Queries | Base | +VHP | delta {metric} | +/- |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['bucket']} | {r['n_queries']} | {r['baseline']:.1f} | "
            f"{r['system']:.1f} | {r['delta']:+.1f} | "
            f"+{r['n_improved']}/-{r['n_degraded']} |")
    return "\n".join(lines)
