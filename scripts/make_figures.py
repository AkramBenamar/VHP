from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TEXT_WIDTH = 6.3
GREY = "#4d4d4d"
DARK = "#1a1a1a"
ACCENT = "#0b6e6e"
ACCENT2 = "#a03a2f"


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 200,
    })


def save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        path = out / f"{name}.{ext}"
        fig.savefig(path, format=ext, bbox_inches="tight", pad_inches=0.02)
        print(f"  {path}")
    plt.close(fig)


CANDIDATES = [
    (0, 9, "center", "bottom"),
    (0, -10, "center", "top"),
    (7, 0, "left", "center"),
    (-7, 0, "right", "center"),
    (7, 8, "left", "bottom"),
    (-7, 8, "right", "bottom"),
    (7, -9, "left", "top"),
    (-7, -9, "right", "top"),
    (0, 20, "center", "bottom"),
    (0, -21, "center", "top"),
]


def place_labels(ax, points, texts, fontsize=7.0, color=DARK,
                 avoid=None) -> list:
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)

    placed = list(avoid or [])
    for x, y in points:
        px, py = ax.transData.transform((x, y))
        placed.append(matplotlib.transforms.Bbox.from_bounds(
            px - 4, py - 4, 8, 8))

    artists = []
    for (x, y), label in zip(points, texts):
        best = None
        for dx, dy, ha, va in CANDIDATES:
            ann = ax.annotate(label, (x, y), textcoords="offset points",
                              xytext=(dx, dy), ha=ha, va=va,
                              fontsize=fontsize, color=color)
            fig.canvas.draw()
            box = ann.get_window_extent(renderer).expanded(1.04, 1.10)
            inside = (box.x0 >= axes_box.x0 - 1 and box.x1 <= axes_box.x1 + 1
                      and box.y0 >= axes_box.y0 - 1
                      and box.y1 <= axes_box.y1 + 1)
            clash = any(box.overlaps(b) for b in placed)
            if inside and not clash:
                best = (ann, box)
                break
            ann.remove()
        if best is None:
            ann = ax.annotate(label, (x, y), textcoords="offset points",
                              xytext=(0, 9), ha="center", va="bottom",
                              fontsize=fontsize, color=color)
            fig.canvas.draw()
            best = (ann, ann.get_window_extent(renderer))
            print(f"  [warn] no free position for label '{label}'")
        placed.append(best[1])
        artists.append(best[0])
    return artists


SHORT_NAMES = {
    "tfidf-svd": "TF-IDF",
    "tfidf": "TF-IDF",
    "tf-idf": "TF-IDF",
    "tf-idf-svd": "TF-IDF",
    "bm25": "BM25",
    "multilingual-e5-base": "mE5",
    "multilingual-e5-small": "mE5-s",
    "e5-base": "mE5",
    "me5": "mE5",
    "bge-m3": "BGE-M3",
    "biencoder-camembert-base-mmarcofr": "CamemBERT",
    "camembert": "CamemBERT",
}


def shorten(name: str) -> str:
    n = name.strip()

    inner = re.match(r"hybrid\s*\((.*)\)\s*$", n,
                     flags=re.IGNORECASE)
    if inner:
        n = inner.group(1)
    if "+" in n:
        return " + ".join(shorten(p) for p in n.split("+"))

    ft = bool(re.match(r"^FT-", n)) or bool(
        re.search(r"(^|[\s-])ft$", n, flags=re.IGNORECASE))
    n = re.sub(r"^(ZS|FT)-", "", n)
    n = re.sub(r"[\s-]ft$", "", n, flags=re.IGNORECASE)
    n = re.sub(r"-bsard$", "", n, flags=re.IGNORECASE)
    n = n.split("/")[-1].strip()

    base = SHORT_NAMES.get(n.lower())
    if base is None:
        base = n
        for pat, rep in (("multilingual-", "m"), ("biencoder-", ""),
                         ("-mmarcoFR", ""), ("-base", "")):
            base = base.replace(pat, rep)
        if len(base) > 12:
            base = base[:12].rstrip("-. ")
    return f"{base} ft" if ft else base


LOCALITY = [
    ("law type",   99.6, 53.062,   2),
    ("code",       96.9,  6.039,  16),
    ("book",       60.8,  1.798,  34),
    ("act",        58.6,  0.468, 125),
    ("chapter",    28.6,  0.122, 234),
    ("section",    23.6,  0.058, 407),
    ("subsection", 22.7,  0.037, 605),
]

GRANULARITY = [
    ("none", 47.2), ("law type", 46.3), ("code", 48.6), ("book", 49.8),
    ("part", 49.8), ("act", 51.1), ("chapter", 52.2), ("section", 51.0),
    ("subsection", 51.3),
]

TRADEOFF = {
    "beta":    [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.6, 2.0],
    "R@100":   [50.2, 51.5, 52.5, 53.6, 55.0, 55.7, 56.9, 55.9, 55.5],
    "R@500":   [64.2, 64.8, 66.1, 67.0, 68.5, 70.1, 70.6, 71.5, 70.3],
    "MAP@100": [16.8, 17.4, 17.9, 18.4, 19.0, 19.7, 19.7, 19.5, 19.2],
    "MRR@100": [25.5, 25.9, 26.5, 26.9, 27.3, 27.5, 26.9, 26.6, 26.0],
    "MRP":     [12.6, 13.0, 13.4, 14.1, 14.6, 15.0, 15.0, 14.7, 14.8],
}

SYSTEMS = [
    ("TF-IDF",         32.6, 35.2, False),
    ("CamemBERT",      35.8, 44.7, False),
    ("mE5",            49.1, 58.8, False),
    ("BM25",           51.5, 58.0, False),
    ("BGE-M3",         59.2, 68.0, False),
    ("BM25 + BGE-M3 ft", 72.7, 78.6, True),
    ("mE5 ft",         86.3, 86.8, True),
    ("BGE-M3 ft",      86.7, 89.0, True),
]

STRATA = ["1", "2-5", "6-20", "21+"]
STRATA_N = [83, 96, 34, 9]
STRATA_BM25 = [2.4, 6.9, 13.7, 11.5]
STRATA_FT = [1.2, 1.8, 2.4, -0.5]


def fig_locality(out: Path) -> None:
    labels = [l for l, _, _, _ in LOCALITY]
    obs = np.array([o for _, o, _, _ in LOCALITY])
    lift = np.array([k for _, _, _, k in LOCALITY], dtype=float)
    y = np.arange(len(labels))[::-1]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH, 2.45), sharey=True,
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.10})

    ax1.barh(y, obs, height=0.62, color=ACCENT, edgecolor="none")
    ax1.set_yticks(y, labels)
    ax1.set_xlabel("relevant pairs co-located (%)")
    ax1.set_xlim(0, 122)
    ax1.set_xticks([0, 25, 50, 75, 100])
    ax1.grid(axis="x", linestyle=":", color=GREY)
    ax1.set_axisbelow(True)
    for yi, v in zip(y, obs):
        ax1.text(v + 2.5, yi, f"{v:.1f}", va="center", fontsize=7.5,
                 color=DARK)

    ax2.barh(y, lift, height=0.62, color=ACCENT2, edgecolor="none")
    ax2.set_xscale("log")
    ax2.set_xlabel("lift over random pairs (log scale)")
    ax2.set_xlim(0.75, float(lift.max()) * 6)
    ax2.grid(axis="x", linestyle=":", color=GREY)
    ax2.set_axisbelow(True)
    for yi, v in zip(y, lift):
        ax2.text(v * 1.2, yi, f"x{v:.0f}", va="center", fontsize=7.5,
                 color=DARK)

    save(fig, out, "fig1_locality")


def fig_granularity_tradeoff(out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.5),
                                   gridspec_kw={"wspace": 0.28})

    labels = [l.replace("_", " ") for l, _ in GRANULARITY]
    vals = [v for _, v in GRANULARITY]
    x = np.arange(len(labels))
    base = vals[0]
    lo, hi = min(vals), max(vals)
    span = hi - lo

    ax1.axhline(base, color=GREY, linestyle="--", linewidth=0.8)
    ax1.plot(x[1:], vals[1:], marker="o", color=ACCENT, zorder=3)
    best = int(np.argmax(vals[1:])) + 1
    ax1.plot(best, vals[best], marker="o", markersize=8, markerfacecolor="none",
             markeredgecolor=ACCENT2, markeredgewidth=1.4, zorder=4)
    ax1.set_ylim(lo - 0.18 * span, hi + 0.30 * span)
    ax1.annotate(f"{vals[best]:.1f}", (best, vals[best]),
                 textcoords="offset points", xytext=(0, 10), ha="center",
                 va="bottom", fontsize=8, color=ACCENT2)
    ax1.text(len(x) - 1, base - 0.06 * span, "no structure", fontsize=7.5,
             color=GREY, ha="right", va="top")
    ax1.set_xticks(x[1:], labels[1:], rotation=40, ha="right",
                   rotation_mode="anchor")
    ax1.set_ylabel("R@100")
    ax1.set_title("(a) one hierarchy level at a time", pad=5)
    ax1.grid(axis="y", linestyle=":", color=GREY)
    ax1.set_axisbelow(True)

    beta = TRADEOFF["beta"]
    styles = [("R@100", "o", "-", ACCENT),
              ("R@500", "s", "-", "#2f7fb0"),
              ("MAP@100", "^", "--", ACCENT2),
              ("MRP", "v", ":", "#7a5195")]
    ymax = 0.0
    for name, mk, ls, col in styles:
        series = np.array(TRADEOFF[name], dtype=float)
        rel = series - series[0]
        ymax = max(ymax, rel.max())
        ax2.plot(beta, rel, marker=mk, linestyle=ls, color=col, label=name)
    ax2.axhline(0, color=GREY, linewidth=0.8)
    ax2.set_ylim(-0.5, ymax * 1.42)
    ax2.axvspan(0.8, 1.2, color=ACCENT, alpha=0.09, lw=0)
    ax2.text(1.0, ymax * 1.38, "all metrics\nimprove", fontsize=7,
             color=GREY, ha="center", va="top")
    ax2.set_xlabel(r"$\beta$")
    ax2.set_ylabel("change vs. base (points)")
    ax2.set_title("(b) strength of the vertical signal", pad=5)
    ax2.legend(loc="lower right", ncol=2, handlelength=1.7,
               columnspacing=0.9, borderaxespad=0.2)
    ax2.grid(axis="y", linestyle=":", color=GREY)
    ax2.set_axisbelow(True)

    save(fig, out, "fig2_granularity_tradeoff")


def fig_gain_analysis(out: Path) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.7),
                                  gridspec_kw={"wspace": 0.26,
                                               "width_ratios": [1.3, 1]})

    systems = sorted(SYSTEMS, key=lambda t: t[1])
    xs = np.array([b for _, b, _, _ in systems], dtype=float)
    gains = np.array([v - b for _, b, v, _ in systems], dtype=float)
    sup = [s for _, _, _, s in systems]
    names = [shorten(n) for n, _, _, _ in systems]

    xpad = 0.10 * (xs.max() - xs.min())
    ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
    gpad = 0.30 * (gains.max() - min(gains.min(), 0.0))
    ax.set_ylim(min(gains.min(), 0.0) - 0.35 * gpad, gains.max() + gpad)

    lo_sup = min([x for x, s in zip(xs, sup) if s], default=None)
    if lo_sup is not None:
        ax.axvspan(lo_sup - 0.35 * xpad, ax.get_xlim()[1],
                   color=ACCENT2, alpha=0.07, lw=0)

    for x, g, s in zip(xs, gains, sup):
        col = ACCENT2 if s else ACCENT
        ax.plot([x, x], [0, g], color=col, linewidth=1.0, alpha=0.45, zorder=1)
        ax.plot(x, g, marker="s" if s else "o", color=col, zorder=3)
    ax.axhline(0, color=GREY, linewidth=0.8)

    handles = [
        plt.Line2D([], [], marker="o", color=ACCENT, linestyle="none",
                   label="no supervision"),
        plt.Line2D([], [], marker="s", color=ACCENT2, linestyle="none",
                   label="fine-tuned"),
    ]
    leg = ax.legend(handles=handles, loc="upper right", handletextpad=0.4,
                    borderaxespad=0.3)
    fig.canvas.draw()
    leg_box = leg.get_window_extent(fig.canvas.get_renderer()).expanded(1.05,
                                                                       1.15)

    place_labels(ax, list(zip(xs, gains)), names, fontsize=6.9,
                 avoid=[leg_box])

    ax.set_xlabel("first-stage R@100")
    ax.set_ylabel("R@100 gain (points)")
    ax.set_title("(a) by strength of the first stage", pad=5)
    ax.grid(axis="y", linestyle=":", color=GREY)
    ax.set_axisbelow(True)

    x = np.arange(len(STRATA))
    w = 0.38
    ax2.bar(x - w / 2, STRATA_BM25, w, color=ACCENT, edgecolor="none",
            label="BM25")
    ax2.bar(x + w / 2, STRATA_FT, w, color=ACCENT2, edgecolor="none",
            label="BGE-M3 ft")
    top = max(max(STRATA_BM25), max(STRATA_FT))
    bot = min(min(STRATA_BM25), min(STRATA_FT), 0.0)
    ax2.set_ylim(bot - 0.13 * top, top * 1.40)
    for xi, v in zip(x - w / 2, STRATA_BM25):
        ax2.text(xi, v + 0.02 * top, f"+{v:.1f}", ha="center", va="bottom",
                 fontsize=6.9, color=DARK)
    for xi, v in zip(x + w / 2, STRATA_FT):
        if v >= 0:
            ax2.text(xi, v + 0.02 * top, f"+{v:.1f}", ha="center",
                     va="bottom", fontsize=6.9, color=DARK)
        else:
            ax2.text(xi, v - 0.02 * top, f"{v:.1f}", ha="center", va="top",
                     fontsize=6.9, color=DARK)
    ax2.axhline(0, color=GREY, linewidth=0.8)
    ax2.set_xticks(x, [f"{s}\n({n})" for s, n in zip(STRATA, STRATA_N)])
    ax2.set_xlabel("relevant articles per question")
    ax2.set_title("(b) by size of the answer set", pad=5)
    ax2.legend(loc="upper left", handlelength=1.1, borderaxespad=0.3)
    ax2.grid(axis="y", linestyle=":", color=GREY)
    ax2.set_axisbelow(True)

    save(fig, out, "fig3_gain_analysis")


def load_from_results(results: Path, figures: Path) -> None:
    global GRANULARITY, TRADEOFF, SYSTEMS, STRATA, STRATA_N
    global STRATA_BM25, STRATA_FT

    figs = json.loads(figures.read_text(encoding="utf-8"))
    if figs.get("granularity"):
        GRANULARITY = [("none" if r["level"].startswith("(")
                        else r["level"], r["R@100"])
                       for r in figs["granularity"]]
    if figs.get("tradeoff_beta"):
        rows = figs["tradeoff_beta"]
        TRADEOFF = {"beta": [r["beta"] for r in rows]}
        for m in ("R@100", "R@500", "MAP@100", "MRR@100", "MRP"):
            if m in rows[0]:
                TRADEOFF[m] = [r[m] for r in rows]

    raw = json.loads(results.read_text(encoding="utf-8"))
    systems, strat_zs, strat_ft = [], None, None
    for name, d in raw.get("systems", {}).items():
        variants = d.get("variants", {})
        if not variants:
            continue
        best = max(variants.values(),
                   key=lambda v: (v.get("pareto", False),
                                  v.get("n_improved", 0),
                                  v.get("metrics", {}).get("R@100", 0)))
        ft = name.startswith("FT-") or "ft" in name.lower().split()
        systems.append((name, d["base"]["R@100"], best["metrics"]["R@100"],
                        bool(ft)))
        if best.get("stratified"):
            if name == "BM25":
                strat_zs = best["stratified"]
            elif name.startswith("FT-"):
                strat_ft = best["stratified"]

    if systems:
        SYSTEMS = sorted(systems, key=lambda t: t[1])
        print(f"  [data] {len(SYSTEMS)} systems: "
              + ", ".join(shorten(n) for n, _, _, _ in SYSTEMS))
    if strat_zs and strat_ft and len(strat_zs) == len(strat_ft):
        STRATA = [r["bucket"] for r in strat_zs]
        STRATA_N = [r["n_queries"] for r in strat_zs]
        STRATA_BM25 = [r["delta"] for r in strat_zs]
        STRATA_FT = [r["delta"] for r in strat_ft]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    ap.add_argument("--from-results", default=None,
                    help="results directory, to read results.json and "
                         "figures.json instead of the hard-coded values")
    args = ap.parse_args()

    setup_style()
    if args.from_results:
        d = Path(args.from_results)
        load_from_results(d / "results.json", d / "figures.json")

    out = Path(args.out)
    print("Figures:")
    fig_locality(out)
    fig_granularity_tradeoff(out)
    fig_gain_analysis(out)


if __name__ == "__main__":
    main()
