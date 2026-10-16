from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bsard_vhp import analysis, data as data_mod, stats
from bsard_vhp.embed_propagate import (EmbeddingSmoother, VHPEParams,
                                       apply_vhpe, tune_vhpe)
from bsard_vhp.metrics import METRIC_ORDER, evaluate, format_table
from bsard_vhp.propagate import VHP, VHPParams
from bsard_vhp.retrievers import (BM25Config, BM25Retriever,
                                  DenseConfig, DenseRetriever, TfidfSvdRetriever,
                                  zfuse)
from bsard_vhp.structure import build_structure
from bsard_vhp.tune import (cross_validate, granularity_curve,
                            sensitivity_curve, staged_search, tradeoff_curve)

OBJECTIVES = ["R@100", "MAP@100"]

CONSTRAINED_METRICS = ["MAP@100", "MAP", "MRP", "R@200", "R@500"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def retriever_signature(name: str, args) -> str:
    parts = [name, str(args.data_dir),
             f"msl={args.max_seq_length}",
             f"ambig={args.citation_ambiguity}"]
    if name in ("BM25",):
        parts += [f"k1={args.bm25_k1}", f"b={args.bm25_b}",
                  f"idf={args.idf_scheme}",
                  f"stem={not args.no_stemming}",
                  f"num={not args.keep_numbers}"]
    return "|".join(parts)


def cache_path(cache_dir: Path, key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    return cache_dir / f"{h}.npz"


def cached_scores(cache_dir: Path, key: str, fn, force: bool = False):
    p = cache_path(cache_dir, key)
    if p.exists() and not force:
        z = np.load(p, allow_pickle=True)
        log(f"  (cache) {key}")
        return {k: z[k] for k in z.files}
    out = fn()
    np.savez_compressed(p, **out)
    return out


def latex_table(rows: list[dict], metrics: list[str], caption: str,
                label: str) -> str:
    head = ("\\begin{table}[t]\n\\centering\n\\small\n"
            "\\begin{tabular}{l" + "c" * len(metrics) + "}\n\\hline\n"
            "System & " + " & ".join(metrics) + " \\\\\n\\hline\n")
    body = ""
    for r in rows:
        cells = " & ".join(
            f"{r[m]:.1f}" if isinstance(r.get(m), (int, float)) else "--"
            for m in metrics)
        name = str(r["system"]).replace("_", "\\_")
        body += f"{name} & {cells} \\\\\n"
    tail = ("\\hline\n\\end{tabular}\n"
            + "\\caption{" + caption + "}\n"
            + "\\label{" + label + "}\n"
            + "\\end{table}\n")
    return head + body + tail


def build_retrievers(args) -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    out.append(("BM25", BM25Retriever(BM25Config(
        k1=args.bm25_k1, b=args.bm25_b, use_stemming=not args.no_stemming,
        remove_numbers=not args.keep_numbers, idf_scheme=args.idf_scheme))))

    if args.tfidf:
        out.append(("TFIDF-SVD", TfidfSvdRetriever()))

    zs = list(args.dense_models or [])
    ft = list(args.finetuned_models or [])
    if zs or ft:
        try:
            import sentence_transformers
        except ImportError:
            log("  ! dense models skipped: "
                "pip install torch sentence-transformers")
            zs, ft = [], []

    for model in zs:
        short = model.rstrip("/").split("/")[-1]
        out.append((f"ZS-{short}", DenseRetriever(
            DenseConfig(model_name=model, device=args.device,
                        batch_size=args.batch_size,
                        max_seq_length=args.max_seq_length))))
    for model in ft:
        short = model.rstrip("/").rstrip("\\").replace("\\", "/").split("/")[-1]
        out.append((f"FT-{short}", DenseRetriever(
            DenseConfig(model_name=model, device=args.device,
                        batch_size=args.batch_size,
                        max_seq_length=args.max_seq_length))))
    return out


def has_embeddings(retr) -> bool:
    return (getattr(retr, "doc_emb", None) is not None
            and hasattr(retr, "encode_queries"))


def eval_system(name: str, scores_test: np.ndarray, qrels, pq_base=None,
                objective: str = "R@100", latency_ms: float | None = None,
                agg_base: dict | None = None) -> dict:
    agg, pq = evaluate(scores_test, qrels)
    rec = {"system": name, **{m: agg[m] for m in METRIC_ORDER},
           "_pq": pq, "_agg": agg}
    if latency_ms is not None:
        rec["_latency_ms_per_query"] = latency_ms / max(len(qrels), 1)
    if pq_base is not None and agg_base is not None:
        cmp_res = stats.compare(pq_base, pq, metrics=METRIC_ORDER)
        rec["_significance"] = cmp_res
        rec["_stratified"] = stats.stratify_by_gold_size(
            pq_base, pq, qrels, metric=objective)
        rec["_n_improved"] = sum(1 for m in METRIC_ORDER
                                 if agg[m] > agg_base[m])
        rec["_pareto"] = all(agg[m] >= agg_base[m] for m in METRIC_ORDER)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/bsard")
    ap.add_argument("--out-dir", default="paper_results")
    ap.add_argument("--dense-models", nargs="*", default=None,
                    help="one or more zero-shot encoders, for example "
                         "intfloat/multilingual-e5-base BAAI/bge-m3")
    ap.add_argument("--finetuned-models", nargs="*", default=None,
                    help="one or more models fine-tuned on BSARD (local paths)")
    ap.add_argument("--dense-model", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--finetuned-model", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq-length", type=int, default=512,
                    help="article truncation length. 512 is the full protocol; "
                         "256 roughly halves the encoding cost and is useful "
                         "without a GPU, but should then be reported")
    ap.add_argument("--threads", type=int, default=None,
                    help="number of CPU threads for torch (without a GPU, set "
                         "it to the number of physical cores)")
    ap.add_argument("--tfidf", action="store_true",
                    help="add a TF-IDF/SVD baseline (no torch required)")
    ap.add_argument("--bm25-k1", type=float, default=0.9)
    ap.add_argument("--bm25-b", type=float, default=0.4)
    ap.add_argument("--no-stemming", action="store_true")
    ap.add_argument("--keep-numbers", action="store_true")
    ap.add_argument("--idf-scheme", default="bsard", choices=["bsard", "lucene"])
    ap.add_argument("--citation-ambiguity", default="nearest")
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--no-cross-validate", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    args = ap.parse_args()
    if args.dense_model:
        args.dense_models = (args.dense_models or []) + [args.dense_model]
    if args.finetuned_model:
        args.finetuned_models = (args.finetuned_models or []) + [args.finetuned_model]

    if args.threads:
        try:
            import torch
            torch.set_num_threads(args.threads)
            log(f"torch: {args.threads} CPU threads")
        except ImportError:
            pass

    out = Path(args.out_dir)
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    md: list[str] = ["# Results", ""]
    tex: list[str] = []
    figures: dict = {}
    raw: dict = {}

    log("Loading BSARD")
    corpus, train, test = data_mod.load_bsard(args.data_dir)
    log(f"  {len(corpus)} articles | {len(train)} train | {len(test)} test")

    log("Extracting the legislative structure")
    structure = build_structure(corpus, ambiguity=args.citation_ambiguity)
    vhp = VHP(structure)
    smoother = EmbeddingSmoother(structure)

    raw["corpus"] = {
        "n_articles": len(corpus), "n_train": len(train), "n_test": len(test),
        "levels": structure.levels,
        "level_sizes": {l: int(len(np.unique(g)))
                        for l, g in structure.groups.items()},
        "citation_stats": structure.citation_stats,
    }

    log("Table 1: structural locality of the labels")
    md += ["## Table 1 - Structural locality of BSARD labels", "",
           analysis.run_full_analysis(corpus, train, structure, "train"), "",
           analysis.run_full_analysis(corpus, test, structure, "test"), ""]
    loc = analysis.colocation_rates(test, structure)
    tex.append(latex_table(
        [{"system": l, "Observed": v["observed"],
          "Random": v["random_baseline"], "Lift": v["lift"]}
         for l, v in loc.items()],
        ["Observed", "Random", "Lift"],
        "Co-location of articles relevant to the same question, by level of "
        "the legislative plan (BSARD test split).", "tab:locality"))
    raw["locality"] = {"test": loc, "train": analysis.colocation_rates(train, structure),
                       "citations_test": analysis.citation_connectivity(test, structure)}

    log("Table 2: main results")
    main_rows: list[dict] = []
    sig_blocks: list[str] = []
    raw["systems"] = {}

    retrievers = build_retrievers(args)
    scores_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    emb_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for name, retr in retrievers:
        log(f"Retriever {name}")
        t0 = time.time()

        def compute(retr=retr):
            retr.index(corpus.texts)
            d = {"train": retr.score(train.texts),
                 "test": retr.score(test.texts)}
            if has_embeddings(retr):
                d["doc_emb"] = retr.doc_emb
                d["q_train"] = retr.encode_queries(train.texts)
                d["q_test"] = retr.encode_queries(test.texts)
            return d

        blob = cached_scores(cache, retriever_signature(name, args), compute,
                             args.force)
        s_train, s_test = blob["train"], blob["test"]
        scores_cache[name] = (s_train, s_test)
        if "doc_emb" in blob:
            emb_cache[name] = (blob["doc_emb"], blob["q_train"], blob["q_test"])
        log(f"  indexing + scoring: {time.time() - t0:.1f}s")

        base_rec = eval_system(name, s_test, test.qrels)
        pq_base, agg_base = base_rec["_pq"], base_rec["_agg"]
        main_rows.append(base_rec)
        raw["systems"][name] = {"base": agg_base, "variants": {}}

        tuned = {}
        for obj in OBJECTIVES:
            log(f"  calibrating VHP-S ({obj}) on the training split")
            p, tr = staged_search(vhp, s_train, train.qrels, objective=obj,
                                  verbose=False)
            tuned[obj] = p
            raw["systems"][name][f"tuning_{obj}"] = tr
            log(f"    {p.as_dict()}")

        log("  calibrating VHP-S (constrained recall) on the training split")
        p_c, tr_c = staged_search(
            vhp, s_train, train.qrels, objective="R@100",
            constrain_metrics=tuple(CONSTRAINED_METRICS), verbose=False)
        tuned["constrained"] = p_c
        raw["systems"][name]["tuning_constrained"] = tr_c
        log(f"    {p_c.as_dict()}")

        best = tuned[OBJECTIVES[0]]
        variants: list[tuple[str, object, str]] = [
            (f"{name} + VHP-S vertical only", replace(best, gamma=0.0), "s"),
            (f"{name} + VHP-S horizontal only", replace(best, beta=0.0), "s"),
        ]
        for obj in OBJECTIVES:
            variants.append((f"{name} + VHP-S (calibrated {obj})", tuned[obj], "s"))
        variants.append(
            (f"{name} + VHP-S (constrained recall)", tuned["constrained"], "s"))

        if name in emb_cache:
            doc_emb, q_train, q_test = emb_cache[name]
            for obj in OBJECTIVES:
                log(f"  calibrating VHP-E ({obj}) on the training split")
                pe, tre = tune_vhpe(smoother, doc_emb, q_train, train.qrels,
                                    objective=obj, verbose=False)
                raw["systems"][name][f"tuning_vhpe_{obj}"] = tre
                cand = []
                for norm in (False, True):
                    pn = replace(pe, normalize=norm)
                    a, _ = evaluate(apply_vhpe(smoother, doc_emb, q_train, pn),
                                    train.qrels)
                    cand.append((a[obj], pn))
                pe = max(cand, key=lambda x: x[0])[1]
                variants.append((f"{name} + VHP-E (calibrated {obj})", pe, "e"))
                log(f"    {pe.as_dict()}")

            target = f"{name} + VHP-E (calibrated {OBJECTIVES[0]})"
            pe_best = [v[1] for v in variants if v[0] == target]
            if pe_best:
                variants.append((f"{name} + VHP-E + VHP-S",
                                 (pe_best[0], tuned[OBJECTIVES[0]]), "es"))

        for label, p, kind in variants:
            t0 = time.time()
            if kind == "s":
                sc = vhp(s_test, params=p)
            elif kind == "e":
                doc_emb, _, q_test = emb_cache[name]
                sc = apply_vhpe(smoother, doc_emb, q_test, p)
            else:
                pe, ps = p
                doc_emb, _, q_test = emb_cache[name]
                sc = vhp(apply_vhpe(smoother, doc_emb, q_test, pe), params=ps)
            elapsed = (time.time() - t0) * 1000
            rec = eval_system(label, sc, test.qrels, pq_base, OBJECTIVES[0],
                              elapsed, agg_base)
            main_rows.append(rec)
            sig_blocks.append(stats.format_comparison(f"{name} -> {label}",
                                                      rec["_significance"]))
            sig_blocks.append(
                f"  {rec['_n_improved']}/{len(METRIC_ORDER)} metrics improved"
                + ("  [PARETO IMPROVEMENT]" if rec["_pareto"] else ""))
            sig_blocks.append(
                "\n" + stats.format_stratified(rec["_stratified"],
                                               OBJECTIVES[0]) + "\n")
            raw["systems"][name]["variants"][label] = {
                "metrics": rec["_agg"],
                "pareto": rec["_pareto"],
                "n_improved": rec["_n_improved"],
                "latency_ms_per_query": rec.get("_latency_ms_per_query"),
                "stratified": rec["_stratified"],
                "significance": {k: {kk: float(vv) for kk, vv in v.items()}
                                 for k, v in rec["_significance"].items()},
            }
            log(f"  {label}: {rec['_n_improved']}/{len(METRIC_ORDER)} "
                f"metrics improved"
                + (" [PARETO]" if rec["_pareto"] else ""))

    dense_names = [n for n, _ in retrievers
                   if n.startswith("ZS-") or n.startswith("FT-")]
    if dense_names and "BM25" in scores_cache:
        best_dense = dense_names[-1]
        log(f"Hybrid BM25 + {best_dense}")
        htr = zfuse(scores_cache["BM25"][0], scores_cache[best_dense][0], 0.5)
        hte = zfuse(scores_cache["BM25"][1], scores_cache[best_dense][1], 0.5)
        hname = f"Hybrid (BM25 + {best_dense})"
        base_rec = eval_system(hname, hte, test.qrels)
        pq_base, agg_base = base_rec["_pq"], base_rec["_agg"]
        main_rows.append(base_rec)
        raw["systems"][hname] = {"base": agg_base, "variants": {}}
        for obj in OBJECTIVES:
            p, _ = staged_search(vhp, htr, train.qrels, objective=obj,
                                 verbose=False)
            rec = eval_system(f"{hname} + VHP-S (calibrated {obj})",
                              vhp(hte, params=p), test.qrels, pq_base, obj,
                              None, agg_base)
            main_rows.append(rec)
            sig_blocks.append(stats.format_comparison(
                f"{hname} -> +VHP-S ({obj})", rec["_significance"]))
            raw["systems"][hname]["variants"][obj] = {
                "metrics": rec["_agg"], "pareto": rec["_pareto"],
                "params": p.as_dict()}

    summary: list[dict] = []
    by_system: dict[str, list[dict]] = {}
    current = None
    for r in main_rows:
        if "_significance" not in r:
            current = r["system"]
            by_system[current] = []
        elif current:
            by_system[current].append(r)

    for base_row in main_rows:
        if "_significance" in base_row:
            continue
        name = base_row["system"]
        cands = by_system.get(name, [])
        if not cands:
            continue
        best_v = max(cands, key=lambda r: (r.get("_pareto", False),
                                           r.get("_n_improved", 0),
                                           r.get("R@100", 0)))
        summary.append({"system": name,
                        **{m: base_row[m] for m in METRIC_ORDER}})
        sig = best_v["_significance"]
        star = ("***" if sig["R@100"]["p_perm"] < 0.001 else
                "**" if sig["R@100"]["p_perm"] < 0.01 else
                "*" if sig["R@100"]["p_perm"] < 0.05 else "")
        label = best_v["system"].replace(name + " + ", "")
        summary.append({
            "system": f"  + {label}"
                      + ("  [Pareto]" if best_v.get("_pareto") else "")
                      + f"  {star}",
            **{m: best_v[m] for m in METRIC_ORDER}})

    md += ["## Table 2b - Summary: best configuration per first stage",
           "", format_table(summary, METRIC_ORDER), "",
           "For each first stage, the configuration shown is the one improving "
           "the most metrics. `[Pareto]` marks a simultaneous improvement on "
           "all seven. Stars refer to R@100 (paired permutation test).", ""]
    tex.append(latex_table(summary, METRIC_ORDER,
                           "Summary: best VHP configuration per first stage on "
                           "the BSARD test split.", "tab:summary"))
    raw["summary"] = summary

    clean_rows = [{k: v for k, v in r.items() if not k.startswith("_")}
                  for r in main_rows]
    md += ["## Table 2 - Main results (BSARD test split)", "",
           format_table(clean_rows, METRIC_ORDER), "",
           "VHP-S acts on scores at query time; VHP-E smooths the "
           "representations once at indexing time and therefore costs nothing "
           "per query.", ""]
    tex.append(latex_table(clean_rows, METRIC_ORDER,
                           "Results on the BSARD test split. VHP-S operates on "
                           "scores, VHP-E on representations.",
                           "tab:main"))

    if not args.no_ablations:
        log("Table 3: ablations")
        s_train, s_test = scores_cache["BM25"]
        base_rec = eval_system("BM25", s_test, test.qrels)
        best, _ = staged_search(vhp, s_train, train.qrels,
                                objective=OBJECTIVES[0], verbose=False)
        abl_rows = [{"system": "BM25 (baseline)",
                     **{m: base_rec["_agg"][m] for m in METRIC_ORDER}},
                    {"system": "Full VHP-S", **{
                        m: evaluate(vhp(s_test, params=best), test.qrels)[0][m]
                        for m in METRIC_ORDER}}]

        ablations = [
            ("without self-exclusion (leave-one-out removed)",
             replace(best, exclude_self=False)),
            ("vertical axis removed (beta = 0)", replace(best, beta=0.0)),
            ("horizontal axis removed (gamma = 0)", replace(best, gamma=0.0)),
            ("aggregation over 5 members (top_m = 5)", replace(best, top_m=5)),
            ("undirected references (direction = sym)",
             replace(best, direction="sym")),
            ("reversed references (direction = in)", replace(best, direction="in")),
            ("two-hop propagation", replace(best, n_hops=2)),
            ("finest level only", replace(
                best, level_weights={l: float(i == len(structure.levels) - 1)
                                     for i, l in enumerate(structure.levels)})),
            ("uniformly weighted levels", replace(
                best, level_weights={l: 1.0 for l in structure.levels})),
        ]
        for label, p in ablations:
            agg, _ = evaluate(vhp(s_test, params=p), test.qrels)
            abl_rows.append({"system": label,
                             **{m: agg[m] for m in METRIC_ORDER}})

        for policy in ("skip", "all"):
            if policy == args.citation_ambiguity:
                continue
            st2 = build_structure(corpus, ambiguity=policy, verbose=False)
            agg, _ = evaluate(VHP(st2)(s_test, params=best), test.qrels)
            abl_rows.append({
                "system": f"ambiguous references: policy '{policy}'",
                **{m: agg[m] for m in METRIC_ORDER}})

        md += ["## Table 3 - Ablations (on BM25, BSARD test split)", "",
               format_table(abl_rows, METRIC_ORDER), ""]
        tex.append(latex_table(abl_rows, METRIC_ORDER,
                               "Ablations of the VHP-S components.",
                               "tab:ablation"))
        raw["ablations"] = abl_rows

        log("Curves for the figures")
        figures["granularity"] = granularity_curve(vhp, s_train, train.qrels,
                                                   best, objective=OBJECTIVES[0])
        figures["tradeoff_beta"] = tradeoff_curve(vhp, s_train, train.qrels, best)
        figures["sensitivity_gamma"] = sensitivity_curve(
            vhp, s_train, train.qrels, best, axis="gamma",
            values=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8),
            objective=OBJECTIVES[0])
        md += ["## Figure 1 - Performance by granularity of the legislative plan",
               "",
               format_table([{"system": r["level"],
                              "groups": r["n_groups"],
                              OBJECTIVES[0]: r[OBJECTIVES[0]]}
                             for r in figures["granularity"]],
                            ["groups", OBJECTIVES[0]]), "",
               "## Figure 2 - Recall/precision trade-off against beta "
               "(training split)", "",
               format_table([{"system": f"beta = {r['beta']}", **{
                   k: v for k, v in r.items() if k != "beta"}}
                   for r in figures["tradeoff_beta"]],
                   ["R@100", "R@500", "MAP@100", "MRR@100", "MRP"]), ""]

        if not args.no_cross_validate:
            log("Cross-validation on the training split")
            cv = cross_validate(vhp, s_train, train.qrels,
                                objective=OBJECTIVES[0])
            raw["cross_validation"] = cv
            md += ["## Stability of the calibration", "",
                   f"Five-fold cross-validation on the training split: "
                   f"{OBJECTIVES[0]} gain of {cv['mean_gain']:+.2f} "
                   f"+/- {cv['std_gain']:.2f} points.", ""]

    md += ["## Table 4 - Published results on BSARD (test split)", "",
           "Two protocols coexist: Louis and Spanakis (ACL 2022) report MAP@100 "
           "and MRR@100, while G-DSR and QABISAR report untruncated MAP and "
           "mean R-precision. Only columns from the same protocol are "
           "comparable. These values are taken from the cited papers; they are "
           "not produced by this repository and do not share its "
           "preprocessing.", "",
           "| System | Training | Params | R@100 | R@200 | R@500 | MAP | MRP |",
           "|---|---|---|---|---|---|---|---|",
           "| BM25 | none | - | 49.3 | 57.3 | 63.0 | 16.8 | 13.6 |",
           "| DSR | supervised | 234M | 77.1 | 81.8 | 86.7 | 35.6 | 28.8 |",
           "| DSR + domain-adaptive pre-training | supervised | 234M | 79.8 | 83.9 | 88.9 | 39.5 | 31.3 |",
           "| DSR + data augmentation | supervised | 234M | 82.7 | 88.7 | 92.8 | 35.3 | 27.5 |",
           "| G-DSR | supervised + GNN | 262M | 84.3 | 90.4 | 93.1 | 47.1 | 40.2 |",
           "| QABISAR | supervised + graph | - | 83.7 | 87.9 | 91.3 | 43.1 | 35.6 |",
           "",
           "Sources: Louis and Spanakis, ACL 2022; Louis, van Dijck and "
           "Spanakis, EACL 2023; Santosh, Sarwat and Grabmair, COLING 2025.",
           ""]

    md += ["## Significance and stratified gains", "",
           "Paired bootstrap (10,000 resamples, 95% CI) and paired permutation "
           "test. `***` p < 0.001, `**` p < 0.01, `*` p < 0.05.",
           "", *sig_blocks, ""]

    checks: list[tuple[bool, str]] = [
        (bool(raw.get("locality")), "Table 1 - locality analysis"),
        (bool(summary), "Table 2b - summary per first stage"),
        (len(retrievers) >= 3,
         f"at least 3 first stages ({len(retrievers)} present: "
         + ", ".join(n for n, _ in retrievers) + ")"),
        (any(n.startswith("FT-") for n, _ in retrievers),
         "a fine-tuned retriever (required to compare against the published "
         "supervised baselines)"),
        (any(n.startswith("ZS-") for n, _ in retrievers),
         "at least one semantic encoder"),
        (bool(emb_cache), "VHP-E evaluated (requires a vector-based retriever)"),
        (not args.no_ablations, "Table 3 - ablations"),
        (bool(figures.get("granularity")), "Figure 1 - granularity"),
        (bool(figures.get("tradeoff_beta")), "Figure 2 - beta trade-off"),
        (bool(raw.get("cross_validation")),
         "cross-validation (stability of the calibration)"),
        (args.max_seq_length >= 512,
         f"articles encoded over 512 tokens (current: {args.max_seq_length}; "
         "below 512 this should be reported as a limitation)"),
        (any(v.get("pareto") for sysd in raw["systems"].values()
             for v in sysd.get("variants", {}).values()),
         "at least one Pareto improvement"),
    ]
    md += ["## Completeness check", ""]
    for ok, label in checks:
        md.append(f"- {'[x]' if ok else '[ ] MISSING:'} {label}")
    md.append("")
    missing = [l for ok, l in checks if not ok]
    if missing:
        log("WARNING - missing items:")
        for m in missing:
            log(f"    - {m}")
    else:
        log("Completeness check: everything is present.")

    out.mkdir(parents=True, exist_ok=True)
    (out / "paper.md").write_text("\n".join(md), encoding="utf-8")
    (out / "tables.tex").write_text("\n\n".join(tex), encoding="utf-8")
    (out / "figures.json").write_text(json.dumps(figures, indent=2,
                                                 ensure_ascii=False),
                                      encoding="utf-8")
    (out / "results.json").write_text(
        json.dumps(raw, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")

    log(f"Done. Tables in {(out / 'paper.md').resolve()}")
    print("\n" + format_table(clean_rows, METRIC_ORDER))


if __name__ == "__main__":
    main()
