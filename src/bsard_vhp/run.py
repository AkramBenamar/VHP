from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import analysis, data as data_mod, diagnose as diag_mod, stats
from .metrics import METRIC_ORDER, aggregate, evaluate, format_table
from .propagate import VHP, VHPParams
from .retrievers import (BM25Config, BM25Retriever, DenseConfig, DenseRetriever,
                         HybridRetriever, TfidfSvdRetriever)
from .structure import build_structure
from .tune import (cross_validate, granularity_curve,
                   sensitivity_curve, staged_search, tradeoff_curve)


def build_retriever(name: str, args) -> object:
    def _bm25_cfg():
        return BM25Config(k1=args.bm25_k1, b=args.bm25_b,
                          use_stemming=not args.no_stemming,
                          remove_numbers=not args.keep_numbers,
                          idf_scheme=args.idf_scheme)

    if name == "bm25":
        return BM25Retriever(_bm25_cfg())
    if name == "dense":
        return DenseRetriever(DenseConfig(model_name=args.dense_model,
                                          device=args.device))
    if name == "tfidf":
        return TfidfSvdRetriever()
    if name == "hybrid":
        return HybridRetriever(
            BM25Retriever(_bm25_cfg()),
            (DenseRetriever(DenseConfig(model_name=args.dense_model,
                                        device=args.device))
             if not args.no_torch else TfidfSvdRetriever()),
            alpha=args.hybrid_alpha,
        )
    raise ValueError(f"unknown retriever: {name}")


def cmd_inspect(args) -> None:
    data_mod.inspect(args.data_dir)


def cmd_diagnose(args) -> None:
    corpus, train, test = data_mod.load_bsard(args.data_dir)
    structure = build_structure(corpus, ambiguity=args.citation_ambiguity,
                                verbose=False)
    print(diag_mod.run_diagnosis(corpus, train, test, structure))


def cmd_citations(args) -> None:
    corpus = data_mod.load_corpus(args.data_dir)
    print(diag_mod.sample_unresolved_citations(corpus, n=args.n))


def cmd_analyze(args) -> None:
    corpus, train, test = data_mod.load_bsard(args.data_dir)
    structure = build_structure(corpus, ambiguity=args.citation_ambiguity)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = "\n\n".join([
        analysis.run_full_analysis(corpus, train, structure, "train"),
        analysis.run_full_analysis(corpus, test, structure, "test"),
    ])
    (out / "locality_analysis.md").write_text(report, encoding="utf-8")
    print(report)


def cmd_experiment(args) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("== Loading ==")
    corpus, train, test = data_mod.load_bsard(
        args.data_dir, append_extra_description=args.use_extra_description,
        include_description_in_text=args.augment_text_with_description)
    print(f"  {len(corpus)} articles | {len(train)} train questions "
          f"| {len(test)} test questions")

    print("== Legislative structure ==")
    structure = build_structure(corpus, ambiguity=args.citation_ambiguity)
    vhp = VHP(structure)

    rows, sig_report, raw = [], [], {}

    for rname in args.retrievers:
        print(f"\n== Retriever: {rname} ==")
        t0 = time.time()
        retr = build_retriever(rname, args)
        retr.index(corpus.texts)
        s_train = retr.score(train.texts)
        s_test = retr.score(test.texts)
        print(f"  indexing + scoring: {time.time() - t0:.1f}s")

        agg_base, pq_base = evaluate(s_test, test.qrels)
        rows.append({"system": retr.name, **{m: agg_base[m] for m in METRIC_ORDER}})

        tuned: dict[str, object] = {}
        for obj in args.objectives:
            print(f"  calibrating VHP on the training split, objective {obj}...")
            t0 = time.time()
            b, tr = staged_search(vhp, s_train, train.qrels,
                                  objective=obj, verbose=args.verbose)
            tuned[obj] = (b, tr)
            print(f"    selected: {b.as_dict()}")
            print(f"    ({time.time() - t0:.1f}s)")
        best, trace = tuned[args.objectives[0]]

        variants = {
            f"{retr.name} + VHP (vertical only)": replace(best, gamma=0.0),
            f"{retr.name} + VHP (horizontal only)": replace(best, beta=0.0),
            f"{retr.name} + VHP (full)": best,
        }
        for obj in args.objectives[1:]:
            variants[f"{retr.name} + VHP (calibrated {obj})"] = tuned[obj][0]

        raw[retr.name] = {"base": agg_base, "params": best.as_dict(),
                          "tuning_trace": trace, "variants": {}}
        for label, p in variants.items():
            t0 = time.time()
            s_vhp = vhp(s_test, params=p)
            elapsed = time.time() - t0
            agg, pq = evaluate(s_vhp, test.qrels)
            rows.append({"system": label, **{m: agg[m] for m in METRIC_ORDER}})

            cmp_res = stats.compare(pq_base, pq, metrics=METRIC_ORDER)
            sig_report.append(stats.format_comparison(
                f"{retr.name} -> {label}", cmp_res))

            strat = stats.stratify_by_gold_size(
                pq_base, pq, test.qrels, metric=args.objectives[0])
            sig_report.append(
                f"\n#### Gain by size of the relevant set "
                f"({label}, {args.objectives[0]})\n"
                + stats.format_stratified(strat, args.objectives[0]))

            pareto = all(agg[m] >= agg_base[m] for m in METRIC_ORDER)
            n_better = sum(1 for m in METRIC_ORDER if agg[m] > agg_base[m])
            sig_report.append(
                f"  -> {n_better}/{len(METRIC_ORDER)} metrics improved"
                + ("  [PARETO IMPROVEMENT]" if pareto else ""))

            print(f"  {label}: {elapsed * 1000:.0f} ms over {len(test)} "
                  f"queries ({elapsed * 1000 / len(test):.1f} ms/query), "
                  f"{n_better}/{len(METRIC_ORDER)} metrics improved"
                  + (" [PARETO]" if pareto else ""))

            raw[retr.name]["variants"][label] = {
                "params": p.as_dict(),
                "metrics": agg,
                "significance": {k: {kk: float(vv) for kk, vv in v.items()}
                                 for k, v in cmp_res.items()},
                "stratified": strat,
                "pareto_improvement": bool(pareto),
                "n_metrics_improved": n_better,
                "latency_ms_total": elapsed * 1000,
                "latency_ms_per_query": elapsed * 1000 / len(test),
            }

        if args.sensitivity:
            gc = granularity_curve(vhp, s_train, train.qrels, best,
                                   objective=args.objectives[0])
            raw[retr.name]["granularity_curve"] = gc
            print("  granularity (vertical axis only, training split):")
            for row in gc:
                print(f"    {row['level']:<12} {row['n_groups']:>6} groups  "
                      f"{args.objectives[0]}={row[args.objectives[0]]:.2f}")
            tc = tradeoff_curve(vhp, s_train, train.qrels, best)
            raw[retr.name]["tradeoff_curve"] = tc
            print("  recall/precision trade-off against beta (training split):")
            print("    beta   R@100  R@500  MAP@100  MRR@100    MRP")
            for r in tc:
                print(f"    {r['beta']:<6.2f} {r['R@100']:5.1f}  "
                      f"{r['R@500']:5.1f}  {r['MAP@100']:6.1f}  "
                      f"{r['MRR@100']:6.1f}  {r['MRP']:6.1f}")
            raw[retr.name]["sensitivity_beta"] = sensitivity_curve(
                vhp, s_train, train.qrels, best, axis="beta",
                objective=args.objectives[0])
            raw[retr.name]["sensitivity_gamma"] = sensitivity_curve(
                vhp, s_train, train.qrels, best, axis="gamma",
                values=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8),
                objective=args.objectives[0])

        if args.cross_validate:
            cv = cross_validate(vhp, s_train, train.qrels,
                                objective=args.objectives[0])
            print(f"  training-split CV, {args.objectives[0]} gain: "
                  f"{cv['mean_gain']:+.2f} +/- {cv['std_gain']:.2f}")
            raw[retr.name]["cv"] = cv

    table = format_table(rows, METRIC_ORDER)
    report = "\n\n".join(["# BSARD results (test split)", table,
                          "## Significance (paired bootstrap and permutation)",
                          "\n".join(sig_report), literature_table()])
    (out / "results.md").write_text(report, encoding="utf-8")
    (out / "results.json").write_text(json.dumps(raw, indent=2,
                                                 ensure_ascii=False),
                                      encoding="utf-8")
    (out / "results.tex").write_text(latex_table(rows), encoding="utf-8")
    print("\n" + report)
    print(f"\nWritten to {out.resolve()}")


def literature_table() -> str:
    return """## Published reference points on BSARD (test split)

Two protocols coexist in the literature: Louis and Spanakis (ACL 2022) report
MAP@100 and MRR@100, while the G-DSR and QABISAR line reports untruncated MAP
and mean R-precision. Only columns from the same protocol are comparable, and
none of these systems shares the preprocessing used here.

| System | Params | R@100 | R@200 | R@500 | MAP | MRP |
|---|---|---|---|---|---|---|
| BM25 | - | 49.3 | 57.3 | 63.0 | 16.8 | 13.6 |
| DSR | 234M | 77.1 | 81.8 | 86.7 | 35.6 | 28.8 |
| DSR + domain-adaptive pre-training | 234M | 79.8 | 83.9 | 88.9 | 39.5 | 31.3 |
| DSR + data augmentation | 234M | 82.7 | 88.7 | 92.8 | 35.3 | 27.5 |
| G-DSR | 262M | 84.3 | 90.4 | 93.1 | 47.1 | 40.2 |
| QABISAR | - | 83.7 | 87.9 | 91.3 | 43.1 | 35.6 |

Sources: Louis and Spanakis, ACL 2022; Louis, van Dijck and Spanakis, EACL
2023; Santosh, Sarwat and Grabmair, COLING 2025."""


def latex_table(rows: list[dict]) -> str:
    head = ("\\begin{tabular}{l" + "c" * len(METRIC_ORDER) + "}\n\\hline\n"
            "System & " + " & ".join(METRIC_ORDER) + " \\\\\n\\hline\n")
    body = "".join(
        f"{r['system']} & " + " & ".join(f"{r[m]:.1f}" for m in METRIC_ORDER)
        + " \\\\\n" for r in rows)
    return head + body + "\\hline\n\\end{tabular}\n"


def main() -> None:
    ap = argparse.ArgumentParser(prog="bsard_vhp")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", default="data/bsard")
    common.add_argument("--out-dir", default="results")
    common.add_argument("--citation-ambiguity", default="nearest",
                        choices=["nearest", "skip", "all"],
                        help="disambiguation policy for references whose "
                             "number matches several articles")

    p = sub.add_parser("inspect", parents=[common]); p.set_defaults(fn=cmd_inspect)
    p = sub.add_parser("diagnose", parents=[common]); p.set_defaults(fn=cmd_diagnose)
    p = sub.add_parser("citations", parents=[common])
    p.add_argument("--n", type=int, default=20)
    p.set_defaults(fn=cmd_citations)
    p = sub.add_parser("analyze", parents=[common]); p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("experiment", parents=[common])
    p.add_argument("--retrievers", nargs="+", default=["bm25"],
                   choices=["bm25", "dense", "hybrid", "tfidf"])
    p.add_argument("--dense-model", default="intfloat/multilingual-e5-base")
    p.add_argument("--device", default=None)
    p.add_argument("--bm25-k1", type=float, default=0.9)
    p.add_argument("--bm25-b", type=float, default=0.4)
    p.add_argument("--no-stemming", action="store_true")
    p.add_argument("--idf-scheme", default="bsard", choices=["bsard", "lucene"])
    p.add_argument("--keep-numbers", action="store_true",
                   help="keep numeric tokens (the reference BSARD "
                        "preprocessing removes them)")
    p.add_argument("--no-torch", action="store_true")
    p.add_argument("--hybrid-alpha", type=float, default=0.5)
    p.add_argument("--objectives", nargs="+", default=["R@100", "MAP@100"],
                   help="calibration objectives; the first one defines the main "
                        "configuration, the others add a row to the table "
                        "(recall/precision trade-off)")
    p.add_argument("--use-extra-description", action="store_true",
                   help="append extra_description to the question")
    p.add_argument("--augment-text-with-description", action="store_true",
                   help="index 'description + article' instead of 'article' "
                        "alone; the reference BSARD code indexes the article "
                        "alone")
    p.add_argument("--cross-validate", action="store_true")
    p.add_argument("--sensitivity", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=cmd_experiment)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
