from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bsard_vhp.data import Corpus, extract_article_number
from bsard_vhp.metrics import evaluate, per_query_metrics, rank_documents
from bsard_vhp.propagate import (VHP, VerticalAggregator, VHPParams,
                                 horizontal_signal, zscore_rows)
from bsard_vhp.retrievers import BM25Retriever
from bsard_vhp.structure import (LegislativeStructure,
                                 build_citation_graph, normalize_adjacency)


def test_metrics_hand_computed():
    ranking = np.array([[0, 1, 3, 2, 4]])
    qrels = [{0, 3}]
    m = per_query_metrics(ranking, qrels, recall_ks=(1, 2, 3),
                          map_ks=(), mrr_ks=())

    assert m["R@1"][0] == 0.5
    assert m["R@2"][0] == 0.5
    assert m["R@3"][0] == 1.0
    assert abs(m["MAP"][0] - (1.0 + 2 / 3) / 2) < 1e-9
    assert abs(m["MRP"][0] - 0.5) < 1e-9
    print("  ok  metrics match the hand computation")


def test_ranking_matches_full_sort():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(7, 500)).astype(np.float32)
    full = rank_documents(s, top_k=None)
    trunc = rank_documents(s, top_k=50)
    assert np.array_equal(full[:, :50], trunc)
    print("  ok  truncated argpartition equals the full sort on the prefix")


def _naive_loo(z: np.ndarray, groups: np.ndarray, top_m: int) -> np.ndarray:
    q, n = z.shape
    out = np.zeros((q, n))
    for i in range(n):
        members = [j for j in range(n) if groups[j] == groups[i] and j != i]
        if not members:
            continue
        m = min(top_m, len(members))
        for k in range(q):
            vals = sorted((z[k, j] for j in members), reverse=True)[:m]
            out[k, i] = float(np.mean(vals))
    return out


def test_vertical_leave_one_out_matches_naive():
    rng = np.random.default_rng(1)
    n, q = 40, 6
    groups = rng.integers(0, 5, size=n)
    z = rng.normal(size=(q, n)).astype(np.float64)
    agg = VerticalAggregator(groups)
    for top_m in (1, 2, 3):
        fast = agg(z, top_m=top_m, exclude_self=True)
        slow = _naive_loo(z, groups, top_m)
        err = np.abs(fast - slow).max()
        assert err < 1e-6, f"top_m={top_m} max error {err}"
    print("  ok  vectorised leave-one-out equals the naive reference")


def test_vertical_excludes_self_information():
    groups = np.array([0, 0, 1])
    z = np.array([[5.0, 1.0, 9.0]])
    out = VerticalAggregator(groups)(z, top_m=1, exclude_self=True)
    assert out[0, 2] == 0.0
    assert out[0, 0] == 1.0
    assert out[0, 1] == 5.0
    print("  ok  no self-amplification (leave-one-out is effective)")


def test_horizontal_promotes_silent_article():
    n = 5
    a = sp.csr_matrix(([1.0], ([0], [2])), shape=(n, n))
    base = np.array([[3.0, 0.1, -1.0, 0.0, 0.05]], dtype=np.float32)

    structure = LegislativeStructure(
        n_articles=n, levels=[], groups={},
        citations=a, n_citation_edges=1,
    )
    vhp = VHP(structure)
    p = VHPParams(beta=0.0, gamma=0.8, direction="sym", n_hops=1)

    rank_before = list(rank_documents(base)[0]).index(2)
    rank_after = list(rank_documents(vhp(base, params=p))[0]).index(2)
    assert rank_after < rank_before, (rank_before, rank_after)
    print(f"  ok  silent article promoted from rank {rank_before + 1} "
          f"to rank {rank_after + 1} by the reference graph")


def test_normalize_adjacency_is_row_stochastic():
    a = sp.csr_matrix(np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=float))
    m = normalize_adjacency(a, direction="out", scheme="row")
    sums = np.asarray(m.sum(axis=1)).ravel()
    assert abs(sums[0] - 1.0) < 1e-6 and abs(sums[1] - 1.0) < 1e-6
    assert sums[2] == 0.0
    print("  ok  row-stochastic normalisation")


def test_zscore_is_scale_invariant():
    rng = np.random.default_rng(2)
    s = rng.normal(size=(4, 100))
    assert np.abs(zscore_rows(s) - zscore_rows(1000 * s + 7)).max() < 1e-6
    print("  ok  VHP is invariant to the scale of the retriever scores")


def test_citation_regex():
    assert extract_article_number("Code civil, art. 1382") == "1382"
    assert extract_article_number("Code civil, article 24bis") == "24bis"
    assert extract_article_number("Code civil, art. 24 bis") == "24bis"
    assert extract_article_number("C. jud., art. 1017/1") == "1017/1"
    print("  ok  article numbers extracted")


def test_citation_graph_resolution():
    corpus = Corpus(
        ids=np.array([0, 1, 2]),
        texts=[
            "Le bailleur est tenu, conformément à l'article 1720 du Code civil, "
            "de délivrer la chose.",
            "Sans préjudice des articles 1719 à 1721, le preneur doit payer.",
            "Disposition isolée sans renvoi.",
        ],
        titles=["", "", ""],
        references=["Code civil, art. 1719", "Code civil, art. 1728",
                    "Code civil, art. 1720"],
        numbers=["1719", "1728", "1720"],
        hierarchy={},
    )
    corpus.raw = None
    c, cstats = build_citation_graph(corpus, verbose=False)
    assert c[0, 2] == 1, "explicit reference 1719 -> 1720 not resolved"
    assert c[1, 0] == 1, "range '1719 a 1721' not expanded to 1719"
    assert c[1, 2] == 1, "range '1719 a 1721' not expanded to 1720"
    assert c[2].nnz == 0
    print(f"  ok  reference graph: {c.nnz} edges, "
          f"resolution rate {cstats['resolution_rate']:.0%}")


def test_bm25_sanity():
    docs = [
        "le bailleur doit delivrer la chose louee au preneur",
        "la peine d emprisonnement est prononcee par le tribunal",
        "le contrat de bail est resilie moyennant un preavis",
    ]
    r = BM25Retriever().index(docs)
    s = r.score(["resiliation du bail et preavis du preneur"])
    assert s.argmax(axis=1)[0] == 2, s
    assert (s >= 0).all()
    print("  ok  BM25 ranks the right document first")


def test_bm25_matches_reference_formula():
    from bsard_vhp.retrievers import BM25Config

    docs = ["chat chat chien", "chien"]
    n, avgdl, df_chat, tf, dl = 2, 2.0, 1, 2, 3

    r = BM25Retriever(BM25Config(idf_scheme="lucene", use_stemming=False,
                                 remove_stopwords=False, remove_numbers=False))
    r.index(docs)
    k1, b = r.cfg.k1, r.cfg.b
    idf = np.log(1 + (n - df_chat + 0.5) / (df_chat + 0.5))
    expected = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    got = r.score(["chat"])[0, 0]
    assert abs(got - expected) < 1e-4, (got, expected)

    r2 = BM25Retriever(BM25Config(idf_scheme="bsard", use_stemming=False,
                                  remove_stopwords=False, remove_numbers=False))
    r2.index(docs)
    idf2 = np.log10((n - df_chat + 0.5) / (df_chat + 0.5))
    expected2 = idf2 * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    got2 = r2.score(["chat"])[0, 0]
    assert abs(got2 - expected2) < 1e-4, (got2, expected2)
    print(f"  ok  BM25 matches the reference formulas (lucene {got:.4f}, "
          f"bsard {got2:.4f})")


def test_bm25_removes_numbers_like_official_preprocessing():
    from bsard_vhp.retrievers import tokenize_fr

    t = "conformement a l article 1382 du code civil"
    assert "1382" in tokenize_fr(t, None, False, remove_numbers=False)
    assert "1382" not in tokenize_fr(t, None, False, remove_numbers=True)
    print("  ok  numeric tokens removed, as in the reference preprocessing")


def test_end_to_end_vhp_recovers_planted_signal():
    rng = np.random.default_rng(3)
    n_sections, per_section = 30, 8
    n = n_sections * per_section
    groups = np.repeat(np.arange(n_sections), per_section)

    n_q = 60
    base = rng.normal(0, 0.3, size=(n_q, n)).astype(np.float32)
    qrels = []
    for q in range(n_q):
        sec = q % n_sections
        members = np.flatnonzero(groups == sec)
        visible, silent = members[0], members[1]
        base[q, visible] += 6.0
        qrels.append({int(visible), int(silent)})

    structure = LegislativeStructure(
        n_articles=n, levels=["section"], groups={"section": groups},
        citations=sp.csr_matrix((n, n)), n_citation_edges=0,
    )
    vhp = VHP(structure)
    p = VHPParams(beta=1.5, gamma=0.0, top_m=1, level_weights={"section": 1.0})

    before, _ = evaluate(base, qrels, recall_ks=(2, 10),
                         map_ks=(), mrr_ks=())
    after, _ = evaluate(vhp(base, params=p), qrels,
                        recall_ks=(2, 10), map_ks=(), mrr_ks=())
    assert after["R@10"] > before["R@10"] + 5, (before, after)
    print(f"  ok  planted structural signal recovered: R@10 "
          f"{before['R@10']:.1f} -> {after['R@10']:.1f}")


class OfficialEvaluator:

    def precision(self, ground_truths, results, k=None):
        k = len(results) if k is None else k
        rel = [1 if d in ground_truths else 0 for d in results[:k]]
        return sum(rel) / len(results[:k])

    def recall(self, ground_truths, results, k=None):
        k = len(results) if k is None else k
        rel = [1 if d in ground_truths else 0 for d in results[:k]]
        return sum(rel) / len(ground_truths)

    def reciprocal_rank(self, ground_truths, results, k=None):
        k = len(results) if k is None else k
        return max([1 / (i + 1) if d in ground_truths else 0.0
                    for i, d in enumerate(results[:k])])

    def average_precision(self, ground_truths, results, k=None):
        k = len(results) if k is None else k
        p_k = [self.precision(ground_truths, results, k=i + 1)
               if d in ground_truths else 0
               for i, d in enumerate(results[:k])]
        return sum(p_k) / len(ground_truths)


def test_matches_official_bsard_evaluator():
    from bsard_vhp.metrics import aggregate, per_query_metrics

    rng = np.random.default_rng(7)
    n_docs, n_q = 800, 40
    scores = rng.normal(size=(n_q, n_docs)).astype(np.float32)
    qrels = [set(rng.choice(n_docs, size=int(rng.integers(1, 12)),
                            replace=False).tolist()) for _ in range(n_q)]

    ranking = rank_documents(scores)
    mine = aggregate(per_query_metrics(ranking, qrels,
                                       recall_ks=(100, 200, 500),
                                       map_ks=(100,), mrr_ks=(100,)))

    ref = OfficialEvaluator()
    lists = [list(map(int, r)) for r in ranking]
    golds = [sorted(g) for g in qrels]
    for k in (100, 200, 500):
        exp = np.mean([ref.recall(g, r, k) for g, r in zip(golds, lists)]) * 100
        assert abs(mine[f"R@{k}"] - exp) < 1e-9, (k, mine[f"R@{k}"], exp)
    exp_map = np.mean([ref.average_precision(g, r, 100)
                       for g, r in zip(golds, lists)]) * 100
    exp_mrr = np.mean([ref.reciprocal_rank(g, r, 100)
                       for g, r in zip(golds, lists)]) * 100
    assert abs(mine["MAP@100"] - exp_map) < 1e-9, (mine["MAP@100"], exp_map)
    assert abs(mine["MRR@100"] - exp_mrr) < 1e-9, (mine["MRR@100"], exp_mrr)
    print(f"  ok  identical to the reference BSARD evaluator "
          f"(R@100={mine['R@100']:.3f}, MAP@100={mine['MAP@100']:.3f}, "
          f"MRR@100={mine['MRR@100']:.3f})")


def test_hierarchy_from_description_path():
    from bsard_vhp.data import parse_description_paths

    descs = [
        "Code civil - Livre III - Titre VIII - Chapitre II - Section 1",
        "Code civil - Livre III - Titre VIII - Chapitre II - Section 2",
        "Code civil - Livre I - Titre V",
        "Code judiciaire - Partie IV - Livre II - Chapitre I - Section 1",
    ]
    levels = parse_description_paths(descs)
    assert len(levels) >= 3, levels
    assert levels[0][1][0] == "Code civil"
    assert levels[0][1][3] == "Code judiciaire"
    assert levels[1][1][:2] == ["Livre III", "Livre III"]
    print(f"  ok  {len(levels)} levels derived from the textual path")


def test_ambiguous_citation_resolved_by_hierarchical_proximity():
    hier = {
        "code": np.array([0, 0, 0, 0]),
        "book": np.array([0, 0, 1, 1]),
    }
    corpus = Corpus(
        ids=np.array([1, 2, 3, 4]),
        texts=[
            "Le preneur applique l'article 5 pour le calcul.",
            "Disposition du Livre I portant le numero 5.",
            "Autre disposition du Livre II.",
            "Disposition du Livre II portant aussi le numero 5.",
        ],
        titles=[""] * 4,
        references=["Art. 9, Code test", "Art. 5, Code test",
                    "Art. 8, Code test", "Art. 5, Code test"],
        numbers=["9", "5", "8", "5"],
        hierarchy=hier,
    )
    import pandas as pd
    corpus.raw = pd.DataFrame({"code": ["Code test"] * 4})

    c_all, s_all = build_citation_graph(corpus, ambiguity="all", verbose=False)
    c_near, s_near = build_citation_graph(corpus, ambiguity="nearest", verbose=False)
    c_skip, s_skip = build_citation_graph(corpus, ambiguity="skip", verbose=False)

    assert s_all["ambiguous"] == 1
    assert c_all[0].nnz == 2, "policy 'all' must link both homonyms"
    assert c_near[0].nnz == 1, "policy 'nearest' must pick one"
    assert c_near[0, 1] == 1, "must point to article 5 of the same book"
    assert c_skip[0].nnz == 0, "policy 'skip' must discard the reference"
    print(f"  ok  disambiguation by hierarchical proximity "
          f"(all={c_all.nnz}, nearest={c_near.nnz}, skip={c_skip.nnz} edges)")


def test_vhpe_leave_one_out_is_exact_division_mean():
    from bsard_vhp.embed_propagate import EmbeddingSmoother

    rng = np.random.default_rng(11)
    n, d = 60, 16
    groups = np.repeat(np.arange(12), 5)
    st = LegislativeStructure(n, ["section"], {"section": groups},
                              sp.csr_matrix((n, n)), 0)
    emb = rng.normal(size=(n, d)).astype(np.float32)
    v = EmbeddingSmoother(st).level_signal(emb, "section")
    for i in (0, 3, 17, 59):
        peers = [j for j in range(n) if groups[j] == groups[i] and j != i]
        assert np.abs(v[i] - emb[peers].mean(0)).max() < 1e-5
    print("  ok  VHP-E: exact leave-one-out mean over the division")


def test_vhpe_singleton_division_gets_no_signal():
    from bsard_vhp.embed_propagate import EmbeddingSmoother

    rng = np.random.default_rng(12)
    n, d = 20, 8
    groups = np.arange(n)
    st = LegislativeStructure(n, ["x"], {"x": groups}, sp.csr_matrix((n, n)), 0)
    emb = rng.normal(size=(n, d)).astype(np.float32)
    assert np.abs(EmbeddingSmoother(st).level_signal(emb, "x")).max() == 0.0
    print("  ok  VHP-E: singleton divisions receive no signal")


def test_vhpe_bilinear_decomposition_matches_explicit_smoothing():
    from bsard_vhp.embed_propagate import (EmbeddingSmoother, VHPEParams,
                                           VHPEScorer)

    rng = np.random.default_rng(13)
    n, d = 60, 16
    groups = np.repeat(np.arange(12), 5)
    cit = sp.csr_matrix(([1.0, 1.0], ([0, 5], [7, 9])), shape=(n, n))
    st = LegislativeStructure(n, ["section"], {"section": groups}, cit, 2)
    sm = EmbeddingSmoother(st)
    emb = rng.normal(size=(n, d)).astype(np.float32)
    q = rng.normal(size=(9, d)).astype(np.float32)

    p = VHPEParams(beta=0.4, gamma=0.25, level_weights={"section": 1.0},
                   direction="out", normalize=False)
    direct = q @ sm.smooth(emb, p).T
    decomposed = VHPEScorer(sm, emb, q, direction="out").score(p)
    err = np.abs(direct - decomposed).max()
    assert err < 1e-3, err
    print(f"  ok  VHP-E: bilinear decomposition is exact (error {err:.2e})")


def test_constrained_search_respects_its_constraints():
    from bsard_vhp.metrics import evaluate
    from bsard_vhp.tune import staged_search

    rng = np.random.default_rng(21)
    n_sec, per = 40, 10
    n = n_sec * per
    groups = np.repeat(np.arange(n_sec), per)
    st = LegislativeStructure(n, ["section"], {"section": groups},
                              sp.csr_matrix((n, n)), 0)
    vhp = VHP(st)

    scores = rng.normal(0, 0.5, size=(150, n)).astype(np.float32)
    qrels = []
    for q in range(150):
        mem = np.flatnonzero(groups == q % n_sec)
        scores[q, mem[0]] += 5.0
        qrels.append({int(mem[0]), int(mem[1])})

    metrics = ("MAP@100", "MRP")
    ref, _ = evaluate(scores, qrels)

    p, _ = staged_search(vhp, scores, qrels, objective="R@100",
                         constrain_metrics=metrics, verbose=False)
    got, _ = evaluate(vhp(scores, params=p), qrels)
    for m in metrics:
        assert got[m] >= ref[m] - 1e-6, (m, got[m], ref[m])

    p2, _ = staged_search(vhp, scores, qrels, objective="R@100",
                          constrain_metrics=metrics,
                          constraint_slack=-1e6, verbose=False)
    assert p2.beta == 0.0 and p2.gamma == 0.0
    print("  ok  constrained search: constraints respected, falls back to the "
          "identity when infeasible")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"\n{t.__name__}")
        t()
    print(f"\n{len(tests)} tests passed.")
