# VHP: Vertical-Horizontal Propagation for Statutory Article Retrieval

Training-free score regularization over the explicit structure of legislation.
Reference implementation and full experimental pipeline for *How Much of the
Legislative Structure Can We Use Without Training? Score Regularization for
Statutory Article Retrieval*, evaluated on the Belgian Statutory Article
Retrieval Dataset (BSARD).

Statutory articles are not independent documents. Their meaning depends on their
position in the plan of the law and on the cross-references they contain, yet
first-stage retrievers score each article in isolation. VHP reinjects both
structures after the fact, with no trained parameter, at a median cost of about
three milliseconds per query, and improves every first stage we tested.

## What the method does

Two structures are read once from the corpus, offline, with no annotation and no
learning:

- **Vertical** -- the legislator's own taxonomy. Each article carries a path of
  up to eight labels (law type, code, book, part, act, chapter, section,
  subsection). A level partitions the corpus by *prefix equality*, so a block is
  defined by the whole path down to that level: "Chapter I" means nothing outside
  its code and book.
- **Horizontal** -- the network of explicit cross-references between articles,
  extracted by rule from the article text and resolved to `(code, number)` pairs.

Two instruments apply them, at different points of the pipeline:

**VHP-S** acts on the scores of any first stage, lexical retrievers included.
With `z` the per-query standardised scores,

```
s* = z + beta * V(z) + gamma * H(z)
```

`V` is the vertical signal: each article receives the mean of the `m` best scores
in its division, at every level at once, weighted by `w_l`. The article is
excluded from its own pool: without that leave-one-out, `V` would be constant
within a block and would leave the ranking inside a division unchanged. `H` is
the horizontal signal, a decayed `kappa`-step propagation over the
row-stochastic reference graph. With `beta = gamma = 0` the base ranking is
recovered exactly, so the layer extends the identity.

**VHP-E** acts one stage earlier, on the representations a dense retriever
indexes:

```
E* = E + beta * sum_l w_l V_l(E) + gamma * H(E)
```

This is a graph convolution stripped of its learned weights and non-linearities.
Being linear, it runs once at indexing time and costs nothing per query, and the
score decomposes into precomputed matrices, which makes the coefficient search
cheap. VHP-E requires embeddings and does not apply to BM25.

## Installation

```bash
git clone https://github.com/AkramBenamar/VHP.git vhp
cd vhp
pip install -e .
```

Optional extras:

```bash
pip install -e ".[lexical]"   # nltk, for French stemming in BM25
pip install -e ".[dense]"     # torch and sentence-transformers
pip install -e ".[data]"      # datasets, to download BSARD
pip install -e ".[figures]"   # matplotlib
pip install -e ".[all]"       # everything, including pytest
```

Stemming is part of the reported protocol: without `nltk` the BM25 numbers are
not comparable to the ones in the paper.

## Getting the data

```bash
python scripts/download_bsard.py --out data/bsard
```

This writes `articles_fr.csv`, `questions_fr_train.csv`,
`questions_fr_test.csv`, and `questions_fr_synthetic.csv` when available. BSARD
is released by Louis and Spanakis under its own licence; see the
[dataset repository](https://github.com/maastrichtlawtech/bsard).

Column names vary across releases, so check what was found before anything else:

```bash
python -m bsard_vhp.run inspect  --data-dir data/bsard
python -m bsard_vhp.run diagnose --data-dir data/bsard
```

`diagnose` returns a verdict per component rather than raw numbers: whether
enough hierarchy levels are usable, how many references resolve, and how many
questions have at least two relevant articles, which is the population VHP can
help.

No data at hand? A synthetic corpus in the same format, with the two properties
VHP exploits planted in it, validates the pipeline end to end:

```bash
python scripts/make_synthetic.py --out data/synthetic
python -m bsard_vhp.run experiment --data-dir data/synthetic
```

## Reproducing the paper

One command runs everything: fine-tuning, then all experiments, tables and
figure data.

```bash
python scripts/run_everything.py --data-dir data/bsard --out-dir paper_results
```

It is resumable. A model already trained is not retrained and score matrices are
cached, so relaunching after an interruption picks up where it stopped; pass
`--force` to recompute. Without a GPU it reduces the sequence length and batch
size automatically and trains only the smaller model.

To skip training and produce the training-free results alone, which is about an
hour of CPU:

```bash
python scripts/run_everything.py --data-dir data/bsard --skip-finetune
```

The experiments stage can also be driven directly:

```bash
python scripts/run_paper.py --data-dir data/bsard --out-dir paper_results \
    --tfidf \
    --dense-models intfloat/multilingual-e5-base BAAI/bge-m3 \
                   antoinelouis/biencoder-camembert-base-mmarcoFR \
    --finetuned-models models/e5-base-bsard models/bge-m3-bsard
```

Outputs in `--out-dir`:

| File | Contents |
|---|---|
| `paper.md` | every table in Markdown, plus a completeness check |
| `tables.tex` | the same tables as LaTeX, ready to include |
| `figures.json` | curve data: granularity, trade-off, sensitivity |
| `results.json` | everything, including the calibration traces |
| `cache/` | score matrices, so a rerun skips the encoding |

Then the figures, in PDF and SVG:

```bash
python scripts/make_figures.py --out figures --from-results paper_results
```

Without `--from-results` the script draws the values reported in the paper.

## Protocol

- **Calibration uses the training split only.** The test split is evaluated
  once, with the final configurations.
- **Hyperparameters.** `beta` in [0.1, 2.4], `gamma` in [0.05, 0.5], `m` in
  {1, 3, 5}, `kappa` in {1, 2}, three directions for the reference graph
  (`out`, `in`, `sym`), and thirteen fixed weight schemes over the levels. The
  decay `rho` is held at 0.5 and is not searched. The hierarchy therefore
  contributes one discrete choice rather than eight free parameters.
- **Staged search.** The vertical axis alone, then the horizontal axis with the
  vertical one frozen, then a joint refinement of `(beta, gamma)`. A
  configuration is kept only if it improves on the best single-axis one, so an
  axis contributing nothing is switched off rather than fitted to noise.
- **Three operating points** are reported: maximum R@100, maximum MAP@100, and
  maximum R@100 subject to no other metric falling below its baseline. The
  third is the one we recommend.
- **Metrics.** Both conventions in the literature: Recall@100/200/500, MAP@100
  and MRR@100 as in Louis and Spanakis (ACL 2022), alongside untruncated MAP and
  mean R-precision as in the G-DSR and QABISAR line. `tests/test_core.py`
  checks numerical equivalence with a direct port of the reference BSARD
  evaluator.
- **Significance.** Confidence intervals come from a paired bootstrap over
  10,000 resamples; every p-value comes from a paired permutation test with
  10,000 permutations, with Holm-Bonferroni correction available for multiple
  comparisons.
- **Published figures are never mixed with ours.** They sit in a separate table
  and do not share this preprocessing.

## Repository layout

```
src/bsard_vhp/
    data.py              BSARD loading, column detection, cumulative hierarchy
    structure.py         reference extraction, resolution, adjacency matrices
    propagate.py         VHP-S: vertical and horizontal signals over scores
    embed_propagate.py   VHP-E: the same operator over representations
    retrievers.py        BM25, dense bi-encoders, TF-IDF/SVD, fusion
    metrics.py           both evaluation protocols
    stats.py             paired bootstrap, permutation test, stratified gains
    tune.py              staged and constrained calibration, curves
    analysis.py          structural locality of the labels
    diagnose.py          per-component verdicts on a loaded corpus
    run.py               command-line interface
scripts/
    download_bsard.py    fetch the dataset
    make_synthetic.py    synthetic corpus in the same format
    finetune_dense.py    fine-tune a bi-encoder on BSARD
    run_paper.py         all tables and figure data
    run_everything.py    fine-tuning then experiments, resumable
    make_figures.py      figures in PDF and SVG
tests/test_core.py       20 tests
```

## Implementation notes

A few choices are load-bearing and easy to get wrong when reimplementing.

**Cumulative blocks.** Levels are accumulated along the path, so two "Chapter I"
from different books never merge. Flat per-level grouping silently inflates
block sizes and destroys the signal.

**Leave-one-out.** Both `VerticalAggregator` and `EmbeddingSmoother` exclude the
article from its own pool. Without it the layer amplifies the first stage instead
of adding information, and an article alone in its division receives its own
score back.

**Reference disambiguation.** The same number recurs across books of one code,
so `(code, number)` can match several articles. The default policy keeps the
candidate whose address agrees with the citing article over the longest prefix,
on the reasoning that an internal reference targets the nearest division. The
`skip` and `all` policies are available as ablations.

**Article numbering.** Belgian numbering is richer than a decimal sequence:
`1382`, `1.1.1` in regional codes, `N1.1` in annexes, `1er`, `24bis`, `1017/1`.
Whitespace is normalised so that `24bis` and `24 bis` give the same key, while
the slash is kept, since article 1017/1 is not article 10171.

**IDF scheme.** `bsard` reproduces the reference implementation,
`log10((N - df + 0.5) / (df + 0.5))` with no additive smoothing, which goes
negative for terms present in more than half the corpus. That is the original
behaviour and it is kept for comparability; `lucene` is the standard variant.
Numeric tokens are removed by default, as in the reference preprocessing, which
on legal text means article numbers leave the lexical index; `--keep-numbers`
inverts it and makes a good ablation.

**Calibration cost.** The structural signals do not depend on `beta` or `gamma`,
so `VHPScorer` computes them once per `(level, m)` and the search over
coefficients becomes a linear combination of cached matrices. For VHP-E the
score is bilinear in the embeddings, giving the same saving; `test_core.py`
checks that the decomposition matches explicit smoothing.

## Tests

```bash
python -m pytest tests/ -v
```

or without pytest:

```bash
python tests/test_core.py
```

The suite checks the metrics against hand computations and against the reference
BSARD evaluator, the vectorised leave-one-out against a naive implementation,
reference resolution including ranges and ambiguity policies, the BM25 formulas
against both IDF schemes, the VHP-E bilinear decomposition, and that constrained
calibration either respects its constraints or falls back to the identity.

## Citation

```bibtex
@inproceedings{benamar2026vhp,
  title     = {How Much of the Legislative Structure Can We Use Without
               Training? Score Regularization for Statutory Article Retrieval},
  author    = {Benamar, Akram and Dilmi, Mohamed Djallel and
               Chaieb-Chakchouk, Faten},
  booktitle = {Legal Knowledge and Information Systems (JURIX)},
  year      = {2026},
}
```

Please also cite the dataset:

```bibtex
@inproceedings{louis2022bsard,
  title     = {A Statutory Article Retrieval Dataset in French},
  author    = {Louis, Antoine and Spanakis, Gerasimos},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for
               Computational Linguistics},
  pages     = {6789--6803},
  year      = {2022},
}
```

## License

MIT, see `LICENSE`. The BSARD dataset is distributed separately under its own
terms.
