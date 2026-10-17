# Results and Configurations

Every result behind *How Much of the Legislative Structure Can We Use Without
Training? Score Regularization for Statutory Article Retrieval* (JURIX 2026),
including the configurations, ablations and significance tests that did not fit
in the paper's ten pages.

All figures are produced by this repository. Calibration uses the BSARD
training split alone; the test split is evaluated once, with the final
configurations. Metrics follow both conventions in the literature:
Recall@100/200/500, MAP@100 and MRR@100 as in Louis and Spanakis (ACL 2022),
alongside untruncated MAP and mean R-precision as in the G-DSR and QABISAR
line.

To regenerate everything:

```bash
python scripts/run_everything.py --data-dir data/bsard --out-dir paper_results
```

The underlying `results.json` carries the same numbers in machine-readable
form, with the full calibration traces. See [README.md](README.md) for the
method and the protocol.

---

## Corpus and Extracted Structure

The corpus holds 22,633 articles, with 886 training and 222 test questions. the table below gives the number of blocks of each cumulative partition.

**Cumulative partitions induced by the plan of the law**

| Level | Blocks |
|:---|---:|
| law_type | 2 |
| code | 34 |
| book | 161 |
| part | 182 |
| act | 864 |
| chapter | 2,670 |
| section | 4,658 |
| subsection | 5,626 |

Reference extraction detects 31,430 references and resolves 24,051 of them (76.5%), of which 4,420 (14.1%) match several articles within the target code and are settled by hierarchical proximity (policy `nearest`). The resulting graph has 18,107 edges, that is 0.75 per resolved reference, the difference coming from duplicate references to the same target within one article.

## Structural Locality

**Structural co-location of relevant articles, train split**

| Level | Pairs | Co-located (%) | Random (%) | Lift |
|:---|---:|---:|---:|---:|
| law_type | 87,415 | 99.9 | 53.062 | x2 |
| code | 87,415 | 91.6 | 6.039 | x15 |
| book | 87,415 | 70.2 | 1.798 | x39 |
| part | 87,415 | 70.2 | 1.748 | x40 |
| act | 87,415 | 69.5 | 0.468 | x148 |
| chapter | 87,415 | 43.0 | 0.122 | x352 |
| section | 87,415 | 30.8 | 0.058 | x531 |
| subsection | 87,415 | 27.3 | 0.037 | x729 |

**Structural co-location of relevant articles, test split**

| Level | Pairs | Co-located (%) | Random (%) | Lift |
|:---|---:|---:|---:|---:|
| law_type | 11,399 | 99.6 | 53.062 | x2 |
| code | 11,399 | 96.9 | 6.039 | x16 |
| book | 11,399 | 60.8 | 1.798 | x34 |
| part | 11,399 | 60.8 | 1.748 | x35 |
| act | 11,399 | 58.6 | 0.468 | x125 |
| chapter | 11,399 | 28.6 | 0.122 | x234 |
| section | 11,399 | 23.6 | 0.058 | x407 |
| subsection | 11,399 | 22.7 | 0.037 | x605 |

On the test split, 3.7% of the 4,000 relevant pairs are connected by a direct reference against 0.00% of random pairs, and 11.5% within two hops against 0.03%.

## Selected Hyperparameters

Every configuration below is selected on the training split alone. The last column reports the gain of the tuning objective on that split.

**Hyperparameters retained by the staged search**

| First stage | Objective | Selected configuration | Train gain |
|:---|:---|:---|---:|
| BM25 | R@100 | beta=1.2, gamma=0.3, m=1, dir. out, kappa=1, increasing | +9.72 |
| BM25 | MAP@100 | beta=1.8, gamma=0.15, m=5, dir. out, kappa=2, increasing | +4.94 |
| BM25 | constrained | beta=1.2, gamma=0.3, m=3, dir. out, kappa=2, increasing | +9.43 |
| TF-IDF-SVD | R@100 | beta=0.6, gamma=1.0, m=3, dir. out, kappa=2, only subsection | +8.66 |
| TF-IDF-SVD | MAP@100 | beta=2.4, gamma=0.2, m=5, dir. sym, kappa=2, uniform over 2 levels | +3.74 |
| TF-IDF-SVD | constrained | beta=0.6, gamma=0.75, m=3, dir. out, kappa=2, only subsection | +8.60 |
| TF-IDF-SVD | VHP-E, R@100 | beta=0.8, gamma=0.3, dir. sym, only act | +5.28 |
| TF-IDF-SVD | VHP-E, MAP@100 | beta=0.8, gamma=0.05, dir. out, only section | +3.14 |
| mE5 (zero-shot) | R@100 | beta=2.4, gamma=0.2, m=3, dir. sym, kappa=1, uniform over 8 levels | +9.32 |
| mE5 (zero-shot) | MAP@100 | beta=0.6, gamma=0.05, m=1, dir. out, kappa=1, mixed | +4.56 |
| mE5 (zero-shot) | constrained | beta=2.4, gamma=0.1, m=3, dir. sym, kappa=2, uniform over 8 levels | +9.18 |
| mE5 (zero-shot) | VHP-E, R@100 | beta=0.5, gamma=0.0, only chapter | +4.79 |
| mE5 (zero-shot) | VHP-E, MAP@100 | beta=0.8, gamma=0.05, dir. sym, increasing | +3.36 |
| BGE-M3 (zero-shot) | R@100 | beta=1.2, gamma=0.15, m=3, dir. out, kappa=1, increasing | +7.72 |
| BGE-M3 (zero-shot) | MAP@100 | beta=0.6, gamma=0.05, m=1, dir. out, kappa=2, only chapter | +4.80 |
| BGE-M3 (zero-shot) | constrained | beta=1.2, gamma=0.1, m=3, dir. out, kappa=2, increasing | +7.45 |
| BGE-M3 (zero-shot) | VHP-E, R@100 | beta=0.8, gamma=0.05, dir. sym, only chapter | +5.16 |
| BGE-M3 (zero-shot) | VHP-E, MAP@100 | beta=0.8, gamma=0.05, dir. sym, only section | +4.33 |
| CamemBERT (zero-shot) | R@100 | beta=1.2, gamma=0.2, m=3, dir. out, kappa=2, uniform over 8 levels | +12.26 |
| CamemBERT (zero-shot) | MAP@100 | beta=0.8, gamma=0.05, m=1, dir. out, kappa=2, mixed | +3.01 |
| CamemBERT (zero-shot) | constrained | beta=1.8, gamma=0.15, m=3, dir. out, kappa=2, uniform over 8 levels | +11.40 |
| CamemBERT (zero-shot) | VHP-E, R@100 | beta=0.8, gamma=0.3, dir. sym, only section | +9.69 |
| CamemBERT (zero-shot) | VHP-E, MAP@100 | beta=0.8, gamma=0.2, dir. sym, only section | +4.19 |
| mE5 (fine-tuned) | R@100 | beta=0.6, gamma=0.1, m=1, dir. sym, kappa=2, only code | +0.29 |
| mE5 (fine-tuned) | MAP@100 | beta=0.2, gamma=0.0, m=5, only chapter | +0.54 |
| mE5 (fine-tuned) | constrained | beta=0.2, gamma=0.0, m=3, only chapter | +0.05 |
| mE5 (fine-tuned) | VHP-E, R@100 | beta=0.2, gamma=0.2, dir. sym, mixed | +0.20 |
| mE5 (fine-tuned) | VHP-E, MAP@100 | beta=0.2, gamma=0.0, uniform over 8 levels | +0.79 |
| BGE-M3 (fine-tuned) | R@100 | beta=0.5, gamma=0.05, m=5, dir. sym, kappa=2, increasing | +0.27 |
| BGE-M3 (fine-tuned) | MAP@100 | beta=0.2, gamma=0.0, m=5, uniform over 8 levels | +0.96 |
| BGE-M3 (fine-tuned) | constrained | beta=0.5, gamma=0.0, m=5, uniform over 8 levels | +0.23 |
| BGE-M3 (fine-tuned) | VHP-E, R@100 | beta=0.5, gamma=0.0, increasing | +0.29 |
| BGE-M3 (fine-tuned) | VHP-E, MAP@100 | beta=0.2, gamma=0.0, increasing | +1.39 |

## Complete Results

 the table below lists every configuration evaluated, on the test split. † marks a Pareto improvement over the corresponding baseline. The last column gives the added latency per query in milliseconds; VHP-E adds nothing at query time, its cost being paid once at indexing.

**All configurations on the BSARD test split**

| System | R@100 | R@200 | R@500 | MAP@100 | MRR@100 | MAP | MRP | Improved | ms/query |
|:---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| **BM25** | 51.5 | 57.2 | 66.1 | 17.5 | 24.6 | 17.8 | 13.6 | — | — |
| &nbsp;&nbsp;+ VHP-S, vertical only | 56.7 | 63.4 | 74.3 | 18.2 | 24.2 | 18.5 | 14.0 | 6/7 | 3.4 |
| &nbsp;&nbsp;+ VHP-S, horizontal only | 54.6 | 60.7 | 67.9 | 15.9 | 22.5 | 16.2 | 10.1 | 3/7 | 0.2 |
| &nbsp;&nbsp;+ VHP-S, tuned for R@100 | 59.5 | 65.3 | 73.8 | 18.1 | 25.4 | 18.4 | 12.4 | 6/7 | 3.4 |
| &nbsp;&nbsp;+ VHP-S, tuned for MAP@100† | 58.0 | 63.7 | 75.7 | 19.6 | 24.8 | 19.9 | 14.5 | 7/7 | 3.4 |
| &nbsp;&nbsp;+ VHP-S, constrained | 58.3 | 65.6 | 75.4 | 18.1 | 23.8 | 18.5 | 12.9 | 5/7 | 3.5 |
| | | | | | | | | | |
| **TF-IDF-SVD** | 32.6 | 43.7 | 58.1 | 6.7 | 11.4 | 7.1 | 4.7 | — | — |
| &nbsp;&nbsp;+ VHP-S, vertical only† | 35.2 | 44.9 | 59.4 | 8.3 | 12.5 | 8.7 | 5.8 | 7/7 | 0.9 |
| &nbsp;&nbsp;+ VHP-S, horizontal only | 41.4 | 52.8 | 64.3 | 5.9 | 9.6 | 6.2 | 2.8 | 3/7 | 0.2 |
| &nbsp;&nbsp;+ VHP-S, tuned for R@100 | 46.3 | 56.1 | 66.3 | 6.2 | 9.9 | 6.5 | 2.9 | 3/7 | 1.0 |
| &nbsp;&nbsp;+ VHP-S, tuned for MAP@100 | 33.1 | 41.9 | 57.9 | 7.8 | 11.2 | 8.1 | 5.1 | 4/7 | 1.6 |
| &nbsp;&nbsp;+ VHP-S, constrained | 46.5 | 56.4 | 65.4 | 7.2 | 11.0 | 7.5 | 3.6 | 5/7 | 1.0 |
| &nbsp;&nbsp;+ VHP-E, tuned for R@100 | 40.5 | 50.0 | 65.3 | 7.2 | 10.9 | 7.6 | 4.6 | 5/7 | 0.2 |
| &nbsp;&nbsp;+ VHP-E, tuned for MAP@100 | 36.3 | 46.5 | 61.5 | 8.1 | 11.1 | 8.4 | 5.3 | 6/7 | 0.2 |
| &nbsp;&nbsp;+ VHP-E then VHP-S | 42.5 | 56.2 | 68.9 | 5.9 | 9.4 | 6.4 | 3.3 | 3/7 | 1.2 |
| | | | | | | | | | |
| **mE5 (zero-shot)** | 49.1 | 57.0 | 68.6 | 14.1 | 24.5 | 14.4 | 10.6 | — | — |
| &nbsp;&nbsp;+ VHP-S, vertical only | 54.9 | 65.2 | 76.0 | 14.2 | 20.9 | 14.6 | 10.0 | 5/7 | 3.3 |
| &nbsp;&nbsp;+ VHP-S, horizontal only | 54.6 | 60.4 | 69.1 | 7.9 | 12.2 | 8.2 | 3.5 | 3/7 | 0.2 |
| &nbsp;&nbsp;+ VHP-S, tuned for R@100 | 59.0 | 66.8 | 78.1 | 13.2 | 17.8 | 13.6 | 8.9 | 3/7 | 3.3 |
| &nbsp;&nbsp;+ VHP-S, tuned for MAP@100 | 53.1 | 64.4 | 74.5 | 15.7 | 23.6 | 16.1 | 10.8 | 6/7 | 3.2 |
| &nbsp;&nbsp;+ VHP-S, constrained | 58.8 | 66.1 | 76.9 | 15.2 | 20.8 | 15.5 | 11.1 | 6/7 | 3.3 |
| &nbsp;&nbsp;+ VHP-E, tuned for R@100 | 54.4 | 61.9 | 70.8 | 15.2 | 24.4 | 15.6 | 11.1 | 6/7 | 0.8 |
| &nbsp;&nbsp;+ VHP-E, tuned for MAP@100 | 49.4 | 55.6 | 63.9 | 16.6 | 26.1 | 16.8 | 12.6 | 5/7 | 3.7 |
| &nbsp;&nbsp;+ VHP-E then VHP-S | 57.4 | 65.1 | 74.7 | 12.9 | 17.0 | 13.3 | 8.1 | 3/7 | 4.1 |
| | | | | | | | | | |
| **BGE-M3 (zero-shot)** | 59.2 | 68.7 | 78.5 | 18.3 | 30.5 | 18.6 | 15.1 | — | — |
| &nbsp;&nbsp;+ VHP-S, vertical only | 65.3 | 74.3 | 83.6 | 19.4 | 27.6 | 19.8 | 14.4 | 5/7 | 3.2 |
| &nbsp;&nbsp;+ VHP-S, horizontal only | 63.6 | 71.6 | 81.2 | 17.1 | 25.6 | 17.5 | 11.9 | 3/7 | 0.2 |
| &nbsp;&nbsp;+ VHP-S, tuned for R@100 | 66.9 | 74.9 | 85.2 | 21.0 | 29.1 | 21.4 | 15.5 | 6/7 | 3.1 |
| &nbsp;&nbsp;+ VHP-S, tuned for MAP@100† | 68.0 | 73.9 | 83.0 | 21.8 | 31.8 | 22.1 | 16.7 | 7/7 | 0.7 |
| &nbsp;&nbsp;+ VHP-S, constrained | 68.1 | 75.1 | 84.7 | 20.9 | 29.3 | 21.2 | 15.8 | 6/7 | 3.2 |
| &nbsp;&nbsp;+ VHP-E, tuned for R@100 | 65.8 | 72.5 | 81.2 | 19.7 | 31.1 | 20.0 | 14.0 | 6/7 | 1.3 |
| &nbsp;&nbsp;+ VHP-E, tuned for MAP@100 | 61.6 | 70.9 | 77.8 | 24.3 | 34.9 | 24.6 | 19.9 | 6/7 | 1.1 |
| &nbsp;&nbsp;+ VHP-E then VHP-S | 64.3 | 72.2 | 81.2 | 19.3 | 26.5 | 19.6 | 13.7 | 5/7 | 4.5 |
| | | | | | | | | | |
| **CamemBERT (zero-shot)** | 35.8 | 41.5 | 56.2 | 10.3 | 18.2 | 10.6 | 9.2 | — | — |
| &nbsp;&nbsp;+ VHP-S, vertical only† | 42.4 | 51.5 | 66.7 | 12.0 | 18.6 | 12.4 | 9.9 | 7/7 | 3.3 |
| &nbsp;&nbsp;+ VHP-S, horizontal only | 42.1 | 48.0 | 63.3 | 7.6 | 10.9 | 7.9 | 4.0 | 3/7 | 0.2 |
| &nbsp;&nbsp;+ VHP-S, tuned for R@100 | 48.4 | 57.4 | 71.5 | 10.3 | 13.6 | 10.7 | 6.4 | 4/7 | 3.2 |
| &nbsp;&nbsp;+ VHP-S, tuned for MAP@100 | 39.3 | 49.6 | 62.5 | 11.9 | 17.8 | 12.3 | 9.8 | 6/7 | 3.1 |
| &nbsp;&nbsp;+ VHP-S, constrained | 44.4 | 56.6 | 70.0 | 11.4 | 14.7 | 11.8 | 8.3 | 5/7 | 3.2 |
| &nbsp;&nbsp;+ VHP-E, tuned for R@100† | 44.5 | 51.7 | 64.8 | 13.3 | 20.6 | 13.6 | 10.0 | 7/7 | 0.9 |
| &nbsp;&nbsp;+ VHP-E, tuned for MAP@100† | 44.7 | 52.8 | 64.5 | 13.2 | 20.6 | 13.6 | 10.2 | 7/7 | 0.8 |
| &nbsp;&nbsp;+ VHP-E then VHP-S | 48.6 | 59.7 | 70.6 | 9.8 | 12.3 | 10.2 | 5.9 | 3/7 | 4.1 |
| | | | | | | | | | |
| **mE5 (fine-tuned)** | 86.3 | 88.8 | 90.9 | 54.6 | 60.7 | 54.7 | 48.3 | — | — |
| &nbsp;&nbsp;+ VHP-S, vertical only | 86.8 | 89.3 | 90.9 | 53.8 | 58.9 | 53.9 | 47.1 | 3/7 | 0.5 |
| &nbsp;&nbsp;+ VHP-S, horizontal only | 86.0 | 89.8 | 91.4 | 47.1 | 52.1 | 47.2 | 36.9 | 2/7 | 0.2 |
| &nbsp;&nbsp;+ VHP-S, tuned for R@100 | 86.8 | 90.0 | 91.6 | 45.2 | 49.8 | 45.4 | 34.2 | 3/7 | 0.4 |
| &nbsp;&nbsp;+ VHP-S, tuned for MAP@100† | 86.8 | 89.1 | 91.1 | 56.1 | 62.4 | 56.2 | 49.5 | 7/7 | 0.6 |
| &nbsp;&nbsp;+ VHP-S, constrained | 87.0 | 89.1 | 90.7 | 56.1 | 62.5 | 56.2 | 49.5 | 6/7 | 0.6 |
| &nbsp;&nbsp;+ VHP-E, tuned for R@100 | 86.6 | 89.3 | 91.4 | 52.7 | 59.5 | 52.8 | 44.0 | 3/7 | 3.7 |
| &nbsp;&nbsp;+ VHP-E, tuned for MAP@100 | 86.6 | 88.8 | 90.7 | 55.4 | 61.8 | 55.5 | 48.2 | 4/7 | 3.5 |
| &nbsp;&nbsp;+ VHP-E then VHP-S | 86.9 | 90.3 | 92.7 | 40.4 | 45.1 | 40.5 | 27.2 | 3/7 | 4.3 |
| | | | | | | | | | |
| **BGE-M3 (fine-tuned)** | 86.7 | 90.5 | 93.2 | 50.5 | 57.6 | 50.6 | 42.2 | — | — |
| &nbsp;&nbsp;+ VHP-S, vertical only† | 88.2 | 91.3 | 93.2 | 50.8 | 58.0 | 50.9 | 42.3 | 7/7 | 3.3 |
| &nbsp;&nbsp;+ VHP-S, horizontal only | 87.1 | 90.7 | 93.2 | 50.0 | 56.4 | 50.1 | 42.7 | 3/7 | 0.2 |
| &nbsp;&nbsp;+ VHP-S, tuned for R@100 | 89.0 | 91.3 | 93.7 | 50.9 | 56.7 | 51.0 | 43.3 | 6/7 | 3.3 |
| &nbsp;&nbsp;+ VHP-S, tuned for MAP@100† | 87.9 | 91.1 | 93.5 | 51.8 | 58.8 | 51.9 | 43.9 | 7/7 | 3.2 |
| &nbsp;&nbsp;+ VHP-S, constrained | 88.3 | 91.0 | 93.7 | 51.0 | 58.1 | 51.1 | 41.9 | 6/7 | 3.1 |
| &nbsp;&nbsp;+ VHP-E, tuned for R@100 | 88.7 | 91.5 | 92.9 | 51.1 | 57.8 | 51.2 | 42.4 | 6/7 | 4.7 |
| &nbsp;&nbsp;+ VHP-E, tuned for MAP@100 | 88.2 | 91.1 | 93.1 | 52.1 | 59.3 | 52.1 | 43.3 | 6/7 | 4.7 |
| &nbsp;&nbsp;+ VHP-E then VHP-S | 88.3 | 91.3 | 94.0 | 49.1 | 55.0 | 49.2 | 42.7 | 4/7 | 8.1 |
| | | | | | | | | | |
| **BM25 + BGE-M3 ft (fusion)** | 72.7 | 78.4 | 89.8 | 31.4 | 42.1 | 31.7 | 26.3 | — | — |
| &nbsp;&nbsp;+ R@100 | 77.5 | 84.2 | 92.1 | 32.1 | 39.6 | 32.3 | 26.1 | 0/7 | — |
| &nbsp;&nbsp;+ MAP@100 | 78.6 | 84.3 | 91.5 | 32.4 | 39.4 | 32.6 | 25.8 | 0/7 | — |

## Complete Ablations

**All ablations on BM25, test split, with every metric**

| Variant | R@100 | R@200 | R@500 | MAP@100 | MRR@100 | MAP | MRP |
|:---|---:|---:|---:|---:|---:|---:|---:|
| BM25 (baseline) | 51.5 | 57.2 | 66.1 | 17.5 | 24.6 | 17.8 | 13.6 |
| Full VHP-S | 59.5 | 65.3 | 73.8 | 18.1 | 25.4 | 18.4 | 12.4 |
| No leave-one-out | 59.6 | 68.0 | 74.1 | 18.4 | 24.9 | 18.7 | 13.2 |
| No vertical axis (beta=0) | 54.6 | 60.7 | 67.9 | 15.9 | 22.5 | 16.2 | 10.1 |
| No horizontal axis (gamma=0) | 56.7 | 63.4 | 74.3 | 18.2 | 24.2 | 18.5 | 14.0 |
| Wider aggregation (m=5) | 57.8 | 64.6 | 74.1 | 18.5 | 24.4 | 18.8 | 13.4 |
| Undirected references | 58.9 | 65.4 | 73.5 | 16.7 | 22.9 | 17.0 | 11.1 |
| Reversed references | 57.0 | 64.4 | 72.6 | 17.9 | 24.4 | 18.2 | 13.1 |
| Two propagation steps | 59.2 | 65.6 | 73.5 | 18.5 | 25.2 | 18.8 | 13.5 |
| Finest level only | 54.9 | 62.4 | 72.4 | 16.8 | 23.9 | 17.1 | 11.8 |
| Uniform level weights | 59.2 | 65.0 | 74.6 | 18.1 | 25.1 | 18.4 | 12.3 |
| Ambiguous references discarded | 59.6 | 65.4 | 73.8 | 17.9 | 25.5 | 18.2 | 11.9 |
| Ambiguous references kept | 59.3 | 65.6 | 73.4 | 18.6 | 26.4 | 18.9 | 13.6 |

## Curves Behind the Figures

Both are computed on the training split with BM25.

**Vertical signal applied at one level at a time**

| Level | Blocks | R@100 |
|:---|---:|---:|
| (none) | 0 | 47.2 |
| law_type | 2 | 46.3 |
| code | 34 | 48.6 |
| book | 161 | 49.8 |
| part | 182 | 49.8 |
| act | 864 | 51.1 |
| chapter | 2670 | 52.2 |
| section | 4658 | 51.0 |
| subsection | 5626 | 51.3 |

**Effect of the strength of the vertical signal**

| beta | R@100 | R@500 | MAP@100 | MRR@100 | MRP |
|:---|---:|---:|---:|---:|---:|
| 0.0 | 50.2 | 64.2 | 16.8 | 25.5 | 12.6 |
| 0.1 | 51.5 | 64.8 | 17.4 | 25.9 | 13.0 |
| 0.2 | 52.5 | 66.1 | 17.9 | 26.5 | 13.4 |
| 0.3 | 53.6 | 67.0 | 18.4 | 26.9 | 14.1 |
| 0.5 | 55.0 | 68.5 | 19.0 | 27.3 | 14.6 |
| 0.8 | 55.7 | 70.1 | 19.7 | 27.5 | 15.0 |
| 1.2 | 56.9 | 70.6 | 19.7 | 26.9 | 15.0 |
| 1.6 | 55.9 | 71.5 | 19.5 | 26.6 | 14.7 |
| 2.0 | 55.5 | 70.3 | 19.2 | 26.0 | 14.8 |

## Gains by Size of the Answer Set

Change in R@100 by number of relevant articles per question, with the number of test questions in each stratum in brackets.

**Stratified R@100 gains on the test split**

| First stage | Configuration | 1 (83) | 2-5 (96) | 6-20 (34) | 21+ (9) |
|:---|:---|---:|---:|---:|---:|
| BM25 | VHP-S, vertical only | +0.0 | +6.4 | +11.3 | +15.9 |
| BM25 | VHP-S, horizontal only | +6.0 | +1.5 | +1.2 | -0.5 |
| BM25 | VHP-S, tuned for R@100 | +3.6 | +9.7 | +11.8 | +14.9 |
| BM25 | VHP-S, tuned for MAP@100 | +2.4 | +6.9 | +13.7 | +11.5 |
| BM25 | VHP-S, constrained | +2.4 | +8.5 | +11.7 | +9.4 |
| TF-IDF-SVD | VHP-S, vertical only | +2.4 | +3.1 | +1.8 | +1.3 |
| TF-IDF-SVD | VHP-S, horizontal only | +16.9 | +5.0 | +2.1 | +0.4 |
| TF-IDF-SVD | VHP-S, tuned for R@100 | +22.9 | +10.5 | +2.8 | +3.9 |
| TF-IDF-SVD | VHP-S, tuned for MAP@100 | -6.0 | +3.1 | +7.9 | +4.6 |
| TF-IDF-SVD | VHP-S, constrained | +22.9 | +10.8 | +3.0 | +4.6 |
| TF-IDF-SVD | VHP-E, tuned for R@100 | +12.0 | +5.6 | +3.9 | +9.0 |
| TF-IDF-SVD | VHP-E, tuned for MAP@100 | +2.4 | +4.7 | +4.2 | +2.4 |
| TF-IDF-SVD | VHP-E then VHP-S | +15.7 | +7.1 | +3.7 | +8.2 |
| mE5 (zero-shot) | VHP-S, vertical only | -6.0 | +13.0 | +11.4 | +15.5 |
| mE5 (zero-shot) | VHP-S, horizontal only | +7.2 | +5.8 | +1.6 | -0.5 |
| mE5 (zero-shot) | VHP-S, tuned for R@100 | +3.6 | +14.1 | +11.3 | +16.2 |
| mE5 (zero-shot) | VHP-S, tuned for MAP@100 | -6.0 | +9.6 | +11.6 | +6.6 |
| mE5 (zero-shot) | VHP-S, constrained | +3.6 | +13.5 | +11.5 | +17.7 |
| mE5 (zero-shot) | VHP-E, tuned for R@100 | +2.4 | +6.6 | +9.3 | +1.2 |
| mE5 (zero-shot) | VHP-E, tuned for MAP@100 | -9.6 | +5.3 | +9.0 | +3.0 |
| mE5 (zero-shot) | VHP-E then VHP-S | -1.2 | +13.3 | +14.6 | +17.7 |
| BGE-M3 (zero-shot) | VHP-S, vertical only | +2.4 | +7.3 | +9.7 | +14.7 |
| BGE-M3 (zero-shot) | VHP-S, horizontal only | +7.2 | +3.6 | +0.8 | -0.3 |
| BGE-M3 (zero-shot) | VHP-S, tuned for R@100 | +4.8 | +8.0 | +12.1 | +14.2 |
| BGE-M3 (zero-shot) | VHP-S, tuned for MAP@100 | +7.2 | +10.5 | +8.0 | +7.6 |
| BGE-M3 (zero-shot) | VHP-S, constrained | +7.2 | +8.4 | +12.2 | +15.7 |
| BGE-M3 (zero-shot) | VHP-E, tuned for R@100 | +6.0 | +7.2 | +6.5 | +5.2 |
| BGE-M3 (zero-shot) | VHP-E, tuned for MAP@100 | -4.8 | +5.1 | +10.9 | +8.4 |
| BGE-M3 (zero-shot) | VHP-E then VHP-S | +0.0 | +5.4 | +14.4 | +13.9 |
| CamemBERT (zero-shot) | VHP-S, vertical only | +1.2 | +8.8 | +11.5 | +12.9 |
| CamemBERT (zero-shot) | VHP-S, horizontal only | +9.6 | +4.5 | +4.9 | -0.1 |
| CamemBERT (zero-shot) | VHP-S, tuned for R@100 | +13.3 | +12.0 | +12.5 | +13.7 |
| CamemBERT (zero-shot) | VHP-S, tuned for MAP@100 | -4.8 | +8.1 | +9.4 | +9.4 |
| CamemBERT (zero-shot) | VHP-S, constrained | +2.4 | +11.9 | +12.5 | +15.2 |
| CamemBERT (zero-shot) | VHP-E, tuned for R@100 | +2.4 | +11.8 | +15.3 | +8.7 |
| CamemBERT (zero-shot) | VHP-E, tuned for MAP@100 | +3.6 | +11.3 | +15.0 | +9.0 |
| CamemBERT (zero-shot) | VHP-E then VHP-S | +6.0 | +16.1 | +19.0 | +16.1 |
| mE5 (fine-tuned) | VHP-S, vertical only | +1.2 | +0.3 | -0.6 | +0.0 |
| mE5 (fine-tuned) | VHP-S, horizontal only | -1.2 | +0.3 | +0.3 | -1.6 |
| mE5 (fine-tuned) | VHP-S, tuned for R@100 | +1.2 | +0.5 | -0.9 | -1.0 |
| mE5 (fine-tuned) | VHP-S, tuned for MAP@100 | +0.0 | +0.3 | +2.1 | -0.6 |
| mE5 (fine-tuned) | VHP-S, constrained | +0.0 | +0.9 | +2.3 | -0.6 |
| mE5 (fine-tuned) | VHP-E, tuned for R@100 | -1.2 | +0.9 | +2.3 | -0.2 |
| mE5 (fine-tuned) | VHP-E, tuned for MAP@100 | +0.0 | +0.3 | +0.5 | +0.0 |
| mE5 (fine-tuned) | VHP-E then VHP-S | -1.2 | +1.8 | +2.0 | -1.6 |
| BGE-M3 (fine-tuned) | VHP-S, vertical only | +1.2 | +1.8 | +2.4 | -0.5 |
| BGE-M3 (fine-tuned) | VHP-S, horizontal only | +0.0 | +1.0 | -0.1 | +0.0 |
| BGE-M3 (fine-tuned) | VHP-S, tuned for R@100 | +2.4 | +2.2 | +3.5 | -0.5 |
| BGE-M3 (fine-tuned) | VHP-S, tuned for MAP@100 | +1.2 | +1.4 | +1.0 | +0.0 |
| BGE-M3 (fine-tuned) | VHP-S, constrained | +1.2 | +2.0 | +1.9 | +0.0 |
| BGE-M3 (fine-tuned) | VHP-E, tuned for R@100 | +2.4 | +2.3 | +0.7 | +0.0 |
| BGE-M3 (fine-tuned) | VHP-E, tuned for MAP@100 | +1.2 | +2.3 | +0.7 | +0.0 |
| BGE-M3 (fine-tuned) | VHP-E then VHP-S | +1.2 | +1.8 | +2.8 | -0.5 |

## Significance Tests

Paired bootstrap over 10,000 resamples for the confidence interval, paired permutation test for the p-value, on the 222 test questions. * p<0.05, ** p<0.01, *** p<0.001.

**Significance against the BM25 baseline**

| Metric | Delta (pts) | 95% CI | p | Better | Worse |
|:---|---:|---:|---:|---:|---:|
| **VHP-S, vertical only** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +5.15 | [+1.41, +8.86] | 0.0059** | 48 | 20 |
| &nbsp;&nbsp;R@200 | +6.12 | [+2.41, +9.89] | 0.0021** | 48 | 14 |
| &nbsp;&nbsp;R@500 | +8.25 | [+5.09, +11.53] | 0.0001*** | 60 | 8 |
| &nbsp;&nbsp;MAP@100 | +0.66 | [-1.98, +3.16] | 0.6192 | 90 | 60 |
| &nbsp;&nbsp;MRR@100 | -0.42 | [-3.64, +2.85] | 0.8047 | 69 | 70 |
| &nbsp;&nbsp;MAP | +0.72 | [-1.92, +3.21] | 0.5782 | 135 | 76 |
| &nbsp;&nbsp;MRP | +0.35 | [-3.17, +3.82] | 0.8428 | 28 | 16 |
| **VHP-S, horizontal only** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +3.09 | [+0.92, +5.48] | 0.0077** | 25 | 13 |
| &nbsp;&nbsp;R@200 | +3.43 | [+1.52, +5.60] | 0.0004*** | 24 | 5 |
| &nbsp;&nbsp;R@500 | +1.85 | [-0.29, +4.02] | 0.1033 | 22 | 4 |
| &nbsp;&nbsp;MAP@100 | -1.61 | [-4.08, +0.83] | 0.2130 | 65 | 85 |
| &nbsp;&nbsp;MRR@100 | -2.18 | [-5.57, +1.21] | 0.2160 | 52 | 84 |
| &nbsp;&nbsp;MAP | -1.59 | [-4.06, +0.83] | 0.2169 | 88 | 124 |
| &nbsp;&nbsp;MRP | -3.48 | [-6.94, -0.21] | 0.0463* | 17 | 26 |
| **VHP-S, tuned for R@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +7.98 | [+4.15, +11.81] | 0.0002*** | 57 | 17 |
| &nbsp;&nbsp;R@200 | +8.11 | [+4.40, +11.79] | 0.0001*** | 57 | 13 |
| &nbsp;&nbsp;R@500 | +7.76 | [+4.16, +11.45] | 0.0001*** | 63 | 11 |
| &nbsp;&nbsp;MAP@100 | +0.57 | [-2.30, +3.25] | 0.6867 | 96 | 58 |
| &nbsp;&nbsp;MRR@100 | +0.77 | [-3.04, +4.60] | 0.7001 | 72 | 63 |
| &nbsp;&nbsp;MAP | +0.63 | [-2.24, +3.32] | 0.6581 | 135 | 78 |
| &nbsp;&nbsp;MRP | -1.16 | [-4.89, +2.41] | 0.5452 | 29 | 21 |
| **VHP-S, tuned for MAP@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +6.44 | [+2.49, +10.40] | 0.0010*** | 55 | 20 |
| &nbsp;&nbsp;R@200 | +6.52 | [+2.81, +10.36] | 0.0006*** | 57 | 18 |
| &nbsp;&nbsp;R@500 | +9.65 | [+6.44, +12.91] | 0.0001*** | 65 | 8 |
| &nbsp;&nbsp;MAP@100 | +2.10 | [-0.45, +4.54] | 0.1023 | 100 | 56 |
| &nbsp;&nbsp;MRR@100 | +0.18 | [-2.91, +3.32] | 0.9154 | 71 | 68 |
| &nbsp;&nbsp;MAP | +2.14 | [-0.41, +4.57] | 0.0947 | 140 | 70 |
| &nbsp;&nbsp;MRP | +0.86 | [-2.64, +4.24] | 0.6391 | 34 | 20 |
| **VHP-S, constrained** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +6.74 | [+3.04, +10.41] | 0.0001*** | 54 | 18 |
| &nbsp;&nbsp;R@200 | +8.34 | [+4.75, +12.00] | 0.0001*** | 58 | 10 |
| &nbsp;&nbsp;R@500 | +9.35 | [+6.19, +12.58] | 0.0001*** | 65 | 6 |
| &nbsp;&nbsp;MAP@100 | +0.61 | [-2.17, +3.27] | 0.6611 | 88 | 67 |
| &nbsp;&nbsp;MRR@100 | -0.89 | [-4.56, +2.83] | 0.6404 | 68 | 72 |
| &nbsp;&nbsp;MAP | +0.67 | [-2.10, +3.32] | 0.6243 | 128 | 84 |
| &nbsp;&nbsp;MRP | -0.72 | [-4.11, +2.50] | 0.6744 | 30 | 20 |

**Significance against the TF-IDF-SVD baseline**

| Metric | Delta (pts) | 95% CI | p | Better | Worse |
|:---|---:|---:|---:|---:|---:|
| **VHP-S, vertical only** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +2.57 | [+0.30, +5.04] | 0.0359* | 24 | 9 |
| &nbsp;&nbsp;R@200 | +1.22 | [-2.29, +4.69] | 0.5020 | 34 | 17 |
| &nbsp;&nbsp;R@500 | +1.30 | [-1.96, +4.39] | 0.4299 | 38 | 15 |
| &nbsp;&nbsp;MAP@100 | +1.56 | [+0.28, +2.85] | 0.0183* | 72 | 35 |
| &nbsp;&nbsp;MRR@100 | +1.05 | [-0.66, +2.80] | 0.2367 | 59 | 37 |
| &nbsp;&nbsp;MAP | +1.59 | [+0.32, +2.88] | 0.0147* | 130 | 88 |
| &nbsp;&nbsp;MRP | +1.11 | [-0.52, +2.85] | 0.2203 | 14 | 6 |
| **VHP-S, horizontal only** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +8.80 | [+3.70, +14.01] | 0.0007*** | 55 | 26 |
| &nbsp;&nbsp;R@200 | +9.13 | [+5.18, +13.28] | 0.0001*** | 54 | 14 |
| &nbsp;&nbsp;R@500 | +6.15 | [+2.79, +9.67] | 0.0001*** | 46 | 14 |
| &nbsp;&nbsp;MAP@100 | -0.88 | [-3.31, +1.47] | 0.4739 | 73 | 67 |
| &nbsp;&nbsp;MRR@100 | -1.83 | [-5.33, +1.70] | 0.3088 | 74 | 65 |
| &nbsp;&nbsp;MAP | -0.85 | [-3.27, +1.49] | 0.4883 | 114 | 107 |
| &nbsp;&nbsp;MRP | -1.92 | [-4.23, +0.24] | 0.1009 | 12 | 19 |
| **VHP-S, tuned for R@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +13.69 | [+9.51, +18.10] | 0.0001*** | 61 | 10 |
| &nbsp;&nbsp;R@200 | +12.47 | [+8.23, +16.97] | 0.0001*** | 70 | 10 |
| &nbsp;&nbsp;R@500 | +8.19 | [+4.72, +11.93] | 0.0001*** | 58 | 10 |
| &nbsp;&nbsp;MAP@100 | -0.57 | [-2.78, +1.44] | 0.5994 | 82 | 56 |
| &nbsp;&nbsp;MRR@100 | -1.48 | [-4.85, +1.83] | 0.3908 | 79 | 58 |
| &nbsp;&nbsp;MAP | -0.55 | [-2.75, +1.46] | 0.6112 | 141 | 80 |
| &nbsp;&nbsp;MRP | -1.78 | [-3.98, +0.14] | 0.1011 | 16 | 15 |
| **VHP-S, tuned for MAP@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +0.47 | [-3.10, +4.02] | 0.8002 | 38 | 26 |
| &nbsp;&nbsp;R@200 | -1.75 | [-6.50, +2.86] | 0.4623 | 46 | 32 |
| &nbsp;&nbsp;R@500 | -0.17 | [-4.80, +4.34] | 0.9438 | 43 | 31 |
| &nbsp;&nbsp;MAP@100 | +1.05 | [-0.86, +2.87] | 0.2737 | 72 | 46 |
| &nbsp;&nbsp;MRR@100 | -0.21 | [-2.62, +2.09] | 0.8624 | 60 | 49 |
| &nbsp;&nbsp;MAP | +1.07 | [-0.83, +2.90] | 0.2654 | 119 | 102 |
| &nbsp;&nbsp;MRP | +0.36 | [-1.59, +2.19] | 0.7164 | 17 | 10 |
| **VHP-S, constrained** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +13.86 | [+9.91, +18.06] | 0.0001*** | 59 | 5 |
| &nbsp;&nbsp;R@200 | +12.75 | [+8.55, +17.22] | 0.0001*** | 73 | 10 |
| &nbsp;&nbsp;R@500 | +7.32 | [+4.04, +10.88] | 0.0001*** | 56 | 10 |
| &nbsp;&nbsp;MAP@100 | +0.43 | [-1.83, +2.55] | 0.7061 | 83 | 51 |
| &nbsp;&nbsp;MRR@100 | -0.40 | [-3.80, +2.90] | 0.8183 | 79 | 53 |
| &nbsp;&nbsp;MAP | +0.44 | [-1.82, +2.55] | 0.7006 | 140 | 81 |
| &nbsp;&nbsp;MRP | -1.17 | [-3.51, +1.01] | 0.3250 | 20 | 14 |
| **VHP-E, tuned for R@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +7.88 | [+4.81, +11.27] | 0.0001*** | 38 | 4 |
| &nbsp;&nbsp;R@200 | +6.36 | [+2.77, +10.07] | 0.0004*** | 44 | 7 |
| &nbsp;&nbsp;R@500 | +7.16 | [+4.34, +10.17] | 0.0001*** | 51 | 6 |
| &nbsp;&nbsp;MAP@100 | +0.51 | [-1.26, +2.19] | 0.5702 | 81 | 35 |
| &nbsp;&nbsp;MRR@100 | -0.49 | [-2.57, +1.46] | 0.6390 | 68 | 36 |
| &nbsp;&nbsp;MAP | +0.53 | [-1.26, +2.20] | 0.5603 | 159 | 60 |
| &nbsp;&nbsp;MRP | -0.07 | [-2.19, +1.97] | 0.9443 | 16 | 11 |
| **VHP-E, tuned for MAP@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +3.69 | [+1.32, +6.29] | 0.0035** | 30 | 6 |
| &nbsp;&nbsp;R@200 | +2.79 | [-0.33, +5.98] | 0.0869 | 39 | 13 |
| &nbsp;&nbsp;R@500 | +3.36 | [-0.12, +6.83] | 0.0574 | 42 | 12 |
| &nbsp;&nbsp;MAP@100 | +1.33 | [-0.47, +3.14] | 0.1466 | 71 | 38 |
| &nbsp;&nbsp;MRR@100 | -0.28 | [-2.39, +1.77] | 0.7939 | 55 | 45 |
| &nbsp;&nbsp;MAP | +1.37 | [-0.43, +3.17] | 0.1354 | 142 | 76 |
| &nbsp;&nbsp;MRP | +0.55 | [-1.39, +2.34] | 0.5758 | 12 | 6 |
| **VHP-E then VHP-S** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +9.82 | [+4.93, +14.94] | 0.0005*** | 62 | 22 |
| &nbsp;&nbsp;R@200 | +12.55 | [+8.32, +16.93] | 0.0001*** | 74 | 12 |
| &nbsp;&nbsp;R@500 | +10.75 | [+6.90, +14.85] | 0.0001*** | 64 | 10 |
| &nbsp;&nbsp;MAP@100 | -0.79 | [-3.07, +1.26] | 0.4844 | 82 | 57 |
| &nbsp;&nbsp;MRR@100 | -2.06 | [-5.49, +1.28] | 0.2342 | 77 | 61 |
| &nbsp;&nbsp;MAP | -0.70 | [-2.97, +1.34] | 0.5280 | 142 | 80 |
| &nbsp;&nbsp;MRP | -1.47 | [-3.71, +0.53] | 0.1880 | 17 | 14 |

**Significance against the mE5 (zero-shot) baseline**

| Metric | Delta (pts) | 95% CI | p | Better | Worse |
|:---|---:|---:|---:|---:|---:|
| **VHP-S, vertical only** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +5.74 | [+0.95, +10.30] | 0.0181* | 68 | 28 |
| &nbsp;&nbsp;R@200 | +8.21 | [+3.96, +12.54] | 0.0004*** | 69 | 25 |
| &nbsp;&nbsp;R@500 | +7.43 | [+3.83, +11.05] | 0.0001*** | 54 | 10 |
| &nbsp;&nbsp;MAP@100 | +0.13 | [-1.89, +1.98] | 0.8969 | 85 | 73 |
| &nbsp;&nbsp;MRR@100 | -3.59 | [-6.42, -0.89] | 0.0130* | 59 | 74 |
| &nbsp;&nbsp;MAP | +0.21 | [-1.80, +2.05] | 0.8364 | 116 | 95 |
| &nbsp;&nbsp;MRP | -0.57 | [-2.86, +1.56] | 0.6224 | 20 | 18 |
| **VHP-S, horizontal only** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +5.44 | [+2.63, +8.46] | 0.0002*** | 34 | 15 |
| &nbsp;&nbsp;R@200 | +3.46 | [+1.56, +5.66] | 0.0005*** | 23 | 11 |
| &nbsp;&nbsp;R@500 | +0.50 | [-1.85, +2.70] | 0.6760 | 22 | 13 |
| &nbsp;&nbsp;MAP@100 | -6.15 | [-8.71, -3.82] | 0.0001*** | 65 | 104 |
| &nbsp;&nbsp;MRR@100 | -12.29 | [-16.35, -8.39] | 0.0001*** | 62 | 104 |
| &nbsp;&nbsp;MAP | -6.17 | [-8.73, -3.84] | 0.0001*** | 96 | 125 |
| &nbsp;&nbsp;MRP | -7.03 | [-10.24, -4.14] | 0.0001*** | 13 | 36 |
| **VHP-S, tuned for R@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +9.83 | [+5.45, +14.16] | 0.0001*** | 70 | 22 |
| &nbsp;&nbsp;R@200 | +9.80 | [+5.48, +14.21] | 0.0001*** | 72 | 18 |
| &nbsp;&nbsp;R@500 | +9.53 | [+6.29, +12.87] | 0.0001*** | 59 | 5 |
| &nbsp;&nbsp;MAP@100 | -0.81 | [-3.56, +1.77] | 0.5592 | 91 | 74 |
| &nbsp;&nbsp;MRR@100 | -6.73 | [-10.66, -2.93] | 0.0006*** | 69 | 84 |
| &nbsp;&nbsp;MAP | -0.75 | [-3.49, +1.83] | 0.5897 | 122 | 95 |
| &nbsp;&nbsp;MRP | -1.62 | [-4.95, +1.52] | 0.3501 | 27 | 25 |
| **VHP-S, tuned for MAP@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +3.92 | [+0.03, +7.65] | 0.0453* | 54 | 16 |
| &nbsp;&nbsp;R@200 | +7.42 | [+3.95, +10.91] | 0.0002*** | 60 | 9 |
| &nbsp;&nbsp;R@500 | +5.92 | [+2.79, +9.00] | 0.0004*** | 52 | 6 |
| &nbsp;&nbsp;MAP@100 | +1.63 | [+0.20, +3.00] | 0.0195* | 98 | 51 |
| &nbsp;&nbsp;MRR@100 | -0.91 | [-2.99, +1.11] | 0.3811 | 68 | 53 |
| &nbsp;&nbsp;MAP | +1.69 | [+0.27, +3.05] | 0.0150* | 143 | 67 |
| &nbsp;&nbsp;MRP | +0.20 | [-1.65, +1.84] | 0.8228 | 21 | 9 |
| **VHP-S, constrained** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +9.66 | [+5.24, +13.97] | 0.0001*** | 68 | 21 |
| &nbsp;&nbsp;R@200 | +9.13 | [+4.99, +13.33] | 0.0001*** | 69 | 19 |
| &nbsp;&nbsp;R@500 | +8.29 | [+4.79, +11.86] | 0.0001*** | 56 | 8 |
| &nbsp;&nbsp;MAP@100 | +1.13 | [-1.40, +3.49] | 0.3748 | 98 | 66 |
| &nbsp;&nbsp;MRR@100 | -3.72 | [-7.08, -0.57] | 0.0261* | 69 | 73 |
| &nbsp;&nbsp;MAP | +1.17 | [-1.34, +3.50] | 0.3559 | 130 | 86 |
| &nbsp;&nbsp;MRP | +0.51 | [-2.53, +3.49] | 0.7453 | 27 | 18 |
| **VHP-E, tuned for R@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +5.23 | [+1.90, +8.69] | 0.0025** | 44 | 14 |
| &nbsp;&nbsp;R@200 | +4.89 | [+1.68, +8.13] | 0.0025** | 45 | 14 |
| &nbsp;&nbsp;R@500 | +2.18 | [-1.19, +5.58] | 0.1980 | 42 | 20 |
| &nbsp;&nbsp;MAP@100 | +1.17 | [-0.04, +2.42] | 0.0614 | 94 | 56 |
| &nbsp;&nbsp;MRR@100 | -0.07 | [-1.96, +1.73] | 0.9456 | 65 | 58 |
| &nbsp;&nbsp;MAP | +1.18 | [-0.02, +2.43] | 0.0597 | 132 | 76 |
| &nbsp;&nbsp;MRP | +0.51 | [-1.03, +2.04] | 0.5227 | 16 | 7 |
| **VHP-E, tuned for MAP@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +0.21 | [-4.54, +4.72] | 0.9311 | 54 | 32 |
| &nbsp;&nbsp;R@200 | -1.35 | [-6.17, +3.28] | 0.5759 | 49 | 39 |
| &nbsp;&nbsp;R@500 | -4.69 | [-9.74, +0.16] | 0.0627 | 37 | 42 |
| &nbsp;&nbsp;MAP@100 | +2.50 | [+0.50, +4.49] | 0.0123* | 106 | 48 |
| &nbsp;&nbsp;MRR@100 | +1.65 | [-1.29, +4.51] | 0.2716 | 86 | 45 |
| &nbsp;&nbsp;MAP | +2.45 | [+0.46, +4.44] | 0.0136* | 138 | 69 |
| &nbsp;&nbsp;MRP | +2.07 | [-0.00, +4.16] | 0.0503 | 29 | 9 |
| **VHP-E then VHP-S** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +8.25 | [+3.11, +13.25] | 0.0018** | 75 | 28 |
| &nbsp;&nbsp;R@200 | +8.10 | [+3.31, +12.88] | 0.0011** | 71 | 28 |
| &nbsp;&nbsp;R@500 | +6.08 | [+2.02, +10.15] | 0.0040** | 55 | 20 |
| &nbsp;&nbsp;MAP@100 | -1.14 | [-3.84, +1.38] | 0.3873 | 89 | 80 |
| &nbsp;&nbsp;MRR@100 | -7.47 | [-11.52, -3.60] | 0.0002*** | 70 | 88 |
| &nbsp;&nbsp;MAP | -1.11 | [-3.81, +1.39] | 0.3958 | 112 | 105 |
| &nbsp;&nbsp;MRP | -2.50 | [-5.66, +0.46] | 0.1134 | 27 | 26 |

**Significance against the BGE-M3 (zero-shot) baseline**

| Metric | Delta (pts) | 95% CI | p | Better | Worse |
|:---|---:|---:|---:|---:|---:|
| **VHP-S, vertical only** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +6.13 | [+2.38, +9.90] | 0.0011** | 54 | 13 |
| &nbsp;&nbsp;R@200 | +5.61 | [+2.40, +8.88] | 0.0009*** | 47 | 14 |
| &nbsp;&nbsp;R@500 | +5.10 | [+2.47, +7.80] | 0.0002*** | 50 | 7 |
| &nbsp;&nbsp;MAP@100 | +1.13 | [-0.86, +3.01] | 0.2635 | 100 | 74 |
| &nbsp;&nbsp;MRR@100 | -2.84 | [-5.69, -0.19] | 0.0431* | 68 | 79 |
| &nbsp;&nbsp;MAP | +1.19 | [-0.80, +3.07] | 0.2391 | 121 | 87 |
| &nbsp;&nbsp;MRP | -0.76 | [-3.25, +1.67] | 0.5539 | 23 | 20 |
| **VHP-S, horizontal only** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +4.38 | [+2.05, +7.07] | 0.0003*** | 26 | 9 |
| &nbsp;&nbsp;R@200 | +2.89 | [+0.52, +5.41] | 0.0210* | 23 | 11 |
| &nbsp;&nbsp;R@500 | +2.66 | [+1.11, +4.56] | 0.0001*** | 22 | 4 |
| &nbsp;&nbsp;MAP@100 | -1.13 | [-4.04, +1.82] | 0.4565 | 57 | 122 |
| &nbsp;&nbsp;MRR@100 | -4.92 | [-9.11, -0.67] | 0.0246* | 55 | 116 |
| &nbsp;&nbsp;MAP | -1.13 | [-4.04, +1.81] | 0.4564 | 71 | 141 |
| &nbsp;&nbsp;MRP | -3.27 | [-6.90, +0.29] | 0.0744 | 19 | 39 |
| **VHP-S, tuned for R@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +7.69 | [+3.94, +11.48] | 0.0001*** | 62 | 11 |
| &nbsp;&nbsp;R@200 | +6.23 | [+3.00, +9.56] | 0.0006*** | 48 | 13 |
| &nbsp;&nbsp;R@500 | +6.70 | [+4.22, +9.38] | 0.0001*** | 54 | 4 |
| &nbsp;&nbsp;MAP@100 | +2.69 | [-0.33, +5.73] | 0.0859 | 100 | 80 |
| &nbsp;&nbsp;MRR@100 | -1.36 | [-5.59, +3.10] | 0.5554 | 71 | 89 |
| &nbsp;&nbsp;MAP | +2.76 | [-0.26, +5.83] | 0.0773 | 116 | 96 |
| &nbsp;&nbsp;MRP | +0.35 | [-3.63, +4.29] | 0.8630 | 37 | 30 |
| **VHP-S, tuned for MAP@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +8.79 | [+5.40, +12.33] | 0.0001*** | 60 | 6 |
| &nbsp;&nbsp;R@200 | +5.20 | [+2.08, +8.44] | 0.0011** | 44 | 9 |
| &nbsp;&nbsp;R@500 | +4.53 | [+2.46, +6.85] | 0.0001*** | 44 | 6 |
| &nbsp;&nbsp;MAP@100 | +3.49 | [+1.22, +5.77] | 0.0026** | 114 | 59 |
| &nbsp;&nbsp;MRR@100 | +1.33 | [-1.85, +4.61] | 0.4178 | 72 | 69 |
| &nbsp;&nbsp;MAP | +3.47 | [+1.20, +5.76] | 0.0025** | 134 | 73 |
| &nbsp;&nbsp;MRP | +1.57 | [-1.60, +4.58] | 0.3199 | 31 | 15 |
| **VHP-S, constrained** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +8.84 | [+5.13, +12.65] | 0.0001*** | 63 | 10 |
| &nbsp;&nbsp;R@200 | +6.36 | [+3.16, +9.67] | 0.0004*** | 48 | 13 |
| &nbsp;&nbsp;R@500 | +6.15 | [+3.49, +8.98] | 0.0001*** | 54 | 6 |
| &nbsp;&nbsp;MAP@100 | +2.58 | [-0.12, +5.36] | 0.0642 | 103 | 79 |
| &nbsp;&nbsp;MRR@100 | -1.17 | [-5.19, +3.00] | 0.5937 | 75 | 86 |
| &nbsp;&nbsp;MAP | +2.62 | [-0.07, +5.41] | 0.0596 | 119 | 94 |
| &nbsp;&nbsp;MRP | +0.65 | [-3.05, +4.24] | 0.7328 | 35 | 24 |
| **VHP-E, tuned for R@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +6.55 | [+2.96, +10.23] | 0.0004*** | 50 | 11 |
| &nbsp;&nbsp;R@200 | +3.80 | [+0.44, +7.19] | 0.0239* | 42 | 14 |
| &nbsp;&nbsp;R@500 | +2.68 | [-0.28, +5.67] | 0.0814 | 42 | 13 |
| &nbsp;&nbsp;MAP@100 | +1.42 | [-0.32, +3.07] | 0.1081 | 106 | 70 |
| &nbsp;&nbsp;MRR@100 | +0.65 | [-2.03, +3.25] | 0.6430 | 79 | 69 |
| &nbsp;&nbsp;MAP | +1.42 | [-0.33, +3.06] | 0.1057 | 124 | 85 |
| &nbsp;&nbsp;MRP | -1.12 | [-3.18, +0.73] | 0.2727 | 20 | 20 |
| **VHP-E, tuned for MAP@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +2.40 | [-1.62, +6.17] | 0.2239 | 49 | 17 |
| &nbsp;&nbsp;R@200 | +2.23 | [-1.56, +5.89] | 0.2428 | 43 | 18 |
| &nbsp;&nbsp;R@500 | -0.72 | [-4.68, +2.96] | 0.7081 | 45 | 18 |
| &nbsp;&nbsp;MAP@100 | +5.99 | [+3.33, +8.75] | 0.0001*** | 108 | 56 |
| &nbsp;&nbsp;MRR@100 | +4.42 | [+0.84, +8.05] | 0.0186* | 78 | 57 |
| &nbsp;&nbsp;MAP | +6.01 | [+3.35, +8.77] | 0.0001*** | 133 | 70 |
| &nbsp;&nbsp;MRP | +4.81 | [+1.69, +8.12] | 0.0026** | 36 | 17 |
| **VHP-E then VHP-S** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +5.12 | [+0.28, +9.87] | 0.0309* | 64 | 24 |
| &nbsp;&nbsp;R@200 | +3.50 | [-0.93, +7.90] | 0.1171 | 56 | 28 |
| &nbsp;&nbsp;R@500 | +2.74 | [-1.14, +6.61] | 0.1820 | 50 | 23 |
| &nbsp;&nbsp;MAP@100 | +0.99 | [-2.31, +4.28] | 0.5630 | 95 | 92 |
| &nbsp;&nbsp;MRR@100 | -3.96 | [-8.51, +0.63] | 0.0950 | 68 | 101 |
| &nbsp;&nbsp;MAP | +1.01 | [-2.29, +4.28] | 0.5578 | 107 | 109 |
| &nbsp;&nbsp;MRP | -1.39 | [-5.38, +2.51] | 0.5059 | 38 | 34 |

**Significance against the CamemBERT (zero-shot) baseline**

| Metric | Delta (pts) | 95% CI | p | Better | Worse |
|:---|---:|---:|---:|---:|---:|
| **VHP-S, vertical only** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +6.56 | [+3.01, +10.01] | 0.0003*** | 53 | 7 |
| &nbsp;&nbsp;R@200 | +10.03 | [+6.00, +14.00] | 0.0001*** | 67 | 8 |
| &nbsp;&nbsp;R@500 | +10.53 | [+7.08, +14.14] | 0.0001*** | 66 | 3 |
| &nbsp;&nbsp;MAP@100 | +1.72 | [+0.63, +2.80] | 0.0015** | 81 | 40 |
| &nbsp;&nbsp;MRR@100 | +0.34 | [-1.52, +2.21] | 0.7225 | 58 | 44 |
| &nbsp;&nbsp;MAP | +1.78 | [+0.71, +2.85] | 0.0009*** | 155 | 57 |
| &nbsp;&nbsp;MRP | +0.66 | [-0.63, +1.94] | 0.3202 | 20 | 7 |
| **VHP-S, horizontal only** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +6.29 | [+3.45, +9.52] | 0.0001*** | 35 | 7 |
| &nbsp;&nbsp;R@200 | +6.57 | [+3.57, +9.81] | 0.0001*** | 35 | 9 |
| &nbsp;&nbsp;R@500 | +7.12 | [+3.73, +10.73] | 0.0001*** | 37 | 6 |
| &nbsp;&nbsp;MAP@100 | -2.64 | [-4.88, -0.46] | 0.0179* | 60 | 76 |
| &nbsp;&nbsp;MRR@100 | -7.31 | [-10.84, -3.90] | 0.0001*** | 55 | 77 |
| &nbsp;&nbsp;MAP | -2.64 | [-4.89, -0.45] | 0.0177* | 104 | 116 |
| &nbsp;&nbsp;MRP | -5.19 | [-8.24, -2.32] | 0.0003*** | 9 | 37 |
| **VHP-S, tuned for R@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +12.59 | [+8.51, +16.74] | 0.0001*** | 68 | 5 |
| &nbsp;&nbsp;R@200 | +15.97 | [+11.74, +20.28] | 0.0001*** | 77 | 3 |
| &nbsp;&nbsp;R@500 | +15.28 | [+11.58, +19.23] | 0.0001*** | 80 | 1 |
| &nbsp;&nbsp;MAP@100 | -0.01 | [-2.28, +2.30] | 0.9953 | 83 | 64 |
| &nbsp;&nbsp;MRR@100 | -4.64 | [-7.98, -1.42] | 0.0048** | 68 | 71 |
| &nbsp;&nbsp;MAP | +0.06 | [-2.21, +2.36] | 0.9615 | 147 | 73 |
| &nbsp;&nbsp;MRP | -2.85 | [-5.92, +0.08] | 0.0672 | 17 | 25 |
| **VHP-S, tuned for MAP@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +3.52 | [-0.59, +7.40] | 0.0826 | 48 | 15 |
| &nbsp;&nbsp;R@200 | +8.16 | [+3.99, +12.27] | 0.0001*** | 62 | 11 |
| &nbsp;&nbsp;R@500 | +6.35 | [+2.63, +10.03] | 0.0005*** | 60 | 12 |
| &nbsp;&nbsp;MAP@100 | +1.65 | [-0.02, +3.27] | 0.0486* | 78 | 42 |
| &nbsp;&nbsp;MRR@100 | -0.41 | [-3.01, +2.25] | 0.7590 | 55 | 50 |
| &nbsp;&nbsp;MAP | +1.70 | [+0.05, +3.32] | 0.0420* | 145 | 68 |
| &nbsp;&nbsp;MRP | +0.55 | [-1.52, +2.61] | 0.6110 | 21 | 11 |
| **VHP-S, constrained** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +8.60 | [+4.26, +12.87] | 0.0001*** | 59 | 13 |
| &nbsp;&nbsp;R@200 | +15.10 | [+10.40, +19.76] | 0.0001*** | 83 | 8 |
| &nbsp;&nbsp;R@500 | +13.81 | [+10.05, +17.72] | 0.0001*** | 77 | 3 |
| &nbsp;&nbsp;MAP@100 | +1.12 | [-1.11, +3.50] | 0.3392 | 78 | 60 |
| &nbsp;&nbsp;MRR@100 | -3.55 | [-6.68, -0.46] | 0.0249* | 61 | 67 |
| &nbsp;&nbsp;MAP | +1.22 | [-1.01, +3.60] | 0.2952 | 148 | 70 |
| &nbsp;&nbsp;MRP | -0.89 | [-3.94, +2.21] | 0.5703 | 22 | 20 |
| **VHP-E, tuned for R@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +8.72 | [+4.52, +12.77] | 0.0002*** | 62 | 11 |
| &nbsp;&nbsp;R@200 | +10.24 | [+6.03, +14.44] | 0.0001*** | 67 | 10 |
| &nbsp;&nbsp;R@500 | +8.58 | [+4.67, +12.49] | 0.0001*** | 65 | 9 |
| &nbsp;&nbsp;MAP@100 | +3.02 | [+0.68, +5.38] | 0.0121* | 98 | 35 |
| &nbsp;&nbsp;MRR@100 | +2.43 | [-1.19, +6.09] | 0.1899 | 81 | 37 |
| &nbsp;&nbsp;MAP | +3.02 | [+0.69, +5.37] | 0.0115* | 162 | 55 |
| &nbsp;&nbsp;MRP | +0.80 | [-1.82, +3.54] | 0.5533 | 21 | 13 |
| **VHP-E, tuned for MAP@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +8.90 | [+4.91, +12.76] | 0.0001*** | 64 | 10 |
| &nbsp;&nbsp;R@200 | +11.39 | [+7.13, +15.59] | 0.0001*** | 68 | 9 |
| &nbsp;&nbsp;R@500 | +8.29 | [+4.16, +12.33] | 0.0002*** | 64 | 10 |
| &nbsp;&nbsp;MAP@100 | +2.96 | [+0.67, +5.25] | 0.0113* | 96 | 38 |
| &nbsp;&nbsp;MRR@100 | +2.36 | [-1.25, +5.98] | 0.2035 | 80 | 41 |
| &nbsp;&nbsp;MAP | +2.97 | [+0.68, +5.26] | 0.0106* | 157 | 60 |
| &nbsp;&nbsp;MRP | +1.00 | [-1.62, +3.73] | 0.4658 | 25 | 13 |
| **VHP-E then VHP-S** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +12.78 | [+7.58, +17.91] | 0.0001*** | 80 | 16 |
| &nbsp;&nbsp;R@200 | +18.25 | [+13.14, +23.40] | 0.0001*** | 92 | 10 |
| &nbsp;&nbsp;R@500 | +14.42 | [+9.94, +18.91] | 0.0001*** | 83 | 11 |
| &nbsp;&nbsp;MAP@100 | -0.45 | [-3.09, +2.14] | 0.7377 | 85 | 66 |
| &nbsp;&nbsp;MRR@100 | -5.92 | [-9.79, -2.16] | 0.0020** | 70 | 72 |
| &nbsp;&nbsp;MAP | -0.39 | [-3.03, +2.19] | 0.7726 | 142 | 79 |
| &nbsp;&nbsp;MRP | -3.36 | [-6.62, -0.28] | 0.0400* | 24 | 33 |

**Significance against the mE5 (fine-tuned) baseline**

| Metric | Delta (pts) | 95% CI | p | Better | Worse |
|:---|---:|---:|---:|---:|---:|
| **VHP-S, vertical only** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +0.50 | [-0.72, +1.87] | 0.4945 | 7 | 5 |
| &nbsp;&nbsp;R@200 | +0.45 | [-0.62, +1.50] | 0.4149 | 9 | 5 |
| &nbsp;&nbsp;R@500 | +0.07 | [-0.55, +0.56] | 0.8804 | 4 | 1 |
| &nbsp;&nbsp;MAP@100 | -0.86 | [-2.57, +0.68] | 0.3184 | 71 | 32 |
| &nbsp;&nbsp;MRR@100 | -1.77 | [-3.76, +0.01] | 0.0704 | 35 | 19 |
| &nbsp;&nbsp;MAP | -0.85 | [-2.56, +0.69] | 0.3224 | 85 | 37 |
| &nbsp;&nbsp;MRP | -1.18 | [-3.65, +1.07] | 0.3292 | 13 | 10 |
| **VHP-S, horizontal only** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | -0.33 | [-1.65, +0.77] | 0.6207 | 4 | 5 |
| &nbsp;&nbsp;R@200 | +0.97 | [+0.01, +2.25] | 0.0970 | 7 | 2 |
| &nbsp;&nbsp;R@500 | +0.51 | [-0.13, +1.61] | 0.5013 | 2 | 1 |
| &nbsp;&nbsp;MAP@100 | -7.53 | [-11.00, -4.11] | 0.0002*** | 49 | 118 |
| &nbsp;&nbsp;MRR@100 | -8.58 | [-12.59, -4.57] | 0.0001*** | 39 | 88 |
| &nbsp;&nbsp;MAP | -7.49 | [-10.96, -4.07] | 0.0002*** | 62 | 123 |
| &nbsp;&nbsp;MRP | -11.33 | [-16.35, -6.34] | 0.0001*** | 18 | 67 |
| **VHP-S, tuned for R@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +0.50 | [-0.49, +1.74] | 0.4469 | 6 | 5 |
| &nbsp;&nbsp;R@200 | +1.20 | [+0.22, +2.50] | 0.0231* | 9 | 3 |
| &nbsp;&nbsp;R@500 | +0.71 | [-0.25, +1.93] | 0.2675 | 6 | 1 |
| &nbsp;&nbsp;MAP@100 | -9.38 | [-13.16, -5.72] | 0.0001*** | 58 | 111 |
| &nbsp;&nbsp;MRR@100 | -10.88 | [-15.20, -6.64] | 0.0001*** | 43 | 86 |
| &nbsp;&nbsp;MAP | -9.34 | [-13.11, -5.67] | 0.0001*** | 73 | 114 |
| &nbsp;&nbsp;MRP | -14.01 | [-19.25, -8.87] | 0.0001*** | 18 | 69 |
| **VHP-S, tuned for MAP@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +0.44 | [-0.32, +1.22] | 0.3208 | 6 | 2 |
| &nbsp;&nbsp;R@200 | +0.29 | [-0.38, +0.90] | 0.4080 | 6 | 1 |
| &nbsp;&nbsp;R@500 | +0.20 | [-0.84, +1.48] | 0.8089 | 4 | 3 |
| &nbsp;&nbsp;MAP@100 | +1.52 | [+0.54, +2.55] | 0.0012** | 76 | 26 |
| &nbsp;&nbsp;MRR@100 | +1.73 | [+0.35, +3.21] | 0.0133* | 40 | 19 |
| &nbsp;&nbsp;MAP | +1.52 | [+0.54, +2.55] | 0.0012** | 90 | 33 |
| &nbsp;&nbsp;MRP | +1.23 | [-0.47, +3.07] | 0.1703 | 18 | 4 |
| **VHP-S, constrained** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +0.70 | [+0.15, +1.40] | 0.0224* | 7 | 1 |
| &nbsp;&nbsp;R@200 | +0.29 | [-0.38, +0.90] | 0.4080 | 6 | 1 |
| &nbsp;&nbsp;R@500 | -0.20 | [-0.95, +0.41] | 0.6262 | 4 | 2 |
| &nbsp;&nbsp;MAP@100 | +1.54 | [+0.52, +2.59] | 0.0015** | 75 | 28 |
| &nbsp;&nbsp;MRR@100 | +1.86 | [+0.44, +3.42] | 0.0103* | 37 | 20 |
| &nbsp;&nbsp;MAP | +1.53 | [+0.52, +2.58] | 0.0015** | 89 | 35 |
| &nbsp;&nbsp;MRP | +1.27 | [-0.42, +3.11] | 0.1537 | 19 | 4 |
| **VHP-E, tuned for R@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +0.29 | [-1.05, +1.44] | 0.6928 | 8 | 3 |
| &nbsp;&nbsp;R@200 | +0.53 | [+0.11, +1.06] | 0.0313* | 6 | 0 |
| &nbsp;&nbsp;R@500 | +0.56 | [+0.06, +1.32] | 0.0626 | 5 | 0 |
| &nbsp;&nbsp;MAP@100 | -1.91 | [-4.55, +0.55] | 0.1411 | 77 | 70 |
| &nbsp;&nbsp;MRR@100 | -1.21 | [-4.42, +1.92] | 0.4504 | 49 | 48 |
| &nbsp;&nbsp;MAP | -1.92 | [-4.57, +0.55] | 0.1387 | 88 | 77 |
| &nbsp;&nbsp;MRP | -4.25 | [-8.41, -0.24] | 0.0402* | 28 | 34 |
| **VHP-E, tuned for MAP@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +0.22 | [-0.49, +0.94] | 0.6170 | 4 | 2 |
| &nbsp;&nbsp;R@200 | -0.02 | [-0.61, +0.41] | 1.0000 | 4 | 1 |
| &nbsp;&nbsp;R@500 | -0.13 | [-0.68, +0.19] | 1.0000 | 2 | 1 |
| &nbsp;&nbsp;MAP@100 | +0.83 | [-0.37, +2.03] | 0.1703 | 79 | 29 |
| &nbsp;&nbsp;MRR@100 | +1.10 | [-0.27, +2.54] | 0.1224 | 39 | 16 |
| &nbsp;&nbsp;MAP | +0.82 | [-0.38, +2.03] | 0.1702 | 93 | 35 |
| &nbsp;&nbsp;MRP | -0.05 | [-2.27, +2.11] | 0.9683 | 15 | 6 |
| **VHP-E then VHP-S** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +0.56 | [-0.75, +1.74] | 0.4237 | 12 | 5 |
| &nbsp;&nbsp;R@200 | +1.49 | [+0.40, +2.91] | 0.0065** | 10 | 3 |
| &nbsp;&nbsp;R@500 | +1.86 | [+0.53, +3.54] | 0.0072** | 8 | 0 |
| &nbsp;&nbsp;MAP@100 | -14.26 | [-18.20, -10.42] | 0.0001*** | 56 | 131 |
| &nbsp;&nbsp;MRR@100 | -15.61 | [-20.15, -11.11] | 0.0001*** | 46 | 101 |
| &nbsp;&nbsp;MAP | -14.23 | [-18.19, -10.39] | 0.0001*** | 68 | 136 |
| &nbsp;&nbsp;MRP | -21.05 | [-26.43, -15.81] | 0.0001*** | 17 | 92 |

**Significance against the BGE-M3 (fine-tuned) baseline**

| Metric | Delta (pts) | 95% CI | p | Better | Worse |
|:---|---:|---:|---:|---:|---:|
| **VHP-S, vertical only** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +1.58 | [-0.31, +3.60] | 0.1207 | 15 | 6 |
| &nbsp;&nbsp;R@200 | +0.75 | [-0.56, +1.94] | 0.2734 | 11 | 1 |
| &nbsp;&nbsp;R@500 | +0.03 | [-1.09, +0.88] | 1.0000 | 5 | 2 |
| &nbsp;&nbsp;MAP@100 | +0.26 | [-1.99, +2.37] | 0.8140 | 96 | 52 |
| &nbsp;&nbsp;MRR@100 | +0.37 | [-2.05, +2.72] | 0.7576 | 50 | 36 |
| &nbsp;&nbsp;MAP | +0.26 | [-1.98, +2.38] | 0.8105 | 103 | 58 |
| &nbsp;&nbsp;MRP | +0.14 | [-3.24, +3.19] | 0.9293 | 35 | 17 |
| **VHP-S, horizontal only** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +0.43 | [-0.26, +1.26] | 0.3717 | 4 | 2 |
| &nbsp;&nbsp;R@200 | +0.16 | [+0.00, +0.36] | 0.2559 | 3 | 0 |
| &nbsp;&nbsp;R@500 | +0.00 | [+0.00, +0.00] | 1.0000 | 0 | 0 |
| &nbsp;&nbsp;MAP@100 | -0.52 | [-2.57, +1.47] | 0.6219 | 60 | 85 |
| &nbsp;&nbsp;MRR@100 | -1.19 | [-3.95, +1.54] | 0.4029 | 33 | 53 |
| &nbsp;&nbsp;MAP | -0.52 | [-2.57, +1.47] | 0.6194 | 72 | 91 |
| &nbsp;&nbsp;MRP | +0.55 | [-2.64, +3.75] | 0.7425 | 19 | 24 |
| **VHP-S, tuned for R@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +2.36 | [+0.73, +4.23] | 0.0047** | 16 | 3 |
| &nbsp;&nbsp;R@200 | +0.75 | [-0.56, +1.94] | 0.2734 | 11 | 1 |
| &nbsp;&nbsp;R@500 | +0.48 | [+0.08, +1.02] | 0.0618 | 5 | 1 |
| &nbsp;&nbsp;MAP@100 | +0.35 | [-2.11, +2.80] | 0.7836 | 88 | 75 |
| &nbsp;&nbsp;MRR@100 | -0.86 | [-3.91, +2.19] | 0.5952 | 47 | 59 |
| &nbsp;&nbsp;MAP | +0.35 | [-2.12, +2.79] | 0.7896 | 97 | 78 |
| &nbsp;&nbsp;MRP | +1.16 | [-2.90, +5.20] | 0.5762 | 43 | 25 |
| **VHP-S, tuned for MAP@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +1.22 | [+0.30, +2.48] | 0.0145* | 7 | 0 |
| &nbsp;&nbsp;R@200 | +0.54 | [+0.15, +1.03] | 0.0342* | 6 | 0 |
| &nbsp;&nbsp;R@500 | +0.33 | [+0.03, +0.75] | 0.1245 | 4 | 0 |
| &nbsp;&nbsp;MAP@100 | +1.29 | [+0.54, +2.13] | 0.0005*** | 94 | 31 |
| &nbsp;&nbsp;MRR@100 | +1.19 | [+0.27, +2.29] | 0.0174* | 42 | 17 |
| &nbsp;&nbsp;MAP | +1.28 | [+0.52, +2.12] | 0.0005*** | 107 | 36 |
| &nbsp;&nbsp;MRP | +1.78 | [+0.60, +3.23] | 0.0036** | 26 | 6 |
| **VHP-S, constrained** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +1.61 | [-0.22, +3.54] | 0.0945 | 13 | 4 |
| &nbsp;&nbsp;R@200 | +0.48 | [-0.75, +1.53] | 0.4799 | 9 | 1 |
| &nbsp;&nbsp;R@500 | +0.51 | [+0.13, +1.04] | 0.0618 | 5 | 0 |
| &nbsp;&nbsp;MAP@100 | +0.51 | [-1.25, +2.16] | 0.5568 | 96 | 46 |
| &nbsp;&nbsp;MRR@100 | +0.54 | [-1.45, +2.50] | 0.5937 | 47 | 30 |
| &nbsp;&nbsp;MAP | +0.51 | [-1.26, +2.16] | 0.5610 | 107 | 50 |
| &nbsp;&nbsp;MRP | -0.27 | [-3.27, +2.48] | 0.8571 | 33 | 15 |
| **VHP-E, tuned for R@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +2.02 | [+0.35, +3.98] | 0.0270* | 12 | 5 |
| &nbsp;&nbsp;R@200 | +0.99 | [+0.34, +1.81] | 0.0046** | 9 | 0 |
| &nbsp;&nbsp;R@500 | -0.28 | [-1.47, +0.60] | 0.7163 | 5 | 2 |
| &nbsp;&nbsp;MAP@100 | +0.61 | [-1.74, +2.94] | 0.5982 | 96 | 49 |
| &nbsp;&nbsp;MRR@100 | +0.23 | [-2.34, +2.80] | 0.8583 | 50 | 33 |
| &nbsp;&nbsp;MAP | +0.60 | [-1.74, +2.94] | 0.6007 | 107 | 52 |
| &nbsp;&nbsp;MRP | +0.23 | [-3.40, +3.75] | 0.8989 | 38 | 20 |
| **VHP-E, tuned for MAP@100** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +1.55 | [+0.45, +2.98] | 0.0083** | 8 | 0 |
| &nbsp;&nbsp;R@200 | +0.56 | [+0.17, +1.06] | 0.0179* | 7 | 0 |
| &nbsp;&nbsp;R@500 | -0.12 | [-1.20, +0.63] | 1.0000 | 4 | 1 |
| &nbsp;&nbsp;MAP@100 | +1.55 | [+0.31, +2.80] | 0.0136* | 97 | 34 |
| &nbsp;&nbsp;MRR@100 | +1.70 | [+0.18, +3.30] | 0.0328* | 43 | 20 |
| &nbsp;&nbsp;MAP | +1.54 | [+0.29, +2.78] | 0.0142* | 111 | 37 |
| &nbsp;&nbsp;MRP | +1.19 | [-1.06, +3.40] | 0.3021 | 28 | 7 |
| **VHP-E then VHP-S** |  |  |  |  |  |
| &nbsp;&nbsp;R@100 | +1.66 | [-1.05, +4.46] | 0.2446 | 16 | 11 |
| &nbsp;&nbsp;R@200 | +0.80 | [-0.63, +2.15] | 0.2874 | 12 | 3 |
| &nbsp;&nbsp;R@500 | +0.85 | [-0.75, +2.52] | 0.3356 | 9 | 2 |
| &nbsp;&nbsp;MAP@100 | -1.38 | [-4.78, +2.00] | 0.4354 | 93 | 83 |
| &nbsp;&nbsp;MRR@100 | -2.62 | [-6.54, +1.29] | 0.1960 | 52 | 67 |
| &nbsp;&nbsp;MAP | -1.38 | [-4.78, +1.98] | 0.4345 | 101 | 86 |
| &nbsp;&nbsp;MRP | +0.56 | [-4.13, +5.18] | 0.8117 | 47 | 30 |

## Cross-Validation

Five-fold cross-validation on the training split, BM25, objective R@100. Per fold gains: +9.27, +10.17, +7.71, +9.53, +10.32, that is +9.40 on average with a standard deviation of 0.93. The configuration retained on each fold is given below; the variation across folds is what the standard deviation measures.

**Configurations selected on each training fold**

| Fold | Selected configuration |
|:---|:---|
| fold 1 | beta=1.2, gamma=0.5, m=3, dir. out, kappa=1, increasing |
| fold 2 | beta=1.2, gamma=0.3, m=1, dir. out, kappa=1, increasing |
| fold 3 | beta=1.2, gamma=0.3, m=1, dir. out, kappa=1, increasing |
| fold 4 | beta=1.2, gamma=0.5, m=1, dir. out, kappa=1, increasing |
| fold 5 | beta=1.2, gamma=0.3, m=1, dir. out, kappa=1, increasing |
