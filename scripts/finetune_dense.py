from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bsard_vhp import data as data_mod
from bsard_vhp.retrievers import (BM25Config, BM25Retriever, PREFIX_PRESETS)


def mine_hard_negatives(corpus, qs, n_neg: int, exclude_same_section: bool = False):
    bm25 = BM25Retriever(BM25Config()).index(corpus.texts)
    scores = bm25.score(qs.texts)
    section = corpus.hierarchy.get("section")
    negs = []
    for i, gold in enumerate(qs.qrels):
        order = np.argsort(-scores[i])
        picked = []
        for j in order:
            j = int(j)
            if j in gold:
                continue
            if (exclude_same_section and section is not None
                    and any(section[j] == section[g] for g in gold)):
                continue
            picked.append(j)
            if len(picked) >= n_neg:
                break
        negs.append(picked)
    return negs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/bsard")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--out", default="models/dense-bsard")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--hard-negatives", type=int, default=4)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-positives", type=int, default=0,
                    help="cap the number of positive articles kept per "
                         "question (0 = all). This matters without a GPU: "
                         "BSARD has 6.5 positives per question on average and "
                         "up to 109, so a handful of questions generate "
                         "thousands of triplets on their own.")
    ap.add_argument("--threads", type=int, default=None,
                    help="CPU threads for torch")
    ap.add_argument("--use-synthetic", action="store_true",
                    help="add the synthetic BSARD questions to training. These "
                         "are what allow the original authors to reach 74.8 "
                         "R@100: 886 annotated questions are not enough for a "
                         "bi-encoder, so they are required for a fair "
                         "comparison with the published supervised baselines.")
    ap.add_argument("--synthetic-max", type=int, default=30000,
                    help="number of synthetic questions sampled (113,165 are "
                         "available; 30,000 is usually enough and roughly "
                         "halves the training time)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    import torch
    if args.threads:
        torch.set_num_threads(args.threads)

    from sentence_transformers import (InputExample, SentenceTransformer,
                                       losses)
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from torch.utils.data import DataLoader

    corpus = data_mod.load_corpus(args.data_dir)
    train = data_mod.load_questions(args.data_dir, "train", corpus)

    synthetic = None
    if args.use_synthetic:
        try:
            synthetic = data_mod.load_questions(args.data_dir, "synthetic",
                                                corpus)
            if args.synthetic_max and len(synthetic) > args.synthetic_max:
                keep = random.sample(range(len(synthetic)), args.synthetic_max)
                synthetic.texts = [synthetic.texts[i] for i in keep]
                synthetic.qrels = [synthetic.qrels[i] for i in keep]
                synthetic.ids = synthetic.ids[keep]
            print(f"{len(synthetic)} synthetic questions added")
        except FileNotFoundError:
            print("[warn] questions_fr_synthetic.csv not found; "
                  "run scripts/download_bsard.py again")

    qpx, dpx = PREFIX_PRESETS.get(args.model, ("", ""))

    print(f"Mining hard negatives with BM25: {args.hard_negatives} per question")
    negs = mine_hard_negatives(corpus, train, args.hard_negatives)
    syn_negs = None

    idx = list(range(len(train)))
    random.shuffle(idx)
    n_dev = max(int(len(idx) * args.dev_frac), 1)
    dev_idx, tr_idx = set(idx[:n_dev]), idx[n_dev:]

    examples = []
    for i in tr_idx:
        q = qpx + train.texts[i]
        golds = sorted(train.qrels[i])
        if args.max_positives:
            golds = golds[:args.max_positives]
        for g in golds:
            pos = dpx + corpus.texts[g]
            if args.hard_negatives:
                for nj in negs[i]:
                    examples.append(InputExample(texts=[q, pos, dpx + corpus.texts[nj]]))
            else:
                examples.append(InputExample(texts=[q, pos]))
    if synthetic is not None:
        for i in range(len(synthetic)):
            q = qpx + synthetic.texts[i]
            for g in sorted(synthetic.qrels[i])[:1]:
                examples.append(InputExample(texts=[q, dpx + corpus.texts[g]]))

    print(f"{len(examples)} training examples "
          f"({len(tr_idx)} annotated questions"
          + (f" + {len(synthetic)} synthetic" if synthetic is not None else "")
          + f", {n_dev} held out for dev)")

    model = SentenceTransformer(args.model)
    model.max_seq_length = args.max_seq_length

    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size,
                        drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model)

    dev_qrels, dev_queries = {}, {}
    needed = set()
    for i in dev_idx:
        dev_queries[str(i)] = qpx + train.texts[i]
        dev_qrels[str(i)] = {str(g) for g in train.qrels[i]}
        needed |= set(train.qrels[i])
    pool = list(needed) + random.sample(range(len(corpus)), min(5000, len(corpus)))
    dev_corpus = {str(j): dpx + corpus.texts[j] for j in set(pool)}
    evaluator = InformationRetrievalEvaluator(
        dev_queries, dev_corpus, dev_qrels, name="bsard-dev",
        accuracy_at_k=[10, 100], precision_recall_at_k=[100], map_at_k=[100],
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    use_amp = torch.cuda.is_available()
    if not use_amp:
        print("[info] no GPU detected: training will be slow. Consider "
              "--max-positives 2 --epochs 5 --max-seq-length 256, or a "
              "smaller model such as multilingual-e5-small.")

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=int(0.1 * len(loader) * args.epochs),
        optimizer_params={"lr": args.lr},
        evaluator=evaluator,
        evaluation_steps=max(len(loader), 1),
        output_path=args.out,
        save_best_model=True,
        use_amp=use_amp,
    )
    print(f"Model written to {args.out}")
    print("Next:  python -m bsard_vhp.run experiment --retrievers dense "
          f"--dense-model {args.out}")


if __name__ == "__main__":
    main()
