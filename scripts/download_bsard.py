from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = "maastrichtlawtech/bsard"


def _to_frame(ds):
    import pandas as pd
    return pd.DataFrame(ds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/bsard")
    ap.add_argument("--repo", default=REPO)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Install the library first: pip install datasets")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Corpus from {args.repo} ...")
    corpus = load_dataset(args.repo, name="corpus")
    split = "corpus" if "corpus" in corpus else list(corpus.keys())[0]
    articles = _to_frame(corpus[split])
    articles.to_csv(out / "articles_fr.csv", index=False)
    print(f"  {len(articles)} articles -> articles_fr.csv")
    print(f"  columns: {list(articles.columns)}")

    for split_name, filename in [("train", "questions_fr_train.csv"),
                                 ("test", "questions_fr_test.csv"),
                                 ("synthetic", "questions_fr_synthetic.csv")]:
        try:
            q = _to_frame(load_dataset(args.repo, name="questions",
                                       split=split_name))
        except Exception as exc:
            print(f"  (no '{split_name}' split: {type(exc).__name__})")
            continue
        q.to_csv(out / filename, index=False)
        print(f"  {len(q)} '{split_name}' questions -> {filename}")
        if split_name == "train":
            print(f"  columns: {list(q.columns)}")

    for split_name in ("train", "synthetic"):
        try:
            n = _to_frame(load_dataset(args.repo, name="negatives",
                                       split=split_name))
        except Exception:
            continue
        n.to_csv(out / f"negatives_{split_name}.csv", index=False)
        print(f"  '{split_name}' negatives -> negatives_{split_name}.csv")

    ids = articles[articles.columns[0]]
    print(f"\nArticle identifiers: min={ids.min()}, max={ids.max()}, "
          f"contiguous={'yes' if len(ids) == int(ids.max()) - int(ids.min()) + 1 else 'no'}")
    print(f"\nWritten to {out.resolve()}")
    print("Next step:  python -m bsard_vhp.run inspect --data-dir "
          f"{args.out}")


if __name__ == "__main__":
    main()
