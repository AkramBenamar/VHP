from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

CODES = ["Code civil", "Code judiciaire", "Code pénal", "Code de commerce"]
BOOKS = ["Livre I", "Livre II", "Livre III"]
CHAPTERS = ["Chapitre I", "Chapitre II", "Chapitre III"]
SECTIONS = ["Section 1", "Section 2", "Section 3"]

TOPIC_WORDS = {
    "bail": "bail loyer locataire bailleur preavis resiliation immeuble",
    "divorce": "divorce epoux mariage separation pension alimentaire",
    "travail": "employeur travailleur contrat licenciement remuneration",
    "succession": "succession heritier testament legs reserve heritage",
    "vente": "vente acheteur vendeur prix garantie vice cache livraison",
    "societe": "societe associe capital gerant assemblee dissolution",
    "penal": "infraction peine amende emprisonnement prevenu poursuite",
    "procedure": "citation delai juridiction competence appel signification",
}
FILLER = ("Le présent article dispose que les parties peuvent convenir de "
          "modalités particulières dans les limites prévues par la loi.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n-articles", type=int, default=1200)
    ap.add_argument("--n-questions", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    topics = list(TOPIC_WORDS)
    articles, section_members, section_topic = [], {}, {}

    for i in range(args.n_articles):
        code = CODES[i % len(CODES)]
        book = BOOKS[(i // 100) % len(BOOKS)]
        chapter = CHAPTERS[(i // 30) % len(CHAPTERS)]
        section = SECTIONS[(i // 10) % len(SECTIONS)]
        key = (code, book, chapter, section)
        topic = section_topic.setdefault(key, rng.choice(topics))
        num = 100 + i
        words = TOPIC_WORDS[topic].split()
        rng.shuffle(words)
        body = f"{' '.join(words[:5])}. {FILLER}"
        articles.append({
            "id": i,
            "article": body,
            "reference": f"{code}, art. {num}",
            "law_type": "code",
            "code": code,
            "book": book,
            "chapter": chapter,
            "section": section,
            "description": f"{code} - {book} - {chapter} - {section}",
            "_num": num,
            "_topic": topic,
        })
        section_members.setdefault(key, []).append(i)

    df = pd.DataFrame(articles)

    silent = set(rng.sample(range(args.n_articles), args.n_articles // 12))
    for i in silent:
        df.loc[i, "article"] = FILLER
    for i in silent:
        key = (df.loc[i, "code"], df.loc[i, "book"],
               df.loc[i, "chapter"], df.loc[i, "section"])
        peers = [j for j in section_members[key] if j != i and j not in silent]
        if not peers:
            continue
        src = rng.choice(peers)
        tgt_num = int(df.loc[i, "_num"])
        df.loc[src, "article"] = (
            f"{df.loc[src, 'article']} Conformément à l'article {tgt_num} "
            f"du {df.loc[i, 'code']}, les mêmes règles s'appliquent."
        )

    questions = []
    for q in range(args.n_questions):
        key = rng.choice(list(section_members))
        members = section_members[key]
        topic = section_topic[key]
        n_gold = rng.choice([1, 2, 2, 3, 4])
        gold = rng.sample(members, min(n_gold, len(members)))
        words = TOPIC_WORDS[topic].split()
        rng.shuffle(words)
        questions.append({
            "id": q,
            "question": f"Que prévoit la loi concernant {' '.join(words[:4])} ?",
            "category": topic,
            "subcategory": topic,
            "extra_description": "",
            "article_ids": ",".join(str(g) for g in gold),
        })

    qdf = pd.DataFrame(questions)
    split = int(len(qdf) * 0.75)
    df.drop(columns=["_num", "_topic"]).to_csv(out / "articles_fr.csv", index=False)
    qdf.iloc[:split].to_csv(out / "questions_fr_train.csv", index=False)
    qdf.iloc[split:].to_csv(out / "questions_fr_test.csv", index=False)
    print(f"Written to {out.resolve()}: {len(df)} articles, "
          f"{split} train questions, {len(qdf) - split} test questions")


if __name__ == "__main__":
    main()
