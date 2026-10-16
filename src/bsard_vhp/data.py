from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ID_CANDIDATES = ["id", "article_id", "aid"]
TEXT_CANDIDATES = ["article", "text", "content", "article_text", "body"]
REF_CANDIDATES = ["reference", "ref", "article_reference"]
DESC_CANDIDATES = ["description", "descr", "heading", "context"]
HIERARCHY_CANDIDATES = [
    "law_type", "code", "book", "part", "act", "chapter",
    "section", "subsection",
]

QID_CANDIDATES = ["id", "question_id", "qid"]
QTEXT_CANDIDATES = ["question", "query", "text"]
QGOLD_CANDIDATES = ["article_ids", "articles", "relevant_articles", "gold"]
QEXTRA_CANDIDATES = ["extra_description", "description", "context"]
QCAT_CANDIDATES = ["category", "topic"]


def _pick(cols: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def norm_key(s: str) -> str:
    s = strip_accents(str(s)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class Corpus:
    ids: np.ndarray
    texts: list[str]
    titles: list[str]
    references: list[str]
    numbers: list[str]
    hierarchy: dict[str, np.ndarray]
    hierarchy_labels: dict[str, list[str]] = field(default_factory=dict)
    raw: pd.DataFrame | None = None

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def id_to_index(self) -> dict[int, int]:
        if not hasattr(self, "_id2idx"):
            self._id2idx = {int(a): i for i, a in enumerate(self.ids)}
        return self._id2idx


@dataclass
class QuerySet:
    ids: np.ndarray
    texts: list[str]
    qrels: list[set[int]]
    categories: list[str] | None = None

    def __len__(self) -> int:
        return len(self.texts)


_ART_NUM_CORE = (
    r"(?:[A-Za-z]{1,2}\.?)?"
    r"\d+(?:\.\d+)*"
    r"(?:\s*er\b|\s*ère\b)?"
    r"(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?"
    r"(?:\s*/\s*\d+)?"
)
_ART_NUM_RE = re.compile(
    r"\bart(?:icle)?s?\.?\s*(" + _ART_NUM_CORE + r")",
    flags=re.IGNORECASE,
)


def normalize_article_number(num: str) -> str:
    s = strip_accents(str(num)).lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"(\d)(er|ere)$", r"\1", s)
    s = re.sub(r"[^a-z0-9/.]", "", s)
    return s.strip(".")


def extract_article_number(reference: str, text: str = "") -> str:
    for src in (reference, text[:120]):
        if not src:
            continue
        m = _ART_NUM_RE.search(str(src))
        if m:
            return normalize_article_number(m.group(1))
    return ""


PATH_SEPARATORS = [" - ", " — ", " > ", " | ", " / ", "\n", ";"]
MAX_DERIVED_LEVELS = 6
MIN_PATH_COVERAGE = 0.5


def split_path(text: str, sep: str) -> list[str]:
    return [seg.strip() for seg in str(text).split(sep) if seg.strip()]


def choose_separator(descriptions: list[str]) -> str | None:
    best, best_cov = None, 0.0
    for sep in PATH_SEPARATORS:
        cov = sum(1 for d in descriptions if len(split_path(d, sep)) >= 2)
        cov /= max(len(descriptions), 1)
        if cov > best_cov:
            best, best_cov = sep, cov
    return best if best_cov >= MIN_PATH_COVERAGE else None


def parse_description_paths(
    descriptions: list[str],
) -> list[tuple[str, list[str]]]:
    sep = choose_separator(descriptions)
    if sep is None:
        return []

    paths = [split_path(d, sep) for d in descriptions]
    depth = min(max((len(p) for p in paths), default=0), MAX_DERIVED_LEVELS)
    if depth < 2:
        return []

    levels: list[tuple[str, list[str]]] = []
    for d in range(depth):
        vals = []
        for p in paths:
            if not p:
                vals.append("_NA_")
            elif d < len(p):
                vals.append(p[d])
            else:
                vals.append(p[-1])
        n_groups = len(set(vals))
        if n_groups <= 1 or n_groups >= 0.9 * len(descriptions):
            continue
        levels.append((f"level{d + 1}", vals))
    return levels


def _find_file(data_dir: Path, patterns: list[str]) -> Path:
    for pat in patterns:
        hits = sorted(data_dir.rglob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No file matching {patterns} under {data_dir}. Check --data-dir."
    )


def inspect(data_dir: str | Path) -> None:
    data_dir = Path(data_dir)
    print(f"Root: {data_dir.resolve()}")
    for p in sorted(data_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(p, nrows=3)
        except Exception as exc:
            print(f"  {p.relative_to(data_dir)}: unreadable ({exc})")
            continue
        print(f"\n  {p.relative_to(data_dir)}  ({len(pd.read_csv(p))} rows)")
        print(f"    columns: {list(df.columns)}")
        print(f"    sample : {df.iloc[0].to_dict()}")


def load_corpus(
    data_dir: str | Path,
    articles_file: str | None = None,
    include_description_in_text: bool = False,
    verbose: bool = True,
) -> Corpus:
    data_dir = Path(data_dir)
    path = (Path(articles_file) if articles_file
            else _find_file(data_dir, ["articles_fr.csv", "articles*.csv"]))
    df = pd.read_csv(path)
    cols = list(df.columns)

    id_col = _pick(cols, ID_CANDIDATES)
    text_col = _pick(cols, TEXT_CANDIDATES)
    if id_col is None or text_col is None:
        raise KeyError(
            f"Could not identify the id/text columns in {path}.\n"
            f"Columns found: {cols}\n"
            "Adjust ID_CANDIDATES / TEXT_CANDIDATES in data.py."
        )
    ref_col = _pick(cols, REF_CANDIDATES)
    desc_col = _pick(cols, DESC_CANDIDATES)

    levels = [c for c in HIERARCHY_CANDIDATES if _pick(cols, [c])]
    levels = [_pick(cols, [c]) for c in levels]

    df = df.reset_index(drop=True)
    texts_raw = df[text_col].fillna("").astype(str).tolist()
    descs = (df[desc_col].fillna("").astype(str).tolist()
             if desc_col else [""] * len(df))
    refs = (df[ref_col].fillna("").astype(str).tolist()
            if ref_col else [""] * len(df))

    if include_description_in_text:
        texts = [f"{d}. {t}".strip(". ") if d else t
                 for d, t in zip(descs, texts_raw)]
    else:
        texts = texts_raw

    numbers = [extract_article_number(r, t) for r, t in zip(refs, texts_raw)]

    level_series: list[tuple[str, list[str]]] = []
    if len(levels) >= 3:
        level_series = [(lvl, df[lvl].fillna("_NA_").astype(str).tolist())
                        for lvl in levels]
    else:
        parsed = parse_description_paths(descs)
        if parsed:
            base = [(lvl, df[lvl].fillna("_NA_").astype(str).tolist())
                    for lvl in levels]
            level_series = base + parsed
            if verbose:
                print(f"[data] hierarchy derived from '{desc_col}': "
                      f"{len(parsed)} levels ({', '.join(n for n, _ in parsed)})")
        else:
            level_series = [(lvl, df[lvl].fillna("_NA_").astype(str).tolist())
                            for lvl in levels]

    if not level_series:
        print("[warn] no hierarchy detected: the vertical axis of VHP will have "
              "no effect. Check the columns with `run inspect`.")

    hierarchy: dict[str, np.ndarray] = {}
    hierarchy_labels: dict[str, list[str]] = {}
    path_acc = [""] * len(df)
    for lvl, vals in level_series:
        path_acc = [f"{p}||{v}" for p, v in zip(path_acc, vals)]
        uniq: dict[str, int] = {}
        gid = np.empty(len(df), dtype=np.int64)
        labels: list[str] = []
        for i, p in enumerate(path_acc):
            if p not in uniq:
                uniq[p] = len(uniq)
                labels.append(p)
            gid[i] = uniq[p]
        hierarchy[lvl] = gid
        hierarchy_labels[lvl] = labels

    return Corpus(
        ids=df[id_col].to_numpy(),
        texts=texts,
        titles=descs,
        references=refs,
        numbers=numbers,
        hierarchy=hierarchy,
        hierarchy_labels=hierarchy_labels,
        raw=df,
    )


def _parse_gold(cell) -> list[int]:
    if isinstance(cell, (list, tuple, np.ndarray)):
        return [int(x) for x in cell]
    s = str(cell)
    return [int(x) for x in re.findall(r"\d+", s)]


def load_questions(
    data_dir: str | Path,
    split: str,
    corpus: Corpus,
    questions_file: str | None = None,
    append_extra_description: bool = False,
) -> QuerySet:
    data_dir = Path(data_dir)
    path = (Path(questions_file) if questions_file
            else _find_file(data_dir, [f"questions_fr_{split}.csv",
                                       f"questions*{split}*.csv"]))
    df = pd.read_csv(path)
    cols = list(df.columns)

    qid_col = _pick(cols, QID_CANDIDATES)
    qtext_col = _pick(cols, QTEXT_CANDIDATES)
    gold_col = _pick(cols, QGOLD_CANDIDATES)
    if qtext_col is None or gold_col is None:
        raise KeyError(
            f"Question/gold columns not found in {path}. Columns: {cols}"
        )
    extra_col = _pick(cols, QEXTRA_CANDIDATES)
    cat_col = _pick(cols, QCAT_CANDIDATES)

    texts = df[qtext_col].fillna("").astype(str).tolist()
    if append_extra_description and extra_col and extra_col != qtext_col:
        extras = df[extra_col].fillna("").astype(str).tolist()
        texts = [f"{q} {e}".strip() for q, e in zip(texts, extras)]

    id2idx = corpus.id_to_index
    qrels, missing = [], 0
    for cell in df[gold_col]:
        idxs = set()
        for aid in _parse_gold(cell):
            if aid in id2idx:
                idxs.add(id2idx[aid])
            else:
                missing += 1
        qrels.append(idxs)
    if missing:
        print(f"[warn] {missing} gold article identifiers are absent from the "
              f"corpus ({path.name}); check that the files match.")

    keep = [i for i, s in enumerate(qrels) if s]
    if len(keep) < len(qrels):
        print(f"[warn] {len(qrels) - len(keep)} questions without a valid gold "
              "article were discarded.")

    return QuerySet(
        ids=(df[qid_col].to_numpy()[keep] if qid_col
             else np.arange(len(df))[keep]),
        texts=[texts[i] for i in keep],
        qrels=[qrels[i] for i in keep],
        categories=([str(df[cat_col].iloc[i]) for i in keep]
                    if cat_col else None),
    )


def load_bsard(
    data_dir: str | Path,
    append_extra_description: bool = False,
    include_description_in_text: bool = False,
) -> tuple[Corpus, QuerySet, QuerySet]:
    corpus = load_corpus(
        data_dir, include_description_in_text=include_description_in_text)
    train = load_questions(data_dir, "train", corpus,
                           append_extra_description=append_extra_description)
    test = load_questions(data_dir, "test", corpus,
                          append_extra_description=append_extra_description)
    return corpus, train, test
