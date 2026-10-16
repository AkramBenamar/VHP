from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

FRENCH_STOPWORDS = set("""
au aux avec ce ces dans de des du elle en et eux il je la le les leur lui ma
mais me meme mes moi mon ne nos notre nous on ou par pas pour qu que qui sa se
ses son sur ta te tes toi ton tu un une vos votre vous c d j l a m n s t y ete
etee etees etes etant suis es est sommes etes sont serai seras sera serons
serez seront avoir avais avait avons avez ont ai as a aura aurons plus tres
lorsque lorsqu dont ainsi cette cet celui celle ceux alors donc toute tous
toutes autre autres meme aussi entre chaque leurs
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[/-][a-z0-9]+)*")


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def _get_stemmer():
    try:
        from nltk.stem.snowball import FrenchStemmer
        return FrenchStemmer()
    except Exception:
        return None


def tokenize_fr(text: str, stemmer=None, remove_stopwords: bool = True,
                remove_numbers: bool = False) -> list[str]:
    toks = _TOKEN_RE.findall(_strip_accents(str(text).lower()))
    if remove_numbers:
        toks = [t for t in toks if not any(c.isdigit() for c in t)]
    if remove_stopwords:
        toks = [t for t in toks if t not in FRENCH_STOPWORDS and len(t) > 1]
    if stemmer is not None:
        toks = [stemmer.stem(t) for t in toks]
    return toks


@dataclass
class BM25Config:
    k1: float = 0.9
    b: float = 0.4
    use_stemming: bool = True
    remove_stopwords: bool = True
    remove_numbers: bool = True
    idf_scheme: str = "bsard"
    min_df: int = 1


class BM25Retriever:

    name = "BM25"

    def __init__(self, config: BM25Config | None = None):
        self.cfg = config or BM25Config()
        self.stemmer = _get_stemmer() if self.cfg.use_stemming else None
        if self.cfg.use_stemming and self.stemmer is None:
            print("[bm25] nltk unavailable: stemming disabled "
                  "(pip install nltk to reproduce the reported results).")
        self.vocab: dict[str, int] = {}
        self.weights: sp.csr_matrix | None = None

    def _tok(self, text: str) -> list[str]:
        return tokenize_fr(text, self.stemmer, self.cfg.remove_stopwords,
                           self.cfg.remove_numbers)

    def index(self, texts: list[str]) -> "BM25Retriever":
        docs = [self._tok(t) for t in texts]
        vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        for i, doc in enumerate(docs):
            counts: dict[int, int] = {}
            for tok in doc:
                j = vocab.setdefault(tok, len(vocab))
                counts[j] = counts.get(j, 0) + 1
            rows.extend([i] * len(counts))
            cols.extend(counts.keys())
            vals.extend(counts.values())

        n, v = len(docs), len(vocab)
        tf = sp.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)),
                           shape=(n, v))
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = dl.mean() if n else 1.0
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        if self.cfg.idf_scheme == "bsard":
            idf = np.log10((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        elif self.cfg.idf_scheme == "lucene":
            idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        else:
            raise ValueError(f"unknown idf_scheme: {self.cfg.idf_scheme}")

        k1, b = self.cfg.k1, self.cfg.b
        norm = (k1 * (1 - b + b * dl / max(avgdl, 1e-9))).astype(np.float32)

        w = tf.tocoo()
        denom = w.data + norm[w.row]
        data = idf[w.col] * w.data * (k1 + 1.0) / np.maximum(denom, 1e-9)
        self.weights = sp.csr_matrix((data.astype(np.float32), (w.row, w.col)),
                                     shape=(n, v))
        self.vocab = vocab
        return self

    def score(self, queries: list[str]) -> np.ndarray:
        assert self.weights is not None, "index() must be called first"
        v = len(self.vocab)
        rows, cols, vals = [], [], []
        for i, q in enumerate(queries):
            counts: dict[int, int] = {}
            for tok in self._tok(q):
                j = self.vocab.get(tok)
                if j is not None:
                    counts[j] = counts.get(j, 0) + 1
            rows.extend([i] * len(counts))
            cols.extend(counts.keys())
            vals.extend(counts.values())
        qm = sp.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)),
                           shape=(len(queries), v))
        return np.asarray((qm @ self.weights.T).todense(), dtype=np.float32)


@dataclass
class DenseConfig:
    model_name: str = "intfloat/multilingual-e5-base"
    query_prefix: str = "query: "
    doc_prefix: str = "passage: "
    batch_size: int = 32
    max_seq_length: int = 512
    normalize: bool = True
    device: str | None = None


PREFIX_PRESETS = {
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "BAAI/bge-m3": ("", ""),
    "OrdalieTech/Solon-embeddings-large-0.1": ("query: ", "passage: "),
    "antoinelouis/biencoder-camembert-base-mmarcoFR": ("", ""),
    "dangvantuan/sentence-camembert-large": ("", ""),
}


class DenseRetriever:

    def __init__(self, config: DenseConfig | None = None):
        self.cfg = config or DenseConfig()
        if self.cfg.model_name in PREFIX_PRESETS:
            self.cfg.query_prefix, self.cfg.doc_prefix = PREFIX_PRESETS[self.cfg.model_name]
        self.name = f"Dense({self.cfg.model_name.split('/')[-1]})"
        self._model = None
        self.doc_emb: np.ndarray | None = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "pip install sentence-transformers torch for dense retrieval"
                ) from exc
            self._model = SentenceTransformer(self.cfg.model_name,
                                              device=self.cfg.device)
            self._model.max_seq_length = self.cfg.max_seq_length
        return self._model

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        return self.model.encode(
            [prefix + t for t in texts],
            batch_size=self.cfg.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.cfg.normalize,
            show_progress_bar=True,
        ).astype(np.float32)

    def index(self, texts: list[str]) -> "DenseRetriever":
        self.doc_emb = self._encode(texts, self.cfg.doc_prefix)
        return self

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self._encode(queries, self.cfg.query_prefix)

    def score(self, queries: list[str]) -> np.ndarray:
        assert self.doc_emb is not None, "index() must be called first"
        return self.encode_queries(queries) @ self.doc_emb.T


class TfidfSvdRetriever:

    name = "TFIDF-SVD"

    def __init__(self, n_components: int = 256, random_state: int = 0):
        self.n_components = n_components
        self.random_state = random_state
        self.vec = None
        self.svd = None
        self.doc_emb: np.ndarray | None = None

    def index(self, texts: list[str]) -> "TfidfSvdRetriever":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(
            tokenizer=lambda t: tokenize_fr(t, None, True),
            lowercase=False, token_pattern=None, min_df=1,
        )
        x = self.vec.fit_transform(texts)
        k = min(self.n_components, min(x.shape) - 1)
        self.svd = TruncatedSVD(n_components=max(k, 2),
                                random_state=self.random_state)
        emb = self.svd.fit_transform(x)
        self.doc_emb = _l2(emb.astype(np.float32))
        return self

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return _l2(self.svd.transform(
            self.vec.transform(queries)).astype(np.float32))

    def score(self, queries: list[str]) -> np.ndarray:
        return self.encode_queries(queries) @ self.doc_emb.T


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-9)


def zfuse(scores_a: np.ndarray, scores_b: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    from .propagate import zscore_rows
    return (1 - alpha) * zscore_rows(scores_a) + alpha * zscore_rows(scores_b)


def rrf(score_list: list[np.ndarray], k: int = 60) -> np.ndarray:
    out = np.zeros_like(score_list[0], dtype=np.float32)
    for s in score_list:
        order = np.argsort(-s, axis=1, kind="stable")
        ranks = np.empty_like(order)
        rows = np.arange(s.shape[0])[:, None]
        ranks[rows, order] = np.arange(s.shape[1])[None, :]
        out += (1.0 / (k + ranks + 1)).astype(np.float32)
    return out


class HybridRetriever:
    def __init__(self, a, b, alpha: float = 0.5, mode: str = "zfuse"):
        self.a, self.b, self.alpha, self.mode = a, b, alpha, mode
        self.name = f"Hybrid({a.name}+{b.name})"

    def index(self, texts: list[str]) -> "HybridRetriever":
        self.a.index(texts)
        self.b.index(texts)
        return self

    def score(self, queries: list[str]) -> np.ndarray:
        sa, sb = self.a.score(queries), self.b.score(queries)
        return (zfuse(sa, sb, self.alpha) if self.mode == "zfuse"
                else rrf([sa, sb]))
