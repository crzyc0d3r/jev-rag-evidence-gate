"""Stage 1: broad, recall-oriented candidate retrieval.

Two independent retrievers rank the corpus and reciprocal rank fusion (RRF)
merges their ranked lists. Everything here is standard library only so the
pipeline runs anywhere; swap `HashedDenseEncoder` for a real embedding model
(e.g. sentence-transformers) by implementing the same `encode()` method.
"""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from dataclasses import dataclass

STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i in is it its
    of on or so that the this to what when where which who why will with without
    my me we you your our after before than then there their them they into""".split()
)


@dataclass(frozen=True)
class Passage:
    id: str
    title: str
    text: str

    @property
    def content(self) -> str:
        return f"{self.title}. {self.text}"


def stem(token: str) -> str:
    """Tiny suffix stripper: good enough to match 'rotate'/'rotation', 'objects'/'object'."""
    for suffix in ("ations", "ation", "ing", "ed", "es", "s", "e"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [stem(w) for w in words if w not in STOPWORDS]


class BM25:
    """Okapi BM25 keyword retriever."""

    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(p.content)) for p in passages]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)
        df = Counter(term for d in self.docs for term in d)
        n = len(passages)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        terms = tokenize(query)
        scores = []
        for p, doc, length in zip(self.passages, self.docs, self.lengths):
            s = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                    s += self.idf.get(t, 0.0) * tf * (self.k1 + 1) / norm
            scores.append((p.id, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]


class HashedDenseEncoder:
    """Stand-in dense encoder: hashed word + character-trigram features, L2-normalised.

    It captures fuzzy lexical similarity (a crude proxy for embedding search).
    Replace with a real embedding model for production use.
    """

    def __init__(self, dim: int = 2048):
        self.dim = dim

    def encode(self, text: str) -> dict[int, float]:
        vec: dict[int, float] = {}
        for tok in tokenize(text):
            feats = [f"w:{tok}"] + [f"c:{tok[i:i + 3]}" for i in range(max(len(tok) - 2, 1))]
            for f in feats:
                h = zlib.crc32(f.encode()) % self.dim
                vec[h] = vec.get(h, 0.0) + (1.0 if f.startswith("w:") else 0.5)
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {k: v / norm for k, v in vec.items()}


class DenseRetriever:
    def __init__(self, passages: list[Passage], encoder: HashedDenseEncoder | None = None):
        self.passages = passages
        self.encoder = encoder or HashedDenseEncoder()
        self.vectors = [self.encoder.encode(p.content) for p in passages]

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = self.encoder.encode(query)
        scores = [
            (p.id, sum(w * v.get(i, 0.0) for i, w in q.items()))
            for p, v in zip(self.passages, self.vectors)
        ]
        scores.sort(key=lambda x: -x[1])
        return scores[:k]


def reciprocal_rank_fusion(rankings: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    """Merge ranked lists: score(d) = sum over lists of 1 / (k + rank(d))."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


class HybridRetriever:
    """BM25 + dense search fused with RRF. Optimised for recall, not precision."""

    def __init__(self, passages: list[Passage]):
        self.by_id = {p.id: p for p in passages}
        self.bm25 = BM25(passages)
        self.dense = DenseRetriever(passages)

    def retrieve(self, query: str, k: int = 10, per_retriever: int = 20) -> list[Passage]:
        fused = reciprocal_rank_fusion(
            [self.bm25.search(query, per_retriever), self.dense.search(query, per_retriever)]
        )
        return [self.by_id[doc_id] for doc_id, _ in fused[:k]]
