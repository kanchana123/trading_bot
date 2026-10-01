import math
import re
from typing import Dict, List, Optional, Sequence

import numpy as np

from rag.corpus import SEED_FILINGS


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


class HybridRetriever:
    """Lexical BM25 plus hashed bag-of-words cosine (pgvector stand-in)."""

    def __init__(self, documents: Optional[Sequence[Dict]] = None, dim: int = 256):
        self.dim = dim
        self.documents: List[Dict] = list(documents if documents is not None else SEED_FILINGS)
        self._rebuild()

    def _rebuild(self):
        self._tokenized = [_tokenize(d["text"]) for d in self.documents]
        self.n = len(self.documents)
        df: Dict[str, int] = {}
        for tokens in self._tokenized:
            for tok in set(tokens):
                df[tok] = df.get(tok, 0) + 1
        self.df = df
        self.avgdl = (
            sum(len(t) for t in self._tokenized) / self.n if self.n else 1.0
        )
        self._vectors = np.vstack(
            [self._embed(tokens) for tokens in self._tokenized]
        ) if self.n else np.zeros((0, self.dim))

    def ingest(self, document: Dict):
        self.documents.append(document)
        self._rebuild()

    def _embed(self, tokens: Sequence[str]) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=float)
        for tok in tokens:
            vec[hash(tok) % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        if norm:
            vec /= norm
        return vec

    def _bm25(self, query_tokens: List[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        scores = np.zeros(self.n)
        n = max(self.n, 1)
        for i, doc_tokens in enumerate(self._tokenized):
            tf: Dict[str, int] = {}
            for tok in doc_tokens:
                tf[tok] = tf.get(tok, 0) + 1
            dl = len(doc_tokens) or 1
            score = 0.0
            for q in query_tokens:
                f = tf.get(q, 0)
                if not f:
                    continue
                n_q = self.df.get(q, 0)
                idf = math.log(1 + (n - n_q + 0.5) / (n_q + 0.5))
                denom = f + k1 * (1 - b + b * dl / self.avgdl)
                score += idf * (f * (k1 + 1)) / denom
            scores[i] = score
        return scores

    def search(self, query: str, symbol: Optional[str] = None, top_k: int = 5) -> List[Dict]:
        q_tokens = _tokenize(query)
        if not self.n or not q_tokens:
            return []
        bm25 = self._bm25(q_tokens)
        q_vec = self._embed(q_tokens)
        dense = self._vectors @ q_vec
        if bm25.max() > 0:
            bm25 = bm25 / (bm25.max() + 1e-9)
        dense = (dense + 1) / 2
        hybrid = 0.6 * bm25 + 0.4 * dense
        ranked = np.argsort(-hybrid)
        out: List[Dict] = []
        for idx in ranked:
            doc = dict(self.documents[idx])
            if symbol and doc.get("symbol") != symbol:
                continue
            doc["score"] = float(hybrid[idx])
            out.append(doc)
            if len(out) >= top_k:
                break
        return out
