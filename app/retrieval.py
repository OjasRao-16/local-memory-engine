"""Hybrid retrieval: semantic cosine similarity fused with BM25 keyword score.

final = alpha * cosine + (1 - alpha) * bm25_normalised
Only ACTIVE memories are searched; replaced ones stay in the DB as history.
A memory is returned only if its fused score clears the relevance threshold.
"""
import numpy as np
from .bm25 import BM25
from .storage import SQLiteMemoryStore

DEFAULT_ALPHA = 0.7


class MemoryRetriever:
    def __init__(self, store: SQLiteMemoryStore, embedder, alpha: float = DEFAULT_ALPHA):
        self.store = store
        self.embedder = embedder
        self.alpha = alpha

    def recall(self, query: str, top_k: int = 5, threshold: float = 0.25,
               alpha: float | None = None):
        alpha = self.alpha if alpha is None else alpha
        memories = self.store.active()
        if not memories:
            return []
        q = self.embedder.encode([query])[0]
        vectors = np.asarray([m["embedding"] for m in memories], dtype="float32")
        denom = np.linalg.norm(vectors, axis=1) * (np.linalg.norm(q) or 1.0)
        cos = np.divide(vectors @ q, denom, out=np.zeros(len(vectors)), where=denom != 0)
        cos = np.clip(cos, 0.0, 1.0)
        kw = np.asarray(BM25([m["text"] for m in memories]).normalised(query))
        fused = alpha * cos + (1 - alpha) * kw
        results = []
        for idx in np.argsort(fused)[::-1]:
            score = float(fused[idx])
            if score < threshold:
                break                      # sorted, so everything after is lower too
            m = memories[int(idx)].copy()
            for hidden in ("embedding", "slot"):
                m.pop(hidden, None)
            m["score"] = round(score, 4)
            results.append(m)
            if len(results) >= top_k:
                break
        return results
