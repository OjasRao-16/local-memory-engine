from functools import lru_cache
import numpy as np

class SentenceEmbedder:
    """Local sentence embeddings; model downloads once on first run."""
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False
        )
        return np.asarray(vectors, dtype="float32")

class HashingEmbedder:
    """Deterministic lightweight fallback for tests/offline smoke checks only.
    Do not use this fallback to claim semantic retrieval quality.
    """
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def encode(self, texts: list[str]) -> np.ndarray:
        import hashlib
        out = []
        for text in texts:
            vec = np.zeros(self.dimensions, dtype="float32")
            tokens = text.lower().split()
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dimensions
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm:
                vec /= norm
            out.append(vec)
        return np.asarray(out, dtype="float32")


class OllamaEmbedder:
    """Embeddings from a local Ollama server (e.g. `ollama pull nomic-embed-text`)."""
    def __init__(self, model: str = "nomic-embed-text", host: str = "http://127.0.0.1:11434"):
        self.model, self.host = model, host.rstrip("/")

    def encode(self, texts: list[str]) -> np.ndarray:
        import json
        import urllib.request
        req = urllib.request.Request(
            f"{self.host}/api/embed",
            data=json.dumps({"model": self.model, "input": texts}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            vecs = np.asarray(json.load(resp)["embeddings"], dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1.0, norms)


def build_embedder(kind: str):
    """kind: 'st' (sentence-transformers), 'ollama', or 'hash' (tests only)."""
    from .config import MODEL_NAME
    if kind == "st":
        return SentenceEmbedder(MODEL_NAME)
    if kind == "ollama":
        return OllamaEmbedder()
    if kind == "hash":
        return HashingEmbedder()
    raise ValueError(f"unknown embedder {kind!r}")
