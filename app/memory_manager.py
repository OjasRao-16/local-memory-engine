from .ingestion import extract_memories
from .retrieval import MemoryRetriever

class MemoryManager:
    def __init__(self, store, embedder, threshold: float = 0.25):
        self.store = store
        self.embedder = embedder
        self.retriever = MemoryRetriever(store, embedder)
        self.threshold = threshold

    def ingest(self, message: str):
        candidates = extract_memories(message)
        if not candidates:
            return {"stored": [], "ignored": True,
                    "message": "No useful memory detected; message ignored."}
        stored = []
        for candidate in candidates:
            old = self.store.find_active_slot(candidate.slot) if candidate.slot else None
            # Avoid duplicating an identical active memory, including candidates
            # that do not yet have a recognized contradiction slot.
            duplicate = old if old and old["text"].casefold() == candidate.text.casefold() else None
            if not duplicate:
                duplicate = next((m for m in self.store.active()
                                  if m["text"].casefold() == candidate.text.casefold()), None)
            if duplicate:
                stored.append({k: duplicate[k] for k in ("id", "text", "type", "timestamp", "status")})
                continue
            vector = self.embedder.encode([candidate.text])[0].tolist()
            near = self._near_duplicate(vector, candidate) if not old else None
            if near:                      # merge: same meaning already stored
                stored.append({k: near[k] for k in ("id", "text", "type", "timestamp", "status")})
                continue
            new = self.store.add(candidate.text, candidate.memory_type, message, vector, candidate.slot)
            if old:
                self.store.mark_replaced(old["id"], new["id"])
            stored.append({k: new[k] for k in ("id", "text", "type", "timestamp", "status")})
        return {"stored": stored, "ignored": False, "message": "Memory processed."}

    DUP_COSINE = 0.97

    def _near_duplicate(self, vector, candidate):
        """An active memory of the same type whose embedding is almost identical."""
        import numpy as np
        v = np.asarray(vector, dtype="float32")
        for m in self.store.active():
            if m["type"] != candidate.memory_type:
                continue
            u = np.asarray(m["embedding"], dtype="float32")
            d = float(np.linalg.norm(u) * np.linalg.norm(v))
            if d and float(u @ v) / d >= self.DUP_COSINE:
                return m
        return None

    def recall(self, query: str, top_k: int = 5, threshold: float | None = None):
        return self.retriever.recall(query, top_k, self.threshold if threshold is None else threshold)
