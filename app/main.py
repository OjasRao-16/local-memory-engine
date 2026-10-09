import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from .config import DB_PATH, DEFAULT_RELEVANCE_THRESHOLD
from .schemas import MemoryRequest, RecallRequest
from .storage import SQLiteMemoryStore
from .embeddings import build_embedder
from .memory_manager import MemoryManager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model loads locally; first run downloads it if it is not cached.
    # MEMORY_EMBEDDER: st (default, sentence-transformers) | ollama | hash (tests)
    store = SQLiteMemoryStore(os.getenv("MEMORY_DB", DB_PATH))
    embedder = build_embedder(os.getenv("MEMORY_EMBEDDER", "st"))
    app.state.manager = MemoryManager(store, embedder, DEFAULT_RELEVANCE_THRESHOLD)
    yield

app = FastAPI(title="Local Chatbot Memory Engine", version="1.0.0", lifespan=lifespan)

def manager():
    if not hasattr(app.state, "manager"):
        raise HTTPException(status_code=503, detail="Memory engine is still initializing.")
    return app.state.manager

@app.get("/")
def root():
    return {"name": "Local Chatbot Memory Engine", "docs": "/docs"}

@app.post("/memory")
def add_memory(request: MemoryRequest):
    return manager().ingest(request.message)

@app.post("/recall")
def recall(request: RecallRequest):
    memories = manager().recall(request.query, request.top_k, request.threshold)
    return {"memories": [
        {k: m[k] for k in ("id", "text", "type", "timestamp", "source_message", "status", "score")}
        for m in memories
    ]}

@app.get("/memories")
def list_memories(include_replaced: bool = True):
    rows = manager().store.list_all(include_replaced=include_replaced)
    for row in rows:
        row.pop("embedding", None)
        row.pop("slot", None)
    return {"memories": rows}

@app.delete("/memory/{memory_id}")
def delete_memory(memory_id: int):
    deleted = manager().store.delete(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"deleted": True, "id": memory_id}
