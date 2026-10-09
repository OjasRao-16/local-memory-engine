# Local Chatbot Memory Engine

A small, locally runnable memory engine for AI chatbots. It extracts useful facts from conversation messages, stores them with timestamps and source messages, updates supported facts when they change, and retrieves relevant memories for a query.

- **No paid API required.**
- **Semantic embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`) or an optional Ollama embedder.
- **Persistence:** SQLite.
- **Retrieval:** embedding similarity combined with BM25 keyword search.

## Architecture

```text
POST /memory
    └── app/ingestion.py       Rule-based extraction and small-talk filtering
          └── app/memory_manager.py
                ├── slot updates / contradiction handling
                ├── near-duplicate handling
                └── app/storage.py  ── SQLite source of truth

POST /recall
    └── app/retrieval.py
          ├── embedding cosine similarity
          ├── BM25 keyword score
          └── fused relevance score → threshold → top-k memories

GET /memories and DELETE /memory/{id}
    └── app/storage.py
```

| Module | Responsibility |
|---|---|
| `app/ingestion.py` | Rule-based extraction into supported memory slots; filters small talk |
| `app/memory_manager.py` | Memory creation, supported slot updates, and near-duplicate handling |
| `app/storage.py` | SQLite persistence, memory status, and history |
| `app/embeddings.py` | Sentence Transformers, Ollama, and test-only hashing embedder |
| `app/bm25.py` | BM25 keyword retrieval |
| `app/retrieval.py` | Hybrid ranking and relevance threshold |
| `app/main.py` | FastAPI endpoints |
| `eval/` | 38-case evaluation set and evaluation runner |
| `tests/` | Unit and API tests |

## Requirements and setup

Python 3.10 or newer is recommended.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

The first run with Sentence Transformers downloads `all-MiniLM-L6-v2`; after the model is cached, inference can run locally without a paid API. The first download requires internet access.

Open interactive API documentation at <http://127.0.0.1:8000/docs>.

### Configuration

- `MEMORY_EMBEDDER=st` uses Sentence Transformers (default).
- `MEMORY_EMBEDDER=ollama` uses a local Ollama embedding service.
- `MEMORY_EMBEDDER=hash` uses a lightweight lexical/hash baseline for smoke tests; it is **not** equivalent to semantic embeddings.
- `MEMORY_DB=path/to/db` selects the SQLite database file.

For Ollama, install and start Ollama, pull an embedding model such as `nomic-embed-text`, and set `MEMORY_EMBEDDER=ollama`.

## API examples

Add a memory:

```bash
curl -X POST http://127.0.0.1:8000/memory \
  -H "Content-Type: application/json" \
  -d '{"message":"My project deadline is October 15."}'
```

Add an update:

```bash
curl -X POST http://127.0.0.1:8000/memory \
  -H "Content-Type: application/json" \
  -d '{"message":"Actually, my project deadline moved to November 2."}'
```

Recall memories:

```bash
curl -X POST http://127.0.0.1:8000/recall \
  -H "Content-Type: application/json" \
  -d '{"query":"When is my project deadline?","top_k":5}'
```

List memories (including replaced history by default):

```bash
curl "http://127.0.0.1:8000/memories"
curl "http://127.0.0.1:8000/memories?include_replaced=false"
```

Delete a memory by ID:

```bash
curl -X DELETE http://127.0.0.1:8000/memory/1
```

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Basic service response |
| `POST` | `/memory` | Ingest a message and store extracted memories |
| `POST` | `/recall` | Retrieve relevant memories for a query |
| `GET` | `/memories` | List stored memories; supports `include_replaced` |
| `DELETE` | `/memory/{memory_id}` | Delete a memory |

## Evaluation

The evaluation set contains **38 cases** covering direct facts, preferences, project details, updates, contradictions, queries with multiple relevant memories, and unknown queries. The evaluator measures retrieval accuracy, Recall@5, unknown-query rejection, and latency percentiles. Each case is evaluated against a fresh database.

Run the evaluation with the real embedding model and run the test suite:

```bash
python -m eval.run_eval --embedder st --sweep --out eval/results_st.json
python -m pytest -q
```

### Sentence Transformers results

The following values were observed in one run of the 38-case evaluation. Results are specific to this test set and environment; they are not a guarantee of performance on unseen conversations.

| Threshold | Accuracy | Recall@5 | Unknown-query rejection | p50 latency | p95 latency |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.842 (84.2%) | 0.857 (85.7%) | 0.714 (71.4%) | 24.0 ms | 31.43 ms |
| **0.25** | **0.842 (84.2%)** | 0.771 (77.1%) | **1.000 (100%)** | 18.67 ms | 27.34 ms |
| 0.30 | 0.763 (76.3%) | 0.686 (68.6%) | 1.000 (100%) | 18.5 ms | 27.56 ms |
| 0.35 | 0.763 (76.3%) | 0.686 (68.6%) | 1.000 (100%) | 24.79 ms | 30.33 ms |
| 0.40 | 0.684 (68.4%) | 0.571 (57.1%) | 1.000 (100%) | 18.66 ms | 27.37 ms |
| 0.45 | 0.684 (68.4%) | 0.571 (57.1%) | 1.000 (100%) | 18.54 ms | 22.09 ms |
| 0.50 | 0.658 (65.8%) | 0.514 (51.4%) | 1.000 (100%) | 24.49 ms | 31.89 ms |

**Provisional threshold choice: `0.25`.** In this evaluation it matched the highest observed accuracy (0.842), achieved 100% unknown-query rejection, and had lower measured latency than the 0.20 run. Its Recall@5 (0.771) was lower than at 0.20 (0.857), so this choice prioritizes rejecting unknown queries without reducing measured accuracy. Revisit the choice if the evaluation set or desired trade-off changes.

The application default threshold is `0.25`, matching the selected evaluation setting. Re-run the tests and evaluation after any configuration change.

For comparison, the `hash` embedder is a lexical/hash baseline, **not semantic search**; do not present its metrics as Sentence Transformers results.

## Design decisions

**Why SQLite?** It is a single-file database, needs no separate server, survives restarts, and can retain replaced memories as history. Stored vectors and memory metadata stay together in one source of truth.

**Why hybrid retrieval?** Embedding similarity can help retrieve paraphrases, while BM25 can help with exact terms such as names and versions. The current fusion is described in the code as `0.7 * cosine + 0.3 * BM25`; check the implementation if you change either component.

**What becomes a memory?** Extraction is rule-based and supports only the slots and phrasing patterns implemented in `app/ingestion.py`. It filters some small talk and may preserve generic first-person statements as slot-less facts.

**How are updates handled?** For supported single-valued slots, a new value can become active while the prior value is marked replaced and linked through `replaced_by`. Replaced memories should remain visible in the memory list but should not be returned as active recall results. Near-duplicate merging is also implemented.

**What happens when no relevant memory is stored?** Retrieval uses a relevance threshold and can return `{"memories": []}` when no candidate clears it.

**What would need work at very large scale?** The current retrieval approach scans active memories and computes scores, which is appropriate for a small student project but will become expensive at very large scale. An approximate-nearest-neighbour index, persistent keyword index, batched embedding, and per-user partitioning would be possible next steps.

## Limitations

- Extraction is rule-based, so unseen phrasings can be missed.
- Only supported slots can be automatically updated or contradicted.
- Negations may not deactivate an earlier memory correctly.
- The current design is single-user and does not provide per-user isolation.
- Retrieval and evaluation are small-scale; 38 cases are not enough to establish general performance.
- There is no web UI or Docker setup.

## Implemented extensions

- Hybrid retrieval using embeddings and a BM25 implementation.
- Near-duplicate handling.
- Evaluation latency reporting (p50 and p95).
- Optional Ollama embedder.

## LLM assistance disclosure

This project was developed with AI assistance during planning, implementation, testing, and documentation. Review this disclosure and edit it to accurately describe the tools used and the parts you personally wrote, tested, and understand. Do not claim authorship of code you did not write, and be prepared to explain the design, implementation, evaluation, and limitations during the interview.
