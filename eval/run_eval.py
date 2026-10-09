"""Evaluate the memory engine on eval/cases.json.

Usage:  python -m eval.run_eval --embedder st            # real semantic model
        python -m eval.run_eval --embedder hash --sweep  # offline smoke run
Each case runs against a fresh temporary database.
"""
import argparse
import json
import statistics
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from app.embeddings import build_embedder
from app.memory_manager import MemoryManager
from app.storage import SQLiteMemoryStore

CASES = Path(__file__).with_name("cases.json")


def run_case(case, embedder, threshold, top_k, alpha):
    with tempfile.TemporaryDirectory() as tmp:
        mgr = MemoryManager(SQLiteMemoryStore(Path(tmp) / "eval.db"), embedder, threshold)
        for msg in case["messages"]:
            mgr.ingest(msg)
        t0 = time.perf_counter()
        res = mgr.retriever.recall(case["query"], top_k, threshold, alpha)
        latency = (time.perf_counter() - t0) * 1000
    blob = " | ".join(m["text"] for m in res).casefold()
    if case.get("expect_none"):
        ok = not res
        found = None
    else:
        found = [e for e in case["expect"] if e.casefold() in blob]
        ok = len(found) == len(case["expect"]) and not any(f.casefold() in blob for f in case.get("forbid", []))
    return ok, found, res, latency


def evaluate(embedder, threshold, top_k, alpha, verbose=False):
    cases = json.loads(CASES.read_text())
    by_cat = defaultdict(lambda: [0, 0])
    lat, hits, expected_total = [], 0, 0
    failures = []
    for c in cases:
        ok, found, res, ms = run_case(c, embedder, threshold, top_k, alpha)
        lat.append(ms)
        by_cat[c["category"]][1] += 1
        by_cat[c["category"]][0] += ok
        if not c.get("expect_none"):
            hits += len(found)
            expected_total += len(c["expect"])
        if not ok:
            failures.append((c["id"], c["query"], [(m["text"], m["score"]) for m in res]))
    n_ok = sum(v[0] for v in by_cat.values())
    unk = by_cat.get("unknown", [0, 0])
    lat.sort()
    out = {
        "threshold": threshold, "alpha": alpha, "top_k": top_k, "cases": len(cases),
        "accuracy": round(n_ok / len(cases), 3),
        f"recall@{top_k}": round(hits / expected_total, 3) if expected_total else None,
        "unknown_rejection": round(unk[0] / unk[1], 3) if unk[1] else None,
        "per_category": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_cat.items())},
        "latency_ms_p50": round(statistics.median(lat), 2),
        "latency_ms_p95": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 2),
        "failures": failures,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="hash", choices=["st", "ollama", "hash"])
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--sweep", action="store_true", help="try several thresholds")
    ap.add_argument("--out", default=None, help="write JSON results here")
    a = ap.parse_args()
    emb = build_embedder(a.embedder)
    thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] if a.sweep else [a.threshold]
    results = []
    for t in thresholds:
        r = evaluate(emb, t, a.top_k, a.alpha)
        results.append(r)
        print(f"embedder={a.embedder} threshold={t} alpha={a.alpha}: accuracy={r['accuracy']} "
              f"recall@{a.top_k}={r[f'recall@{a.top_k}']} unknown_rejection={r['unknown_rejection']} "
              f"p50={r['latency_ms_p50']}ms p95={r['latency_ms_p95']}ms")
        if not a.sweep:
            print("per category:", r["per_category"])
            for f in r["failures"]:
                print("  FAIL", f)
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
