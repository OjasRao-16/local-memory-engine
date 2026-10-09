import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_EMBEDDER", "hash")
    monkeypatch.setenv("MEMORY_DB", str(tmp_path / "api.db"))
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_full_api_flow(client):
    r = client.post("/memory", json={"message": "My project deadline is October 15."})
    assert r.status_code == 200 and r.json()["ignored"] is False
    client.post("/memory", json={"message": "Actually, the deadline moved to November 2."})
    rec = client.post("/recall", json={"query": "project deadline", "top_k": 5, "threshold": 0.1}).json()
    assert len(rec["memories"]) == 1 and "November 2" in rec["memories"][0]["text"]
    assert {"text", "type", "score"} <= set(rec["memories"][0])
    allm = client.get("/memories").json()["memories"]
    assert [m["status"] for m in allm] == ["replaced", "active"]
    assert client.get("/memories?include_replaced=false").json()["memories"][0]["status"] == "active"


def test_small_talk_ignored_via_api(client):
    assert client.post("/memory", json={"message": "ok thanks"}).json()["ignored"] is True
    assert client.get("/memories").json()["memories"] == []


def test_delete_and_404(client):
    mid = client.post("/memory", json={"message": "I live in Pune."}).json()["stored"][0]["id"]
    assert client.delete(f"/memory/{mid}").status_code == 200
    assert client.delete(f"/memory/{mid}").status_code == 404


def test_unrelated_query_returns_nothing(client):
    client.post("/memory", json={"message": "I live in Pune."})
    assert client.post("/recall", json={"query": "favourite colour"}).json()["memories"] == []


def test_validation_rejects_empty_message(client):
    assert client.post("/memory", json={"message": ""}).status_code == 422
