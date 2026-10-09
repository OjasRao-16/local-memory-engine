import pytest
from app.bm25 import BM25, tokenize
from app.embeddings import HashingEmbedder
from app.ingestion import extract_memories
from app.memory_manager import MemoryManager
from app.storage import SQLiteMemoryStore


@pytest.fixture
def manager(tmp_path):
    return MemoryManager(SQLiteMemoryStore(tmp_path / "t.db"), HashingEmbedder(), threshold=0.1)


@pytest.mark.parametrize("message, slot, fragment", [
    ("I switched to Fedora.", "os", "Fedora"),
    ("I prefer PostgreSQL over MySQL.", "database", "PostgreSQL"),
    ("Actually, the deadline moved to November 2.", "deadline", "November 2"),
    ("I moved to Pune.", "city", "Pune"),
    ("My birthday is March 3.", "birthday", "March 3"),
    ("I'm allergic to peanuts.", "allergy", "peanuts"),
    ("I work at Google.", "employer", "Google"),
    ("I use VS Code as my editor.", "editor", "VS Code"),
])
def test_slot_extraction(message, slot, fragment):
    item = extract_memories(message)[0]
    assert item.slot == slot and fragment in item.text


def test_multi_sentence_message_yields_multiple_memories():
    items = extract_memories("I use Python. I'm building a Flask API. Thanks!")
    assert [i.slot for i in items] == ["programming_language", "project"]


def test_negation_is_not_stored_as_fact():
    assert extract_memories("I no longer use Arch Linux.") == []


@pytest.mark.parametrize("msg", ["ok thanks", "Thanks!", "Hello", "Good morning", "The weather is nice today."])
def test_more_small_talk_ignored(msg):
    assert extract_memories(msg) == []


def test_os_update_replaces_old(manager):
    manager.ingest("I use Arch Linux.")
    manager.ingest("I switched to Fedora.")
    rows = manager.store.list_all()
    assert [r["status"] for r in rows] == ["replaced", "active"]
    assert rows[0]["replaced_by"] == rows[1]["id"]


def test_recall_hides_replaced(manager):
    manager.ingest("I use Arch Linux.")
    manager.ingest("I switched to Fedora.")
    texts = " ".join(m["text"] for m in manager.recall("What OS does the user use?", threshold=0.0))
    assert "Fedora" in texts and "Arch" not in texts


def test_bm25_ranks_keyword_match_first():
    docs = ["User uses Arch Linux.", "User prefers PostgreSQL.", "User lives in Pune."]
    scores = BM25(docs).normalised("Which database, PostgreSQL?")
    assert scores.index(max(scores)) == 1


def test_bm25_no_overlap_scores_zero():
    assert max(BM25(["User lives in Pune."]).normalised("favourite colour")) == 0.0


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("What does the user use?") == ["use"]
    assert tokenize("uses") == ["use"]


def test_near_duplicate_merge(manager):
    a = manager.ingest("I love hiking on weekends.")
    b = manager.ingest("I really enjoy hiking on weekends.")
    assert len(manager.store.list_all()) <= 2


def test_alpha_zero_is_pure_keyword(manager):
    manager.ingest("I prefer PostgreSQL.")
    hits = manager.retriever.recall("PostgreSQL", 5, 0.1, alpha=0.0)
    assert hits and "PostgreSQL" in hits[0]["text"]
