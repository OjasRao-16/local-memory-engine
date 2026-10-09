import numpy as np
import pytest
from app.embeddings import HashingEmbedder
from app.storage import SQLiteMemoryStore
from app.memory_manager import MemoryManager
from app.ingestion import extract_memories

@pytest.fixture
def manager(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "test.db")
    embedder = HashingEmbedder()
    return MemoryManager(store, embedder, threshold=0.15)

@pytest.mark.parametrize("message", [
    "ok", "thanks", "thank you", "cool", "got it", "sure"
])
def test_small_talk_ignored(message):
    assert extract_memories(message) == []

@pytest.mark.parametrize("message, expected_type", [
    ("I prefer Python for programming.", "preference"),
    ("I use PostgreSQL.", "preference"),
    ("I'm building a Next.js app.", "project"),
    ("My goal is to become a better programmer.", "goal"),
    ("My favourite food is dosa.", "preference"),
    ("I study computer science.", "fact"),
    ("I live in Nagpur.", "fact"),
    ("I have a project.", "fact"),
])
def test_useful_messages_extracted(message, expected_type):
    items = extract_memories(message)
    assert items
    assert items[0].memory_type == expected_type

def test_store_and_list(manager):
    result = manager.ingest("I prefer Python for programming.")
    assert result["ignored"] is False
    assert len(manager.store.list_all()) == 1

def test_small_talk_not_stored(manager):
    result = manager.ingest("ok thanks")
    assert result["ignored"] is True
    assert manager.store.list_all() == []

def test_update_preserves_old_memory(manager):
    manager.ingest("My project deadline is October 15.")
    manager.ingest("My project deadline moved to November 2.")
    rows = manager.store.list_all()
    assert len(rows) == 2
    assert rows[0]["status"] == "replaced"
    assert rows[0]["replaced_by"] == rows[1]["id"]
    assert rows[1]["status"] == "active"
    assert "November 2" in rows[1]["text"]

def test_active_only_returns_current(manager):
    manager.ingest("My project deadline is October 15.")
    manager.ingest("My project deadline moved to November 2.")
    assert len(manager.store.list_all(include_replaced=False)) == 1

def test_duplicate_does_not_create_new_row(manager):
    manager.ingest("I prefer Python for programming.")
    manager.ingest("I prefer Python for programming.")
    assert len(manager.store.list_all()) == 1

def test_unknown_query_returns_empty_with_high_threshold(manager):
    manager.ingest("My project deadline is October 15.")
    assert manager.recall("zzzz unrelated mystery query", threshold=0.99) == []

def test_delete_existing_memory(manager):
    result = manager.ingest("I live in Nagpur.")
    memory_id = result["stored"][0]["id"]
    assert manager.store.delete(memory_id) is True
    assert manager.store.get(memory_id) is None

def test_delete_missing_memory(manager):
    assert manager.store.delete(9999) is False

def test_source_message_preserved(manager):
    original = "I live in Nagpur."
    manager.ingest(original)
    assert manager.store.list_all()[0]["source_message"] == original

def test_timestamp_present(manager):
    manager.ingest("I live in Nagpur.")
    assert manager.store.list_all()[0]["timestamp"]

def test_memory_type_preserved(manager):
    manager.ingest("I prefer Python for programming.")
    assert manager.store.list_all()[0]["type"] == "preference"

def test_restart_persistence(tmp_path):
    path = tmp_path / "persist.db"
    store1 = SQLiteMemoryStore(path)
    manager1 = MemoryManager(store1, HashingEmbedder())
    manager1.ingest("I live in Nagpur.")
    store2 = SQLiteMemoryStore(path)
    assert len(store2.list_all()) == 1

def test_recall_returns_score_field(manager):
    manager.ingest("I prefer Python for programming.")
    results = manager.recall("Python programming", threshold=0.0)
    assert results
    assert "score" in results[0]

def test_top_k_limit(manager):
    manager.ingest("I prefer Python for programming.")
    manager.ingest("I live in Nagpur.")
    assert len(manager.recall("Python", top_k=1, threshold=0.0)) <= 1

def test_replaced_memory_not_recalled(manager):
    manager.ingest("My project deadline is October 15.")
    manager.ingest("My project deadline moved to November 2.")
    results = manager.recall("project deadline November 2", threshold=0.0)
    assert all(m["status"] == "active" for m in results)

def test_unrecognized_message_ignored(manager):
    assert manager.ingest("The weather is nice today.")["ignored"] is True

def test_database_preference_extraction():
    item = extract_memories("I prefer PostgreSQL.")[0]
    assert item.slot == "database"
    assert "PostgreSQL" in item.text

def test_os_extraction():
    item = extract_memories("I use Arch Linux.")[0]
    assert item.slot == "os"
    assert "Arch Linux" in item.text

def test_project_slot_update(manager):
    manager.ingest("I'm building a Next.js app.")
    manager.ingest("I'm building a Django app.")
    rows = manager.store.list_all()
    assert rows[0]["status"] == "replaced"
    assert rows[1]["status"] == "active"

def test_independent_slots_do_not_replace(manager):
    manager.ingest("I use Arch Linux.")
    manager.ingest("I prefer PostgreSQL.")
    assert all(m["status"] == "active" for m in manager.store.list_all())

def test_delete_replacement_target_keeps_db_valid(manager):
    manager.ingest("My project deadline is October 15.")
    manager.ingest("My project deadline moved to November 2.")
    newest = manager.store.list_all()[-1]["id"]
    assert manager.store.delete(newest)
    assert manager.store.list_all()[0]["replaced_by"] is None

def test_memory_has_id(manager):
    result = manager.ingest("I live in Nagpur.")
    assert isinstance(result["stored"][0]["id"], int)

def test_message_whitespace_normalized():
    item = extract_memories("  I prefer Python for programming.  ")
    assert item and item[0].text.endswith(".")

def test_case_insensitive_duplicate(manager):
    manager.ingest("I prefer Python for programming.")
    manager.ingest("I PREFER PYTHON FOR PROGRAMMING.")
    assert len(manager.store.list_all()) == 1

def test_recall_empty_database(manager):
    assert manager.recall("anything") == []

def test_replaced_history_visible(manager):
    manager.ingest("My project deadline is October 15.")
    manager.ingest("My project deadline moved to November 2.")
    assert len(manager.store.list_all(include_replaced=True)) == 2
