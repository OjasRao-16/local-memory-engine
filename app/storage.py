import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

class SQLiteMemoryStore:
    """Persistent memory metadata and vectors. SQLite is the source of truth."""
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def initialize(self):
        with self.connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source_message TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active'
                        CHECK(status IN ('active', 'replaced')),
                    replaced_by INTEGER NULL,
                    embedding TEXT NOT NULL,
                    slot TEXT NULL,
                    FOREIGN KEY(replaced_by) REFERENCES memories(id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_slot ON memories(slot)")

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat()

    def add(self, text: str, memory_type: str, source_message: str,
            embedding: list[float], slot: str | None = None) -> dict[str, Any]:
        with self.connect() as conn:
            cur = conn.execute(
                """INSERT INTO memories(text,type,timestamp,source_message,status,embedding,slot)
                   VALUES(?,?,?,?, 'active', ?, ?)""",
                (text, memory_type, self.now(), source_message,
                 json.dumps([float(x) for x in embedding]), slot))
            row = conn.execute("SELECT * FROM memories WHERE id=?", (cur.lastrowid,)).fetchone()
            return self._row(row)

    def get(self, memory_id: int):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            return self._row(row) if row else None

    def list_all(self, include_replaced: bool = True):
        sql = "SELECT * FROM memories"
        if not include_replaced:
            sql += " WHERE status='active'"
        sql += " ORDER BY id"
        with self.connect() as conn:
            return [self._row(r) for r in conn.execute(sql).fetchall()]

    def active(self):
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM memories WHERE status='active' ORDER BY id").fetchall()
            return [self._row(r) for r in rows]

    def find_active_slot(self, slot: str):
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE slot=? AND status='active' ORDER BY id DESC LIMIT 1",
                (slot,)).fetchone()
            return self._row(row) if row else None

    def mark_replaced(self, old_id: int, new_id: int):
        with self.connect() as conn:
            conn.execute("UPDATE memories SET status='replaced', replaced_by=? WHERE id=?",
                         (new_id, old_id))

    def delete(self, memory_id: int) -> bool:
        with self.connect() as conn:
            row = conn.execute("SELECT id FROM memories WHERE id=?", (memory_id,)).fetchone()
            if not row:
                return False
            # Preserve referential integrity when deleting a replacement target.
            conn.execute("UPDATE memories SET replaced_by=NULL WHERE replaced_by=?", (memory_id,))
            conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            return True

    @staticmethod
    def _row(row):
        if row is None:
            return None
        item = dict(row)
        item["embedding"] = json.loads(item["embedding"])
        return item
