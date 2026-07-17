"""
minion/memory/store.py

SQLite-backed memory store with FTS5 full-text search.

Schema:
  memories      — the canonical table, one row per memory
  memories_fts  — FTS5 virtual table shadowing content + tags for fast recall

All I/O is synchronous (sqlite3 is not async-capable). The MemoryManager
runs this in a thread executor when called from async context.

The DB file lives at config.db_path (~/.minion/minion.db) — a single
inspectable file you can open with any SQLite browser.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from minion.memory.models import Memory, MemoryType

# FTS5 rank weight: content matters more than tags
_FTS_QUERY = """
    SELECT m.id, m.type, m.content, m.tags, m.created_at, m.updated_at, m.recalled_count
    FROM memories m
    JOIN memories_fts f ON m.id = f.rowid
    WHERE memories_fts MATCH ?
    ORDER BY rank
    LIMIT ?
"""


class MemoryStore:
    """
    Low-level SQLite store. Operates synchronously — callers that need async
    should wrap calls with asyncio.to_thread().
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")  # better concurrency
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                type          TEXT    NOT NULL,
                content       TEXT    NOT NULL,
                tags          TEXT    NOT NULL DEFAULT '[]',
                created_at    TEXT    NOT NULL,
                updated_at    TEXT    NOT NULL,
                recalled_count INTEGER NOT NULL DEFAULT 0,
                deleted       INTEGER NOT NULL DEFAULT 0
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content,
                tags,
                content='memories',
                content_rowid='id'
            );

            CREATE TRIGGER IF NOT EXISTS memories_ai
            AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content, tags)
                VALUES (new.id, new.content, new.tags);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_au
            AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, tags)
                VALUES ('delete', old.id, old.content, old.tags);
                INSERT INTO memories_fts(rowid, content, tags)
                VALUES (new.id, new.content, new.tags);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_ad
            AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, tags)
                VALUES ('delete', old.id, old.content, old.tags);
            END;
        """)
        self._conn.commit()

    # ── Write operations ──────────────────────────────────────────────────

    def insert(self, memory: Memory) -> Memory:
        """Persist a new memory. Returns the memory with id populated."""
        now = datetime.now(timezone.utc).isoformat()
        tags_json = json.dumps(memory.tags)
        cur = self._conn.execute(
            """
            INSERT INTO memories (type, content, tags, created_at, updated_at, recalled_count)
            VALUES (?, ?, ?, ?, ?, 0)
            """,
            (memory.type.value, memory.content, tags_json, now, now),
        )
        self._conn.commit()
        return memory.model_copy(update={"id": cur.lastrowid})

    def update(self, memory_id: int, content: str) -> bool:
        """Update the content of an existing memory. Returns True if found."""
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            "UPDATE memories SET content = ?, updated_at = ? WHERE id = ? AND deleted = 0",
            (content, now, memory_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete(self, memory_id: int) -> bool:
        """Soft-delete a memory. Returns True if found."""
        cur = self._conn.execute(
            "UPDATE memories SET deleted = 1 WHERE id = ? AND deleted = 0",
            (memory_id,),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def increment_recalled(self, memory_id: int) -> None:
        self._conn.execute(
            "UPDATE memories SET recalled_count = recalled_count + 1 WHERE id = ?",
            (memory_id,),
        )
        self._conn.commit()

    # ── Read operations ───────────────────────────────────────────────────

    def search(self, query: str, limit: int = 10) -> list[Memory]:
        """FTS5 full-text search over content and tags."""
        # Sanitise query: FTS5 is strict about syntax
        safe_query = self._sanitise_fts_query(query)
        try:
            rows = self._conn.execute(_FTS_QUERY, (safe_query, limit)).fetchall()
        except sqlite3.OperationalError:
            # Fallback to LIKE search if FTS query is malformed
            rows = self._conn.execute(
                """
                SELECT id, type, content, tags, created_at, updated_at, recalled_count
                FROM memories
                WHERE deleted = 0 AND (content LIKE ? OR tags LIKE ?)
                LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_all(self, memory_type: MemoryType | None = None) -> list[Memory]:
        if memory_type:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE deleted = 0 AND type = ? ORDER BY updated_at DESC",
                (memory_type.value,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE deleted = 0 ORDER BY updated_at DESC"
            ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE deleted = 0"
        ).fetchone()
        return row[0]

    def get_by_id(self, memory_id: int) -> Memory | None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ? AND deleted = 0", (memory_id,)
        ).fetchone()
        return self._row_to_memory(row) if row else None

    # ── Helpers ───────────────────────────────────────────────────────────

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            type=MemoryType(row["type"]),
            content=row["content"],
            tags=json.loads(row["tags"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            recalled_count=row["recalled_count"],
        )

    @staticmethod
    def _sanitise_fts_query(query: str) -> str:
        """
        Wrap each word with prefix matching (word*) so partial terms match.
        Quoted to avoid FTS5 syntax errors from special characters.
        """
        words = query.split()
        return " OR ".join(f'"{w}"*' for w in words if w)

    def close(self) -> None:
        self._conn.close()
