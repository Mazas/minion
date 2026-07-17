"""
minion/memory/store.py

SQLite-backed memory store with FTS5 full-text search and vector embeddings.

Schema:
  memories      — canonical table (includes embedding + importance + last_recalled_at)
  memories_fts  — FTS5 virtual table for keyword search

Migration: _migrate_schema() safely adds new columns to existing DBs using
ALTER TABLE ... IF NOT EXISTS equivalent (try/except per column). Safe to run
on any existing minion.db.

All I/O is synchronous (sqlite3 is not async-capable). The MemoryManager
runs calls in asyncio.to_thread() where needed.

TODO: migrate to sqlite-vec for O(1) similarity search when memory count
grows large (hundreds → thousands). Current pure-Python cosine similarity
loads all embeddings into memory — fast for personal use, not for scale.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from minion.memory.models import Memory, MemoryType
from minion.llm.embeddings import cosine_similarity

_FTS_QUERY = """
    SELECT m.id, m.type, m.content, m.tags, m.created_at, m.updated_at,
           m.recalled_count, m.importance, m.last_recalled_at, m.embedding
    FROM memories m
    JOIN memories_fts f ON m.id = f.rowid
    WHERE memories_fts MATCH ?
    ORDER BY rank
    LIMIT ?
"""


class MemoryStore:
    """
    Low-level SQLite store. Operates synchronously.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        self._migrate_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                type            TEXT    NOT NULL,
                content         TEXT    NOT NULL,
                tags            TEXT    NOT NULL DEFAULT '[]',
                created_at      TEXT    NOT NULL,
                updated_at      TEXT    NOT NULL,
                recalled_count  INTEGER NOT NULL DEFAULT 0,
                deleted         INTEGER NOT NULL DEFAULT 0,
                importance      INTEGER NOT NULL DEFAULT 3,
                last_recalled_at TEXT,
                embedding       TEXT
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

    def _migrate_schema(self) -> None:
        """Add new columns to existing DBs without breaking them."""
        migrations = [
            "ALTER TABLE memories ADD COLUMN importance INTEGER NOT NULL DEFAULT 3",
            "ALTER TABLE memories ADD COLUMN last_recalled_at TEXT",
            "ALTER TABLE memories ADD COLUMN embedding TEXT",
        ]
        for sql in migrations:
            try:
                self._conn.execute(sql)
            except sqlite3.OperationalError:
                pass  # column already exists
        self._conn.commit()

    # ── Write operations ──────────────────────────────────────────────────

    def insert(self, memory: Memory) -> Memory:
        """Persist a new memory. Returns the memory with id populated."""
        now = datetime.now(timezone.utc).isoformat()
        tags_json = json.dumps(memory.tags)
        cur = self._conn.execute(
            """
            INSERT INTO memories
                (type, content, tags, created_at, updated_at, recalled_count,
                 importance, last_recalled_at, embedding)
            VALUES (?, ?, ?, ?, ?, 0, ?, NULL, NULL)
            """,
            (memory.type.value, memory.content, tags_json, now, now, memory.importance),
        )
        self._conn.commit()
        return memory.model_copy(update={"id": cur.lastrowid})

    def update(self, memory_id: int, content: str) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            "UPDATE memories SET content = ?, updated_at = ?, embedding = NULL "
            "WHERE id = ? AND deleted = 0",
            (content, now, memory_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete(self, memory_id: int) -> bool:
        cur = self._conn.execute(
            "UPDATE memories SET deleted = 1 WHERE id = ? AND deleted = 0",
            (memory_id,),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def increment_recalled(self, memory_id: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE memories SET recalled_count = recalled_count + 1, "
            "last_recalled_at = ? WHERE id = ?",
            (now, memory_id),
        )
        self._conn.commit()

    def save_embedding(self, memory_id: int, vector: list[float]) -> None:
        self._conn.execute(
            "UPDATE memories SET embedding = ? WHERE id = ?",
            (json.dumps(vector), memory_id),
        )
        self._conn.commit()

    def update_importance(self, memory_id: int, importance: int) -> bool:
        cur = self._conn.execute(
            "UPDATE memories SET importance = ? WHERE id = ? AND deleted = 0",
            (max(1, min(5, importance)), memory_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ── Read operations ───────────────────────────────────────────────────

    def search(self, query: str, limit: int = 10) -> list[Memory]:
        """FTS5 full-text search over content and tags."""
        safe_query = self._sanitise_fts_query(query)
        try:
            rows = self._conn.execute(_FTS_QUERY, (safe_query, limit)).fetchall()
        except sqlite3.OperationalError:
            rows = self._conn.execute(
                """
                SELECT id, type, content, tags, created_at, updated_at,
                       recalled_count, importance, last_recalled_at, embedding
                FROM memories
                WHERE deleted = 0 AND (content LIKE ? OR tags LIKE ?)
                LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def search_semantic(
        self, query_vector: list[float], limit: int = 10
    ) -> list[Memory]:
        """
        Cosine similarity search over all embedded memories.
        Loads embeddings into Python memory — fast for personal use.
        TODO: replace with sqlite-vec at scale.
        """
        rows = self._conn.execute(
            """
            SELECT id, type, content, tags, created_at, updated_at,
                   recalled_count, importance, last_recalled_at, embedding
            FROM memories
            WHERE deleted = 0 AND embedding IS NOT NULL
            """
        ).fetchall()

        scored: list[tuple[float, Memory]] = []
        for row in rows:
            vector = json.loads(row["embedding"])
            score = cosine_similarity(query_vector, vector)
            scored.append((score, self._row_to_memory(row)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    def get_without_embeddings(self) -> list[Memory]:
        """Return all active memories that have no embedding yet."""
        rows = self._conn.execute(
            """
            SELECT id, type, content, tags, created_at, updated_at,
                   recalled_count, importance, last_recalled_at, embedding
            FROM memories
            WHERE deleted = 0 AND embedding IS NULL
            """
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_all(self, memory_type: MemoryType | None = None) -> list[Memory]:
        if memory_type:
            rows = self._conn.execute(
                """
                SELECT id, type, content, tags, created_at, updated_at,
                       recalled_count, importance, last_recalled_at, embedding
                FROM memories WHERE deleted = 0 AND type = ? ORDER BY updated_at DESC
                """,
                (memory_type.value,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT id, type, content, tags, created_at, updated_at,
                       recalled_count, importance, last_recalled_at, embedding
                FROM memories WHERE deleted = 0 ORDER BY updated_at DESC
                """
            ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE deleted = 0"
        ).fetchone()
        return row[0]

    def get_by_id(self, memory_id: int) -> Memory | None:
        row = self._conn.execute(
            """
            SELECT id, type, content, tags, created_at, updated_at,
                   recalled_count, importance, last_recalled_at, embedding
            FROM memories WHERE id = ? AND deleted = 0
            """,
            (memory_id,),
        ).fetchone()
        return self._row_to_memory(row) if row else None

    def decay_stale(
        self,
        context_days: int = 30,
        project_days: int = 90,
    ) -> int:
        """
        Soft-delete stale memories based on type, age, and recall frequency.

        Decay rules:
          context  — not recalled in context_days AND recalled_count < 2
          project  — not recalled in project_days AND recalled_count < 2
          fact     — never decayed (long-lived ground truth)
          preference — never decayed (long-lived user identity)

        importance >= 4 is always spared regardless of type.

        Returns the number of memories soft-deleted.
        """
        now = datetime.now(timezone.utc)

        def _cutoff(days: int) -> str:
            from datetime import timedelta
            return (now - timedelta(days=days)).isoformat()

        # Decay context memories
        cur = self._conn.execute(
            """
            UPDATE memories SET deleted = 1
            WHERE deleted = 0
              AND type = 'context'
              AND importance < 4
              AND recalled_count < 2
              AND (
                last_recalled_at IS NULL AND created_at < ?
                OR last_recalled_at < ?
              )
            """,
            (_cutoff(context_days), _cutoff(context_days)),
        )
        context_deleted = cur.rowcount

        # Decay project memories
        cur = self._conn.execute(
            """
            UPDATE memories SET deleted = 1
            WHERE deleted = 0
              AND type = 'project'
              AND importance < 4
              AND recalled_count < 2
              AND (
                last_recalled_at IS NULL AND created_at < ?
                OR last_recalled_at < ?
              )
            """,
            (_cutoff(project_days), _cutoff(project_days)),
        )
        project_deleted = cur.rowcount

        self._conn.commit()
        return context_deleted + project_deleted

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
            importance=row["importance"],
        )

    @staticmethod
    def _sanitise_fts_query(query: str) -> str:
        words = query.split()
        return " OR ".join(f'"{w}"*' for w in words if w)

    def close(self) -> None:
        self._conn.close()
