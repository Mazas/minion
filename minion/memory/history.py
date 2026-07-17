"""
minion/memory/history.py

Persists conversation history across sessions.

Each "session" is a named conversation thread stored in the same SQLite DB
as memories. On startup minion loads the most recent session automatically.
Sessions are stored as JSON-serialised PydanticAI ModelMessage lists.

Schema:
  sessions     — one row per conversation session
  messages     — one row per exchange (request + response pair), linked to session
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter


class HistoryStore:
    """
    Stores and retrieves conversation history.
    Operates synchronously — wrap in asyncio.to_thread() for async callers.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL REFERENCES sessions(id),
                messages    TEXT NOT NULL,
                saved_at    TEXT NOT NULL
            );
        """)
        self._conn.commit()

    # ── Sessions ──────────────────────────────────────────────────────────

    def create_session(self) -> int:
        """Create a new session and return its ID."""
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            "INSERT INTO sessions (created_at, updated_at) VALUES (?, ?)",
            (now, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_latest_session_id(self) -> int | None:
        """Return the ID of the most recently updated session, or None."""
        row = self._conn.execute(
            "SELECT id FROM sessions ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    def touch_session(self, session_id: int) -> None:
        """Update the session's updated_at timestamp."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (now, session_id),
        )
        self._conn.commit()

    # ── Messages ──────────────────────────────────────────────────────────

    def save_messages(self, session_id: int, messages: list[ModelMessage]) -> None:
        """Persist the full message history for a session (replaces previous save)."""
        encoded = ModelMessagesTypeAdapter.dump_json(messages).decode("utf-8")
        now = datetime.now(timezone.utc).isoformat()
        # Upsert: delete old snapshot, insert fresh one
        self._conn.execute(
            "DELETE FROM session_messages WHERE session_id = ?",
            (session_id,),
        )
        self._conn.execute(
            "INSERT INTO session_messages (session_id, messages, saved_at) VALUES (?, ?, ?)",
            (session_id, encoded, now),
        )
        self._conn.commit()
        self.touch_session(session_id)

    def load_messages(self, session_id: int) -> list[ModelMessage]:
        """Load the saved message history for a session."""
        row = self._conn.execute(
            "SELECT messages FROM session_messages WHERE session_id = ? "
            "ORDER BY saved_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if not row:
            return []
        return ModelMessagesTypeAdapter.validate_json(row[0])
