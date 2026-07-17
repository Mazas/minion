"""
minion/memory/manager.py

High-level memory interface used by the agent tools.

MemoryManager sits between the agent tools and the raw MemoryStore. It handles:
  - async wrapping (store is sync, agent is async)
  - formatting recalled memories for injection into the agent context
  - keeping the store open for the lifetime of the app
"""

from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path

from minion.memory.models import Memory, MemoryType
from minion.memory.store import MemoryStore


class MemoryManager:
    def __init__(self, db_path: Path) -> None:
        self._store = MemoryStore(db_path)

    # ── Async wrappers ────────────────────────────────────────────────────

    async def remember(
        self,
        content: str,
        type: str = "fact",
        tags: list[str] | None = None,
    ) -> Memory:
        """
        Store a new memory. Called by the agent's remember tool.

        Args:
            content: The information to remember.
            type: One of fact | preference | project | context.
            tags: Optional keywords to improve recall.
        """
        mem_type = MemoryType(type) if type in MemoryType.__members__.values() else MemoryType.FACT
        memory = Memory(type=mem_type, content=content, tags=tags or [])
        return await asyncio.to_thread(self._store.insert, memory)

    async def recall(self, query: str, limit: int = 5) -> list[Memory]:
        """
        Search memories by relevance to query.
        Increments recalled_count on each returned memory.
        """
        memories = await asyncio.to_thread(self._store.search, query, limit)
        for m in memories:
            if m.id is not None:
                await asyncio.to_thread(self._store.increment_recalled, m.id)
        return memories

    async def forget(self, memory_id: int) -> bool:
        """Soft-delete a memory by ID. Returns True if it existed."""
        return await asyncio.to_thread(self._store.delete, memory_id)

    async def update(self, memory_id: int, content: str) -> bool:
        """Replace the content of an existing memory."""
        return await asyncio.to_thread(self._store.update, memory_id, content)

    async def count(self) -> int:
        return await asyncio.to_thread(self._store.count)

    async def get_all(self, memory_type: MemoryType | None = None) -> list[Memory]:
        return await asyncio.to_thread(self._store.get_all, memory_type)

    # ── Formatting helpers (for agent context injection) ──────────────────

    @staticmethod
    def format_for_context(memories: list[Memory]) -> str:
        """
        Render a list of memories as a compact block for the system prompt.
        """
        if not memories:
            return "(no relevant memories found)"
        lines = []
        for m in memories:
            tag_str = f" [{', '.join(m.tags)}]" if m.tags else ""
            lines.append(f"- [{m.type}]{tag_str} {m.content}")
        return "\n".join(lines)

    def close(self) -> None:
        self._store.close()
