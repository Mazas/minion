"""
minion/memory/manager.py

High-level memory interface used by the agent tools and CLI.

Responsibilities:
  - Async wrapping of synchronous store operations
  - Hybrid recall: FTS5 keyword + cosine vector search, merged by score
  - Embedding backfill: generate embeddings for memories that lack them
  - Memory decay: silently soft-delete stale low-importance memories on startup
  - Session history via HistoryStore (shared SQLite connection)
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic_ai.messages import ModelMessage

from minion.llm.embeddings import embed
from minion.memory.history import HistoryStore
from minion.memory.models import Memory, MemoryType
from minion.memory.store import MemoryStore


class MemoryManager:
    def __init__(self, db_path: Path, config=None) -> None:
        self._store = MemoryStore(db_path)
        self._history = HistoryStore(self._store._conn)
        self._config = config  # optional — enables semantic search when present

    # ── Memory write operations ───────────────────────────────────────────

    async def remember(
        self,
        content: str,
        type: str = "fact",
        tags: list[str] | None = None,
        importance: int = 3,
    ) -> Memory:
        """Store a new memory and generate its embedding asynchronously."""
        mem_type = MemoryType(type) if type in MemoryType._value2member_map_ else MemoryType.FACT
        importance = max(1, min(5, importance))
        memory = Memory(type=mem_type, content=content, tags=tags or [], importance=importance)
        memory = await asyncio.to_thread(self._store.insert, memory)

        # Generate embedding in background — don't block the agent
        if self._semantic_enabled:
            asyncio.create_task(self._embed_memory(memory))

        return memory

    async def _embed_memory(self, memory: Memory) -> None:
        """Generate and save an embedding for a single memory."""
        assert memory.id is not None
        vector = await embed(
            memory.content,
            model=self._config.embed_model,
            base_url=self._config.ollama_embed_base_url,
        )
        if vector is not None:
            await asyncio.to_thread(self._store.save_embedding, memory.id, vector)

    async def forget(self, memory_id: int) -> bool:
        return await asyncio.to_thread(self._store.delete, memory_id)

    async def update(self, memory_id: int, content: str) -> bool:
        updated = await asyncio.to_thread(self._store.update, memory_id, content)
        # Re-embed after content change
        if updated and self._semantic_enabled:
            memory = await asyncio.to_thread(self._store.get_by_id, memory_id)
            if memory:
                asyncio.create_task(self._embed_memory(memory))
        return updated

    async def update_importance(self, memory_id: int, importance: int) -> bool:
        return await asyncio.to_thread(self._store.update_importance, memory_id, importance)

    # ── Recall ────────────────────────────────────────────────────────────

    async def recall(self, query: str, limit: int = 5) -> list[Memory]:
        """
        Hybrid recall: FTS5 keyword search + cosine vector search.

        Both result sets are merged by ID, deduplicated, and returned.
        FTS5 handles exact/partial keyword matches; vectors handle semantic
        similarity ("I hate verbose code" matches "code style preferences").
        Falls back to FTS5-only if semantic search is disabled or unavailable.
        """
        fts_results = await asyncio.to_thread(self._store.search, query, limit)

        if self._semantic_enabled:
            vector = await embed(
                query,
                model=self._config.embed_model,
                base_url=self._config.ollama_embed_base_url,
            )
            if vector is not None:
                semantic_results = await asyncio.to_thread(
                    self._store.search_semantic, vector, limit
                )
                results = _merge_results(fts_results, semantic_results, limit)
            else:
                results = fts_results
        else:
            results = fts_results

        for m in results:
            if m.id is not None:
                await asyncio.to_thread(self._store.increment_recalled, m.id)

        return results

    # ── Background tasks (called from cli.py on startup) ──────────────────

    async def backfill_embeddings(self) -> None:
        """Generate embeddings for any memory that lacks one. Silent."""
        if not self._semantic_enabled:
            return
        memories = await asyncio.to_thread(self._store.get_without_embeddings)
        for memory in memories:
            await self._embed_memory(memory)

    async def decay_stale(self) -> None:
        """Soft-delete stale low-importance memories. Silent."""
        if self._config is None:
            return
        await asyncio.to_thread(
            self._store.decay_stale,
            self._config.memory_decay_context_days,
            self._config.memory_decay_project_days,
        )

    # ── Misc ──────────────────────────────────────────────────────────────

    async def count(self) -> int:
        return await asyncio.to_thread(self._store.count)

    async def get_all(self, memory_type: MemoryType | None = None) -> list[Memory]:
        return await asyncio.to_thread(self._store.get_all, memory_type)

    @property
    def _semantic_enabled(self) -> bool:
        return self._config is not None and self._config.enable_semantic_search

    @staticmethod
    def format_for_context(memories: list[Memory]) -> str:
        if not memories:
            return "(no relevant memories found)"
        lines = []
        for m in memories:
            tag_str = f" [{', '.join(m.tags)}]" if m.tags else ""
            lines.append(f"- [{m.type}]{tag_str} {m.content}")
        return "\n".join(lines)

    # ── Session history ───────────────────────────────────────────────────

    async def create_session(self) -> int:
        return await asyncio.to_thread(self._history.create_session)

    async def get_latest_session_id(self) -> int | None:
        return await asyncio.to_thread(self._history.get_latest_session_id)

    async def save_messages(self, session_id: int, messages: list[ModelMessage]) -> None:
        await asyncio.to_thread(self._history.save_messages, session_id, messages)

    async def load_messages(self, session_id: int) -> list[ModelMessage]:
        return await asyncio.to_thread(self._history.load_messages, session_id)

    def close(self) -> None:
        self._store.close()


def _merge_results(
    fts: list[Memory],
    semantic: list[Memory],
    limit: int,
) -> list[Memory]:
    """
    Merge FTS5 and semantic results, deduplicating by ID.
    FTS results appear first (typically higher precision for exact matches),
    then semantic results that weren't already included.
    """
    seen: set[int] = set()
    merged: list[Memory] = []
    for m in fts + semantic:
        if m.id not in seen:
            seen.add(m.id)
            merged.append(m)
        if len(merged) >= limit:
            break
    return merged
