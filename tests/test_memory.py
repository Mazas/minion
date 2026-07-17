"""
tests/test_memory.py

Tests for the memory store and manager.
Uses a temp directory so nothing touches ~/.minion.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from minion.memory.manager import MemoryManager
from minion.memory.models import Memory, MemoryType
from minion.memory.store import MemoryStore


# ── MemoryStore (sync) ────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(tmp_path / "test.db")
    yield s
    s.close()


def test_store_insert_and_get(store: MemoryStore) -> None:
    mem = Memory.preference("Prefers dark mode")
    saved = store.insert(mem)
    assert saved.id is not None
    fetched = store.get_by_id(saved.id)
    assert fetched is not None
    assert fetched.content == "Prefers dark mode"
    assert fetched.type == MemoryType.PREFERENCE


def test_store_count(store: MemoryStore) -> None:
    assert store.count() == 0
    store.insert(Memory.fact("User lives in Berlin"))
    store.insert(Memory.fact("User speaks German"))
    assert store.count() == 2


def test_store_soft_delete(store: MemoryStore) -> None:
    mem = store.insert(Memory.fact("Temporary fact"))
    assert store.count() == 1
    store.delete(mem.id)
    assert store.count() == 0
    assert store.get_by_id(mem.id) is None


def test_store_update(store: MemoryStore) -> None:
    mem = store.insert(Memory.fact("User is 30 years old"))
    store.update(mem.id, "User is 31 years old")
    fetched = store.get_by_id(mem.id)
    assert fetched.content == "User is 31 years old"


def test_store_fts_search(store: MemoryStore) -> None:
    store.insert(Memory.preference("Prefers terminal apps over web UIs", tags=["ui", "terminal"]))
    store.insert(Memory.fact("User's name is Alex"))
    store.insert(Memory.project("Building a Rust CLI called fenix", tags=["rust", "cli"]))

    results = store.search("terminal")
    assert len(results) == 1
    assert "terminal" in results[0].content.lower()


def test_store_fts_search_by_tag(store: MemoryStore) -> None:
    store.insert(Memory.project("Building a Rust CLI", tags=["rust", "cli"]))
    results = store.search("rust")
    assert len(results) == 1


def test_store_fts_no_results(store: MemoryStore) -> None:
    store.insert(Memory.fact("User's name is Alex"))
    results = store.search("python")
    assert results == []


def test_store_fts_special_chars_dont_crash(store: MemoryStore) -> None:
    store.insert(Memory.fact("User likes C++ and Rust"))
    # These characters would break raw FTS5 queries
    results = store.search("C++")
    # Should not raise, results may be empty or not depending on sanitisation
    assert isinstance(results, list)


def test_store_get_all(store: MemoryStore) -> None:
    store.insert(Memory.fact("fact one"))
    store.insert(Memory.preference("preference one"))
    store.insert(Memory.fact("fact two"))

    all_memories = store.get_all()
    assert len(all_memories) == 3

    facts = store.get_all(MemoryType.FACT)
    assert len(facts) == 2
    assert all(m.type == MemoryType.FACT for m in facts)


def test_store_recalled_count(store: MemoryStore) -> None:
    mem = store.insert(Memory.fact("User's timezone is UTC+2"))
    assert mem.recalled_count == 0
    store.increment_recalled(mem.id)
    store.increment_recalled(mem.id)
    fetched = store.get_by_id(mem.id)
    assert fetched.recalled_count == 2


# ── MemoryManager (async) ─────────────────────────────────────────────────────


@pytest.fixture
def manager(tmp_path: Path) -> MemoryManager:
    m = MemoryManager(tmp_path / "test.db")
    yield m
    m.close()


@pytest.mark.asyncio
async def test_manager_remember_and_recall(manager: MemoryManager) -> None:
    await manager.remember("Prefers vim over emacs", type="preference", tags=["editor"])
    results = await manager.recall("editor preferences")
    assert len(results) == 1
    assert "vim" in results[0].content


@pytest.mark.asyncio
async def test_manager_forget(manager: MemoryManager) -> None:
    mem = await manager.remember("Temporary info", type="context")
    assert await manager.count() == 1
    removed = await manager.forget(mem.id)
    assert removed is True
    assert await manager.count() == 0


@pytest.mark.asyncio
async def test_manager_update(manager: MemoryManager) -> None:
    mem = await manager.remember("User is a Python developer", type="fact")
    await manager.update(mem.id, "User is a Python and Rust developer")
    results = await manager.recall("developer")
    assert "Rust" in results[0].content


@pytest.mark.asyncio
async def test_manager_count(manager: MemoryManager) -> None:
    assert await manager.count() == 0
    await manager.remember("fact one", type="fact")
    await manager.remember("fact two", type="fact")
    assert await manager.count() == 2


@pytest.mark.asyncio
async def test_manager_format_for_context(manager: MemoryManager) -> None:
    await manager.remember("Prefers dark themes", type="preference", tags=["ui"])
    memories = await manager.recall("theme")
    formatted = MemoryManager.format_for_context(memories)
    assert "preference" in formatted
    assert "dark themes" in formatted


@pytest.mark.asyncio
async def test_manager_format_empty(manager: MemoryManager) -> None:
    memories = await manager.recall("nothing here")
    formatted = MemoryManager.format_for_context(memories)
    assert "no relevant memories" in formatted
