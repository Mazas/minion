"""
minion/memory/models.py

Data models for the memory system. Each Memory is a typed, tagged piece of
information about the user that persists across sessions.

Memory types:
  fact        — objective facts ("user's name is Alex")
  preference  — stated preferences ("prefers terminal apps over web UIs")
  project     — ongoing work ("building a Rust CLI called fenix")
  context     — situational context ("currently learning Neovim")
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    PROJECT = "project"
    CONTEXT = "context"


class Memory(BaseModel):
    id: int | None = None
    type: MemoryType
    content: str
    tags: list[str] = Field(default_factory=list)
    importance: int = Field(default=3, ge=1, le=5)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    recalled_count: int = 0

    @classmethod
    def fact(cls, content: str, tags: list[str] | None = None, importance: int = 3) -> Self:
        return cls(type=MemoryType.FACT, content=content, tags=tags or [], importance=importance)

    @classmethod
    def preference(cls, content: str, tags: list[str] | None = None, importance: int = 3) -> Self:
        return cls(type=MemoryType.PREFERENCE, content=content, tags=tags or [], importance=importance)

    @classmethod
    def project(cls, content: str, tags: list[str] | None = None, importance: int = 3) -> Self:
        return cls(type=MemoryType.PROJECT, content=content, tags=tags or [], importance=importance)

    @classmethod
    def context(cls, content: str, tags: list[str] | None = None, importance: int = 3) -> Self:
        return cls(type=MemoryType.CONTEXT, content=content, tags=tags or [], importance=importance)

    def tag_string(self) -> str:
        """Space-separated tags for FTS indexing."""
        return " ".join(self.tags)
