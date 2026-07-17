"""
minion/tools/search.py

Web search abstraction. The agent calls `web_search` and gets back formatted
results — it never touches the provider directly.

Architecture:
  SearchResult    — Pydantic model for a single result
  SearchProvider  — Protocol (interface) any provider must satisfy
  DuckDuckGoProvider — Default implementation, no API key required
  get_search_provider() — Factory, reads config.search_provider

Adding a new provider (e.g. Brave Search):
  1. Implement SearchProvider protocol.
  2. Add an elif branch in get_search_provider().
  3. Set MINION_SEARCH_PROVIDER=brave in ~/.minion/.env.
"""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from ddgs import DDGS
from pydantic import BaseModel

from minion.config import Config


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str

    def format(self) -> str:
        return f"**{self.title}**\n{self.url}\n{self.snippet}"


@runtime_checkable
class SearchProvider(Protocol):
    async def search(self, query: str, limit: int = 5) -> list[SearchResult]: ...


class DuckDuckGoProvider:
    """
    Privacy-respecting search via DuckDuckGo. No API key required.

    Uses the `ddgs` package (successor to duckduckgo-search). Runs the
    synchronous client in a thread executor to avoid blocking the event loop.
    """

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        return await asyncio.to_thread(self._search_sync, query, limit)

    def _search_sync(self, query: str, limit: int) -> list[SearchResult]:
        results: list[SearchResult] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=limit):
                results.append(
                    SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                    )
                )
        return results


def get_search_provider(config: Config) -> SearchProvider:
    """Factory — returns the configured search provider."""
    provider = config.search_provider.lower()
    if provider == "duckduckgo":
        return DuckDuckGoProvider()
    raise ValueError(
        f"Unknown search provider: {provider!r}. "
        "Valid options: duckduckgo"
    )


def format_results(results: list[SearchResult]) -> str:
    """Render results as a markdown block for the agent to read."""
    if not results:
        return "No results found."
    parts = [f"{i + 1}. {r.format()}" for i, r in enumerate(results)]
    return "\n\n".join(parts)
