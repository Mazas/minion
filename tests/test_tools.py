"""
tests/test_tools.py

Tests for tool modules. Search tests use a mock provider so no real
network calls are made — fast, deterministic, offline-safe.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from minion.tools.search import (
    DuckDuckGoProvider,
    SearchResult,
    format_results,
    get_search_provider,
)
from minion.config import Config


# ── SearchResult ──────────────────────────────────────────────────────────────


def test_search_result_format() -> None:
    r = SearchResult(
        title="Python 3.14 Released",
        url="https://python.org/news",
        snippet="The Python team announced version 3.14 today.",
    )
    formatted = r.format()
    assert "Python 3.14 Released" in formatted
    assert "https://python.org/news" in formatted
    assert "announced version 3.14" in formatted


def test_search_result_model_fields() -> None:
    r = SearchResult(title="Title", url="https://example.com", snippet="A snippet.")
    assert r.title == "Title"
    assert r.url == "https://example.com"
    assert r.snippet == "A snippet."


# ── format_results ────────────────────────────────────────────────────────────


def test_format_results_empty() -> None:
    assert format_results([]) == "No results found."


def test_format_results_numbered() -> None:
    results = [
        SearchResult(title="First", url="https://first.com", snippet="First result."),
        SearchResult(title="Second", url="https://second.com", snippet="Second result."),
    ]
    formatted = format_results(results)
    assert "1." in formatted
    assert "2." in formatted
    assert "First" in formatted
    assert "Second" in formatted


def test_format_results_single() -> None:
    results = [SearchResult(title="Only", url="https://only.com", snippet="Only result.")]
    formatted = format_results(results)
    assert "1." in formatted
    assert "Only" in formatted


# ── DuckDuckGoProvider ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_duckduckgo_provider_returns_results() -> None:
    """Mock the DDGS client so no real network call is made."""
    mock_results = [
        {"title": "Result One", "href": "https://one.com", "body": "Snippet one."},
        {"title": "Result Two", "href": "https://two.com", "body": "Snippet two."},
    ]

    mock_ddgs = MagicMock()
    mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
    mock_ddgs.__exit__ = MagicMock(return_value=False)
    mock_ddgs.text = MagicMock(return_value=mock_results)

    with patch("minion.tools.search.DDGS", return_value=mock_ddgs):
        provider = DuckDuckGoProvider()
        results = await provider.search("test query", limit=2)

    assert len(results) == 2
    assert results[0].title == "Result One"
    assert results[0].url == "https://one.com"
    assert results[0].snippet == "Snippet one."
    assert results[1].title == "Result Two"


@pytest.mark.asyncio
async def test_duckduckgo_provider_respects_limit() -> None:
    mock_results = [
        {"title": f"Result {i}", "href": f"https://r{i}.com", "body": f"Snippet {i}."}
        for i in range(10)
    ]

    mock_ddgs = MagicMock()
    mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
    mock_ddgs.__exit__ = MagicMock(return_value=False)
    mock_ddgs.text = MagicMock(return_value=mock_results[:3])

    with patch("minion.tools.search.DDGS", return_value=mock_ddgs):
        provider = DuckDuckGoProvider()
        results = await provider.search("test", limit=3)

    # Verify limit was passed through
    mock_ddgs.text.assert_called_once_with("test", max_results=3)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_duckduckgo_provider_empty_results() -> None:
    mock_ddgs = MagicMock()
    mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
    mock_ddgs.__exit__ = MagicMock(return_value=False)
    mock_ddgs.text = MagicMock(return_value=[])

    with patch("minion.tools.search.DDGS", return_value=mock_ddgs):
        provider = DuckDuckGoProvider()
        results = await provider.search("something obscure", limit=5)

    assert results == []


@pytest.mark.asyncio
async def test_duckduckgo_provider_handles_missing_fields() -> None:
    """Results with missing keys should not crash — default to empty string."""
    mock_results = [{"title": "Partial"}]  # missing href and body

    mock_ddgs = MagicMock()
    mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
    mock_ddgs.__exit__ = MagicMock(return_value=False)
    mock_ddgs.text = MagicMock(return_value=mock_results)

    with patch("minion.tools.search.DDGS", return_value=mock_ddgs):
        provider = DuckDuckGoProvider()
        results = await provider.search("test", limit=1)

    assert len(results) == 1
    assert results[0].title == "Partial"
    assert results[0].url == ""
    assert results[0].snippet == ""


# ── get_search_provider factory ───────────────────────────────────────────────


def test_get_search_provider_duckduckgo() -> None:
    cfg = Config(search_provider="duckduckgo")
    provider = get_search_provider(cfg)
    assert isinstance(provider, DuckDuckGoProvider)


def test_get_search_provider_unknown_raises() -> None:
    cfg = Config(search_provider="unknown_engine")
    with pytest.raises(ValueError, match="Unknown search provider"):
        get_search_provider(cfg)
