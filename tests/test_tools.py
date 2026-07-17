"""
tests/test_tools.py

Tests for tool modules. Search tests use a mock provider so no real
network calls are made — fast, deterministic, offline-safe.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from minion.tools.filesystem import file_read, file_write, list_dir
from minion.tools.git import git_branches, git_commit, git_diff, git_log, git_status
from minion.tools.shell import BlockedCommandError, shell_exec, _check_blocklist
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


# ── Filesystem tools ──────────────────────────────────────────────────────────


def test_file_read_existing(tmp_path: Path) -> None:
    f = tmp_path / "hello.txt"
    f.write_text("Hello, world!")
    result = file_read(str(f))
    assert result == "Hello, world!"


def test_file_read_not_found(tmp_path: Path) -> None:
    result = file_read(str(tmp_path / "missing.txt"))
    assert "not found" in result


def test_file_read_not_a_file(tmp_path: Path) -> None:
    result = file_read(str(tmp_path))
    assert "not a file" in result


def test_file_read_truncates_large_file(tmp_path: Path) -> None:
    f = tmp_path / "big.txt"
    f.write_bytes(b"x" * 40_000)
    result = file_read(str(f))
    assert "truncated" in result
    assert len(result) < 40_000


def test_file_write_dry_run(tmp_path: Path) -> None:
    f = tmp_path / "out.txt"
    result = file_write(str(f), "some content", confirm=False)
    assert "Dry run" in result
    assert not f.exists()


def test_file_write_confirmed(tmp_path: Path) -> None:
    f = tmp_path / "out.txt"
    result = file_write(str(f), "hello file", confirm=True)
    assert "Written" in result
    assert f.read_text() == "hello file"


def test_file_write_creates_parents(tmp_path: Path) -> None:
    f = tmp_path / "a" / "b" / "c.txt"
    file_write(str(f), "nested", confirm=True)
    assert f.read_text() == "nested"


def test_list_dir_basic(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("hi")
    (tmp_path / "subdir").mkdir()
    result = list_dir(str(tmp_path))
    assert "file.txt" in result
    assert "subdir/" in result


def test_list_dir_not_found(tmp_path: Path) -> None:
    result = list_dir(str(tmp_path / "nope"))
    assert "not found" in result


def test_list_dir_not_a_directory(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("x")
    result = list_dir(str(f))
    assert "not a directory" in result


def test_list_dir_shows_sizes(tmp_path: Path) -> None:
    (tmp_path / "data.bin").write_bytes(b"x" * 2048)
    result = list_dir(str(tmp_path))
    assert "data.bin" in result
    assert "KB" in result or "B" in result


# ── Shell tool ────────────────────────────────────────────────────────────────


def test_check_blocklist_safe_command() -> None:
    # Should not raise
    _check_blocklist("ls -la", [])


def test_check_blocklist_blocked_pattern() -> None:
    with pytest.raises(BlockedCommandError, match="blocked"):
        _check_blocklist("rm -rf /", [])


def test_check_blocklist_custom_pattern() -> None:
    with pytest.raises(BlockedCommandError):
        _check_blocklist("drop table users", ["drop table"])


def test_check_blocklist_case_insensitive() -> None:
    with pytest.raises(BlockedCommandError):
        _check_blocklist("RM -RF /", [])


@pytest.mark.asyncio
async def test_shell_exec_simple_command() -> None:
    result = await shell_exec("echo hello")
    assert "hello" in result


@pytest.mark.asyncio
async def test_shell_exec_nonzero_exit() -> None:
    result = await shell_exec("bash -c 'exit 1'")
    assert "exit code 1" in result


@pytest.mark.asyncio
async def test_shell_exec_blocked_command() -> None:
    with pytest.raises(BlockedCommandError):
        await shell_exec("rm -rf /")


@pytest.mark.asyncio
async def test_shell_exec_timeout() -> None:
    result = await shell_exec("sleep 10", timeout=1)
    assert "timed out" in result


@pytest.mark.asyncio
async def test_shell_exec_workdir(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("found")
    result = await shell_exec("ls", workdir=str(tmp_path))
    assert "marker.txt" in result


@pytest.mark.asyncio
async def test_shell_exec_stderr_captured() -> None:
    result = await shell_exec("echo error >&2")
    assert "error" in result


# ── Git tools ─────────────────────────────────────────────────────────────────


def _init_git_repo(path: Path) -> None:
    """Create a minimal git repo with one commit for testing."""
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@test.com"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True, capture_output=True)
    (path / "README.md").write_text("hello")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "init"], check=True, capture_output=True)


@pytest.mark.asyncio
async def test_git_status_clean_repo(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    result = await git_status(cwd=str(tmp_path))
    assert "nothing to commit" in result or "working tree clean" in result


@pytest.mark.asyncio
async def test_git_status_dirty_repo(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "new_file.txt").write_text("untracked")
    result = await git_status(cwd=str(tmp_path))
    assert "new_file.txt" in result


@pytest.mark.asyncio
async def test_git_log_shows_commits(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    result = await git_log(cwd=str(tmp_path), limit=5)
    assert "init" in result


@pytest.mark.asyncio
async def test_git_diff_no_changes(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    result = await git_diff(cwd=str(tmp_path))
    # No changes — empty diff or no output
    assert "$ git diff" in result


@pytest.mark.asyncio
async def test_git_diff_with_changes(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "README.md").write_text("modified content")
    result = await git_diff(cwd=str(tmp_path))
    assert "README.md" in result or "modified" in result


@pytest.mark.asyncio
async def test_git_branches_shows_branch(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    result = await git_branches(cwd=str(tmp_path))
    # Should show master or main
    assert "master" in result or "main" in result


@pytest.mark.asyncio
async def test_git_commit_dry_run(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "file.txt").write_text("content")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True, capture_output=True)
    result = await git_commit("test commit", cwd=str(tmp_path), confirm=False)
    assert "Dry run" in result
    assert "test commit" in result
    # File should still be staged, not committed
    log = await git_log(cwd=str(tmp_path), limit=5)
    assert "test commit" not in log


@pytest.mark.asyncio
async def test_git_commit_confirmed(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "file.txt").write_text("content")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True, capture_output=True)
    result = await git_commit("my commit", cwd=str(tmp_path), confirm=True)
    assert "my commit" in result or "master" in result or "main" in result
    log = await git_log(cwd=str(tmp_path), limit=5)
    assert "my commit" in log


@pytest.mark.asyncio
async def test_git_status_not_a_repo(tmp_path: Path) -> None:
    result = await git_status(cwd=str(tmp_path))
    assert "not a git repository" in result.lower() or "exit code" in result


# ── History store ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_history_roundtrip(tmp_path: Path) -> None:
    from minion.memory.manager import MemoryManager
    from pydantic_ai.messages import ModelRequest, UserPromptPart, ModelResponse, TextPart

    manager = MemoryManager(tmp_path / "test.db")

    sid = await manager.create_session()
    msgs = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(parts=[TextPart(content="hi there")], model_name="qwen3:8b"),
    ]
    await manager.save_messages(sid, msgs)
    loaded = await manager.load_messages(sid)

    assert len(loaded) == 2
    assert loaded[0].parts[0].content == "hello"
    assert loaded[1].parts[0].content == "hi there"
    manager.close()


@pytest.mark.asyncio
async def test_history_latest_session(tmp_path: Path) -> None:
    from minion.memory.manager import MemoryManager

    manager = MemoryManager(tmp_path / "test.db")

    assert await manager.get_latest_session_id() is None
    sid1 = await manager.create_session()
    sid2 = await manager.create_session()
    latest = await manager.get_latest_session_id()
    # Most recently created session should be latest
    assert latest == sid2
    manager.close()


@pytest.mark.asyncio
async def test_history_empty_session(tmp_path: Path) -> None:
    from minion.memory.manager import MemoryManager

    manager = MemoryManager(tmp_path / "test.db")
    sid = await manager.create_session()
    msgs = await manager.load_messages(sid)
    assert msgs == []
    manager.close()


# ── Delegation tool ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delegate_known_role(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock, patch
    from minion.tools.delegate import run_delegate
    from minion.config import Config

    cfg = Config(
        orchestrator_model="qwen3:4b",
        delegate_models={"reasoning": "qwen3:8b", "code": "qwen2.5-coder:7b"},
    )

    mock_result = AsyncMock()
    mock_result.output = "The answer is 42."

    with patch("minion.tools.delegate.Agent") as MockAgent:
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run = AsyncMock(return_value=mock_result)
        MockAgent.return_value = mock_agent_instance

        result = await run_delegate(
            role="reasoning",
            task="What is the meaning of life?",
            context="",
            config=cfg,
        )

    assert result == "The answer is 42."


@pytest.mark.asyncio
async def test_delegate_unknown_role() -> None:
    from minion.tools.delegate import run_delegate
    from minion.config import Config

    cfg = Config(
        orchestrator_model="qwen3:4b",
        delegate_models={"reasoning": "qwen3:8b"},
    )
    result = await run_delegate(
        role="nonexistent",
        task="Do something",
        context="",
        config=cfg,
    )
    assert "No specialist model configured" in result
    assert "nonexistent" in result


@pytest.mark.asyncio
async def test_delegate_uses_correct_model() -> None:
    from unittest.mock import AsyncMock, patch, MagicMock
    from minion.tools.delegate import run_delegate
    from minion.config import Config

    cfg = Config(
        orchestrator_model="qwen3:4b",
        delegate_models={"code": "qwen2.5-coder:7b"},
    )

    captured_model_name: list[str] = []

    mock_result = AsyncMock()
    mock_result.output = "def hello(): pass"

    def fake_get_model(model_override=None):
        captured_model_name.append(model_override or "default")
        return MagicMock()

    with patch("minion.tools.delegate.get_provider") as mock_provider_fn:
        mock_provider = MagicMock()
        mock_provider.get_model = fake_get_model
        mock_provider_fn.return_value = mock_provider

        with patch("minion.tools.delegate.Agent") as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_agent_instance

            await run_delegate(
                role="code",
                task="Write a hello function",
                context="",
                config=cfg,
            )

    assert "qwen2.5-coder:7b" in captured_model_name


@pytest.mark.asyncio
async def test_delegate_includes_context_in_task() -> None:
    from unittest.mock import AsyncMock, patch
    from minion.tools.delegate import run_delegate
    from minion.config import Config

    cfg = Config(
        orchestrator_model="qwen3:4b",
        delegate_models={"code": "qwen2.5-coder:7b"},
    )

    received_task: list[str] = []

    mock_result = AsyncMock()
    mock_result.output = "done"

    with patch("minion.tools.delegate.Agent") as MockAgent:
        mock_agent_instance = AsyncMock()

        async def capture_run(task):
            received_task.append(task)
            return mock_result

        mock_agent_instance.run = capture_run
        MockAgent.return_value = mock_agent_instance

        await run_delegate(
            role="code",
            task="Write a function",
            context="Use Python 3.11+",
            config=cfg,
        )

    assert len(received_task) == 1
    assert "Write a function" in received_task[0]
    assert "Use Python 3.11+" in received_task[0]


# ── Cosine similarity ─────────────────────────────────────────────────────────


def test_cosine_similarity_identical() -> None:
    from minion.llm.embeddings import cosine_similarity
    v = [1.0, 0.5, 0.3]
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal() -> None:
    from minion.llm.embeddings import cosine_similarity
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-6


def test_cosine_similarity_zero_vector() -> None:
    from minion.llm.embeddings import cosine_similarity
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
