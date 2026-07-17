"""
minion/tools/git.py

Git tools: status, log, diff, commit, and branch info.

All operations run via subprocess (git CLI) so they work with any repo
regardless of language or structure. The working directory defaults to
cwd but can be overridden per-call so the agent can work on any repo.

git_commit requires confirm=True (same dry-run pattern as write_file and
run_shell) to prevent accidental commits.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path


_GIT = shutil.which("git") or "git"
_TIMEOUT = 15  # seconds
_MAX_OUTPUT = 16_384  # 16 KB


async def _git(
    *args: str,
    cwd: str | None = None,
) -> tuple[int, str]:
    """Run a git command, return (returncode, combined output)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            _GIT,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return -1, f"Error: git command timed out after {_TIMEOUT}s"
    except OSError as e:
        return -1, f"Error: could not run git: {e}"

    combined = (stdout + stderr)[:_MAX_OUTPUT].decode("utf-8", errors="replace")
    return proc.returncode, combined


def _format(returncode: int, output: str, command: str) -> str:
    lines = [f"$ git {command}"]
    if returncode not in (0, 1):  # git diff exits 1 when there are differences
        lines.append(f"(exit code {returncode})")
    lines.append(output.strip() if output.strip() else "(no output)")
    return "\n".join(lines)


async def git_status(cwd: str | None = None) -> str:
    """
    Show the working tree status of a git repository.

    Args:
        cwd: Path to the git repository (defaults to current directory).
    """
    rc, out = await _git("status", cwd=cwd)
    return _format(rc, out, f"status{' ' + cwd if cwd else ''}")


async def git_log(cwd: str | None = None, limit: int = 10) -> str:
    """
    Show recent commit history.

    Args:
        cwd: Path to the git repository (defaults to current directory).
        limit: Number of commits to show (default 10).
    """
    rc, out = await _git(
        "log",
        f"--max-count={limit}",
        "--oneline",
        "--decorate",
        cwd=cwd,
    )
    return _format(rc, out, f"log -{limit}")


async def git_diff(
    cwd: str | None = None,
    staged: bool = False,
    path: str | None = None,
) -> str:
    """
    Show changes in the working tree or staging area.

    Args:
        cwd: Path to the git repository (defaults to current directory).
        staged: If True, show staged (cached) changes instead of unstaged.
        path: Limit diff to a specific file or directory.
    """
    args = ["diff"]
    if staged:
        args.append("--staged")
    args.extend(["--stat", "--patch"])
    if path:
        args.extend(["--", path])
    rc, out = await _git(*args, cwd=cwd)
    label = "diff" + (" --staged" if staged else "") + (f" -- {path}" if path else "")
    return _format(rc, out, label)


async def git_branches(cwd: str | None = None) -> str:
    """
    List local branches and indicate the current branch.

    Args:
        cwd: Path to the git repository (defaults to current directory).
    """
    rc, out = await _git("branch", "-v", cwd=cwd)
    return _format(rc, out, "branch -v")


async def git_commit(
    message: str,
    cwd: str | None = None,
    confirm: bool = False,
) -> str:
    """
    Commit staged changes with a message.

    Always call with confirm=False first to show a preview, then call
    with confirm=True only after the user explicitly approves.

    Args:
        message: Commit message.
        cwd: Path to the git repository (defaults to current directory).
        confirm: Must be True to actually commit.
    """
    if not confirm:
        # Show what would be committed
        _, status_out = await _git("status", "--short", cwd=cwd)
        _, diff_out = await _git("diff", "--staged", "--stat", cwd=cwd)
        return (
            f"Dry run — would commit with message:\n\n  {message!r}\n\n"
            f"Staged changes:\n{status_out or '(nothing staged)'}\n\n"
            f"{diff_out or ''}\n"
            "Call again with confirm=True to actually commit."
        )

    rc, out = await _git("commit", "-m", message, cwd=cwd)
    return _format(rc, out, f'commit -m "{message}"')
