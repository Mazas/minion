"""
minion/tools/shell.py

Shell execution tool with safety controls.

Safety model:
- Shell is disabled by default (MINION_ENABLE_SHELL=false).
- A blocklist of dangerous command patterns is checked before execution.
- Commands time out after a configurable limit (default 30s).
- Output is capped to avoid flooding the context window.

The blocklist is intentionally conservative. If a legitimate command is
blocked, the user can adjust MINION_SHELL_BLOCKLIST in ~/.minion/.env
(comma-separated patterns to add) or MINION_SHELL_ALLOWLIST to override.

This is not a sandbox — it provides a reasonable safety net against
accidental damage, not against a determined adversary.
"""

from __future__ import annotations

import asyncio
import shlex
import subprocess


# Commands/patterns that are blocked regardless of config.
# Matched as substrings in the full command string (after lowercasing).
DEFAULT_BLOCKLIST: list[str] = [
    "rm -rf /",
    "rm -rf ~",
    "mkfs",
    "dd if=",
    ":(){:|:&};:",  # fork bomb
    "chmod -r /",
    "chown -r /",
    "> /dev/sd",
    "mv / ",
    "mv ~/ /",
]

MAX_OUTPUT_BYTES = 16_384  # 16 KB
DEFAULT_TIMEOUT = 30  # seconds


class BlockedCommandError(Exception):
    """Raised when a command matches the blocklist."""


def _check_blocklist(command: str, extra_blocked: list[str]) -> None:
    """Raise BlockedCommandError if the command matches any blocked pattern."""
    lower = command.lower()
    for pattern in DEFAULT_BLOCKLIST + extra_blocked:
        if pattern.lower() in lower:
            raise BlockedCommandError(
                f"Command blocked — matches unsafe pattern: {pattern!r}\n"
                "If this is intentional, run it in your own terminal."
            )


async def shell_exec(
    command: str,
    workdir: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    extra_blocked: list[str] | None = None,
) -> str:
    """
    Execute a shell command and return its output.

    Args:
        command: The shell command to run.
        workdir: Working directory for the command (defaults to cwd).
        timeout: Seconds before the command is killed (default 30).
        extra_blocked: Additional patterns to block, on top of the defaults.

    Returns:
        Combined stdout + stderr output, or an error message.
    """
    _check_blocklist(command, extra_blocked or [])

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return f"Error: command timed out after {timeout}s: {command!r}"

    except OSError as e:
        return f"Error: failed to start command: {e}"

    combined = (stdout + stderr)[:MAX_OUTPUT_BYTES]
    output = combined.decode("utf-8", errors="replace")

    if len(stdout + stderr) > MAX_OUTPUT_BYTES:
        output += f"\n[output truncated at {MAX_OUTPUT_BYTES} bytes]"

    result_lines = [f"$ {command}"]
    if proc.returncode != 0:
        result_lines.append(f"(exit code {proc.returncode})")
    if output.strip():
        result_lines.append(output)
    else:
        result_lines.append("(no output)")

    return "\n".join(result_lines)
