"""
minion/tools/filesystem.py

Filesystem tools: read files, write files, list directories.

Safety principles:
- Paths are always resolved and validated before access.
- file_write requires explicit confirmation from the agent (the agent must
  set confirm=True), so accidental writes are harder to trigger.
- No path traversal: resolved paths are returned to the agent so it can
  show the user exactly what was accessed.
"""

from __future__ import annotations

from pathlib import Path


# Maximum bytes to read in a single call — prevents the context window
# from being flooded by huge files.
MAX_READ_BYTES = 32_768  # 32 KB


def _resolve(path: str) -> Path:
    return Path(path).expanduser().resolve()


def file_read(path: str) -> str:
    """
    Read a file and return its contents as a string.

    Args:
        path: Absolute or relative path to the file (~ expanded).

    Returns:
        File contents, or an error message if the file cannot be read.
        If the file exceeds 32 KB only the first 32 KB is returned with
        a truncation notice.
    """
    resolved = _resolve(path)

    if not resolved.exists():
        return f"Error: file not found: {resolved}"
    if not resolved.is_file():
        return f"Error: not a file: {resolved}"

    size = resolved.stat().st_size
    truncated = size > MAX_READ_BYTES

    try:
        raw = resolved.read_bytes()[:MAX_READ_BYTES]
        content = raw.decode("utf-8", errors="replace")
    except OSError as e:
        return f"Error reading {resolved}: {e}"

    if truncated:
        content += f"\n\n[truncated — file is {size} bytes, showed first {MAX_READ_BYTES}]"

    return content


def file_write(path: str, content: str, confirm: bool = False) -> str:
    """
    Write content to a file. Creates parent directories if needed.

    Args:
        path: Absolute or relative path to the file (~ expanded).
        content: Text content to write.
        confirm: Must be True to actually write. If False, returns a dry-run
                 preview so the agent can ask the user before committing.

    Returns:
        Success message, dry-run preview, or error message.
    """
    resolved = _resolve(path)

    if not confirm:
        return (
            f"Dry run — would write {len(content)} characters to {resolved}.\n"
            "Call again with confirm=True to actually write."
        )

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"Written {len(content)} characters to {resolved}"
    except OSError as e:
        return f"Error writing {resolved}: {e}"


def list_dir(path: str = ".", max_entries: int = 200) -> str:
    """
    List the contents of a directory.

    Args:
        path: Directory to list (defaults to current working directory).
        max_entries: Maximum number of entries to return.

    Returns:
        Formatted directory listing, or an error message.
    """
    resolved = _resolve(path)

    if not resolved.exists():
        return f"Error: path not found: {resolved}"
    if not resolved.is_dir():
        return f"Error: not a directory: {resolved}"

    try:
        entries = sorted(resolved.iterdir(), key=lambda p: (p.is_file(), p.name))
    except PermissionError:
        return f"Error: permission denied: {resolved}"

    truncated = len(entries) > max_entries
    entries = entries[:max_entries]

    lines = [f"Directory: {resolved}\n"]
    for entry in entries:
        if entry.is_dir():
            lines.append(f"  {entry.name}/")
        elif entry.is_symlink():
            lines.append(f"  {entry.name} -> {entry.resolve()}")
        else:
            size = entry.stat().st_size
            lines.append(f"  {entry.name}  ({_human_size(size)})")

    if truncated:
        lines.append(f"\n  ... (showing {max_entries} of {len(list(resolved.iterdir()))} entries)")

    return "\n".join(lines)


def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size}{unit}"
        size //= 1024
    return f"{size}TB"
