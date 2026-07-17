# Architecture

This document explains the key architectural decisions in minion and the reasoning behind them.

## Overview

Minion is structured as a set of loosely coupled modules. Each module has a clear responsibility and can be replaced independently.

```
┌──────────────────────────────────────────────┐
│                   TUI (Textual)               │
└───────────────────────┬──────────────────────┘
                        │
┌───────────────────────▼──────────────────────┐
│                 Agent (PydanticAI)            │
│                                              │
│   ┌──────────┐  ┌────────┐  ┌────────────┐  │
│   │  Memory  │  │ Search │  │ Shell / FS │  │
│   │  tools   │  │  tool  │  │   tools    │  │
│   └──────────┘  └────────┘  └────────────┘  │
└───────────────────────┬──────────────────────┘
                        │
┌───────────────────────▼──────────────────────┐
│              LLM Provider (Ollama)            │
└──────────────────────────────────────────────┘
```

## Components

### TUI (`minion/tui/`)

Built with [Textual](https://textual.textualize.io/). Provides:
- Scrollable chat history with markdown rendering
- Fixed input area at the bottom
- Status bar showing model name, memory count, and tool activity

The TUI is a thin layer — it only handles display and input. All logic lives in the agent.

### Agent (`minion/agent/`)

Built with [PydanticAI](https://ai.pydantic.dev/). The agent:
- Receives user messages from the TUI
- Decides which tools to call (memory recall, web search, etc.)
- Streams responses back to the TUI

Tools are plain Python functions registered with the agent. Adding a tool means writing a function — no framework magic.

### LLM Provider (`minion/llm/`)

Ollama exposes an OpenAI-compatible REST API at `http://localhost:11434/v1`. PydanticAI's `OpenAIModel` accepts a custom `base_url`, so we point it at Ollama. Switching to a real OpenAI or Anthropic model later is a one-line config change.

```python
class LLMProvider(Protocol):
    def get_model(self) -> Model: ...
```

### Memory (`minion/memory/`)

SQLite with FTS5 (full-text search). Three layers:

- **`store.py`** — raw SQLite CRUD. Synchronous. Owns the connection, schema, and FTS5 triggers.
- **`manager.py`** — async wrapper around the store. What the agent tools call.
- **`models.py`** — `Memory` Pydantic model and `MemoryType` enum.

Memory types:

| Type | Example |
|---|---|
| `fact` | "User's name is Alex" |
| `preference` | "Prefers terminal apps over web UIs" |
| `project` | "Working on a Rust CLI called fenix" |
| `context` | "Currently learning Neovim" |

The agent has four memory tools: `store_memory`, `recall_memories`, `forget_memory`, `update_memory`. The system prompt instructs it to recall relevant memories before every response and store new information proactively. The DB is a single file at `~/.minion/minion.db` — inspectable with any SQLite viewer.

**Why not vector search?** For hundreds of personal memories, SQLite FTS5 gives good recall with zero additional dependencies. FTS5 uses prefix matching (`"term"*`) so partial words resolve correctly. Vector embeddings can be layered in later as an additional recall path without changing the store interface.

### Tools (`minion/tools/`)

Each tool is a module with a clear interface. Tools are registered with the agent and can be enabled/disabled via config flags in `~/.minion/.env`.

#### Filesystem (`tools/filesystem.py`)

- **`read_file`** — reads up to 32 KB, returns truncation notice if larger
- **`write_file`** — dry-run by default (`confirm=False`), only writes when `confirm=True`; creates parent directories
- **`list_dir`** — sorted listing with file sizes, capped at 200 entries

All paths are resolved via `Path.expanduser().resolve()` before access.

#### Shell (`tools/shell.py`)

Disabled by default (`MINION_ENABLE_SHELL=false`). When enabled:

- A blocklist of dangerous patterns (e.g. `rm -rf /`, `mkfs`, `dd if=`, fork bomb) is checked before execution
- Commands time out after 30 seconds (configurable)
- Output is capped at 16 KB
- The agent tool wrapper adds a `confirm` flag — destructive commands show a dry-run preview first; the agent only executes with `confirm=True` after the user approves in chat

This is a safety net, not a sandbox. It prevents accidents, not determined misuse.

Three-layer design:
- **`SearchResult`** — Pydantic model: `title`, `url`, `snippet`
- **`SearchProvider`** — Protocol (interface) any provider must satisfy: `search(query, limit) -> list[SearchResult]`
- **`DuckDuckGoProvider`** — Default implementation using the `ddgs` package. No API key. Runs the sync client in `asyncio.to_thread()` to avoid blocking the event loop.

The agent's `web_search` tool is only registered when `config.enable_web_search` is `True`. Swapping to a different provider (Brave Search, Tavily, etc.) means implementing the protocol and adding an elif in `get_search_provider()` — the agent tool code is untouched.

| Tool | Module | Status |
|---|---|---|
| `store_memory`, `recall_memories`, `forget_memory`, `update_memory` | `memory/` | done |
| `web_search` | `tools/search.py` | done |
| `read_file`, `write_file`, `list_directory` | `tools/filesystem.py` | done |
| `run_shell` | `tools/shell.py` | done |
| `git_status`, `git_log`, `git_diff` | `tools/git.py` | Milestone 5 |

### Config (`minion/config.py`)

`pydantic-settings` reads from environment variables and `~/.minion/.env`. All config is typed. No config file parsing code to maintain.

## Data Layout

```
~/.minion/
├── .env          # local config (gitignored)
├── minion.db     # SQLite: memories + session history
└── logs/         # debug logs (optional)
```

## Design Principles

**Prefer working software over speculative abstractions.** Each milestone delivers something runnable. No scaffolding that doesn't do anything yet.

**Small files, clear boundaries.** If a file is getting long, it's doing too much.

**Tools are just functions.** No tool base classes, no plugin registries. A tool is a Python function with a docstring. PydanticAI handles the rest.

**The DB is yours.** SQLite means you can open `~/.minion/minion.db` in any viewer, write SQL against it, export it, or delete it. No proprietary format.
