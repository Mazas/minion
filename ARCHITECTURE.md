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

Each tool is a module with a clear interface. Tools are registered with the agent and can be enabled/disabled via config.

| Tool | Module | Status |
|---|---|---|
| `store_memory`, `recall_memories`, `forget_memory`, `update_memory` | `memory/` | done |
| `web_search` | `tools/search.py` | Milestone 3 |
| `file_read`, `file_write`, `list_dir` | `tools/filesystem.py` | Milestone 4 |
| `shell_exec` | `tools/shell.py` | Milestone 4 |
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
