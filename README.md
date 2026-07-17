# minion

A local-first personal AI assistant that lives in your terminal.

Minion runs local LLMs via Ollama, remembers things about you across sessions, searches the web, reads files, runs git commands, and delegates complex tasks to specialist models.

```
┌─────────────────────────────────────────────┐
│  minion                          qwen3:4b   │
├─────────────────────────────────────────────┤
│                                             │
│  You: Can you write a binary search in Rust │
│       and explain the tradeoffs vs linear?  │
│                                             │
│  M: Let me get our specialists on this...   │
│  [delegating to code...]                    │
│  [delegating to reasoning...]               │
│                                             │
│  Here's the implementation: ...             │
│                                             │
├─────────────────────────────────────────────┤
│ > ________________________________________  │
│  qwen3:4b          5 memories · session 3   │
└─────────────────────────────────────────────┘
```

## Goals

- **Local-first** — your data stays on your machine
- **Persistent memory** — remembers facts, preferences, and context across sessions
- **Hybrid recall** — keyword (FTS5) + semantic (vector embeddings) search
- **Tool-capable** — web search, filesystem, shell, git
- **Delegating** — fast orchestrator routes complex tasks to specialist models
- **Hackable** — small files, clear boundaries, replaceable components

## Architecture

```
minion/
├── agent/      # PydanticAI agent, session management, streaming
├── llm/        # Ollama provider, embedding client
├── memory/     # SQLite + FTS5 + vector search, session history, decay
├── tools/      # web search, filesystem, shell, git, delegation
└── tui/        # Textual TUI (chat pane + input + status bar)
```

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally

## Quick Start

```bash
# Pull required models
ollama pull qwen3:4b          # orchestrator (fast coordinator)
ollama pull qwen3:8b          # reasoning specialist
ollama pull qwen2.5-coder:7b  # code specialist
ollama pull nomic-embed-text  # embeddings for semantic memory search

# Install and run
uv sync
uv run minion
```

## Configuration

Copy `.env.example` to `~/.minion/.env` and adjust as needed.

```env
MINION_ORCHESTRATOR_MODEL=qwen3:4b
MINION_DELEGATE_MODELS='{"reasoning":"qwen3:8b","code":"qwen2.5-coder:7b"}'
MINION_EMBED_MODEL=nomic-embed-text
```

## Key bindings

| Key | Action |
|---|---|
| `Enter` | Send message |
| `Ctrl+N` | Start a new session |
| `Ctrl+Q` | Quit |

## Memory

Minion stores four types of memories, all in `~/.minion/minion.db`:

| Type | Example | Decays? |
|---|---|---|
| `fact` | "User's name is Alex" | Never |
| `preference` | "Prefers terminal apps over web UIs" | Never |
| `project` | "Working on a Rust CLI called fenix" | After 90 days inactive |
| `context` | "Currently learning Neovim" | After 30 days inactive |

Recall uses hybrid search: FTS5 keyword matching + cosine vector similarity via `nomic-embed-text`. High-importance memories (importance ≥ 4) never decay. Backfill and decay run silently on startup.

## Session history

Minion automatically saves and restores conversation history. When you restart, your last session resumes. Start fresh with `Ctrl+N`.

## Model delegation

The orchestrator (`qwen3:4b`) coordinates and handles simple tasks. For complex work it delegates to specialists:

- `reasoning` → `qwen3:8b` — deep analysis, multi-step thinking, tradeoffs
- `code` → `qwen2.5-coder:7b` — writing, reviewing, debugging code

Add roles in `~/.minion/.env`:
```env
MINION_DELEGATE_MODELS='{"reasoning":"qwen3:8b","code":"qwen2.5-coder:7b","custom":"llama3.2:3b"}'
```

## Data

```
~/.minion/
├── .env          # local config overrides
└── minion.db     # SQLite: memories, embeddings, session history
```

Nothing is sent to the cloud unless you configure a cloud model provider.

## Backlog

- Live streaming from delegate to TUI (currently synchronous)
- `sqlite-vec` for O(1) vector search at scale (currently pure Python cosine)
- Obsidian integration
- `/commands` slash commands in TUI
- Cloud model provider support (OpenAI, Anthropic)

## License

MIT
