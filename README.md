# minion

A local-first personal AI assistant that lives in your terminal.

Minion runs a local LLM via Ollama, remembers things about you across sessions, and can use tools like web search, file access, and shell commands. It grows with you over time.

```
┌─────────────────────────────────────────────┐
│  minion                        qwen2.5:3b   │
├─────────────────────────────────────────────┤
│                                             │
│  You: I prefer terminal apps over web UIs   │
│                                             │
│  Minion: Got it — I'll keep that in mind.   │
│  [memory stored]                            │
│                                             │
├─────────────────────────────────────────────┤
│ > ________________________________________  │
└─────────────────────────────────────────────┘
```

## Goals

- **Local-first** — your data stays on your machine
- **Persistent memory** — remembers facts, preferences, and context across sessions
- **Tool-capable** — web search, filesystem, shell, git
- **Hackable** — small files, clear boundaries, replaceable components
- **Terminal-native** — proper TUI, not a web app

## Architecture

```
minion/
├── agent/      # PydanticAI agent + session management
├── llm/        # Ollama provider (cloud providers optional later)
├── memory/     # SQLite-backed memory store with FTS search
├── tools/      # web search, filesystem, shell, git
└── tui/        # Textual TUI (chat pane + input + status bar)
```

### Key decisions

- **[PydanticAI](https://ai.pydantic.dev/)** for the agent — type-safe, minimal, tools are plain Python functions
- **Ollama** via its OpenAI-compatible endpoint — easy model swaps, works offline
- **Textual** for the TUI — proper split-pane layout, markdown rendering, keyboard-driven
- **SQLite + FTS5** for memory — zero dependencies, single inspectable file, good enough recall for personal use
- **DuckDuckGo** for web search — no API key required, privacy-respecting

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally

## Quick Start

> Not yet implemented — see roadmap below.

```bash
# Pull a model
ollama pull qwen2.5:3b

# Install and run
uv sync
uv run minion
```

## Configuration

Copy `.env.example` to `~/.minion/.env` and adjust as needed.

```env
MINION_MODEL=qwen2.5:3b
MINION_OLLAMA_BASE_URL=http://localhost:11434
MINION_DATA_DIR=~/.minion
```

## Roadmap

- [x] Repository setup
- [ ] **Milestone 1** — Project scaffold + Textual TUI + basic Ollama chat
- [ ] **Milestone 2** — Persistent memory (SQLite + FTS5)
- [ ] **Milestone 3** — Web search tool (DuckDuckGo)
- [ ] **Milestone 4** — Filesystem + shell tools
- [ ] **Milestone 5** — Git tool + session history + TUI polish

## Data

All app data is stored in `~/.minion/`:

```
~/.minion/
├── .env          # local config overrides
├── minion.db     # SQLite database (memory + session history)
└── logs/         # optional debug logs
```

Nothing is sent to the cloud unless you explicitly configure a cloud model provider.

## License

MIT
