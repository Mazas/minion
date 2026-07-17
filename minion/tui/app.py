"""
minion/tui/app.py

The main Textual application. Owns the layout (chat history + input area +
status bar), handles user input, and drives the agent session.

Key design choices:
- Session is created asynchronously on mount (to support history loading).
- Streaming is done in a Textual worker (background task) so the UI stays
  responsive while the model generates.
- AssistantMessage is mounted empty and updated chunk-by-chunk.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.widget import Widget
from textual.widgets import Static, TextArea

from minion.agent.agent import create_agent
from minion.agent.session import Session, StreamEvent
from minion.config import Config
from minion.memory.manager import MemoryManager
from minion.tools.search import SearchProvider
from minion.tui.widgets import AssistantMessage, StatusBar, UserMessage

CSS_PATH = Path(__file__).parent / "app.tcss"

WELCOME = """\
**Minion** is ready. Type a message and press **Enter** to send.
**Ctrl+N** new session  **Ctrl+Q** quit
"""


class InputArea(Widget):
    """Prompt label + TextArea input row."""

    DEFAULT_CSS = """
    InputArea {
        height: auto;
        max-height: 10;
        padding: 1 1 0 1;
        layout: horizontal;
        background: $surface;
    }
    InputArea .prompt {
        color: $accent;
        text-style: bold;
        width: auto;
        margin-right: 1;
        padding-top: 0;
    }
    InputArea TextArea {
        width: 1fr;
        height: auto;
        max-height: 8;
        background: $surface;
        border: none;
        padding: 0;
    }
    InputArea TextArea:focus {
        border: none;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(">", classes="prompt")
        yield TextArea(id="user-input")


class MinionApp(App[None]):
    """The root Textual application."""

    CSS_PATH = str(CSS_PATH)

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+n", "new_session", "New session"),
    ]

    def __init__(self, config: Config, memory: MemoryManager, search: SearchProvider) -> None:
        super().__init__()
        self._config = config
        self._memory = memory
        self._search = search
        self._agent = create_agent(config, memory)
        self._session: Session | None = None
        self._busy = False

    # ── Layout ────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield ScrollableContainer(id="chat-history")
        yield InputArea(id="input-area")
        yield StatusBar(self._config.orchestrator_model)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def on_mount(self) -> None:
        self.query_one(StatusBar).set_status("loading...")
        self._session = await Session.create(
            self._agent, self._memory, self._search, resume=True
        )
        self._restore_or_welcome()
        self.query_one("#user-input", TextArea).focus()
        await self._refresh_status()
        # Silent startup tasks — run in background, don't block UI
        asyncio.create_task(self._memory.decay_stale())
        asyncio.create_task(self._memory.backfill_embeddings())

    def _restore_or_welcome(self) -> None:
        """Show previous messages if resuming, otherwise show welcome."""
        assert self._session is not None
        history = self._session._history
        if history:
            self._render_history(history)
        else:
            self.query_one("#chat-history").mount(AssistantMessage(WELCOME))

    def _render_history(self, history: list) -> None:
        """Re-render persisted messages into the chat pane."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        chat = self.query_one("#chat-history")
        for msg in history:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, UserPromptPart):
                        chat.mount(UserMessage(part.content))
            elif isinstance(msg, ModelResponse):
                text = "".join(
                    part.content for part in msg.parts if isinstance(part, TextPart)
                )
                if text:
                    chat.mount(AssistantMessage(text))
        chat.scroll_end(animate=False)

    async def _refresh_status(self) -> None:
        count = await self._memory.count()
        session_id = self._session.session_id if self._session else "?"
        self.query_one(StatusBar).set_info(f"{count} memories · session {session_id}")
        self.query_one(StatusBar).set_status("ready")

    # ── Actions ───────────────────────────────────────────────────────────

    async def action_new_session(self) -> None:
        """Start a fresh conversation session."""
        if self._busy:
            return
        self._session = await Session.create(
            self._agent, self._memory, self._search, resume=False
        )
        chat = self.query_one("#chat-history")
        await chat.remove_children()
        chat.mount(AssistantMessage(WELCOME))
        await self._refresh_status()
        self.query_one("#user-input", TextArea).focus()

    # ── Input handling ────────────────────────────────────────────────────

    @on(TextArea.Changed, "#user-input")
    def _on_input_changed(self, event: TextArea.Changed) -> None:
        text = event.text_area.text
        if text.endswith("\n"):
            clean = text.rstrip("\n")
            if clean.strip() and not self._busy and self._session is not None:
                event.text_area.clear()
                self._submit(clean.strip())

    def _submit(self, text: str) -> None:
        assert self._session is not None
        history = self.query_one("#chat-history")
        history.mount(UserMessage(text))
        response_widget = AssistantMessage()
        history.mount(response_widget)
        history.scroll_end(animate=False)
        self._busy = True
        self.query_one(StatusBar).set_status("thinking...")
        self._stream_response(text, response_widget)

    # ── Streaming worker ──────────────────────────────────────────────────

    @work(exclusive=True)
    async def _stream_response(self, text: str, widget: AssistantMessage) -> None:
        assert self._session is not None
        try:
            async for event in self._session.stream(text):
                if event.kind == "thinking":
                    widget.append_thinking(event.content)
                    self.query_one(StatusBar).set_status("responding...")
                elif event.kind == "tool":
                    self.query_one(StatusBar).set_status(event.content)
                else:
                    self.query_one(StatusBar).set_status("responding...")
                    widget.append_text(event.content)
                self.query_one("#chat-history").scroll_end(animate=False)
        except Exception as exc:
            widget.append_text(f"\n\n**Error:** {exc}")
        finally:
            self._busy = False
            await self._refresh_status()
            self.query_one("#user-input", TextArea).focus()
