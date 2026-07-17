"""
minion/tui/app.py

The main Textual application. Owns the layout (chat history + input area +
status bar), handles user input, and drives the agent session.

Key design choices:
- The app creates the Session once and reuses it for the lifetime of the process.
- Streaming is done in a Textual worker (background task) so the UI stays
  responsive while the model generates.
- AssistantMessage is mounted empty and updated chunk-by-chunk via widget.append().
"""

from __future__ import annotations

from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.widget import Widget
from textual.widgets import Static, TextArea

from minion.agent.agent import create_agent
from minion.agent.session import Session
from minion.config import Config
from minion.memory.manager import MemoryManager
from minion.tui.widgets import AssistantMessage, StatusBar, UserMessage

CSS_PATH = Path(__file__).parent / "app.tcss"

WELCOME = """\
**Minion** is ready. Type a message and press **Enter** to send.
Press **Ctrl+Q** to quit.
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
    ]

    def __init__(self, config: Config, memory: MemoryManager) -> None:
        super().__init__()
        self._config = config
        self._memory = memory
        self._agent = create_agent(config, memory)
        self._session = Session(self._agent, memory)
        self._busy = False

    # ── Layout ────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield ScrollableContainer(id="chat-history")
        yield InputArea(id="input-area")
        yield StatusBar(self._config.model)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def on_mount(self) -> None:
        self._post_welcome()
        self.query_one("#user-input", TextArea).focus()
        await self._refresh_memory_count()

    def _post_welcome(self) -> None:
        self.query_one("#chat-history").mount(AssistantMessage(WELCOME))

    async def _refresh_memory_count(self) -> None:
        count = await self._memory.count()
        self.query_one(StatusBar).set_info(f"{count} memories")

    # ── Input handling ────────────────────────────────────────────────────

    @on(TextArea.Changed, "#user-input")
    def _on_input_changed(self, event: TextArea.Changed) -> None:
        text = event.text_area.text
        if text.endswith("\n"):
            clean = text.rstrip("\n")
            if clean.strip() and not self._busy:
                event.text_area.clear()
                self._submit(clean.strip())

    def _submit(self, text: str) -> None:
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
        try:
            async for chunk in self._session.stream(text):
                widget.append(chunk)
                self.query_one("#chat-history").scroll_end(animate=False)
        except Exception as exc:
            widget.append(f"\n\n**Error:** {exc}")
        finally:
            self._busy = False
            self.query_one(StatusBar).set_status("ready")
            self.query_one("#user-input", TextArea).focus()
            await self._refresh_memory_count()
