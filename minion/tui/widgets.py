"""
minion/tui/widgets.py

Custom Textual widgets used by the chat interface.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Markdown, Static


class UserMessage(Widget):
    """A message bubble for user input."""

    DEFAULT_CSS = """
    UserMessage {
        layout: horizontal;
        height: auto;
        margin: 1 0;
        padding: 0 1;
    }
    UserMessage .label {
        color: $accent;
        text-style: bold;
        width: auto;
        margin-right: 1;
    }
    UserMessage .content {
        color: $text;
        width: 1fr;
    }
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def compose(self) -> ComposeResult:
        yield Static("You", classes="label")
        yield Static(self._text, classes="content")


class AssistantMessage(Widget):
    """
    A message bubble for assistant responses.
    Uses Markdown so code blocks, bold, etc. render correctly.
    """

    DEFAULT_CSS = """
    AssistantMessage {
        layout: horizontal;
        height: auto;
        margin: 1 0;
        padding: 0 1;
    }
    AssistantMessage .label {
        color: $success;
        text-style: bold;
        width: auto;
        margin-right: 1;
    }
    AssistantMessage .content {
        width: 1fr;
    }
    """

    def __init__(self, text: str = "") -> None:
        super().__init__()
        self._text = text

    def compose(self) -> ComposeResult:
        yield Static("M", classes="label")
        yield Markdown(self._text, classes="content")

    def append(self, chunk: str) -> None:
        """Stream in a new chunk by updating the Markdown widget."""
        self._text += chunk
        self.query_one(Markdown).update(self._text)


class StatusBar(Widget):
    """
    Bottom status bar showing model name and session info.
    Updated reactively as the session state changes.
    """

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        background: $panel;
        color: $text-muted;
        padding: 0 1;
        layout: horizontal;
    }
    StatusBar .model {
        width: 1fr;
    }
    StatusBar .info {
        width: auto;
        text-align: right;
    }
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self._model_name = model_name
        self._status = "ready"
        self._info = ""

    def compose(self) -> ComposeResult:
        yield Static(f" {self._model_name}", classes="model", id="status-model")
        yield Static("", classes="info", id="status-info")

    def _render_info(self) -> None:
        parts = [p for p in [self._info, self._status] if p]
        self.query_one("#status-info", Static).update("  ".join(parts) + " ")

    def set_status(self, status: str) -> None:
        self._status = status
        self._render_info()

    def set_info(self, info: str) -> None:
        self._info = info
        self._render_info()

    def set_model(self, model_name: str) -> None:
        self._model_name = model_name
        self.query_one("#status-model", Static).update(f" {model_name}")
