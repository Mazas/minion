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

    Renders two sections stacked vertically inside a body container:
    - Thinking block (dimmed, italic) — Qwen3's <think> reasoning
    - Response text (markdown-rendered) — the actual answer

    Thinking appears immediately as tokens arrive, so the user sees activity
    instead of a blank screen during the reasoning phase.
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
    AssistantMessage .body {
        width: 1fr;
        height: auto;
        layout: vertical;
    }
    AssistantMessage .thinking {
        color: $text-muted;
        text-style: italic;
        height: auto;
        margin-bottom: 1;
        padding-left: 1;
        border-left: solid $panel;
    }
    AssistantMessage .thinking.hidden {
        display: none;
    }
    AssistantMessage .response {
        height: auto;
        width: 1fr;
    }
    """

    def __init__(self, text: str = "") -> None:
        super().__init__()
        self._thinking = ""
        self._text = text
        self._thinking_id = f"thinking-{id(self)}"
        self._response_id = f"response-{id(self)}"

    def compose(self) -> ComposeResult:
        yield Static("M", classes="label")
        with Widget(classes="body"):
            # thinking starts hidden; shown as soon as first chunk arrives
            yield Static(
                "",
                classes="thinking hidden",
                id=self._thinking_id,
            )
            yield Markdown(self._text, classes="response", id=self._response_id)

    def append_thinking(self, chunk: str) -> None:
        """Stream in a thinking chunk — shown dimmed above the response."""
        self._thinking += chunk
        widget = self.query_one(f"#{self._thinking_id}", Static)
        # Remove hidden class on first chunk so it becomes visible
        widget.remove_class("hidden")
        widget.update(f"Thinking…\n{self._thinking}")

    def append_text(self, chunk: str) -> None:
        """Stream in a response chunk — shown as markdown."""
        self._text += chunk
        self.query_one(f"#{self._response_id}", Markdown).update(self._text)

    # Keep backward compat for any callers using the old API
    def append(self, chunk: str) -> None:
        self.append_text(chunk)


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
