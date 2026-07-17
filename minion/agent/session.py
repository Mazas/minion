"""
minion/agent/session.py

Manages a single conversation session: holds message history and provides
a clean interface for the TUI to send messages and receive streamed chunks.

Yields typed StreamEvent objects so the TUI can render thinking and text
differently — thinking appears immediately as dimmed text, text as normal.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ThinkingPart, TextPart

from minion.agent.agent import AgentDeps
from minion.memory.manager import MemoryManager
from minion.tools.search import SearchProvider


@dataclass
class StreamEvent:
    """A single streamed chunk from the agent, tagged by type."""
    kind: Literal["thinking", "text"]
    content: str


class Session:
    """
    Wraps an Agent with conversation history.

    PydanticAI agents are stateless — they accept a message_history list on
    each call. The Session owns that list so the TUI doesn't have to.
    """

    def __init__(
        self,
        agent: Agent[AgentDeps, str],
        memory: MemoryManager,
        search: SearchProvider,
    ) -> None:
        self._agent = agent
        self._memory = memory
        self._search = search
        self._history: list[ModelMessage] = []

    async def stream(self, user_message: str) -> AsyncIterator[StreamEvent]:
        """
        Send a user message and yield StreamEvents (thinking + text chunks).

        stream_response() yields full ModelResponse snapshots — each snapshot
        contains ALL content accumulated so far in the current model cycle.
        Each snapshot is a different object, so we cannot use id() to detect
        new cycles.

        Instead we track the last-seen content length per part type. When
        content shrinks (new cycle after a tool call), we reset offsets.
        """
        deps = AgentDeps(memory=self._memory, search=self._search)
        async with self._agent.run_stream(
            user_message,
            deps=deps,
            message_history=self._history,
        ) as streamed:
            seen_thinking = 0  # byte offset into thinking content seen so far
            seen_text = 0      # byte offset into text content seen so far

            async for response in streamed.stream_response(debounce_by=0.05):
                for part in response.parts:
                    if isinstance(part, ThinkingPart):
                        current_len = len(part.content)
                        if current_len < seen_thinking:
                            # New model cycle started — reset offset
                            seen_thinking = 0
                        delta = part.content[seen_thinking:]
                        if delta:
                            yield StreamEvent(kind="thinking", content=delta)
                        seen_thinking = current_len

                    elif isinstance(part, TextPart):
                        current_len = len(part.content)
                        if current_len < seen_text:
                            # New model cycle started — reset offset
                            seen_text = 0
                        delta = part.content[seen_text:]
                        if delta:
                            yield StreamEvent(kind="text", content=delta)
                        seen_text = current_len

            self._history.extend(streamed.new_messages())

    def clear(self) -> None:
        """Reset conversation history."""
        self._history = []

    @property
    def message_count(self) -> int:
        return len(self._history)
